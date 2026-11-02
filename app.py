from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
JOBS_DIR = APP_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="ClipForge — Long video to Shorts", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner=False)
def runtime_state() -> dict:
    return {
        "model_cache": {},
        "model_lock": threading.Lock(),
        "job_threads": {},
        "job_cancel": {},
        "job_lock": threading.Lock(),
        "job_file_lock": threading.Lock(),
    }


_RUNTIME = runtime_state()


def find_binary(name: str) -> str | None:
    return (
        shutil.which(name)
        or shutil.which(f"{name}.exe")
        or (str(APP_DIR / "tools" / f"{name}.exe") if (APP_DIR / "tools" / f"{name}.exe").is_file() else None)
    )


def run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, check=True)


def probe_duration(path: Path, ffprobe: str) -> float:
    result = run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)])
    return float(json.loads(result.stdout)["format"]["duration"])


def get_whisper_model(model_name: str, device: str, cpu_threads: int):
    key = (model_name, device, cpu_threads)
    with _RUNTIME["model_lock"]:
        if key not in _RUNTIME["model_cache"]:
            # Keep only the most recently selected model/configuration resident in memory.
            _RUNTIME["model_cache"].clear()
            from faster_whisper import WhisperModel

            _RUNTIME["model_cache"][key] = WhisperModel(
                model_name,
                device=device,
                compute_type="float16" if device == "cuda" else "int8",
                cpu_threads=cpu_threads,
                num_workers=1,
            )
        return _RUNTIME["model_cache"][key]


def cuda_device_count() -> int:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def ffmpeg_has_encoder(ffmpeg: str, encoder: str) -> bool:
    try:
        result = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
        return encoder in result.stdout
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def cached_cuda_device_count() -> int:
    return cuda_device_count()


def upload_identity(uploaded) -> str:
    data = uploaded.getbuffer()
    sample = data[:65536].tobytes() + data[-65536:].tobytes()
    fingerprint = hashlib.sha256(sample).hexdigest()[:16]
    return f"upload:{Path(uploaded.name).name}:{uploaded.size}:{fingerprint}"


@st.cache_data(ttl=15, show_spinner=False)
def folder_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def human_size(size: int) -> str:
    amount = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"


def extract_audio(source: Path, output: Path, ffmpeg: str) -> None:
    temporary = output.with_suffix(".partial.flac")
    temporary.unlink(missing_ok=True)
    run([ffmpeg, "-y", "-i", str(source), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "flac", "-compression_level", "5", str(temporary)])
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("FFmpeg did not extract an audio track from this video.")
    temporary.replace(output)


def read_job(job_dir: Path) -> dict:
    try:
        return json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_json_atomic(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_job(job_dir: Path, **updates) -> dict:
    with _RUNTIME["job_file_lock"]:
        job = read_job(job_dir)
        job.update(updates)
        write_json_atomic(job_dir / "job.json", job)
        return job


def load_job_results(job_dir: Path) -> tuple[list[dict] | None, list[dict] | None]:
    try:
        segments = json.loads((job_dir / "transcript.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        segments = None
    try:
        clips = json.loads((job_dir / "clips.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        clips = None
    return segments, clips


def transcribe(path: Path, model_name: str, language: str, device: str, cpu_threads: int,
               beam_size: int, duration: float, cancel_event: threading.Event,
               report_progress) -> list[dict]:
    if cancel_event.is_set():
        raise InterruptedError("Transcription cancelled by user.")
    model = get_whisper_model(model_name, device, cpu_threads)
    if cancel_event.is_set():
        raise InterruptedError("Transcription cancelled by user.")
    kwargs = {"beam_size": beam_size, "vad_filter": True}
    language_codes = {
        "English": "en", "Hindi": "hi", "Marathi": "mr", "Tamil": "ta",
        "Telugu": "te", "Bengali": "bn", "Gujarati": "gu", "Kannada": "kn",
        "Malayalam": "ml", "Punjabi": "pa",
    }
    if language != "Auto-detect":
        kwargs["language"] = language_codes[language]
    segments, _ = model.transcribe(str(path), word_timestamps=True, **kwargs)
    result = []
    next_report = 0.0
    for segment in segments:
        if cancel_event.is_set():
            raise InterruptedError("Transcription cancelled by user.")
        if segment.text.strip():
            words = [
                {"start": float(word.start), "end": float(word.end), "text": word.word}
                for word in (segment.words or [])
                if word.word.strip()
            ]
            result.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "words": words,
                "avg_logprob": float(segment.avg_logprob),
                "no_speech_prob": float(segment.no_speech_prob),
            })
        progress = min(0.98, max(0.01, float(segment.end) / max(duration, 1)))
        if progress >= next_report:
            report_progress(progress)
            next_report = progress + 0.02
    report_progress(0.99)
    return result


def transcription_worker(job_id: str, job_dir: Path, source: Path, options: dict) -> None:
    cancel_event = _RUNTIME["job_cancel"][job_id]
    try:
        audio_path = job_dir / "audio.flac"
        if not audio_path.is_file() or audio_path.stat().st_size == 0:
            write_job(job_dir, status="running", stage="Extracting audio once", progress=0.01, error=None)
            extract_audio(source, audio_path, options["ffmpeg"])
        if cancel_event.is_set():
            raise InterruptedError("Transcription cancelled by user.")
        write_job(job_dir, status="running", stage="Loading transcription model", progress=0.03, error=None)

        def report(progress: float) -> None:
            write_job(job_dir, status="running", stage="Transcribing audio", progress=progress)

        segments = transcribe(
            audio_path,
            options["model"],
            options["language"],
            options["device"],
            options["cpu_threads"],
            options["beam_size"],
            options["duration"],
            cancel_event,
            report,
        )
        if cancel_event.is_set():
            raise InterruptedError("Transcription cancelled by user.")
        write_json_atomic(job_dir / "transcript.json", segments)
        clips = suggest_clips(segments, options["clip_length"], options["clip_count"])
        write_json_atomic(job_dir / "clips.json", clips)
        write_job(job_dir, status="complete", stage="Ready", progress=1.0, error=None)
    except InterruptedError as exc:
        write_job(job_dir, status="cancelled", stage="Cancelled", error=str(exc))
    except Exception as exc:
        write_job(job_dir, status="failed", stage="Failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        with _RUNTIME["job_lock"]:
            _RUNTIME["job_threads"].pop(job_id, None)
            _RUNTIME["job_cancel"].pop(job_id, None)


def start_transcription(job_id: str, job_dir: Path, source: Path, options: dict) -> bool:
    with _RUNTIME["job_lock"]:
        current = _RUNTIME["job_threads"].get(job_id)
        if current and current.is_alive():
            return False
        if any(thread.is_alive() for other_id, thread in _RUNTIME["job_threads"].items() if other_id != job_id):
            return False
        event = threading.Event()
        _RUNTIME["job_cancel"][job_id] = event
        thread = threading.Thread(
            target=transcription_worker,
            args=(job_id, job_dir, source, options),
            name=f"clipforge-{job_id}",
            daemon=True,
        )
        _RUNTIME["job_threads"][job_id] = thread
        thread.start()
        return True


def cancel_transcription(job_id: str) -> None:
    with _RUNTIME["job_lock"]:
        event = _RUNTIME["job_cancel"].get(job_id)
        if event:
            event.set()


def remove_old_job(job_dir: Path) -> None:
    resolved = job_dir.resolve()
    if resolved.parent != JOBS_DIR.resolve():
        raise ValueError("Refusing to remove a folder outside the jobs directory.")
    shutil.rmtree(resolved)


@st.fragment(run_every="1s")
def show_job_progress(job_id: str, job_dir: Path) -> None:
    job = read_job(job_dir)
    status = job.get("status", "ready")
    observed_key = f"_observed_job_status_{job_id}"
    previous_status = st.session_state.get(observed_key)
    if previous_status != status:
        st.session_state[observed_key] = status
        if previous_status == "running" and status in {"complete", "failed", "cancelled", "interrupted"}:
            st.rerun()
    if status == "running":
        stage = job.get("stage", "Working")
        progress = float(job.get("progress", 0.0))
        st.progress(min(1.0, max(0.0, progress)), text=f"{stage} · {progress:.0%}")
        started_at = float(job.get("started_at", 0.0))
        elapsed = max(0.0, time.time() - started_at) if started_at else 0.0
        if progress >= 0.05 and elapsed >= 5:
            remaining = int(elapsed * (1.0 - progress) / progress)
            hours, remainder = divmod(remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            st.caption(f"Approximate time remaining: {hours:02}:{minutes:02}:{seconds:02} · estimate improves as transcription progresses")
        else:
            st.caption("Estimating time from transcription progress…")
        if st.button("Cancel transcription", key=f"cancel_{job_id}"):
            cancel_transcription(job_id)
            st.info("Cancellation requested. The current audio segment will finish first.")
    elif status == "complete":
        st.success("Transcription complete. Review your saved clip suggestions below.")
    elif status == "failed":
        st.error(f"Transcription failed: {job.get('error', 'Unknown error')}")
    elif status == "cancelled":
        st.warning("Transcription was cancelled. You can restart it from the saved job.")
    elif status == "interrupted":
        st.warning("The app stopped during transcription. Start it again to retry; completed results remain saved.")


def candidate_score(text: str, avg_logprob: float = 0.0, no_speech_prob: float = 0.0) -> float:
    words = re.findall(r"[\w']+", text.lower())
    if not words:
        return 0.0
    hooks = {"but", "because", "secret", "mistake", "best", "worst", "never", "always", "why", "how", "imagine", "finally", "actually", "important", "problem", "solution", "learned", "surprisingly", "instead", "truth"}
    fillers = {"um", "uh", "erm", "hmm", "like", "you", "know"}
    score = min(len(words) / 32, 1.35)
    score += min(sum(w in hooks for w in words) * 0.16, 0.8)
    score += min(sum(w in hooks for w in words[:8]) * 0.24, 0.72)
    score += min(text.count("?") * 0.22 + text.count("!") * 0.16, 0.65)
    score += min(sum(ch.isdigit() for ch in text) * 0.025, 0.3)
    score += 0.25 if re.search(r"[.!?][\"'’)]?\s*$", text) else 0.0
    score -= min(sum(w in fillers for w in words) * 0.08, 0.4)
    score += max(-0.5, min(0.3, (avg_logprob + 0.5) * 0.35))
    score -= min(max(0.0, no_speech_prob - 0.2) * 0.8, 0.5)
    return score


def suggest_clips(segments: list[dict], target_seconds: int, count: int) -> list[dict]:
    if not segments:
        return []
    # Build rolling windows from transcript segments, keeping cuts near speech boundaries.
    candidates = []
    cursor = 0
    while cursor < len(segments):
        start_idx = cursor
        while start_idx > 0 and segments[start_idx - 1]["end"] > segments[cursor]["start"] - 2.0:
            previous_text = segments[start_idx - 1]["text"].rstrip()
            if re.search(r"[.!?][\"'’)]?$", previous_text):
                break
            start_idx -= 1
        start = segments[start_idx]["start"]
        end_idx = cursor
        while end_idx + 1 < len(segments) and segments[end_idx]["end"] - start < target_seconds:
            end_idx += 1
        # Extend a little to end on a sentence boundary when the speaker is finishing a thought.
        extension_count = 0
        while end_idx + 1 < len(segments) and extension_count < 3 and segments[end_idx]["end"] - start < target_seconds * 1.15:
            if re.search(r"[.!?][\"'’)]?$", segments[end_idx]["text"].rstrip()):
                break
            end_idx += 1
            extension_count += 1
        end = segments[end_idx]["end"]
        if end - start >= min(12, target_seconds * 0.55):
            text = " ".join(s["text"] for s in segments[start_idx:end_idx + 1])
            selected_segments = segments[start_idx:end_idx + 1]
            avg_logprob = sum(s.get("avg_logprob", 0.0) for s in selected_segments) / len(selected_segments)
            no_speech = sum(s.get("no_speech_prob", 0.0) for s in selected_segments) / len(selected_segments)
            candidates.append({"start": max(0.0, start - 0.5), "end": end + 0.5, "text": text,
                               "score": candidate_score(text, avg_logprob, no_speech)})
        # Advance about half a window to create alternatives, then suppress overlaps below.
        next_cursor = cursor + 1
        while next_cursor < len(segments) and segments[next_cursor]["start"] < start + target_seconds * 0.55:
            next_cursor += 1
        cursor = max(cursor + 1, next_cursor)
    chosen = []
    for item in sorted(candidates, key=lambda x: x["score"], reverse=True):
        if all(item["end"] <= prior["start"] or item["start"] >= prior["end"] for prior in chosen):
            chosen.append(item)
        if len(chosen) >= count:
            break
    chosen.sort(key=lambda x: x["start"])
    for i, item in enumerate(chosen, 1):
        item["title"] = " ".join(item["text"].split()[:8]).strip(".,!?;:") or f"Clip {i}"
        item["keep"] = True
    return chosen


def srt_time(seconds: float) -> str:
    millis = int(max(0, seconds) * 1000)
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def write_srt(path: Path, segments: list[dict], start: float, end: float) -> None:
    lines = []
    index = 1
    for segment in segments:
        a, b = max(start, segment["start"]), min(end, segment["end"])
        if b <= a:
            continue
        caption = re.sub(r"\s+", " ", segment["text"]).strip()
        lines.extend([str(index), f"{srt_time(a - start)} --> {srt_time(b - start)}", caption, ""])
        index += 1
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def ass_time(seconds: float) -> str:
    centiseconds = int(max(0, seconds) * 100)
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02}:{secs:02}.{cs:02}"


def write_ass(path: Path, segments: list[dict], start: float, end: float,
              font_size: int, caption_position: str, word_highlight: bool) -> None:
    alignment, margin_v = {
        "Bottom (safe area)": (2, 192),
        "Lower third": (2, 360),
        "Center": (5, 0),
    }[caption_position]
    header = [
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920", "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: Default,Arial,{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,1,{alignment},60,60,{margin_v},1", "",
        "[Events]", "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    events = []
    for segment in segments:
        if segment.get("words"):
            words = segment["words"]
        else:
            tokens = segment["text"].split()
            span = max(0.05, segment["end"] - segment["start"])
            step = span / max(1, len(tokens))
            words = [{"start": segment["start"] + i * step, "end": segment["start"] + (i + 1) * step, "text": token}
                     for i, token in enumerate(tokens)]
        timed_words = []
        for word in words:
            word_start = max(start, float(word["start"]))
            word_end = min(end, float(word["end"]))
            text = str(word["text"]).replace("{", "").replace("}", "").replace("\\", "").strip()
            if text and word_end > word_start:
                timed_words.append({"start": word_start, "end": word_end, "text": text})
        chunks = []
        current = []
        current_chars = 0
        for word in timed_words:
            if current and (len(current) >= 6 or current_chars + len(word["text"]) + 1 > 34):
                chunks.append(current)
                current, current_chars = [], 0
            current.append(word)
            current_chars += len(word["text"]) + (1 if current_chars else 0)
        if current:
            chunks.append(current)
        for chunk in chunks:
            start_time = max(start, chunk[0]["start"])
            end_time = min(end, chunk[-1]["end"])
            parts = []
            for word in chunk:
                duration_cs = max(1, int((word["end"] - word["start"]) * 100))
                karaoke_tag = f"{{\\kf{duration_cs}}}" if word_highlight else ""
                parts.append(f"{karaoke_tag}{word['text']}")
            text = " ".join(parts)
            events.append(f"Dialogue: 0,{ass_time(start_time - start)},{ass_time(end_time - start)},Default,,0,0,0,,{text}")
    path.write_text("\n".join(header + events) + "\n", encoding="utf-8")


def render_clip(source: Path, output: Path, subtitle: Path, start: float, end: float, ffmpeg: str,
                encoder: str, preset: str, quality: int, crop_center: float) -> None:
    # Quote/escape the subtitle path for FFmpeg's filter syntax, including Windows drive letters.
    subtitle_path = str(subtitle.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    crop_x = f"max(0\\,min(iw-1080\\,(iw-1080)*{crop_center:.4f}))"
    vf = f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:x={crop_x},subtitles='{subtitle_path}'"
    command = [ffmpeg, "-y", "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end-start:.3f}", "-vf", vf,
               "-c:v", encoder]
    if encoder == "h264_nvenc":
        command += ["-preset", preset, "-cq", str(quality), "-b:v", "0"]
    else:
        command += ["-preset", preset, "-crf", str(quality)]
    command += ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(output)]
    run(command)


def render_preview(source: Path, output: Path, subtitle: Path, start: float, duration: float,
                   ffmpeg: str, crop_center: float) -> None:
    subtitle_path = str(subtitle.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    crop_x = f"max(0\\,min(iw-540\\,(iw-540)*{crop_center:.4f}))"
    vf = f"scale=540:960:force_original_aspect_ratio=increase,crop=540:960:x={crop_x},subtitles='{subtitle_path}'"
    command = [ffmpeg, "-y", "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
               "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32", "-an", "-movflags", "+faststart", str(output)]
    run(command)


def detect_face_crop_center(source: Path, start: float, end: float) -> float | None:
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the source video for face detection.")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            return 0.5
        aspect = width / height
        target_aspect = 9 / 16
        if aspect <= target_aspect:
            return 0.5
        crop_width_ratio = target_aspect / aspect
        classifier_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        classifier = cv2.CascadeClassifier(classifier_path)
        if classifier.empty():
            raise RuntimeError("OpenCV face-detection data could not be loaded.")
        centers = []
        sample_count = max(3, min(12, int(max(1.0, end - start) / 2.0)))
        for sample in range(sample_count):
            timestamp = start + (end - start) * (sample + 0.5) / sample_count
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            frame_width = frame.shape[1]
            resized_width = min(480, frame_width)
            if frame_width > resized_width:
                frame = cv2.resize(frame, (resized_width, max(1, int(frame.shape[0] * resized_width / frame_width))))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
            if len(faces):
                x, y, face_width, face_height = max(faces, key=lambda face: face[2] * face[3])
                face_center = (x + face_width / 2) / frame.shape[1]
                centers.append(max(0.0, min(1.0, (face_center - crop_width_ratio / 2) / (1 - crop_width_ratio))))
        return statistics.median(centers) if centers else None
    finally:
        capture.release()


def crop_center_for_clip(job_dir: Path, source: Path, clip: dict, mode: str) -> tuple[float, bool]:
    if mode != "Face-aware":
        return 0.5, False
    crop_key = hashlib.sha256(f"{clip['start']:.2f}:{clip['end']:.2f}".encode("utf-8")).hexdigest()[:12]
    job = read_job(job_dir)
    centers = job.get("crop_centers", {})
    if crop_key in centers:
        return float(centers[crop_key]), crop_key in job.get("crop_fallbacks", [])
    try:
        center = detect_face_crop_center(source, float(clip["start"]), float(clip["end"]))
    except Exception:
        center = None
    fallbacks = job.get("crop_fallbacks", [])
    fallback = center is None
    if fallback:
        center = 0.5
        fallbacks = sorted(set(fallbacks + [crop_key]))
    centers[crop_key] = center
    write_job(job_dir, crop_centers=centers, crop_fallbacks=fallbacks)
    return center, fallback


st.title("🎬 ClipForge")
st.caption("Turn one long video into a batch of reviewable, vertical Shorts. Files stay on this computer.")

ffmpeg, ffprobe = find_binary("ffmpeg"), find_binary("ffprobe")
if not ffmpeg or not ffprobe:
    st.error("FFmpeg is not installed or not on PATH. Install it using the setup guide, then restart this app.")
    st.stop()

local_path = st.text_input("Video path on this computer (recommended for large files)", placeholder=r"C:\Videos\my_long_video.mp4")
uploaded = st.file_uploader(
    "Or choose a video file (up to 5 GB; large uploads use extra memory)",
    type=["mp4", "mov", "mkv", "webm", "m4v", "avi"],
    max_upload_size=5120,
)
source = None
is_uploaded = False
source_key = None
if local_path.strip():
    candidate_path = Path(local_path.strip().strip('"')).expanduser()
    if candidate_path.is_file():
        source = candidate_path.resolve()
        source_key = f"local:{source}:{source.stat().st_size}:{source.stat().st_mtime}"
    else:
        st.warning("That video path was not found. Check the full path and try again.")
elif uploaded:
    is_uploaded = True
    source_key = upload_identity(uploaded)
    if uploaded.size >= 1024**3:
        st.info("For large videos, paste the file path above to avoid the browser upload copy. This upload is saved once and can resume from its saved job.")

if source_key:
    if st.session_state.get("source_key") != source_key:
        st.session_state["source_key"] = source_key
        st.session_state.pop("segments", None)
        st.session_state.pop("clips", None)
        st.session_state.pop("duration", None)
        st.session_state.pop("duration_source_key", None)
        st.session_state.pop("_loaded_job_id", None)
        st.session_state.pop("_loaded_job_state", None)
    job_id = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:12]
    st.session_state["job_id"] = job_id
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    if is_uploaded:
        source = job_dir / Path(uploaded.name).name
        if not source.is_file() or source.stat().st_size != uploaded.size:
            temporary_source = source.with_name(source.name + ".uploading")
            with temporary_source.open("wb") as output_file:
                output_file.write(uploaded.getbuffer())
            temporary_source.replace(source)

    try:
        job = read_job(job_dir)
        if st.session_state.get("duration_source_key") == source_key:
            duration = st.session_state["duration"]
        elif job.get("duration"):
            duration = float(job["duration"])
        else:
            with st.spinner("Reading video metadata…"):
                duration = probe_duration(source, ffprobe)
        st.session_state["duration"] = duration
        st.session_state["duration_source_key"] = source_key
        write_job(job_dir, source_key=source_key, source_name=source.name, duration=duration)
    except Exception as exc:
        st.error(f"Could not read this video: {exc}")
        st.stop()

    with _RUNTIME["job_lock"]:
        worker = _RUNTIME["job_threads"].get(job_id)
        worker_running = bool(worker and worker.is_alive())
    job = read_job(job_dir)
    if job.get("status") == "running" and not worker_running:
        job = write_job(job_dir, status="interrupted", stage="Interrupted", error="The app restarted during this job.")

    if st.session_state.get("_loaded_job_id") != job_id or st.session_state.get("_loaded_job_state") != job.get("status"):
        saved_segments, saved_clips = load_job_results(job_dir)
        if saved_segments is not None:
            st.session_state["segments"] = saved_segments
            st.session_state["clips"] = saved_clips or []
        elif st.session_state.get("_loaded_job_id") != job_id:
            st.session_state.pop("segments", None)
            st.session_state.pop("clips", None)
        st.session_state["_loaded_job_id"] = job_id
        st.session_state["_loaded_job_state"] = job.get("status")

    st.success(f"Loaded **{source.name}** · {duration/60:.1f} minutes · saved locally")
    cuda_count = cached_cuda_device_count()
    with st.expander("Transcription and clip settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        model_name = col1.selectbox("Transcription model", ["base", "small", "medium"], index=0, help="Base is quicker on most CPUs; larger models may improve accuracy but take longer.")
        language = col2.selectbox("Spoken language", ["Auto-detect", "English", "Hindi", "Marathi", "Tamil", "Telugu", "Bengali", "Gujarati", "Kannada", "Malayalam", "Punjabi"])
        clip_length = col3.slider("Target clip length (seconds)", 20, 90, 45, 5)
        clip_count = st.slider("Number of clips to suggest", 1, 20, 8)
        mode_col1, mode_col2 = st.columns(2)
        device_label = mode_col1.selectbox("Transcription device", ["CPU", "NVIDIA GPU"] if cuda_count else ["CPU"], help="GPU mode appears only when CTranslate2 detects a CUDA-capable NVIDIA GPU.")
        speed_label = mode_col2.selectbox("Transcription speed", ["Fast", "Balanced", "More accurate"], index=0, help="Fast uses a smaller beam and can be slightly less accurate.")
        cpu_threads = st.slider("CPU threads", 1, max(2, min(os.cpu_count() or 4, 16)), min(os.cpu_count() or 4, 8), help="Lower this if you want to keep the computer responsive during transcription.")

    if st.button("1 · Transcribe and find moments", type="primary", use_container_width=True,
                 disabled=job.get("status") == "running"):
        options = {
            "model": model_name,
            "language": language,
            "device": "cuda" if device_label == "NVIDIA GPU" else "cpu",
            "cpu_threads": cpu_threads,
            "beam_size": {"Fast": 1, "Balanced": 3, "More accurate": 5}[speed_label],
            "duration": duration,
            "clip_length": clip_length,
            "clip_count": clip_count,
            "ffmpeg": ffmpeg,
        }
        write_job(job_dir, status="running", stage="Starting", progress=0.0, started_at=time.time(), options=options, error=None)
        if start_transcription(job_id, job_dir, source, options):
            st.session_state["_loaded_job_state"] = "running"
            st.rerun()
        else:
            st.error("Another transcription job is already running. Wait for it or cancel it before starting another.")

    if read_job(job_dir).get("status") in {"running", "failed", "cancelled", "interrupted"}:
        show_job_progress(job_id, job_dir)

    segments = st.session_state.get("segments")
    clips = st.session_state.get("clips")
    if segments is not None and job.get("status") != "running":
        st.subheader(f"2 · Review suggestions ({len(clips or [])})")
        if not clips:
            st.warning("No candidate moments were found. Try a shorter target length or check that the video has clear speech.")
        st.subheader("3 · Export settings")
        with st.expander("Captions, framing, and video quality", expanded=False):
            export_col1, export_col2, export_col3 = st.columns(3)
            font_size = export_col1.slider("Caption size", 28, 72, 48, 2)
            caption_position = export_col2.selectbox("Caption position", ["Bottom (safe area)", "Lower third", "Center"])
            export_mode = export_col3.selectbox("Encoding speed", ["Fastest", "Balanced", "Higher quality"], index=1)
            caption_row = st.columns(3)
            word_highlight = caption_row[0].checkbox("Highlight words as spoken", value=True)
            crop_mode = caption_row[1].selectbox("Vertical framing", ["Center", "Face-aware"])
            quality = caption_row[2].slider("Video quality (lower number = better quality)", 18, 28, 20)
            st.caption("Final Shorts export at Full HD 1080 × 1920. Source videos below 1080p are upscaled; upscaling cannot restore missing detail.")
        cpu_preset = {"Fastest": "ultrafast", "Balanced": "veryfast", "Higher quality": "fast"}[export_mode]
        nvenc_preset = {"Fastest": "p1", "Balanced": "p4", "Higher quality": "p6"}[export_mode]
        nvenc_available = cuda_count > 0 and ffmpeg_has_encoder(ffmpeg, "h264_nvenc")
        encoder_options = ["CPU (H.264)"] + (["NVIDIA GPU (NVENC)"] if nvenc_available else [])
        encoder_label = st.selectbox("Export encoder", encoder_options, help="NVIDIA encoding is shown only when both CUDA and FFmpeg NVENC support are detected.")
        encoder = "h264_nvenc" if encoder_label.startswith("NVIDIA") else "libx264"
        preset = nvenc_preset if encoder == "h264_nvenc" else cpu_preset
        if crop_mode == "Face-aware":
            st.caption("Face-aware framing samples a few frames and uses the most prominent detected face; it falls back to center crop when no face is found.")
        for i, clip in enumerate(clips or []):
            with st.container(border=True):
                st.markdown(f"**Clip {i+1}: {clip['title']}** · suggestion score {clip['score']:.1f}")
                st.caption(clip["text"][:500] + ("…" if len(clip["text"]) > 500 else ""))
                c1, c2, c3 = st.columns([1, 1, 1])
                clip["keep"] = c1.checkbox("Export this clip", value=clip.get("keep", True), key=f"keep_{job_id}_{i}")
                clip["start"] = c2.number_input("Start (seconds)", min_value=0.0, max_value=float(duration), value=float(clip["start"]), step=0.5, key=f"start_{job_id}_{i}")
                clip["end"] = c3.number_input("End (seconds)", min_value=0.0, max_value=float(duration), value=min(float(duration), float(clip["end"])), step=0.5, key=f"end_{job_id}_{i}")
                preview_dir = job_dir / "previews"
                preview_key = hashlib.sha256(f"{clip['start']:.2f}:{clip['end']:.2f}:{font_size}:{caption_position}:{word_highlight}:{crop_mode}".encode("utf-8")).hexdigest()[:8]
                preview_path = preview_dir / f"preview_{i + 1:02}_{preview_key}.mp4"
                if st.button("Create 6-second vertical preview", key=f"preview_{job_id}_{i}_{preview_key}"):
                    preview_dir.mkdir(exist_ok=True)
                    try:
                        crop_center, face_fallback = crop_center_for_clip(job_dir, source, clip, crop_mode)
                        preview_ass = preview_dir / f"preview_{i + 1:02}_{preview_key}.ass"
                        write_ass(preview_ass, segments, clip["start"], min(clip["end"], clip["start"] + 6.0),
                                  font_size, caption_position, word_highlight)
                        render_preview(source, preview_path, preview_ass, clip["start"],
                                       min(6.0, clip["end"] - clip["start"]), ffmpeg, crop_center)
                        st.session_state[f"preview_ready_{job_id}_{i}"] = str(preview_path)
                        st.session_state[f"preview_fallback_{job_id}_{i}"] = face_fallback
                    except Exception as exc:
                        st.error(f"Could not create preview: {exc}")
                if st.session_state.get(f"preview_ready_{job_id}_{i}") == str(preview_path) and preview_path.is_file():
                    if st.session_state.get(f"preview_fallback_{job_id}_{i}"):
                        st.caption("No face was detected for this clip; center crop was used.")
                    st.video(str(preview_path))
        write_json_atomic(job_dir / "clips.json", clips or [])

        export_signature = hashlib.sha256(json.dumps({
            "clips": [(round(c["start"], 2), round(c["end"], 2)) for c in (clips or []) if c.get("keep")],
            "font_size": font_size,
            "caption_position": caption_position,
            "word_highlight": word_highlight,
            "crop_mode": crop_mode,
            "encoder": encoder,
            "preset": preset,
            "quality": quality,
        }, sort_keys=True).encode("utf-8")).hexdigest()[:8]

        if st.button("3 · Export selected Shorts", type="primary", use_container_width=True):
            selected = [c for c in clips or [] if c.get("keep") and c["end"] > c["start"]]
            if not selected:
                st.warning("Select at least one valid clip.")
            else:
                output_dir = job_dir / "shorts"
                output_dir.mkdir(exist_ok=True)
                progress = st.progress(0, text="Preparing exports…")
                try:
                    completed = 0
                    for n, clip in enumerate(selected, 1):
                        base = f"short_{n:02}_{int(clip['start'])}s_{export_signature}"
                        subtitle = output_dir / f"{base}.srt"
                        subtitle_ass = output_dir / f"{base}.ass"
                        video_out = output_dir / f"{base}.mp4"
                        if video_out.is_file() and video_out.stat().st_size > 0:
                            completed += 1
                            progress.progress(n / len(selected), text=f"Already complete: {n} of {len(selected)}")
                            continue
                        temporary_video = output_dir / f"{base}.partial.mp4"
                        temporary_video.unlink(missing_ok=True)
                        write_srt(subtitle, segments, clip["start"], clip["end"])
                        write_ass(subtitle_ass, segments, clip["start"], clip["end"],
                                  font_size, caption_position, word_highlight)
                        crop_center, face_fallback = crop_center_for_clip(job_dir, source, clip, crop_mode)
                        if face_fallback:
                            st.info(f"No face detected in clip {n}; using center crop for this clip.")
                        write_job(job_dir, export_status="running", export_progress=(n - 1) / len(selected), export_stage=f"Rendering {n} of {len(selected)}")
                        with st.spinner(f"Rendering {n} of {len(selected)}…"):
                            render_clip(source, temporary_video, subtitle_ass, clip["start"], clip["end"], ffmpeg,
                                        encoder, preset, quality, crop_center)
                        if not temporary_video.is_file() or temporary_video.stat().st_size == 0:
                            raise RuntimeError(f"FFmpeg did not produce a valid output for {base}.")
                        temporary_video.replace(video_out)
                        completed += 1
                        write_job(job_dir, export_progress=n / len(selected), export_stage=f"Rendered {n} of {len(selected)}")
                        progress.progress(n / len(selected), text=f"Rendered {n} of {len(selected)}")
                    write_job(job_dir, export_status="complete", export_progress=1.0, export_stage="Ready")
                    st.success(f"Export complete: {completed} vertical clips saved in `{output_dir}`")
                    st.info("Completed clips are kept if you refresh or retry this job. Upload the MP4 files manually using YouTube Studio.")
                except Exception as exc:
                    write_job(job_dir, export_status="failed", export_error=f"{type(exc).__name__}: {exc}")
                    st.error("Export failed. Check the FFmpeg error below; completed clips are preserved so you can retry.")
                    st.code(str(exc))
        output_dir = job_dir / "shorts"
        for video in sorted(output_dir.glob("*.mp4")) if output_dir.exists() else []:
            st.video(str(video))
            st.download_button(f"Download {video.name}", video.read_bytes(), file_name=video.name, mime="video/mp4", key=f"download_{job_id}_{video.name}")

st.sidebar.header("Storage")
all_jobs = [path for path in JOBS_DIR.iterdir() if path.is_dir()]
total_bytes = sum(folder_size(path) for path in all_jobs)
st.sidebar.caption(f"ClipForge jobs use **{human_size(total_bytes)}** across {len(all_jobs)} folders.")
current_job = st.session_state.get("job_id")
old_jobs = [path for path in all_jobs if path.name != current_job]
if old_jobs:
    with st.sidebar.expander("Remove old jobs", expanded=False):
        selected_jobs = []
        for old_job in sorted(old_jobs, key=lambda path: path.stat().st_mtime, reverse=True):
            details = read_job(old_job)
            label = details.get("source_name", old_job.name)
            if details.get("status") == "running":
                st.caption(f"{label} · currently processing; cannot remove yet")
            elif st.checkbox(f"{label} · {human_size(folder_size(old_job))}", key=f"delete_job_{old_job.name}"):
                selected_jobs.append(old_job)
        confirm_removal = st.checkbox("I understand selected jobs and their clips will be deleted", key="confirm_job_removal")
        if st.button("Delete selected jobs", disabled=not (selected_jobs and confirm_removal)):
            try:
                for old_job in selected_jobs:
                    remove_old_job(old_job)
                st.success(f"Removed {len(selected_jobs)} old job folder(s).")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not remove selected job: {exc}")

st.divider()
st.caption("Clip suggestions are transcript-based starting points, not guaranteed viral picks. Review the framing, captions, and context before publishing.")
