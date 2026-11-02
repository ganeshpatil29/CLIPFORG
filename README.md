# ClipForge — local long-video to Shorts app

ClipForge runs on your Windows computer and opens a private local page in your browser. It transcribes speech locally, proposes several transcript-based moments, lets you edit timestamps, and exports vertical MP4 clips with burned-in captions.

## Requirements

- Windows 10/11
- Python 3.11 or 3.12
- FFmpeg and FFprobe with the `subtitles` filter enabled (libass); the app can use the local `tools` folder or PATH
- Internet only for the initial Python package/model downloads; video processing is local

## Install

1. Install Python 3.11 from [python.org](https://www.python.org/downloads/windows/) and enable **Add Python to PATH**.
2. Install an FFmpeg build that includes libass. One option, if you use Windows Package Manager, is:

   ```powershell
   winget install Gyan.FFmpeg
   ```

   Close and reopen PowerShell after installation. Check that `ffmpeg -filters` lists `subtitles` and that `ffprobe -version` works. If you use the FFmpeg files bundled in `tools`, this step is already handled.
3. Double-click `setup.bat` once to install the app's Python packages. Or open PowerShell in this `ClipForge` folder and run:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python -m streamlit run app.py
   ```

4. Double-click `run.bat` to launch. Streamlit opens ClipForge in your browser. Paste a local video path (best for large videos) or choose a video file, select the spoken language and clip length, then click **Transcribe and find moments**. On first use, faster-whisper downloads the selected model. Allow time for this download.
5. Transcription runs in the background with progress, a rough remaining-time estimate, and a cancel option. Audio is extracted once to a compact 16 kHz mono FLAC file. Completed transcripts and clip selections are saved under `jobs/<job-id>/`, so they can be restored after a page refresh or reselecting the same source.
6. Review/adjust candidates and create a captioned vertical preview. Clip suggestions use transcript punctuation, opening hooks, and Whisper confidence as lightweight ranking signals; they remain suggestions for human review.
7. Choose center or face-aware framing, caption placement/size/word highlighting, and encoding settings, then click **Export selected Shorts**. Completed clips are kept when you retry an interrupted export. Output appears under `jobs/<job-id>/shorts/`.

If PowerShell blocks activation, run this once in the same PowerShell window and retry activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Notes

- `base` is the default transcription model. Fast mode uses a smaller beam for quicker results; larger beams may improve accuracy but take longer. The app detects a CTranslate2 CUDA device before offering NVIDIA GPU transcription. Model instances are cached for reuse while the app remains open, with only the last selected configuration kept resident in memory.
- NVIDIA NVENC export appears only when both a CUDA device and an FFmpeg NVENC encoder are detected. Otherwise, exports use CPU H.264 encoding.
- Face-aware framing samples a few frames per clip using OpenCV's built-in frontal-face cascade. It falls back to center framing if it cannot detect a face; this is a lightweight crop helper, not full person tracking.
- Suggested moments are selected using transcript content and timestamps. They are starting points for your review, not a promise of virality.
- Vertical output uses a centered crop. If the speaker is off-center, the crop may need manual adjustment in a video editor.
- FFmpeg must include the `subtitles` filter for burned captions. If rendering reports that the filter is unavailable, install a full FFmpeg build with libass and restart the app.
- Video files and generated clips stay in this project’s `jobs` folder. The Storage panel shows disk use and lets you select old jobs for deletion.
- This MVP exports files; it does not upload or publish to YouTube automatically.
- The browser uploader is configured for files up to 5 GB. For very large videos, entering the local file path avoids transferring the video through the browser uploader.
- Uploaded files are copied into the job folder once. Source identity is saved so a reselected file can recover its job; local paths avoid the browser transfer for large videos.
- Word-highlight captions use word timestamps from Whisper and are grouped into short readable chunks. The app also writes an SRT file alongside each ASS caption file for external editing.
