"""Subtitle generation — Whisper transcription into ``.srt`` files.

Transcribe a local video file (or every video in a folder) with faster-whisper
and write a ``.srt`` next to each video, using the same base name
(``video/talk.mkv`` → ``video/talk.srt``). This is what the ``transcribe``
command runs, and the written ``.srt`` doubles as the transcript cache used by
the subtitle burn-in stage.
"""

from pathlib import Path
from typing import Dict, List, Optional

from .config import Settings

# Extensions treated as video files when scanning a directory. Kept broad on
# purpose so common containers are picked up without extra configuration.
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
    ".m4v",
    ".flv",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".ts",
    ".m2ts",
    ".3gp",
    ".ogv",
}


def find_video_files(input_path: str) -> List[str]:
    """Return the video files to process.

    ``input_path`` may be a single video file (returned as-is) or a directory
    (all video files directly inside it, sorted by name).
    """
    path = Path(input_path).expanduser()

    if path.is_file():
        return [str(path)]

    if path.is_dir():
        return sorted(
            str(child)
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS
        )

    raise RuntimeError(f"Input path does not exist: {input_path}")


def subtitle_path_for(media_path: str) -> str:
    """Return the ``.srt`` path next to a video, sharing its base name."""
    return str(Path(media_path).with_suffix(".srt"))


def generate_subtitles(
    settings: Settings,
    input_path: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict:
    """Transcribe one video (or every video in a folder) into ``.srt`` files.

    Each subtitle file is written next to its source video with the same base
    name. The ``.srt`` also serves as the transcription cache, so re-running
    skips Whisper while the file is still newer than the source.

    Returns a structured result::

        {
          "mode": "transcribe",
          "input": str,
          "results": [
            {"source_video", "subtitle_path", "segments", "duration"},
            {"source_video", "subtitle_path": None, "error"},
            ...
          ],
        }
    """
    from .transcriber import transcribe

    source = input_path or settings.input
    if not source:
        raise RuntimeError("No input given. Set INPUT in .env or pass -i/--input.")

    videos = find_video_files(settings.resolve(source))
    if not videos:
        raise RuntimeError(f"No video files found in: {source}")

    print(f"[transcribe] {len(videos)} video file(s) to process", flush=True)

    results: List[Dict] = []
    for i, video in enumerate(videos, 1):
        print(f"[transcribe] {i}/{len(videos)}: {video}", flush=True)
        srt_path = subtitle_path_for(video)
        try:
            transcript = transcribe(
                video, settings, language=language, cache_path=srt_path
            )
            if not transcript.get("segments"):
                raise RuntimeError("Whisper produced no segments for this file.")
            print(f"[transcribe] wrote {srt_path}", flush=True)
            results.append(
                {
                    "source_video": video,
                    "subtitle_path": srt_path,
                    "segments": len(transcript["segments"]),
                    "duration": transcript["duration"],
                }
            )
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"[transcribe] {i} failed: {e}", flush=True)
            results.append(
                {"source_video": video, "subtitle_path": None, "error": str(e)}
            )

    return {"mode": "transcribe", "input": source, "results": results}
