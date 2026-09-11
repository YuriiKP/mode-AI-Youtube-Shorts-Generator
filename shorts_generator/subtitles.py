"""Subtitle-only generation.

Transcribe a local video file (or every video in a directory) with
faster-whisper and write a `.srt` subtitle file next to each video, using the
same base name as the video file (e.g. ``clips/talk.mp4`` → ``clips/talk.srt``).

This bypasses the highlight ranking and cropping stages entirely — useful when
you only need subtitles for one video or a whole folder of videos.
"""

from pathlib import Path
from typing import Dict, List, Optional

# Extensions we treat as video files when scanning a directory. Kept broad on
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

    ``input_path`` may be either a single video file (returned as-is) or a
    directory (all video files directly inside it, sorted by name).
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


def generate_subtitles_for_path(
    input_path: str, language: Optional[str] = None
) -> Dict:
    """Transcribe one video (or every video in a directory) into `.srt` files.

    Each subtitle file is written next to its source video with the same base
    name, e.g. ``video/talk.mkv`` → ``video/talk.srt``. The `.srt` also doubles
    as the transcription cache, so re-running skips Whisper when the file is
    still newer than the source video.

    Returns a structured result:
        {
          "mode": "subtitles",
          "input": str,
          "results": [
            {
              "source_video": str,
              "subtitle_path": str | None,
              "segments": int,
              "duration": float,
              "error": str,          # only present on failure
            },
            ...
          ],
        }
    """
    from .transcriber import transcribe

    videos = find_video_files(input_path)
    if not videos:
        raise RuntimeError(f"No video files found in: {input_path}")

    print(f"[subtitles] {len(videos)} video file(s) to process", flush=True)

    results: List[Dict] = []
    for i, video in enumerate(videos, 1):
        print(f"[subtitles] {i}/{len(videos)}: {video}", flush=True)
        srt_path = subtitle_path_for(video)
        try:
            transcript = transcribe(video, language=language, cache_path=srt_path)
            if not transcript.get("segments"):
                raise RuntimeError("Whisper produced no segments for this file.")
            print(f"[subtitles] wrote {srt_path}", flush=True)
            results.append(
                {
                    "source_video": video,
                    "subtitle_path": srt_path,
                    "segments": len(transcript["segments"]),
                    "duration": transcript["duration"],
                }
            )
        except Exception as e:
            print(f"[subtitles] {i} failed: {e}", flush=True)
            results.append(
                {"source_video": video, "subtitle_path": None, "error": str(e)}
            )

    return {"mode": "subtitles", "input": input_path, "results": results}
