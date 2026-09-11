"""End-to-end orchestrator.

Runs entirely on your machine: `yt-dlp` for the download, `faster-whisper` for
transcription, an LLM (OpenAI / DeepSeek / Gemini, selected by `LLM_PROVIDER`)
for highlight ranking, and `ffmpeg` / OpenCV for the vertical crop.
"""

from typing import Dict, List, Optional

from .clipper import crop_highlights
from .downloader import download_youtube
from .highlights import get_highlights
from .llm import call_llm
from .transcriber import transcribe


def _run(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    face_tracking: bool = True,
) -> Dict:
    source_path = download_youtube(youtube_url, fmt=download_format)

    transcript = transcribe(source_path, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[
        :num_clips
    ]
    print(
        f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates",
        flush=True,
    )

    shorts = crop_highlights(
        source_path, top, aspect_ratio=aspect_ratio, face_tracking=face_tracking
    )

    return {
        "mode": "local",
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def generate_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    face_tracking: bool = True,
) -> Dict:
    """Run the full pipeline and return a structured result.

    Args:
        youtube_url: source URL, ``file://`` URL, or local file path.
        num_clips: how many shorts to render.
        aspect_ratio: e.g. "9:16", "1:1".
        download_format: source resolution ("360" / "480" / "720" / "1080").
        language: ISO-639-1 to force Whisper language detection.
        face_tracking: True (default) tracks faces with OpenCV for the vertical
            crop; False uses a static centre crop.

    Returns:
        {
          "mode": "local",
          "source_video_url": str,   # local path to the source video
          "transcript": {...},
          "highlights": [...],       # all candidates ranked
          "shorts": [...],           # top `num_clips` with local clip paths
        }
    """
    return _run(
        youtube_url,
        num_clips,
        aspect_ratio,
        download_format,
        language,
        face_tracking=face_tracking,
    )


def generate_subtitles(
    input_path: str,
    language: Optional[str] = None,
) -> Dict:
    """Generate `.srt` subtitles only — no highlight ranking, no cropping.

    ``input_path`` may be a single video file or a directory of video files.
    Each video gets its own subtitle file written next to it with the same base
    name (e.g. ``video/talk.mkv`` → ``video/talk.srt``).

    Transcription runs locally via faster-whisper; the written `.srt` also
    serves as the transcript cache for that video.

    Returns a structured result:
        {
          "mode": "subtitles",
          "input": str,
          "results": [{"source_video", "subtitle_path", "segments", "duration"}, ...],
        }
    """
    from .subtitles import generate_subtitles_for_path

    return generate_subtitles_for_path(input_path, language=language)
