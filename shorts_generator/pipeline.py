"""End-to-end orchestrator.

Runs entirely on your machine: `yt-dlp` for the download, `faster-whisper` for
transcription, an LLM (OpenAI / DeepSeek / Gemini, selected by `LLM_PROVIDER`)
for highlight ranking, and `ffmpeg` / OpenCV for the vertical crop.

Everything is driven by a single :class:`~shorts_generator.config.Settings`
object, which is loaded from one ``.env`` file (see ``config.load_settings``).
Pass ``enhance=True`` to :func:`generate_shorts` to also add background music
and burned-in subtitles to every rendered short (the ``all`` command).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from .clipper import crop_highlights
from .config import Settings
from .downloader import download_youtube
from .enhance import enhance_shorts
from .highlights import get_highlights
from .subtitles import find_video_files
from .transcriber import transcribe


def _run(
    settings: Settings,
    enhance: bool = False,
    add_music: bool = True,
    burn_subtitles: bool = True,
) -> Dict:
    if not settings.input:
        raise RuntimeError("No input given. Set INPUT in .env or pass -i/--input.")

    source_path = download_youtube(
        settings.input,
        fmt=settings.download_format,
        out_dir=settings.output_dir,
    )

    transcript = transcribe(source_path, settings)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(
        transcript, settings, num_clips=settings.num_clips
    )
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[
        : settings.num_clips
    ]
    print(
        f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates",
        flush=True,
    )

    shorts = crop_highlights(
        source_path,
        top,
        aspect_ratio=settings.aspect_ratio,
        out_dir=settings.output_dir,
        face_tracking=settings.face_tracking,
    )

    # Optional post-processing: add background music and burned-in subtitles to
    # every rendered short (uses the built-in post-processing engine).
    if enhance:
        enhance_shorts(
            transcript,
            shorts,
            settings,
            add_music=add_music,
            burn_subtitles=burn_subtitles,
        )

    return {
        "mode": "local",
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def generate_shorts(
    settings: Settings,
    enhance: bool = False,
    *,
    add_music: bool = True,
    burn_subtitles: bool = True,
) -> Dict:
    """Run the full clipping pipeline and return a structured result.

    Args:
        settings: resolved configuration (source, output dir, clipping options,
            LLM, Whisper, ...).
        enhance: when ``True``, post-process every rendered short with background
            music and burned-in subtitles (the ``all`` command). Subtitles are
            sliced out of the transcript already computed for ranking, so no
            extra Whisper pass is needed.
        add_music: with ``enhance``, mix in background music (default: on).
        burn_subtitles: with ``enhance``, burn subtitles into each short
            (default: on).

    Returns:
        {
          "mode": "local",
          "source_video_url": str,   # local path to the source video
          "transcript": {...},
          "highlights": [...],       # every candidate, ranked
          "shorts": [...],           # top `num_clips` with local clip paths
        }
    """
    return _run(
        settings,
        enhance=enhance,
        add_music=add_music,
        burn_subtitles=burn_subtitles,
    )


def generate_subtitles(
    settings: Settings,
    input_path: Optional[str] = None,
) -> Dict:
    """Generate ``.srt`` subtitles only — no highlight ranking, no cropping.

    ``input_path`` (defaulting to ``settings.input``) may be a single video file
    or a folder of videos. Each video gets its own ``.srt`` written next to it
    with the same base name (``video/talk.mkv`` → ``video/talk.srt``).

    Transcription runs locally via faster-whisper; the written ``.srt`` also
    serves as the transcript cache for that video.
    """
    from .subtitles import generate_subtitles as _generate

    return _generate(settings, input_path=input_path)


def resolve_input_videos(settings: Settings) -> List[str]:
    """Return the local video files for the current ``INPUT``.

    ``INPUT`` may be a folder (every video file directly inside it is returned),
    a local file, or a YouTube URL (downloaded into ``OUTPUT_DIR`` first).
    """
    source = settings.input
    if not source:
        raise RuntimeError("No input given. Set INPUT in .env or pass -i/--input.")

    local = settings.resolve(source)
    if os.path.isdir(local):
        videos = find_video_files(local)
        if not videos:
            raise RuntimeError(f"No video files found in: {source}")
        return videos

    return [
        download_youtube(
            source, fmt=settings.download_format, out_dir=settings.output_dir
        )
    ]
