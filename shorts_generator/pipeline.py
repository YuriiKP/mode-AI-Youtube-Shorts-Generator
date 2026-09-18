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
from .highlights import get_highlights, snap_highlights_to_transcript
from .subtitles import find_video_files
from .timing import start_timer
from .transcriber import transcribe


def _process_one(
    source_path: str,
    settings: Settings,
    timer,
    *,
    enhance: bool,
    add_music: bool,
    burn_subtitles: bool,
) -> Dict:
    """Run transcribe -> highlights -> crop (-> enhance) for one local video."""
    with timer.stage("transcribe"):
        transcript = transcribe(source_path, settings)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments; the video may have no detectable "
            f"speech: {os.path.basename(source_path)}"
        )

    with timer.stage("highlights"):
        highlights_result = get_highlights(
            transcript, settings, num_clips=settings.num_clips
        )
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError(
            f"Highlight generator returned zero clips for "
            f"{os.path.basename(source_path)}."
        )

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[
        : settings.num_clips
    ]
    print(
        f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates",
        flush=True,
    )

    if settings.clip_snap_to_transcript:
        snap_highlights_to_transcript(
            top,
            transcript,
            start_padding=settings.clip_start_padding,
            end_padding=settings.clip_end_padding,
            max_end=float(transcript.get("duration", 0.0)) or None,
        )

    with timer.stage("crop"):
        shorts = crop_highlights(
            source_path,
            top,
            aspect_ratio=settings.aspect_ratio,
            out_dir=settings.output_dir,
            face_tracking=settings.face_tracking,
        )

    if enhance:
        with timer.stage("enhance"):
            enhance_shorts(
                transcript,
                shorts,
                settings,
                add_music=add_music,
                burn_subtitles=burn_subtitles,
            )

    return {
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def _run(
    settings: Settings,
    enhance: bool = False,
    add_music: bool = True,
    burn_subtitles: bool = True,
) -> Dict:
    if not settings.input:
        raise RuntimeError("No input given. Set INPUT in .env or pass -i/--input.")

    # Time each stage so the CLI can show where the run spends its time.
    timer = start_timer()

    # INPUT may be a YouTube URL, a single file or a folder of videos; turn it
    # into the list of local paths to process (downloading a URL first).
    with timer.stage("download"):
        source_paths = resolve_input_videos(settings)

    videos: List[Dict] = []
    for source_path in source_paths:
        if len(source_paths) > 1:
            print(f"[pipeline] video: {os.path.basename(source_path)}", flush=True)
        videos.append(
            _process_one(
                source_path,
                settings,
                timer,
                enhance=enhance,
                add_music=add_music,
                burn_subtitles=burn_subtitles,
            )
        )

    single = len(videos) == 1
    return {
        "mode": "local",
        "source_video_url": (
            videos[0]["source_video_url"] if single else list(source_paths)
        ),
        "transcript": videos[0]["transcript"] if single else None,
        "highlights": [h for v in videos for h in v["highlights"]],
        "shorts": [s for v in videos for s in v["shorts"]],
        "videos": videos,
        "timings": timer.as_dict(),
    }


def generate_shorts(
    settings: Settings,
    enhance: bool = False,
    *,
    add_music: bool = True,
    burn_subtitles: bool = True,
) -> Dict:
    """Run the full clipping pipeline and return a structured result.

    ``settings.input`` may be a YouTube URL, a single file, or a folder of
    videos; each source video is processed in turn and the results are merged.

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
          "source_video_url": str | list[str],  # one path, or all inputs
          "transcript": {...} | None,           # single input only
          "highlights": [...],   # every candidate, merged across videos
          "shorts": [...],       # top `num_clips` per video, with clip paths
          "videos": [            # per-source breakdown (always present)
            {"source_video_url": str, "transcript": {...},
             "highlights": [...], "shorts": [...]},
            ...
          ],
          "timings": {...},      # per-stage wall-clock measurements
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
