"""Post-processing bridge: re-frame, banner, burn subtitles and mix music.

This is the glue used by the ``all`` command. It reuses the transcript the
pipeline already computed for highlight ranking, so no extra Whisper pass is
needed: for each short the overlapping segments are clipped to the clip's time
range and re-timed to start at zero, kept in memory and burned in by the
built-in post-processing engine (no per-clip ``.srt`` file is written to disk).
In the same pass the clip is
re-framed to the vertical ``FIT_ASPECT_RATIO`` (empty space filled with a
blurred copy of the video) and the configured ``BANNER_*`` overlay is drawn, so
every rendered short comes out as a clean 9:16 video. Background music is mixed
at the same time.

Each short is re-encoded to a temporary file and swapped in atomically, so a
failure never destroys the original clip. Per-clip failures are recorded on the
short dict (``enhance_error``) and do not abort the remaining clips.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from .config import Settings


def build_clip_items(
    transcript: Dict, start: float, end: float
) -> List[Tuple[Tuple[float, float], str]]:
    """Return in-memory subtitle entries for one clip.

    Segments overlapping ``[start, end]`` are clipped to that range and shifted
    so the clip starts at 0. Nothing is written to disk — the entries are handed
    straight to the post-processing engine and burned from memory. Returns an
    empty list when the clip contains no speech.
    """
    segments = transcript.get("segments", []) or []
    items: List[Tuple[Tuple[float, float], str]] = []

    for segment in segments:
        try:
            seg_start = float(segment["start"])
            seg_end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue

        # Skip segments that lie entirely outside the clip.
        if seg_end <= start or seg_start >= end:
            continue

        clipped_start = max(seg_start, start) - start
        clipped_end = min(seg_end, end) - start
        if clipped_end <= clipped_start:
            continue

        text = str(segment.get("text", "")).strip().replace("\r", "").replace("\n", " ")
        if not text:
            continue

        items.append(((clipped_start, clipped_end), text))

    return items


def enhance_shorts(
    transcript: Dict,
    shorts: List[Dict],
    settings: Settings,
    *,
    add_music: bool = True,
    burn_subtitles: bool = True,
) -> None:
    """Post-process each rendered short in place (background music + subtitles).

    Args:
        transcript: the source transcript (``{"duration", "segments"}``) used to
            derive per-clip subtitles.
        shorts: the short dicts produced by the crop stage; modified in place
            with ``enhanced`` / ``enhance_error``.
        settings: resolved configuration (music source/volume, subtitle look,
            encoding). Subtitles are always taken from ``transcript`` here.
        add_music: whether to mix in background music.
        burn_subtitles: whether to burn in subtitles.
    """
    from .postprocess.log import setup_logging
    from .postprocess.pipeline import run as postprocess_run

    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))

    total = len(shorts)
    for i, short in enumerate(shorts, 1):
        clip_url = short.get("clip_url")
        if not clip_url or not os.path.isfile(clip_url):
            continue

        clip_path = os.path.abspath(clip_url)
        print(f"[enhance] {i}/{total}: {os.path.basename(clip_path)}", flush=True)

        # --- per-clip subtitles, reused from the highlight transcript ------
        # Built in memory and handed straight to the engine, so no per-clip
        # ``.srt`` file is written to disk.
        subtitle_items: List[Tuple[Tuple[float, float], str]] = []
        if burn_subtitles:
            subtitle_items = build_clip_items(
                transcript,
                float(short.get("start_time", 0.0)),
                float(short.get("end_time", 0.0)),
            )
            if not subtitle_items:
                print(
                    "[enhance]   no speech in this clip; skipping subtitles", flush=True
                )

        # Write to a temp file first, then swap it in, so a failure never
        # destroys the original clip (the engine refuses in-place overwrites).
        temp_out = clip_path + ".enhancing.mp4"
        try:
            postprocess_run(
                clip_path,
                temp_out,
                settings,
                burn_subtitles=bool(subtitle_items),
                add_music=add_music,
                subtitle_items=subtitle_items,
            )
            os.replace(temp_out, clip_path)
            short["enhanced"] = True

            print("[enhance]   done", flush=True)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"[enhance]   failed: {exc}", flush=True)
            short["enhance_error"] = str(exc)
            try:
                if os.path.isfile(temp_out):
                    os.remove(temp_out)
            except OSError:
                pass
