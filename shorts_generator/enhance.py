"""Post-processing bridge: re-frame, banner, burn subtitles and mix music.

This is the glue used by the ``all`` command. It reuses the transcript the
pipeline already computed for highlight ranking, so no extra Whisper pass is
needed: for each short the overlapping segments are clipped to the clip's time
range and re-timed to start at zero, written as a per-clip ``.srt`` and burned
in by the built-in post-processing engine. In the same pass the clip is
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
from typing import Dict, List

from .config import Settings


def _format_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp ``HH:MM:SS,mmm``."""
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_clip_srt(transcript: Dict, start: float, end: float, out_path: str) -> str:
    """Write a per-clip ``.srt`` derived from the source transcript.

    Segments overlapping ``[start, end]`` are clipped to that range and shifted
    so the clip starts at 0. Returns ``out_path`` on success, or ``""`` when the
    clip contains no speech (in which case no file is written).
    """
    segments = transcript.get("segments", []) or []
    lines: List[str] = []
    index = 0

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

        index += 1
        lines.append(str(index))
        lines.append(
            f"{_format_timestamp(clipped_start)} --> {_format_timestamp(clipped_end)}"
        )
        lines.append(text)
        lines.append("")

    if not lines:
        return ""

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return out_path


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
            with ``enhanced`` / ``subtitle_path`` / ``enhance_error``.
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
        srt_path = ""
        if burn_subtitles:
            candidate = os.path.splitext(clip_path)[0] + ".srt"
            srt_path = build_clip_srt(
                transcript,
                float(short.get("start_time", 0.0)),
                float(short.get("end_time", 0.0)),
                candidate,
            )
            if not srt_path:
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
                burn_subtitles=bool(srt_path),
                add_music=add_music,
                subtitle_file=srt_path or None,
            )
            os.replace(temp_out, clip_path)
            short["enhanced"] = True
            if srt_path:
                short["subtitle_path"] = srt_path
            print("[enhance]   done", flush=True)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"[enhance]   failed: {exc}", flush=True)
            short["enhance_error"] = str(exc)
            try:
                if os.path.isfile(temp_out):
                    os.remove(temp_out)
            except OSError:
                pass
