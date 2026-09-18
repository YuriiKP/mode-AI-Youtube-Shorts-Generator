"""End-to-end processing engine for a single video.

:func:`run` ties the stages together:

1. bake the configured colour/lens effects (``SATURATION`` / ``SHARPNESS`` /
   ``CHROMATIC_ABERRATION``) into the source with a single FFmpeg pass, so they
   land on the video *only* — before any text is drawn;
2. resolve the subtitle source (an existing ``.srt`` next to the video, a given
   ``.srt``, or a Whisper transcription when none is found);
3. re-frame the clip into the vertical frame, filling the empty area with a
   blurred copy of the video when it does not already match the target ratio
   (``FIT_VERTICAL``);
4. burn the subtitles and overlay the banner onto the video;
5. mix a background music track under the existing audio;
6. write the result to disk.

The input's own audio is always kept and the music is mixed *under* it, so an
existing voice-over is never lost. Every stage is optional and independent —
the ``subtitles``, ``music`` and ``all`` commands pick which ones to apply.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from typing import Optional

from moviepy import (
    CompositeAudioClip,
    CompositeVideoClip,
    VideoFileClip,
)

from ..config import Settings
from .banner import banner_kind, build_banner_clips
from .effects import build_filter_chain
from .ffmpeg import configure_ffmpeg
from .fonts import resolve_font_path
from .layout import build_vertical_clip, needs_vertical_fit
from .log import log
from .music import build_music_audio, resolve_music_file
from .subtitles import build_subtitle_clips

_DEFAULT_VIDEO_CODEC = "libx264"

# CRF for the effects pre-pass intermediate: high enough that the extra encode
# generation is visually invisible, while still encoding quickly.
_EFFECTS_CRF = "18"


class ProcessingError(RuntimeError):
    """Raised for user-facing problems (missing files, bad options, ...)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_video_direct(path: str) -> VideoFileClip:
    """Open ``path`` with MoviePy, keeping its original audio track.

    MoviePy 2.x prints reader/decode information straight to stdout. We capture
    and demote it to a debug log so the CLI output stays clean.
    """
    captured = io.StringIO()
    with redirect_stdout(captured):
        clip = VideoFileClip(path, audio=True)
    if captured.getvalue().strip():
        log.debug("suppressed MoviePy reader output for %s", path)
    return clip


def _remove_file(path: str) -> None:
    """Delete ``path`` if it exists, ignoring (but logging) any error."""
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError as exc:  # pragma: no cover - best effort cleanup
        log.debug("failed to remove temporary file %s: %s", path, exc)


def _scrub_input(path: str, ffmpeg_binary: str) -> str:
    """Remux ``path`` into a temporary copy without chapters/extra streams.

    MoviePy 2.x's ffmpeg parser crashes on files that contain exactly one
    chapter per input (its ``FFmpegInfosParser`` indexes an undersized list),
    so when the direct open fails we retry on a cleaned copy. Matroska is used
    as the temporary container because ``-c copy`` accepts almost any codec.
    """
    handle, temp_path = tempfile.mkstemp(prefix="shorts_scrub_", suffix=".mkv")
    os.close(handle)
    _remove_file(temp_path)

    cmd = [
        ffmpeg_binary,
        "-y",
        "-loglevel",
        "error",
        "-i",
        path,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-map_chapters",
        "-1",
        "-sn",
        "-dn",
        "-c",
        "copy",
        temp_path,
    ]
    try:
        _ = subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except Exception:
        _remove_file(temp_path)
        raise

    log.info("created a cleaned copy without chapters/extra streams")
    return temp_path


def _apply_effects_pass(source_path: str, filter_chain: str, ffmpeg_binary: str) -> str:
    """Bake the colour/lens effects into ``source_path`` with one FFmpeg pass.

    The effects must land on the video *only* — before MoviePy composites the
    subtitles and the banner on top — so rather than filtering the final encode
    we rewrite the source once here and hand the result to the rest of the
    pipeline. FFmpeg runs the filters in optimised, multithreaded C, so this
    stays far cheaper than touching every frame in Python. The intermediate is a
    high-quality x264 copy, so the extra encode generation is visually lossless.

    Returns the temporary file path; the caller must delete it when done. On
    failure the temp file is removed and a :class:`ProcessingError` is raised.
    """
    handle, temp_path = tempfile.mkstemp(prefix="shorts_fx_", suffix=".mkv")
    os.close(handle)
    _remove_file(temp_path)

    cmd = [
        ffmpeg_binary,
        "-y",
        "-loglevel",
        "error",
        "-i",
        source_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-map_chapters",
        "-1",
        "-sn",
        "-dn",
        "-vf",
        filter_chain,
        "-c:v",
        _DEFAULT_VIDEO_CODEC,
        "-crf",
        _EFFECTS_CRF,
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        temp_path,
    ]
    log.info("applying video filters to the source: %s", filter_chain)
    try:
        _ = subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as exc:
        _remove_file(temp_path)
        stderr = (exc.stderr or b"").decode("utf-8", "replace").strip()
        raise ProcessingError(f"failed to apply video effects: {stderr}") from exc
    except Exception:
        _remove_file(temp_path)
        raise
    return temp_path


def _open_video(path: str, ffmpeg_binary: str = "ffmpeg") -> tuple[VideoFileClip, str]:
    """Open a video, falling back to a cleaned remux when MoviePy fails.

    Returns ``(clip, temp_path)`` where ``temp_path`` is the temporary cleaned
    copy that must be removed once the clip has been closed. It is ``""`` when
    no copy was needed (the common case).
    """
    try:
        return _open_video_direct(path), ""
    except Exception as exc:  # noqa: BLE001 - retry on a cleaned copy first
        log.debug("direct-open error for %s: %s", path, exc)
        log.warning(
            "could not open %s directly; retrying on a cleaned copy",
            os.path.basename(path),
        )

    cleaned = _scrub_input(path, ffmpeg_binary)
    try:
        return _open_video_direct(cleaned), cleaned
    except Exception:
        _remove_file(cleaned)
        raise


def _default_output_path(input_path: str) -> str:
    stem, _ = os.path.splitext(input_path)
    return f"{stem}_out.mp4"


def sibling_subtitle_path(input_path: str) -> str:
    """Return the ``.srt`` next to a video that shares its base name."""
    stem, _ = os.path.splitext(input_path)
    return f"{stem}.srt"


def _generate_subtitles(settings: Settings, input_path: str) -> str:
    """Transcribe ``input_path`` with Whisper and return the written ``.srt``."""
    from ..transcriber import transcribe

    target = sibling_subtitle_path(input_path)
    log.info("no subtitle file found; transcribing with Whisper")
    transcript = transcribe(input_path, settings, cache_path=target)
    if not transcript.get("segments"):
        raise ProcessingError("Whisper produced no segments for this file.")
    return target


def _resolve_subtitle_file(
    settings: Settings,
    input_path: str,
    subtitle_file: Optional[str],
) -> str:
    """Return the ``.srt`` to burn in, or ``""`` when subtitles are skipped.

    ``subtitle_file`` (when given) is an explicit override used by the ``all``
    pipeline, which slices a per-clip ``.srt`` out of the source transcript.
    Otherwise ``SUBTITLE_SOURCE`` decides:

    * ``auto``    — an explicit ``SUBTITLE_FILE`` or a same-named ``.srt`` next
      to the video; if neither exists, transcribe with Whisper;
    * ``file``    — an explicit ``SUBTITLE_FILE`` or a same-named ``.srt``;
    * ``whisper`` — always transcribe with Whisper;
    * ``none``    — never burn subtitles.
    """
    if subtitle_file:
        path = settings.resolve(subtitle_file)
        if not os.path.isfile(path):
            raise ProcessingError(f"subtitle file not found: {subtitle_file!r}")
        return path

    source = (settings.subtitle_source or "auto").strip().lower()

    if source == "none":
        log.info("SUBTITLE_SOURCE=none; skipping subtitles")
        return ""

    if source == "file":
        candidate = settings.subtitle_file or sibling_subtitle_path(input_path)
        path = settings.resolve(candidate)
        if not os.path.isfile(path):
            raise ProcessingError(
                f"subtitle file not found: {candidate!r}. Set SUBTITLE_FILE to an "
                "existing .srt, use SUBTITLE_SOURCE=auto/whisper, or disable "
                "subtitles with SUBTITLE_SOURCE=none."
            )
        return path

    if source == "whisper":
        return _generate_subtitles(settings, input_path)

    if source == "auto":
        if settings.subtitle_file:
            explicit = settings.resolve(settings.subtitle_file)
            if os.path.isfile(explicit):
                return explicit
        sibling = sibling_subtitle_path(input_path)
        if os.path.isfile(sibling):
            log.info("using subtitle file next to the video: %s", sibling)
            return sibling
        return _generate_subtitles(settings, input_path)

    raise ProcessingError(
        f"unknown SUBTITLE_SOURCE {settings.subtitle_source!r}; "
        "use one of: auto, file, whisper, none"
    )


def _safe_close(clip, _visited: Optional[set] = None) -> None:
    """Close a clip and its children, ignoring (but logging) any errors."""
    if clip is None:
        return
    if _visited is None:
        _visited = set()
    if id(clip) in _visited:
        return
    _visited.add(id(clip))

    try:
        reader = getattr(clip, "reader", None)
        if reader is not None:
            reader.close()
    except Exception as exc:  # pragma: no cover - best effort cleanup
        log.debug("failed to close clip reader: %s", exc)

    try:
        audio = getattr(clip, "audio", None)
        if audio is not None:
            _safe_close(audio, _visited)
    except Exception as exc:  # pragma: no cover - best effort cleanup
        log.debug("failed to close clip audio: %s", exc)

    try:
        for child in list(getattr(clip, "clips", []) or []):
            if child is not clip:
                _safe_close(child, _visited)
    except Exception as exc:  # pragma: no cover - best effort cleanup
        log.debug("failed to close child clips: %s", exc)

    try:
        clip.close()
    except Exception as exc:  # pragma: no cover - best effort cleanup
        log.debug("failed to close clip: %s", exc)


def _temp_audio_dir(output_dir: str) -> str:
    """Pick a safe directory for MoviePy's temporary audio file.

    On Windows, Defender can lock files written into the output directory and
    make MoviePy fail with a PermissionError; the system temp dir avoids it.
    """
    if sys.platform == "win32":
        return tempfile.gettempdir()
    return output_dir


def _write_video(
    clip,
    output_path: str,
    settings: Settings,
    source_fps: float,
    audio_fps: int,
) -> None:
    """Write ``clip`` to ``output_path`` with a libx264 fallback."""
    fps = settings.fps if (settings.fps and settings.fps > 0) else (source_fps or 30)
    output_dir = os.path.dirname(output_path) or "."

    # 0 (or unset) means "let ffmpeg decide"; passing None leaves ``-threads``
    # off entirely so the encoder uses every available core instead of the two
    # the old default pinned it to.
    thread_count = (
        int(settings.threads) if settings.threads and settings.threads > 0 else None
    )

    kwargs = dict(
        audio_codec=settings.audio_codec,
        audio_bitrate=settings.audio_bitrate,
        audio_fps=audio_fps,
        temp_audiofile_path=_temp_audio_dir(output_dir),
        threads=thread_count,
        preset=settings.preset or "fast",
        # Keep the progress bar: without it a long encode prints nothing and
        # looks like it has frozen.
        logger="bar",
        fps=fps,
    )

    codec = settings.video_codec or _DEFAULT_VIDEO_CODEC
    log.info(
        "encoding output: %s (codec=%s, preset=%s, fps=%s, threads=%s)",
        output_path,
        codec,
        settings.preset or "fast",
        fps,
        thread_count if thread_count else "auto",
    )

    try:
        clip.write_videofile(output_path, codec=codec, **kwargs)
    except Exception as exc:
        if codec == _DEFAULT_VIDEO_CODEC:
            raise
        # Hardware encoders depend on the GPU/driver being present at runtime;
        # fall back to a widely supported software encoder exactly once.
        log.warning(
            "video codec %s failed (%s); retrying with %s",
            codec,
            exc,
            _DEFAULT_VIDEO_CODEC,
        )
        clip.write_videofile(output_path, codec=_DEFAULT_VIDEO_CODEC, **kwargs)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(
    input_path: str,
    output_path: str,
    settings: Settings,
    *,
    burn_subtitles: bool = True,
    add_music: bool = True,
    subtitle_file: Optional[str] = None,
) -> str:
    """Process one video and return the output path.

    Args:
        input_path: source video (absolute or relative to ``settings.base_dir``).
        output_path: where to write the result.
        settings: fully resolved configuration (subtitle look, banner, vertical
            frame, music, encoding).
        burn_subtitles: whether to attempt the subtitle stage.
        add_music: whether to attempt the music stage.
        subtitle_file: explicit ``.srt`` to burn (overrides ``SUBTITLE_SOURCE``).
    """
    ffmpeg_binary = configure_ffmpeg(settings.ffmpeg_path)

    source_path = settings.resolve(input_path)
    if not source_path or not os.path.isfile(source_path):
        raise ProcessingError(f"input video not found: {input_path!r}")

    target_path = settings.resolve(output_path) or _default_output_path(source_path)
    if os.path.abspath(target_path) == os.path.abspath(source_path):
        raise ProcessingError("the output path must not overwrite the input video")
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)

    log.info("input video:  %s", source_path)
    log.info("output video: %s", target_path)

    srt_path = (
        _resolve_subtitle_file(settings, source_path, subtitle_file)
        if burn_subtitles
        else ""
    )

    # --- colour / lens effects -------------------------------------------
    # Baked into the source first, so they affect the video but never the text
    # (subtitles/banner) that gets composited on top later. Skipped entirely
    # when every effect is at its neutral value.
    working_path = source_path
    effects_input = ""
    filter_chain = build_filter_chain(settings)
    if filter_chain:
        effects_input = _apply_effects_pass(source_path, filter_chain, ffmpeg_binary)
        working_path = effects_input

    try:
        video_clip, scrubbed_input = _open_video(working_path, ffmpeg_binary)
    except BaseException:
        _remove_file(effects_input)
        raise

    music_source = None
    try:
        width, height = video_clip.size
        video_duration = float(video_clip.duration or 0.0)
        log.info(
            "source: %dx%d, %.2fs, fps=%s",
            int(width),
            int(height),
            video_duration,
            getattr(video_clip, "fps", None),
        )

        final_clip = video_clip
        final_width, final_height = int(width), int(height)

        # --- vertical fit: blurred background fill -----------------------
        if needs_vertical_fit(video_clip, settings):
            final_clip = build_vertical_clip(video_clip, settings)
            final_width, final_height = (int(value) for value in final_clip.size)

        # --- overlays: burned-in subtitles + banner ----------------------
        # The font is only needed when something textual is drawn (subtitles or
        # a text banner), so an image-only banner does not require one.
        needs_font = bool(srt_path) or banner_kind(settings) == "text"
        font_path = resolve_font_path(settings) if needs_font else ""

        subtitle_clips = []
        if srt_path:
            log.info("burning subtitles from: %s", srt_path)
            subtitle_clips = build_subtitle_clips(
                srt_path, settings, final_width, final_height, font_path
            )
            if subtitle_clips:
                log.info("added %d subtitle clip(s)", len(subtitle_clips))
            else:
                log.warning("no subtitle entries were rendered; skipping burn-in")

        banner_clips = build_banner_clips(
            settings, final_width, final_height, video_duration, font_path
        )

        overlays = [*subtitle_clips, *banner_clips]
        if overlays:
            # ``use_bgclip=True`` lets MoviePy treat the opaque, full-frame video
            # as the composite background. Without it, CompositeVideoClip builds
            # a full-canvas transparency mask and recomputes it on every frame —
            # at 1080x1920 that alone costs ~90 ms/frame (more than the rest of
            # the render). The shortcut skips the mask machinery entirely; the
            # pixels are identical because the background already covers the
            # whole frame. It does drop the background clip from the duration
            # computation, though, so the duration is restored explicitly.
            final_clip = CompositeVideoClip(
                [final_clip, *overlays],
                size=(final_width, final_height),
                use_bgclip=True,
            ).with_duration(video_duration)

        # --- background music --------------------------------------------
        original_audio = video_clip.audio
        audio_clip = original_audio

        if add_music:
            music_file = resolve_music_file(settings)
            if music_file:
                music_source, music_clip = build_music_audio(
                    music_file,
                    video_duration,
                    settings.music_volume,
                    max(0.0, float(settings.music_fade_out)),
                )
                if original_audio is not None:
                    log.info(
                        "mixing background music under existing audio (volume=%.2f)",
                        settings.music_volume,
                    )
                    audio_clip = CompositeAudioClip([original_audio, music_clip])
                else:
                    log.info(
                        "input video has no audio; using background music only "
                        "(volume=%.2f)",
                        settings.music_volume,
                    )
                    audio_clip = music_clip

        if audio_clip is not None:
            final_clip = final_clip.with_audio(audio_clip)

        # --- write -------------------------------------------------------
        audio_fps = int(
            getattr(original_audio, "fps", 0) or getattr(audio_clip, "fps", 0) or 44100
        )
        _write_video(
            final_clip,
            target_path,
            settings,
            source_fps=float(getattr(video_clip, "fps", 0) or 0.0),
            audio_fps=audio_fps,
        )
    finally:
        _safe_close(video_clip)
        if music_source is not None:
            _safe_close(music_source)
        _remove_file(scrubbed_input)
        _remove_file(effects_input)

    log.info("done: %s", target_path)
    return target_path
