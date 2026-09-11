"""Background music resolution and mixing helpers.

The music source is a single ``MUSIC`` setting in ``.env`` — it may point at
either a specific audio file or a folder. Resolution rules:

* a **folder** → a random supported track is picked from it;
* a **file** → that exact track is used;
* empty / missing → no music (a warning is logged, the run continues).

The chosen track is looped to fill the whole video, volume-adjusted (music only,
the original voice is untouched) and faded out at the end before being mixed
*under* the existing audio.
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple

from moviepy import AudioFileClip, afx

from ..config import Settings
from .log import log

# Audio containers FFmpeg can decode reliably as background music.
SUPPORTED_AUDIO_EXTENSIONS = (
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
)


def list_music_files(directory: str) -> List[str]:
    """Return the absolute paths of supported audio files in ``directory``.

    Missing directories are treated as empty so that a random pick can degrade
    to "no music" instead of crashing.
    """
    directory = os.path.abspath(os.path.expanduser(directory)) if directory else ""
    if not directory or not os.path.isdir(directory):
        return []

    files: List[str] = []
    for name in sorted(os.listdir(directory), key=str.lower):
        # Skip hidden files and editor/packaging leftovers.
        if name.startswith("."):
            continue
        if os.path.splitext(name)[1].lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        full_path = os.path.join(directory, name)
        if os.path.isfile(full_path):
            files.append(full_path)
    return files


def resolve_music_file(settings: Settings) -> str:
    """Resolve the background music track from the single ``MUSIC`` setting.

    ``MUSIC`` may be a folder (a random track is picked) or a file (used as-is).
    Returns an empty string when no track should be used, and logs a warning
    instead of failing when the configured path cannot be found.
    """
    raw = (settings.music or "").strip()
    if not raw:
        return ""

    path = settings.resolve(raw)

    if os.path.isdir(path):
        files = list_music_files(path)
        if not files:
            log.warning(
                "no music files found in %s; continuing without background music",
                path,
            )
            return ""
        chosen = random.choice(files)
        log.info(
            "picked random background music: %s (from %d file(s))",
            os.path.basename(chosen),
            len(files),
        )
        return chosen

    if os.path.isfile(path):
        log.info("using background music file: %s", path)
        return path

    log.warning(
        "MUSIC path does not exist: %s; continuing without background music", raw
    )
    return ""


def build_music_audio(
    music_file: str,
    duration: float,
    volume: float,
    fade_out: float,
) -> Tuple[AudioFileClip, AudioFileClip]:
    """Load ``music_file`` and prepare it to be mixed under the video.

    Parameters
    ----------
    music_file:
        Path to the audio file.
    duration:
        Target duration in seconds (the length of the final video). Short
        tracks are looped to cover it; longer tracks are trimmed.
    volume:
        Linear volume multiplier applied to the music only.
    fade_out:
        Length of the fade-out applied at the end of the (looped) track.

    Returns
    -------
    (source_clip, processed_clip)
        ``source_clip`` is the raw ``AudioFileClip`` that the caller must close;
        ``processed_clip`` is the volume-adjusted / looped / faded version.
    """
    source = AudioFileClip(music_file)

    try:
        processed = source.with_effects([afx.MultiplyVolume(float(volume))])

        # Loop the track so it fills the whole video. When the source is longer
        # than the video, AudioLoop(duration=...) trims it to the target length.
        target = max(0.0, float(duration))
        if target > 0:
            processed = processed.with_effects([afx.AudioLoop(duration=target)])

        # Guard against the source being longer than the requested target even
        # after looping, so the fade below always lands at the end of the video.
        if target > 0 and processed.duration and processed.duration > target:
            processed = processed.subclipped(0, target)

        fade = max(0.0, float(fade_out))
        if fade > 0 and processed.duration:
            # Never fade over more time than the clip actually has.
            fade = min(fade, float(processed.duration))
            if fade > 0:
                processed = processed.with_effects([afx.AudioFadeOut(fade)])

        return source, processed
    except Exception:
        # If anything goes wrong while preparing the track, release the source
        # so we do not leak an FFmpeg reader.
        try:
            source.close()
        except Exception:
            pass
        raise
