"""Transcription via faster-whisper.

Reads a local media file and returns the same shape the highlight generator
expects: ``{duration, segments[start, end, text]}``. The result is cached as an
``.srt`` file next to the source (or in ``OUTPUT_DIR``), so a second run — and
the subtitle burn-in stage — reuse it instead of paying for Whisper again.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Settings
from .cues import split_segments_into_cues


def _transcript_cache_path(
    media_path: str, settings: Settings, cache_path: Optional[str] = None
) -> Path:
    """Return the ``.srt`` cache path for a media file.

    When ``cache_path`` is given (e.g. an ``.srt`` sitting next to the video),
    that exact path is used. Otherwise the cache lands in ``OUTPUT_DIR`` under
    the video's stem.
    """
    if cache_path:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    cache_dir = Path(settings.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (Path(media_path).stem + ".srt")


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_srt_timestamp(value: str) -> float:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, millis = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + (millis / 1000.0)


def _write_srt_cache(
    media_path: str,
    transcript: Dict,
    settings: Settings,
    cache_path: Optional[str] = None,
) -> Path:
    srt_path: Path = _transcript_cache_path(media_path, settings, cache_path=cache_path)
    lines = []
    for idx, segment in enumerate(transcript.get("segments", []), start=1):
        start = _format_srt_timestamp(float(segment["start"]))
        end = _format_srt_timestamp(float(segment["end"]))
        text = str(segment.get("text", "")).strip().replace("\r", "").replace("\n", " ")
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return srt_path


def _load_srt_cache(cache_path: Path) -> Dict:
    content = cache_path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return {"duration": 0.0, "segments": []}

    segments = []
    for block in re.split(r"\n\s*\n", content):
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if "-->" not in lines[0] and len(lines) > 1 and "-->" in lines[1]:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[0].split("-->", 1)]
        text = "\n".join(lines[1:]).strip()
        segments.append(
            {
                "start": _parse_srt_timestamp(start_raw),
                "end": _parse_srt_timestamp(end_raw),
                "text": text,
            }
        )

    duration = segments[-1]["end"] if segments else 0.0
    return {"duration": duration, "segments": segments}


def _apply_cue_split(transcript: Dict, settings: Settings) -> Dict:
    """Re-chunk whole-sentence segments into short subtitle cues.

    Whisper returns whole sentences per segment, which would otherwise be burned
    in as one long block that hangs over the video. Splitting into short cues here
    — using word-level timings when present — means the highlight ranking, the
    ``.srt`` cache and the burned-in subtitles all share the same neat phrasing.
    """
    cues = split_segments_into_cues(
        transcript.get("segments", []),
        max_chars=settings.subtitle_max_chars,
        max_words=settings.subtitle_max_words,
        max_duration=settings.subtitle_max_duration,
        pause_threshold=settings.subtitle_pause_threshold,
    )
    return {"duration": transcript.get("duration", 0.0), "segments": cues}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            # Verify CUDA really works (catches missing cuBLAS/cuDNN libs).
            torch.zeros(1, device="cuda")
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass
    return "cpu"


def transcribe(
    media_path: str,
    settings: Settings,
    *,
    language: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> Dict:
    """Run faster-whisper on a local file, caching the result as ``.srt``.

    Args:
        media_path: local path to a video/audio file.
        settings: resolved configuration (model, device, VAD, output dir).
        language: ISO-639-1 code to force; defaults to ``settings.whisper_language``.
        cache_path: where to write the ``.srt`` cache (defaults to the video's
            folder / ``OUTPUT_DIR``).
    """
    language = language if language is not None else (settings.whisper_language or None)

    srt_path = _transcript_cache_path(media_path, settings, cache_path=cache_path)
    if srt_path.exists():
        source_mtime = os.path.getmtime(media_path)
        cache_mtime = srt_path.stat().st_mtime
        if cache_mtime >= source_mtime:
            print(f"[transcribe] reusing cached transcript: {srt_path}", flush=True)
            cached = _load_srt_cache(srt_path)
            # Treat an empty cache as invalid (likely a partial run) and re-run.
            if not cached["segments"] or cached["duration"] <= 0.0:
                print(
                    f"[transcribe] cache is empty/invalid, deleting: {srt_path}",
                    flush=True,
                )
                srt_path.unlink(missing_ok=True)
            else:
                cached = _apply_cue_split(cached, settings)
                print(
                    f"[transcribe] {len(cached['segments'])} cached cues, "
                    f"{cached['duration']:.0f}s of audio",
                    flush=True,
                )
                return cached

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is required. Install it with:\n"
            "    pip install -r requirements.txt"
        ) from e

    device = _resolve_device(settings.whisper_device)
    compute_type = "float16" if device == "cuda" else "int8"
    print(
        f"[transcribe] faster-whisper model={settings.whisper_model} device={device}",
        flush=True,
    )

    model = WhisperModel(
        settings.whisper_model, device=device, compute_type=compute_type
    )

    transcribe_kwargs: Dict[str, Any] = {
        "audio": media_path,
        "language": language,
        "beam_size": 5,
        "condition_on_previous_text": False,
        # Word-level timings let the cue splitter follow the actual speech so
        # each on-screen phrase appears and disappears in sync with the voice.
        "word_timestamps": True,
    }
    if settings.whisper_vad_filter:
        transcribe_kwargs["vad_filter"] = True
    else:
        transcribe_kwargs["vad_filter"] = False

    segments_iter, info = model.transcribe(**transcribe_kwargs)

    segments = []
    for s in segments_iter:
        segment = {
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip(),
        }
        words = getattr(s, "words", None)
        if words:
            segment["words"] = [
                {"start": float(w.start), "end": float(w.end), "word": (w.word or "")}
                for w in words
            ]
        segments.append(segment)

    duration = float(getattr(info, "duration", 0.0)) or (
        segments[-1]["end"] if segments else 0.0
    )
    transcript = _apply_cue_split(
        {"duration": duration, "segments": segments}, settings
    )
    print(
        f"[transcribe] {len(transcript['segments'])} cues, {duration:.0f}s of audio",
        flush=True,
    )
    srt_path = _write_srt_cache(media_path, transcript, settings, cache_path=cache_path)
    print(f"[transcribe] wrote cache: {srt_path}", flush=True)
    return transcript
