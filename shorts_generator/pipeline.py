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
from typing import Dict, List, Optional, Tuple

from .clipper import crop_highlights
from .config import Settings
from .downloader import download_youtube
from .enhance import enhance_shorts
from .highlights import (
    dedupe_highlights,
    enrich_highlights_metadata,
    get_highlights,
    select_highlights,
    snap_highlights_to_transcript,
)
from .subtitles import find_video_files
from .timing import start_timer
from .transcriber import transcribe
from .visual_indexer import index_video

# How many boundary words to show per clip edge in the analysis log.
_CUT_LOG_WORDS = 4


def _boundary_words(
    transcript: Dict,
    start: float,
    end: float,
    *,
    count: int = _CUT_LOG_WORDS,
) -> Tuple[str, str]:
    """Return the first and last ``count`` words spoken inside [start, end].

    Words come from the transcript cues overlapping the window (no extra
    Whisper pass), so the values mirror what will actually be cut.
    """
    parts: List[str] = []
    for segment in transcript.get("segments", []) or []:
        try:
            seg_start = float(segment["start"])
            seg_end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if seg_end <= start or seg_start >= end:
            continue
        text = str(segment.get("text", "")).strip()
        if text:
            parts.append(text)
    words = " ".join(parts).split()
    if not words:
        return "", ""
    return " ".join(words[:count]), " ".join(words[-count:])


def _log_cut_points(transcript: Dict, highlights: List[Dict]) -> None:
    """Print each clip's window, duration and boundary words for analysis."""
    for i, h in enumerate(highlights, 1):
        try:
            start = float(h["start_time"])
            end = float(h["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        head, tail = _boundary_words(transcript, start, end)
        print(
            f"[cut] #{i}  {start:.2f}s -> {end:.2f}s  "
            f"(длительность {end - start:.2f}s)  "
            f"score={h.get('score')}  «{h.get('title', '')}»",
            flush=True,
        )
        if head:
            print(f"[cut]   начало ({start:.2f}s): «{head}»", flush=True)
        if tail:
            print(f"[cut]   конец  ({end:.2f}s): «{tail}»", flush=True)


def _process_one(
    source_path: str,
    settings: Settings,
    timer,
    *,
    enhance: bool,
    add_music: bool,
    burn_subtitles: bool,
    start_index: int = 0,
) -> Dict:
    """Run transcribe -> highlights -> crop (-> enhance) for one local video.

    ``start_index`` is the number of shorts already rendered by earlier source
    videos; clip numbering continues from there so the number in each file name
    is a single running order across all inputs.
    """
    with timer.stage("transcribe"):
        transcript = transcribe(source_path, settings)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments; the video may have no detectable "
            f"speech: {os.path.basename(source_path)}"
        )

    # Optional visual indexing (step between transcription and ranking): it
    # describes what happens on screen so the ranker can favour moments where
    # the words are reinforced by a strong visual. Off by default.
    visuals: Dict = {"engine": "none", "scenes": []}
    if settings.visual_indexer_enabled:
        # index_video times its own work on the process-wide timer: the
        # frame-by-frame scan under ``scenes`` and the engine's descriptions
        # (model load included) under ``visual``. Measuring it here instead
        # would fold the scan and the model into one number — and, since the
        # scan is often the slow part, hide where the visual step really
        # spends its time.
        visuals = index_video(source_path, settings)

    with timer.stage("highlights"):
        highlights_result = get_highlights(
            transcript,
            settings,
            num_clips=settings.num_clips,
            visuals=visuals,
        )
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError(
            f"Highlight generator returned zero clips for "
            f"{os.path.basename(source_path)}."
        )

    # Границы выбирает LLM по началам реплик, поэтому привязка к фразам идёт до
    # отбора: разные моменты могут схлопнуться в одно и то же окно, и тогда
    # дедуп ниже отбросит дубликат, а в топ попадёт следующий кандидат (иначе в
    # выдаче оказывались два одинаковых клипа).
    if settings.clip_snap_to_transcript:
        snap_highlights_to_transcript(
            all_highlights,
            transcript,
            start_padding=settings.clip_start_padding,
            end_padding=settings.clip_end_padding,
            # Тот же порог паузы, что и у субтитров: границы клипов привязываются
            # к тем же настоящим паузам в речи, на которых рвутся реплики.
            pause_threshold=settings.subtitle_pause_threshold,
            max_end=float(transcript.get("duration", 0.0)) or None,
        )

    all_highlights = dedupe_highlights(all_highlights)
    # Слишком короткие моменты (LLM порой отдаёт окна в 0.15–0.5 с) здесь либо
    # достраиваются до CLIP_MIN_DURATION, либо отбрасываются: в нарезку не уходит
    # обрубок, а освободившееся место достаётся следующему кандидату по оценке.
    top = select_highlights(
        all_highlights,
        transcript,
        num_clips=settings.num_clips,
        min_duration=settings.clip_min_duration,
        max_end=float(transcript.get("duration", 0.0)) or None,
    )
    print(
        f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates",
        flush=True,
    )

    # Двухэтапный режим: ранжирование отдало только тайминги, оценку и причину,
    # а метаданные (заголовок, описание, теги, метрики) генерируются здесь —
    # одним батч-вызовом и только для клипов, которые реально пойдут в нарезку.
    # Хук и панчлайн этап 2 выводит из финального окна детерминированно.
    if settings.two_stage_analysis:
        with timer.stage("metadata"):
            enrich_highlights_metadata(top, transcript, settings)
        print(
            f"[pipeline] metadata generated for {len(top)} selected clip(s)",
            flush=True,
        )

    # Analysis log: show what was actually selected for cutting — each clip's
    # time window plus a few words from its start and end phrases.
    _log_cut_points(transcript, top)

    with timer.stage("crop"):
        shorts = crop_highlights(
            source_path,
            top,
            aspect_ratio=settings.aspect_ratio,
            out_dir=settings.output_dir,
            face_tracking=settings.face_tracking,
            slide_effect=settings.slide_effect,
            slide_gap=settings.slide_transition_gap,
            slide_range=settings.slide_range,
            start_index=start_index,
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
        "visuals": visuals,
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
    rendered = 0  # running total of shorts produced so far (all videos)
    for source_path in source_paths:
        if len(source_paths) > 1:
            print(f"[pipeline] video: {os.path.basename(source_path)}", flush=True)
        video = _process_one(
            source_path,
            settings,
            timer,
            enhance=enhance,
            add_music=add_music,
            burn_subtitles=burn_subtitles,
            start_index=rendered,
        )
        rendered += len(video.get("shorts") or [])
        videos.append(video)

    single = len(videos) == 1
    return {
        "mode": "local",
        "source_video_url": (
            videos[0]["source_video_url"] if single else list(source_paths)
        ),
        "transcript": videos[0]["transcript"] if single else None,
        "visuals": videos[0].get("visuals") if single else None,
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
