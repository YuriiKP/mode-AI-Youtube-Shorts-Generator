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
from typing import Callable, Dict, List, Optional, Tuple

from .clipper import crop_highlights, reframe_video
from .config import Settings
from .downloader import download_youtube
from .enhance import enhance_shorts
from .highlights import (
    build_clip_text,
    dedupe_highlights,
    enrich_highlights_metadata,
    get_highlights,
    select_highlights,
    snap_highlights_to_transcript,
)
from .montage import (
    build_montage_transcript,
    create_montage,
    generate_montage_metadata,
    select_montage_segments,
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


def _process_montage_one(
    source_path: str,
    settings: Settings,
    timer,
    *,
    enhance: bool,
    add_music: bool,
    burn_subtitles: bool,
    start_index: int = 0,
) -> Dict:
    """Run transcribe -> montage-plan -> stitch -> crop (-> metadata, enhance).

    В отличие от :func:`_process_one` (несколько независимых клипов), здесь из
    одного большого видео собирается ровно один ролик: LLM отбирает самые
    динамичные и связные по смыслу отрезки, скрипт вырезает их и склеивает в
    один файл.

    Дальше монтаж проходит ровно те же шаги, что и обычный хайлайт, чтобы режим
    не «терял» часть пайплайна:

    * **кроп** (:func:`clipper.reframe_video`) — обязательный шаг, как и в
      ``crop_highlights``: склейка выходит в исходных пропорциях, поэтому её
      нужно обрезать до ``ASPECT_RATIO``. Он выполняется всегда, независимо от
      ``enhance`` — иначе под ``MONTAGE_MODE=true`` (маршрут команды ``clip``,
      где ``enhance=False``) ролик остался бы в исходном формате;
    * **метаданные** (``TWO_STAGE_ANALYSIS=true``) — заголовок, описание и теги
      генерируются по реальной речи внутри монтажа;
    * **постобработка** (``enhance``) — вертикальная подгонка до
      ``FIT_ASPECT_RATIO``, цветокор и эффекты, уникализация, баннер,
      субтитры и музыка.
    """
    with timer.stage("transcribe"):
        transcript = transcribe(source_path, settings)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments; the video may have no detectable "
            f"speech: {os.path.basename(source_path)}"
        )

    # Optional visual indexing (same as ``clip``): gives the montage selector the
    # on-screen context so it can favour visually strong moments.
    visuals: Dict = {"engine": "none", "scenes": []}
    if settings.visual_indexer_enabled:
        visuals = index_video(source_path, settings)

    with timer.stage("montage_select"):
        plan = select_montage_segments(transcript, settings, visuals)

    segments: List[Dict] = plan.get("segments", [])
    if not segments:
        raise RuntimeError(
            f"Montage planner returned zero segments for "
            f"{os.path.basename(source_path)}."
        )

    # The montage is written next to the other clips, under its own name so a
    # batch does not overwrite anything. ``start_index`` keeps one running order
    # across several inputs (a single input still yields exactly one clip).
    out_dir = settings.resolve(settings.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    source_stem = os.path.splitext(os.path.basename(source_path))[0]
    number = start_index + 1
    out_path = os.path.join(out_dir, f"montage_{number:02d}_{source_stem}.mp4")

    # Stitch into a temporary file first: the segments come out in the *source*
    # aspect ratio, so the result still has to be re-framed before anything else
    # looks at it.
    stitched_path = out_path + ".stitched.mp4"
    try:
        with timer.stage("montage_cut"):
            create_montage(source_path, segments, stitched_path)

        # Crop step — the montage equivalent of what ``crop_highlights`` does per
        # highlight. It runs even when ``enhance`` is off (the ``clip`` route
        # under ``MONTAGE_MODE``), because the re-framing is not part of the
        # enhance stage: without it the clip would keep the source 16:9 frame and
        # ``ASPECT_RATIO`` would appear to be skipped. The re-frame preserves the
        # timeline frame for frame, so the montage transcript built below (and
        # with it the subtitles and the stage-2 metadata) still lines up.
        with timer.stage("crop"):
            reframe_video(
                stitched_path,
                out_path,
                settings.aspect_ratio,
                face_tracking=settings.face_tracking,
                slide_effect=settings.slide_effect,
                slide_gap=settings.slide_transition_gap,
                slide_range=settings.slide_range,
            )
    finally:
        # The stitched intermediate is only an input to the crop step; on Windows
        # a leftover file also holds a handle that blocks later cleanup.
        if os.path.exists(stitched_path):
            os.remove(stitched_path)

    # The transcript is remapped onto the stitched timeline (each segment shifted
    # by the running total of the previous ones): its window starts at 0, so both
    # the subtitles burned by ``enhance_shorts`` and the stage-2 metadata read the
    # speech of the finished montage instead of the source video's timings.
    montage_transcript = build_montage_transcript(transcript, segments)
    total = float(montage_transcript.get("duration") or 0.0)

    # Metadata for the single clip. The fallback title/description are derived
    # from what is actually said in the montage, never from what the script did;
    # with TWO_STAGE_ANALYSIS the LLM then rewrites title/description/tags from
    # that same speech (one extra call, exactly like stage 2 for regular clips).
    short = generate_montage_metadata(
        plan, build_clip_text(montage_transcript, 0.0, total)
    )
    short["clip_url"] = out_path
    short["start_time"] = 0.0
    short["end_time"] = total
    _log_montage_segments(segments)

    if settings.two_stage_analysis:
        with timer.stage("metadata"):
            enrich_highlights_metadata([short], montage_transcript, settings)

    if enhance:
        with timer.stage("enhance"):
            enhance_shorts(
                montage_transcript,
                [short],
                settings,
                add_music=add_music,
                burn_subtitles=burn_subtitles,
            )

    return {
        "source_video_url": source_path,
        "transcript": transcript,
        "visuals": visuals,
        "montage": plan,
        "highlights": [short],
        "shorts": [short],
    }


def _log_montage_segments(segments: List[Dict]) -> None:
    """Print the montage plan: each segment's window, duration and reason."""
    total = sum(float(s["end_time"]) - float(s["start_time"]) for s in segments)
    print(
        f"[montage] plan: {len(segments)} segment(s), total {total:.1f}s",
        flush=True,
    )
    for i, seg in enumerate(segments, 1):
        start = float(seg["start_time"])
        end = float(seg["end_time"])
        mark = "hook" if seg.get("hook") else "    "
        print(
            f"[montage]   {i}. [{mark}] {start:.2f}s -> {end:.2f}s "
            f"({end - start:.1f}s)  {seg.get('reason', '')}",
            flush=True,
        )


def _run_montage(
    settings: Settings,
    enhance: bool = False,
    add_music: bool = True,
    burn_subtitles: bool = True,
    on_video_done: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """Montage variant of :func:`_run`: one hook clip per source video.

    Each source is cut into connected segments and stitched back into a single
    50–90s video that then follows the normal post-processing path. Failures are
    isolated per source exactly like the ``clip`` pipeline.
    """
    if not settings.input:
        raise RuntimeError("No input given. Set INPUT in .env or pass -i/--input.")

    timer = start_timer()

    with timer.stage("download"):
        source_paths = resolve_input_videos(settings)

    videos: List[Dict] = []
    failures: List[Dict] = []
    rendered = 0  # running total of montage clips produced so far
    for source_path in source_paths:
        if len(source_paths) > 1:
            print(f"[pipeline] video: {os.path.basename(source_path)}", flush=True)
        try:
            video = _process_montage_one(
                source_path,
                settings,
                timer,
                enhance=enhance,
                add_music=add_music,
                burn_subtitles=burn_subtitles,
                start_index=rendered,
            )
        except Exception as exc:  # noqa: BLE001 - one bad video must not sink the rest
            detail = f"{type(exc).__name__}: {exc}"
            print(
                f"[pipeline] video failed: {os.path.basename(source_path)} "
                f"({detail}); continuing with the remaining inputs",
                flush=True,
            )
            failures.append({"source_video_url": source_path, "error": detail})
            if on_video_done is not None:
                on_video_done(_merge_result(source_paths, videos, failures))
            continue
        rendered += len(video.get("shorts") or [])
        videos.append(video)
        if on_video_done is not None:
            on_video_done(_merge_result(source_paths, videos, failures))

    return {
        **_merge_result(source_paths, videos, failures),
        "timings": timer.as_dict(),
    }


def _merge_result(
    source_paths: List[str],
    videos: List[Dict],
    failures: List[Dict],
) -> Dict:
    """Assemble the merged result from the videos processed so far.

    Shared by the final return value and by the incremental snapshots handed to
    ``on_video_done`` after every source (a failed one included), so the caller
    can persist partial progress — the ``shorts_info`` sheet — as the run goes
    instead of waiting for the whole set to succeed.
    """
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
        "failures": failures,
    }


def _run(
    settings: Settings,
    enhance: bool = False,
    add_music: bool = True,
    burn_subtitles: bool = True,
    on_video_done: Optional[Callable[[Dict], None]] = None,
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
    failures: List[Dict] = []
    rendered = 0  # running total of shorts produced so far (all videos)
    for source_path in source_paths:
        if len(source_paths) > 1:
            print(f"[pipeline] video: {os.path.basename(source_path)}", flush=True)
        try:
            video = _process_one(
                source_path,
                settings,
                timer,
                enhance=enhance,
                add_music=add_music,
                burn_subtitles=burn_subtitles,
                start_index=rendered,
            )
        except Exception as exc:  # noqa: BLE001 - one bad video must not sink the rest
            # Isolate the failure: a single source (bad audio, exhausted LLM
            # quota, a broken ffmpeg run) is recorded and the loop moves on to
            # the next input. KeyboardInterrupt/SystemExit are BaseExceptions,
            # so Ctrl+C still stops the whole run as usual.
            detail = f"{type(exc).__name__}: {exc}"
            print(
                f"[pipeline] video failed: {os.path.basename(source_path)} "
                f"({detail}); continuing with the remaining inputs",
                flush=True,
            )
            failures.append({"source_video_url": source_path, "error": detail})
            # Still hand out what has been produced so far so partial results
            # (and their clips) get saved even when a later video fails.
            if on_video_done is not None:
                on_video_done(_merge_result(source_paths, videos, failures))
            continue
        rendered += len(video.get("shorts") or [])
        videos.append(video)
        if on_video_done is not None:
            on_video_done(_merge_result(source_paths, videos, failures))

    return {
        **_merge_result(source_paths, videos, failures),
        "timings": timer.as_dict(),
    }


def generate_shorts(
    settings: Settings,
    enhance: bool = False,
    *,
    add_music: bool = True,
    burn_subtitles: bool = True,
    on_video_done: Optional[Callable[[Dict], None]] = None,
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
        on_video_done: optional callback invoked after every source video is
            processed (successful or failed) with the merged result assembled so
            far. Lets a caller persist partial progress — e.g. write the
            ``shorts_info`` sheet after each video — instead of losing it when a
            later input aborts the run.

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
          "failures": [...],     # sources that failed (kept when the rest ran)
        }
    """
    return _run(
        settings,
        enhance=enhance,
        add_music=add_music,
        burn_subtitles=burn_subtitles,
        on_video_done=on_video_done,
    )


def generate_montage(
    settings: Settings,
    enhance: bool = False,
    *,
    add_music: bool = True,
    burn_subtitles: bool = True,
    on_video_done: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """Cut one video into connected segments and stitch them into one hook clip.

    The LLM picks the most dynamic, semantically connected moments whose total
    runtime lands in the 50–90s window; the script then cuts and joins them into
    a single video that continues through the normal pipeline (vertical frame,
    colour/lens effects, music, burned-in subtitles) — so one long video becomes
    one short, punchy "hook" montage.

    Args:
        settings: resolved configuration. The montage length and the per-segment
            bounds are not settings — they are hardcoded constants
            (``MONTAGE_*_SECONDS``) at the top of ``montage.py``, right beside the
            prompt, so the whole duration policy is tuned in one place by hand.
        enhance: when ``True``, post-process the stitched clip (music + burned-in
            subtitles), mirroring the ``all`` command.
        add_music: with ``enhance``, mix in background music (default: on).
        burn_subtitles: with ``enhance``, burn subtitles into the clip (default:
            on). Cue timings are remapped onto the stitched timeline so the text
            stays in sync with the joined segments.
        on_video_done: optional callback invoked after every source video is
            processed with the merged result assembled so far, so a caller can
            persist partial progress.

    Returns:
        The same shape as :func:`generate_shorts` (``videos`` / ``highlights`` /
        ``shorts`` / ``failures`` / ``timings``); each video carries exactly one
        short whose ``clip_url`` is the stitched montage and whose
        ``montage_segments`` record the plan.
    """
    return _run_montage(
        settings,
        enhance=enhance,
        add_music=add_music,
        burn_subtitles=burn_subtitles,
        on_video_done=on_video_done,
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
