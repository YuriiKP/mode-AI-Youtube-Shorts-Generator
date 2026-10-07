"""Montage mode: cut one video into connected segments and stitch them into one hook clip.

В этом режиме LLM отбирает самые динамичные и связные по смыслу отрезки
с общей длительностью 50–90 секунд, скрипт их вырезает и склеивает вместе,
затем результат идёт дальше по пайплайну (музыка, субтитры, эффекты).

Результат — один короткий «хуковый» ролик вместо нескольких отдельных клипов.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Callable, Dict, List, Optional

from .config import Settings
from .cues import phrase_boundaries
from .llm import call_llm

LLMFn = Callable[[str], str]

# --- Длительность монтажа: настройки зашиты здесь, а не в .env ---------------
# Эти значения намеренно живут рядом с промптом, а не в .env: их правят руками
# вместе с текстом инструкции, поэтому всё, что касается длительности и границ,
# лежит в одном месте. Крутить только здесь — других источников значений нет.
#
# Целевая общая длительность, секунды. На неё ориентируется промпт, и к ней
# доводится сумма отрезков после ответа модели (см. ``_fit_total_duration``).
MONTAGE_TARGET_SECONDS = 60.0
# Рамки одного отрезка: слишком короткие (<3–5с) дают рваный ритм, слишком
# длинные — «провисшую» динамику. Отрезки вне этих рамок отбрасываются.
MONTAGE_MIN_SEGMENT_SECONDS = 5.0
MONTAGE_MAX_SEGMENT_SECONDS = 20.0
# Окно общей длительности. Монтаж короче или длиннее не принимается: сумму
# подтягивают (``_fit_total_duration``), а промпт переспрашивают с уточнением.
MONTAGE_MIN_TOTAL_SECONDS = 50.0
MONTAGE_MAX_TOTAL_SECONDS = 90.0


# 1. Динамика и виральность: Выбирай моменты с максимальным вовлечением — конфликты, эмоции, неожиданные повороты, смешные реакции, интригующие вопросы.
# 5. Длительность отрезков: Каждый отрезок от {min_segment}с до {max_segment}с. Слишком короткие (<3с) создают рваный ритм.
MONTAGE_SYSTEM_PROMPT = """Ты элитный монтажёр коротких вертикальных видео. Твоя задача — отобрать самые динамичные и связные по смыслу отрезки из длинного видео и собрать из них один интересный «хуковый» клип длительностью {target_duration} секунд (допустим диапазон 50–90 секунд).

ПРАВИЛА ОТБОРА ОТРЕЗКОВ:
1. Связность по смыслу: Отрезки должны логически переходить друг в друга. Не прыгай между несвязанными темами. Если это невозможно — бери один длинный связный кусок.
2. Хук в начале: Первый отрезок должен начинаться с мощного хука — провокационная фраза, внезапный конфликт, интригующий вопрос или эпичное действие.
3. Панчлайн в конце: Последний отрезок должен заканчиваться чёткой развязкой — смешная реакция, неожиданный твист, эпичная фраза, яркая эмоция.
4. Общая длительность: Сумма всех отрезков должна быть около {target_duration} секунд (допустим диапазон 50–90с).
5. Границы: Бери точные start_time и end_time из лога [начало - конец]. Начинай и заканчивай на границах фраз, не режь посреди предложения.

{content_hint}

{visual_hint}

Отвечай ТОЛЬКО валидным JSON (без markdown, без пояснений):
{{"segments":[{{"start_time":float,"end_time":float,"reason":"string","hook":bool}}],"total_duration":float,"montage_concept":"string"}}

Где:
- segments: массив отрезков в хронологическом порядке
- start_time, end_time: точные границы в секундах из лога
- reason: одним предложением — почему этот отрезок выбран
- hook: true для первого отрезка (с хуком), false для остальных
- total_duration: сумма длительностей всех отрезков
- montage_concept: одно предложение — общая идея/тема всего монтажа
"""


def _parse_json_loose(raw: str) -> Dict:
    """Strip markdown fences and parse JSON."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise


def _build_transcript_log(transcript: Dict, visuals: Optional[Dict] = None) -> str:
    """Build a text log of the transcript for the LLM, optionally with visuals."""
    from .highlights import build_transcript_log

    return build_transcript_log(transcript, visuals)


def _content_hint(transcript: Dict) -> str:
    """Detect content type hint from transcript metadata if available."""
    content_type = transcript.get("content_type", "").lower()
    if content_type in ("podcast", "interview"):
        return "Контент: разговорный. Ищи конфликты мнений, провокационные вопросы, эмоциональные реакции."
    elif content_type in ("tutorial", "lecture"):
        return "Контент: обучающий. Ищи яркие демонстрации, неожиданные факты, ага-моменты."
    elif content_type == "commentary":
        return "Контент: комментарий. Ищи резкие оценки, споры, смешные реакции."
    elif content_type == "anime":
        return (
            "Контент: аниме. Ищи эпичные битвы, драматичные моменты, смешные реакции."
        )
    return "Контент: общий. Ищи любые моменты с максимальным вовлечением."


def _visual_hint(visuals: Optional[Dict]) -> str:
    """Build visual indexing hint if available."""
    if not visuals or visuals.get("engine") == "none":
        return ""
    return (
        "В логе есть визуальные описания [Visuals: ...]. Учитывай их при отборе: "
        "яркие действия, эмоции на лицах и динамичная картинка усиливают виральность."
    )


def _validate_segments(
    segments: List[Dict],
    duration: float,
    min_segment: float,
    max_segment: float,
) -> List[Dict]:
    """Проверить и отфильтровать отрезки, вернувшиеся от LLM.

    Отбрасывает всё, что не ложится на исходное видео: битые или неполные
    тайминги, выход за пределы ролика, нулевые/обратные окна и отрезки вне
    рамок ``min_segment``/``max_segment``. Затем сортирует отрезки по времени и
    убирает пересечения, оставляя их в хронологическом порядке — так план монтажа
    не может «наложиться» сам на себя.
    """
    valid: List[Dict] = []
    for seg in segments:
        try:
            start = float(seg["start_time"])
            end = float(seg["end_time"])
        except (KeyError, TypeError, ValueError):
            continue

        if start < 0 or end > duration or start >= end:
            continue

        seg_duration = end - start
        if seg_duration < min_segment or seg_duration > max_segment:
            continue

        valid.append(
            {
                "start_time": start,
                "end_time": end,
                "duration": seg_duration,
                "reason": str(seg.get("reason", "")).strip(),
                "hook": bool(seg.get("hook", False)),
            }
        )

    # Sort chronologically and remove overlaps
    valid.sort(key=lambda s: s["start_time"])
    deduped: List[Dict] = []
    for seg in valid:
        if not deduped or seg["start_time"] >= deduped[-1]["end_time"]:
            deduped.append(seg)

    return deduped


def _total_duration(segments: List[Dict]) -> float:
    """Суммарная длительность отрезков монтажа в секундах."""
    return sum(float(seg["end_time"]) - float(seg["start_time"]) for seg in segments)


def _phrase_end_times(transcript: Dict, pause_threshold: float) -> List[float]:
    """Моменты исходного видео, где действительно заканчивается фраза.

    Это концы предложений (``.!?…``) и места настоящих пауз в речи — те же
    границы, по которым рвутся субтитры (см. ``cues.phrase_boundaries``), чтобы
    отрезок монтажа никогда не обрывался посреди мысли.
    """
    return [
        float(time)
        for time, _ in phrase_boundaries(
            transcript.get("segments", []) or [],
            pause_threshold=pause_threshold,
        )
    ]


def _fit_total_duration(
    segments: List[Dict],
    transcript: Dict,
    *,
    target: float,
    min_total: float,
    max_total: float,
    min_segment: float,
    max_segment: float,
    pause_threshold: float = 0.6,
) -> List[Dict]:
    """Довести суммарную длительность монтажа до окна 50–90 секунд.

    LLM отбирает отрезки по смыслу, но не всегда попадает в окно: сумма может
    уйти за верхнюю границу (монтаж провисает и теряет динамику) или не добрать
    до нижней (ролик выходит короче обещанного). Здесь длительность подтягивается
    к ``target``: слишком длинная нарезка подрезается с конца, слишком короткая —
    достраивается целыми фразами. Обе границы двигаются только по настоящим
    границам фраз (``cues.phrase_boundaries``), поэтому отрезок не обрывается
    посреди слова, а ``min_segment``/``max_segment`` держат каждый отрезок в
    заданных рамках.
    """
    # Работаем с копиями: план LLM остаётся нетронутым для логов и инфо-листа.
    segments = [dict(seg) for seg in segments]
    if not segments:
        return segments

    video_end = float(transcript.get("duration") or 0.0)
    ends = _phrase_end_times(transcript, pause_threshold)

    # --- слишком длинный монтаж: подрезаем концовку --------------------------
    # Режем только до верхней границы окна (max_total), а не до target: любая
    # длительность внутри окна 50–90с годится, поэтому незачем выбрасывать
    # хорошие отрезки сверх необходимого.
    while segments and _total_duration(segments) > max_total:
        excess = _total_duration(segments) - max_total
        last = segments[-1]
        start = float(last["start_time"])
        end = float(last["end_time"])
        # Не сжимаем отрезок меньше минимального.
        floor = start + min_segment
        wanted = max(floor, end - excess)
        # Округляем подрезку вниз до конца целой фразы — но не за ``wanted``.
        candidate = max(floor, wanted)
        snapped = max(
            [e for e in ends if floor <= e <= candidate + 1e-6], default=candidate
        )
        if snapped >= end - 1e-6:
            # Подрезать последний отрезок уже нельзя — убираем его целиком.
            segments.pop()
            continue
        last["end_time"] = snapped
        last["duration"] = snapped - start

    if not segments:
        return segments

    # --- слишком короткий монтаж: достраиваем целыми фразами -----------------
    # Концы тянем вперёд по одному предложению за проход; затем, если всё ещё
    # мало, начало самого первого отрезка сдвигаем на фразу назад. Цикл сходится:
    # каждая граница строго монотонна, а рост ограничен max_segment и границами
    # соседних отрезков / концом видео.
    while _total_duration(segments) < min_total:
        grown = False

        for i, seg in enumerate(segments):
            if _total_duration(segments) >= target:
                break
            start = float(seg["start_time"])
            end = float(seg["end_time"])
            limit = (
                float(segments[i + 1]["start_time"])
                if i + 1 < len(segments)
                else video_end
            )
            ceiling = start + max_segment
            if limit <= end:
                continue
            nxt = min([e for e in ends if end + 1e-6 < e <= limit], default=None)
            if nxt is not None and nxt <= ceiling:
                seg["end_time"] = nxt
                seg["duration"] = nxt - start
                grown = True

        if not grown:
            # Начало монтажа назад — целой фразой, но не длиннее max_segment.
            first = segments[0]
            start = float(first["start_time"])
            end = float(first["end_time"])
            prev = max([e for e in ends if e < start - 1e-6], default=None)
            if prev is not None and (end - prev) <= max_segment:
                first["start_time"] = prev
                first["duration"] = end - prev
                grown = True

        if not grown:
            break

    return segments


def select_montage_segments(
    transcript: Dict,
    settings: Settings,
    visuals: Optional[Dict] = None,
) -> Dict:
    """Ask LLM to select dynamic segments for montage and return the plan.

    Длительность и рамки отрезков — не настройки из ``.env``, а константы
    ``MONTAGE_*_SECONDS`` в начале этого модуля: они лежат рядом с промптом,
    потому что правятся руками вместе с текстом инструкции. ``settings`` нужен
    только для провайдера/модели LLM и порога паузы субтитров.

    Returns:
        {
          "segments": [{"start_time", "end_time", "duration", "reason", "hook"}, ...],
          "total_duration": float,
          "montage_concept": str
        }
    """
    duration = float(transcript.get("duration", 0.0))
    if duration == 0:
        raise RuntimeError("Transcript has zero duration")

    target = MONTAGE_TARGET_SECONDS
    min_seg = MONTAGE_MIN_SEGMENT_SECONDS
    max_seg = MONTAGE_MAX_SEGMENT_SECONDS

    text_log = _build_transcript_log(transcript, visuals)
    content = _content_hint(transcript)
    visual = _visual_hint(visuals)

    system = MONTAGE_SYSTEM_PROMPT.format(
        target_duration=int(target),
        min_segment=int(min_seg),
        max_segment=int(max_seg),
        content_hint=content,
        visual_hint=visual,
    )
    prompt = f"{system}\n\nТранскрипт:\n{text_log}"

    print(
        f"[montage] selecting segments for {duration:.0f}s video "
        f"(target: {target:.0f}s, segments: {min_seg:.0f}-{max_seg:.0f}s)",
        flush=True,
    )

    llm_fn: LLMFn = lambda p: call_llm(p, settings)  # noqa: E731
    max_attempts = 3
    last_error = "unknown"

    # Лучший из полученных вариантов. Сюда же попадает нарезка, которая после
    # доводки длительности так и не вошла в окно 50–90с (например, в видео
    # просто нет материала на 50 секунд): вернуть её всё равно полезнее, чем
    # уронить шаг с ошибкой.
    best: Optional[Dict] = None
    best_gap = float("inf")

    def _gap(total: float) -> float:
        """Расстояние суммарной длительности до окна 50–90с (0 — внутри окна)."""
        if total < MONTAGE_MIN_TOTAL_SECONDS:
            return MONTAGE_MIN_TOTAL_SECONDS - total
        if total > MONTAGE_MAX_TOTAL_SECONDS:
            return total - MONTAGE_MAX_TOTAL_SECONDS
        return 0.0

    feedback = ""

    for attempt in range(1, max_attempts + 1):
        # Первая попытка идёт по чистому промпту; на повторах к нему добавляется
        # подсказка, почему предыдущий ответ не подошёл (невалидный JSON или
        # промах по длительности).
        prompt = (
            f"{system}\n\n{feedback}\n\nТранскрипт:\n{text_log}" if feedback else prompt
        )
        raw = llm_fn(prompt)
        try:
            parsed = _parse_json_loose(raw)
            raw_segments = parsed.get("segments", [])
            if not isinstance(raw_segments, list):
                raise ValueError("segments must be an array")

            segments = _validate_segments(raw_segments, duration, min_seg, max_seg)
            if not segments:
                raise ValueError("no valid segments after validation")

            # LLM не всегда попадает в окно 50–90с: сумма отрезков может выйти и
            # слишком длинной, и слишком короткой. Доводим её до цели по
            # настоящим границам фраз, не обрывая мысль на полуслове.
            segments = _fit_total_duration(
                segments,
                transcript,
                target=target,
                min_total=MONTAGE_MIN_TOTAL_SECONDS,
                max_total=MONTAGE_MAX_TOTAL_SECONDS,
                min_segment=min_seg,
                max_segment=max_seg,
                pause_threshold=settings.subtitle_pause_threshold,
            )
            if not segments:
                raise ValueError("no segments left after duration fitting")

            total = _total_duration(segments)
            concept = str(parsed.get("montage_concept", "")).strip()
            plan = {
                "segments": segments,
                "total_duration": total,
                "montage_concept": concept or "динамичный монтаж",
            }

            # В окно попали — это готовый ответ, дальше спрашивать нечего.
            if _gap(total) == 0.0:
                print(
                    f"[montage] selected {len(segments)} segment(s), "
                    f"total duration: {total:.1f}s",
                    flush=True,
                )
                return plan

            # Мимо окна: запоминаем вариант и пробуем ещё раз с уточнением.
            gap = _gap(total)
            print(
                f"[montage] attempt {attempt}/{max_attempts}: "
                f"{len(segments)} segment(s), {total:.1f}s — outside the "
                f"50-90s window (gap {gap:.1f}s)",
                flush=True,
            )
            if gap < best_gap:
                best, best_gap = plan, gap

            if attempt < max_attempts:
                need = (
                    f"не хватает примерно {MONTAGE_MIN_TOTAL_SECONDS - total:.0f}с — "
                    "добавь ещё связных по смыслу отрезков (можно взять другой "
                    "момент видео или рядом стоящие реплики)"
                    if total < MONTAGE_MIN_TOTAL_SECONDS
                    else f"нарезка на {total - MONTAGE_MAX_TOTAL_SECONDS:.0f}с длиннее "
                    "нужного — убери самый слабый/наименее связанный отрезок или "
                    "укороти его"
                )
                feedback = (
                    f"ПРЕДЫДУЩАЯ ПОПЫТКА НЕ ПОДОШЛА: сумма отрезков вышла {total:.0f}с, "
                    f"это вне диапазона 50–90с — {need}. "
                    "Сумма всех отрезков обязана попасть в 50–90 секунд."
                )
                last_error = f"total {total:.1f}s outside the 50-90s window"
                continue

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        print(
            f"[montage] invalid response on attempt {attempt}/{max_attempts} ({last_error})",
            flush=True,
        )

        if attempt < max_attempts:
            feedback = (
                "ПРЕДЫДУЩИЙ ОТВЕТ НЕ РАЗОБРАН. Верни ТОЛЬКО валидный JSON со структурой: "
                '{"segments":[{"start_time":float,"end_time":float,"reason":"string","hook":bool}],'
                '"total_duration":float,"montage_concept":"string"}. '
                "Без markdown-ограждений, без комментариев. Сумма отрезков обязана "
                "попасть в 50–90 секунд."
            )

    # Все попытки промахнулись: отдаём лучший вариант, если он есть — короткий
    # или длинный монтаж лучше, чем отсутствие клипа.
    if best is not None:
        total = best["total_duration"]
        print(
            f"[montage] returning best effort after {max_attempts} attempts: "
            f"{len(best['segments'])} segment(s), {total:.1f}s",
            flush=True,
        )
        return best

    raise RuntimeError(
        f"Montage segment selection failed after {max_attempts} attempts: {last_error}"
    )


def build_montage_transcript(
    transcript: Dict,
    segments: List[Dict],
) -> Dict:
    """Remap transcript cue timings onto the stitched montage timeline.

    Каждый отрезок идёт в монтаже подряд, поэтому его реплики нужно сдвинуть
    на суммарную длительность всех предыдущих отрезков. Полученный транскрипт
    описывает уже склеенное видео: его можно передать в enhance_shorts, и
    субтитры лягут синхронно (см. pipeline), а не по таймингам исходника.

    Returns:
        {"duration": float, "segments": [{"start", "end", "text"}, ...]}
    """
    source_segments = transcript.get("segments", []) or []
    cues: List[Dict] = []
    offset = 0.0

    for seg in segments:
        try:
            seg_start = float(seg["start_time"])
            seg_end = float(seg["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        seg_duration = seg_end - seg_start
        if seg_duration <= 0:
            continue

        for source in source_segments:
            try:
                src_start = float(source["start"])
                src_end = float(source["end"])
            except (KeyError, TypeError, ValueError):
                continue
            # Реплика целиком вне отрезка — пропускаем.
            if src_end <= seg_start or src_start >= seg_end:
                continue

            clipped_start = max(src_start, seg_start) - seg_start + offset
            clipped_end = min(src_end, seg_end) - seg_start + offset
            if clipped_end <= clipped_start:
                continue

            text = (
                str(source.get("text", "")).strip().replace("\r", "").replace("\n", " ")
            )
            if not text:
                continue

            cues.append({"start": clipped_start, "end": clipped_end, "text": text})

        offset += seg_duration

    return {"duration": offset, "segments": cues}


def _cut_segment(source_path: str, start: float, end: float, output_path: str) -> str:
    """Cut one segment from source with ffmpeg.

    ``-ss`` стоит *до* ``-i`` (input seeking): ffmpeg перематывает по индексу, а
    не декодирует видео с самого начала до нужной секунды. Для монтажа это важно,
    потому что отрезков несколько и они могут лежать далеко от начала — иначе
    каждый отрезок тянул бы за собой декодирование всего предшествующего куска
    (вплоть до всего ролика). ``-t`` после ``-i`` задаёт длительность отрезка на
    выходе; при перекодировании input-seek точный (accurate_seek), так что
    границы не разъезжаются на ключевых кадрах.
    """
    duration = max(0.0, float(end) - float(start))
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        source_path,
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-map_chapters",
        "-1",
        "-sn",
        "-dn",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    return output_path


def _stitch_segments(segment_paths: List[str], output_path: str) -> str:
    """Stitch multiple segments into one video with ffmpeg concat demuxer."""
    if not segment_paths:
        raise ValueError("No segments to stitch")

    if len(segment_paths) == 1:
        # Only one segment — just rename/copy it
        import shutil

        shutil.copy2(segment_paths[0], output_path)
        return output_path

    # Create concat list file
    list_path = output_path + ".concat.txt"
    try:
        with open(list_path, "w", encoding="utf-8") as f:
            for path in segment_paths:
                # The concat demuxer resolves the entries *relative to the list
                # file*, not to the working directory (and Windows backslashes
                # would be read as escapes), so write absolute paths with forward
                # slashes. ``-safe 0`` below allows the absolute paths through.
                absolute = os.path.abspath(path).replace(os.sep, "/")
                # Escape single quotes for the ffmpeg concat demuxer.
                escaped = absolute.replace("'", r"'\''")
                f.write(f"file '{escaped}'\n")

        base = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            # Regenerate presentation timestamps so each segment starts where the
            # previous one ended; without it some players stall on the joins.
            "-fflags",
            "+genpts",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
        ]

        # Every segment was encoded by ``_cut_segment`` with the very same codec
        # and parameters, so the streams are compatible: a stream copy joins them
        # without a second full re-encode (the result still gets re-encoded later
        # by the post-processing pass). Falls back to a re-encode if the copy is
        # refused — a mismatched stream (different resolution, broken DTS) would
        # otherwise leave no output at all.
        try:
            subprocess.run(
                [*base, "-c", "copy", "-map_chapters", "-1", "-sn", "-dn", output_path],
                check=True,
            )
        except subprocess.CalledProcessError:
            print(
                "[montage] stream copy concat failed; re-encoding the join",
                flush=True,
            )
            subprocess.run(
                [
                    *base,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-map_chapters",
                    "-1",
                    "-sn",
                    "-dn",
                    output_path,
                ],
                check=True,
            )
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)

    return output_path


def create_montage(
    source_path: str,
    segments: List[Dict],
    output_path: str,
    temp_dir: Optional[str] = None,
) -> str:
    """Cut segments from source and stitch them into one montage video.

    Args:
        source_path: path to source video
        segments: list of {"start_time", "end_time", ...} dicts
        output_path: where to write the final montage
        temp_dir: where to write temporary segment files (default: same as output)

    Returns:
        output_path
    """
    if not segments:
        raise ValueError("No segments to create montage")

    temp_dir = temp_dir or os.path.dirname(output_path)
    os.makedirs(temp_dir, exist_ok=True)

    segment_paths: List[str] = []
    try:
        for i, seg in enumerate(segments, 1):
            start = seg["start_time"]
            end = seg["end_time"]
            temp_path = os.path.join(temp_dir, f"montage_seg_{i:02d}.mp4")
            print(
                f"[montage] cutting segment {i}/{len(segments)}: "
                f"{start:.1f}s -> {end:.1f}s ({end - start:.1f}s)",
                flush=True,
            )
            _cut_segment(source_path, start, end, temp_path)
            segment_paths.append(temp_path)

        print(
            f"[montage] stitching {len(segment_paths)} segment(s) into final video",
            flush=True,
        )
        _stitch_segments(segment_paths, output_path)

    finally:
        # Clean up temporary segment files
        for path in segment_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    return output_path


# Границы предложений: по ним первые фразы монтажа вырезаются в заголовок и
# описание. Нужны только для запасного варианта метаданных — когда их генерирует
# LLM (TWO_STAGE_ANALYSIS=true), этот путь остаётся лишь страховкой.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _first_sentences(text: str, limit: int, max_chars: int) -> str:
    """Первые ``limit`` предложений текста, уложенные в ``max_chars`` символов.

    Текст приводится к одному пробелу и режется по концам предложений
    (``.!?…``); для транскрипта без пунктуации возвращается начало строки.
    Пустая строка на входе даёт пустую строку на выходе.
    """
    text = " ".join(str(text or "").split())
    if not text:
        return ""
    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    if not sentences:
        sentences = [text]

    out = ""
    for sentence in sentences[:limit]:
        candidate = f"{out} {sentence}".strip()
        # Дальше приклеиваем только пока влезаем в лимит: первое предложение
        # берётся всегда, даже если оно само длиннее лимита.
        if out and len(candidate) > max_chars:
            break
        out = candidate
    return out or text[:max_chars]


def generate_montage_metadata(plan: Dict, montage_text: str = "") -> Dict:
    """Заголовок, описание и теги монтажа — про содержание, а не про процесс.

    Запасной вариант (этап 2 выключен, не ответил или вернул пустое поле)
    описывает то, **о чём** ролик, и никогда — что с ним сделал скрипт:
    заголовок берётся из идеи монтажа (``montage_concept`` с этапа плана), иначе
    из первой произнесённой фразы; описание собирается из первых предложений
    реальной речи внутри монтажа; теги — минимальный набор, который затем
    уточняет этап 2.

    ``montage_text`` — речь склейки (обычно ``highlights.build_clip_text`` по
    ремапнутому транскрипту). При ``TWO_STAGE_ANALYSIS=true`` эти значения
    перезаписывает этап 2 (:func:`highlights.enrich_highlights_metadata`) по тому
    же тексту, поэтому запасной вариант нужен лишь как страховка от неудачного
    вызова.

    Returns:
        Словарь клипа, совместимый с остальным пайплайном (музыка, субтитры,
        эффекты) и с публикацией.
    """
    segments = plan.get("segments", [])
    if not segments:
        raise ValueError("No segments in montage plan")

    first = segments[0]
    concept = str(plan.get("montage_concept") or "").strip()
    total = float(plan.get("total_duration") or 0.0)
    if total <= 0:
        total = sum(
            float(seg["end_time"]) - float(seg["start_time"]) for seg in segments
        )

    # Запасные метаданные — из речи монтажа, а не из того, как он собран:
    # «Склейка из N отрезков» описывает работу скрипта, а не ролик, поэтому в
    # заголовок и описание такое больше не попадает. Но в логе (virality_reason)
    # состав склейки остаётся — там он как раз уместен.
    spoken = " ".join(str(montage_text or "").split())
    title = concept or _first_sentences(spoken, 1, 60) or "Нарезка лучших моментов"
    if len(title) > 60:
        title = title[:57].rstrip() + "..."
    description = _first_sentences(spoken, 2, 240) or concept

    # Find the hook segment (marked by LLM)
    hook_segment = next((s for s in segments if s.get("hook")), first)
    hook_reason = hook_segment.get("reason", "")
    # Хук и панчлайн монтажа — первая и последняя фразы склейки (этап 2 уточнит
    # их по своей разметке, если он включён).
    hook_sentence = _first_sentences(spoken, 1, 200) or title
    punchline = _SENTENCE_SPLIT.split(spoken)[-1].strip() if spoken else concept

    return {
        "title": title,
        "description": description,
        "tags": ["shorts"],
        # Окно клипа — это таймлайн уже склеенного видео (он начинается с 0),
        # а не тайминги исходника: по этому окну enhance_shorts режет субтитры
        # из ремапнутого транскрипта (montage.build_montage_transcript).
        "start_time": 0.0,
        "end_time": total,
        "score": 85,  # High score for montage
        "clip_type": "montage",
        "virality_reason": f"Монтаж из {len(segments)} лучших моментов: {hook_reason}",
        "hook_sentence": hook_sentence,
        "punchline": punchline,
        "laugh_score": 3,
        "cringe_score": 1,
        "intrigue_score": 4,
        "montage_segments": segments,  # Store original segments for reference
        "montage_concept": concept,
    }
