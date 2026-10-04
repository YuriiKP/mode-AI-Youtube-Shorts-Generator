"""Find the most viral-worthy highlights in a transcript.

Logic ported from ViralVadoo's transcript_analysis/highlight_generator.py:
  - content-type / density self-detection folded into the highlight call
  - chunking for long videos with overlap
  - virality-criteria prompt
  - score-based dedupe with overlap suppression

The LLM is pluggable via the `llm_fn` argument; :func:`get_highlights` binds it
to the configured provider (OpenAI / DeepSeek / Gemini, selected by
``LLM_PROVIDER`` and read from the resolved settings).
"""

import json
import re
from typing import Callable, Dict, List, Optional, Tuple

from .config import Settings
from .cues import phrase_boundaries
from .llm import call_llm
from .visual_indexer import format_merged_log, merge_transcripts_and_visuals

LLMFn = Callable[[str], str]

# --- Russian prompts ---
CONTENT_TYPE_PROMPT = """
Проанализируй образец транскрипта (диалоги + возможные визуальные описания) и определи тип контента.
Выбери один: podcast, interview, tutorial, lecture, commentary, debate, vlog, anime, other.
Также оцени плотность контента: low (в основном вода/тишина), medium или high (плотная информация/динамика).
Отвечай ТОЛЬКО в формате JSON: {"content_type": "...", "density": "..."}
"""


VIRALITY_CRITERIA = """Ты элитный режиссер монтажа Shorts, Reels и TikTok. Твоя задача — извлечь самые виральные, логически завершенные и цельные моменты из лога видео.

{virality_criteria}

{content_hint}

{visual_hint}

ПРАВИЛА НАРЕЗКИ:
1. Длительность: 20–60 секунд. Игнорируй микро-моменты короче 15 секунд. Если логика сцены требует 60 секунд для полного раскрытия контекста — бери весь фрагмент.
2. Границы клипа: Начинай ровно с хука и заканчивай чётким панчлайном. Бери точные значения start_time и end_time из лога [начало - конец].
3. Изолированность: Хайлайты не должны пересекаться и не должны стоять впритык друг к другу. Выбирай только лучшие моменты с паузами между ними.
4. Оценка 0-100 по виральному потенциалу.
5. {num_clips_instruction}

Отвечай ТОЛЬКО валидным JSON (без markdown, без пояснений) со строгим соблюдением структуры:
{{"content_type":"string","density":"string","highlights":[{{"clip_type":"string","title":"string","description":"string","tags":["string"],"start_time":float,"end_time":float,"score":int,"laugh_score":int,"cringe_score":int,"intrigue_score":int,"hook_sentence":"string","punchline":"string","virality_reason":"string"}}]}}"""

HIGHLIGHT_SYSTEM_PROMPT = """Ты элитный редактор коротких вертикальных видео, изучивший тысячи вирусных клипов в TikTok, Instagram Reels и YouTube Shorts. Ты точно знаешь, что заставляет зрителей прекратить листать, досматривать до конца и делиться.

{virality_criteria}

{content_hint}

{visual_hint}

Твоя задача: определить самые виральные моменты (хайлайты) в логе видео.

Правила:
- В логе КАЖДАЯ реплика помечена своим интервалом [начало - конец] в секундах. Значения start_time и end_time бери РОВНО из границ выбранных реплик: start_time — начало реплики-хука, end_time — конец реплики-панчлайна. Не округляй до соседних строк и не бери окно целиком.
- В окно клипа должны попасть ТОЛЬКО реплики от хука до панчлайна. Не захватывай реплику ПЕРЕД хуком и реплику ПОСЛЕ панчлайна — именно из-за них в клип залезают соседние сцены.
- Начинай клип ровно на хуке и заканчивай ровно на панчлайне. Не тяни разгон, контекст и продолжение «на всякий случай».
- Предпочтительная длительность: 20–50 секунд. Идеальный одиночный one-liner может быть короче; развёрнутая история — до 60 секунд.
- Никогда не обрезай посреди предложения или мысли — каждый клип должен ощущаться завершённым и самодостаточным.
- Клипы не должны существенно пересекаться друг с другом.
- Бесшовность (Loop): по возможности выводи финал обратно к началу.
- Оценка 0-100 по виральному потенциалу (а не по общему качеству).
- {num_clips_instruction}
- Укажи "hook_sentence" — точную первую фразу клипа, которая заставляет остановиться.
- Укажи "punchline" — точную последнюю фразу-развязку.
- Проставь "intrigue_score", "laugh_score" и "cringe_score" от 0 до 5 — силы интриги, юмора и «испанского стыда» в клипе.
- Укажи "clip_type" — короткий тип момента (например: shock, conflict, reveal, joke, emotional).
- Объясни одним предложением, почему этот клип виральный ("virality_reason").
- Дай "description" — короткое цепляющее описание клипа на 1-2 предложения для заполнения при публикации на площадках (YouTube Shorts, TikTok, Instagram Reels). Пиши цепляюще и по сути, как подпись к видео, без хэштегов, эмодзи и markdown.
- Дай "tags" — массив из 3–5 коротких хэштегов (без символа #), точно по теме клипа, для продвижения на YouTube Shorts и TikTok. Пиши теги латиницей, при необходимости добавь тег на языке контента. Пример: ["anime","shorts","аниме"].

Отвечай ТОЛЬКО валидным JSON (без markdown, без пояснений):
{{"content_type":"string","density":"string","highlights":[{{"clip_type":"string","title":"string","description":"string","tags":["string"],"start_time":float,"end_time":float,"score":int,"laugh_score":int,"cringe_score":int,"intrigue_score":int,"hook_sentence":"string","punchline":"string","virality_reason":"string"}}]}}"""


CHUNK_SIZE_SECONDS = 1200  # 20-min chunks for long videos
LONG_VIDEO_THRESHOLD = 1800  # chunk videos longer than 30 min
CHUNK_OVERLAP_SECONDS = 60
MAX_HIGHLIGHT_API_ATTEMPTS = 3

# Насколько далеко край клипа может уехать (в секундах) от границы реплики,
# выбранной LLM, чтобы попасть на настоящую границу фразы — конец предложения
# или паузу. Ограничение не даёт транскрипту без пунктуации и пауз растянуть
# или обрезать клип далеко от задуманного окна.
_MAX_PHRASE_SHIFT = 9.0

# Насколько «ближе» считается граница, закрывающая предложение (с точкой,
# вопросом или восклицанием), по сравнению с границей, за которой лишь пауза.
# Пауза вполне может стоять посреди мысли («… Всё-таки <пауза> она…»), и тогда
# короткий обрубок в концовке тоже заканчивается паузой. Небольшая фора
# заставляет предпочесть стоящий рядом настоящий конец предложения, но при
# этом длинный обрубок, до конца которого LLM явно дотянулось, остаётся.
_SENTENCE_BONUS = 1.5


def _parse_json_loose(raw: str) -> Dict:
    """Some models wrap JSON in markdown fences — strip and parse."""
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


def _preview(text: str, head: int = 400, tail: int = 200) -> str:
    """Compact preview of a raw model response for debug logging."""
    text = (text or "").strip()
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    return f"{text[:head]} … [{omitted} chars omitted] … {text[-tail:]}"


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_tags(value: object) -> List[str]:
    """Normalize a model-provided ``tags`` value into clean hashtag words.

    Accepts a list/tuple/set or a single comma/semicolon/space separated string,
    strips a leading ``#`` from each item, drops empties and removes duplicates
    while preserving order, so the value is safe to hand straight to the
    publisher (which turns each tag into ``#tag``).
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw = [str(item) for item in value]
    else:
        raw = str(value).replace(",", " ").replace(";", " ").split()

    tags: List[str] = []
    for item in raw:
        tag = item.strip().lstrip("#").strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _sanitize_highlights(raw_highlights: object, duration: float) -> List[Dict]:
    """Normalize model output into the expected shape; skip invalid entries."""
    if not isinstance(raw_highlights, list):
        return []

    max_end = duration if duration > 0 else float("inf")
    cleaned: List[Dict] = []
    for item in raw_highlights:
        if not isinstance(item, dict):
            continue

        start = _coerce_float(item.get("start_time"), default=-1.0)
        end = _coerce_float(item.get("end_time"), default=-1.0)
        if start < 0 or end <= start:
            continue

        if max_end != float("inf"):
            start = min(start, max_end)
            end = min(end, max_end)
            if end <= start:
                continue

        cleaned.append(
            {
                "title": str(item.get("title") or "Untitled Highlight").strip(),
                "description": str(item.get("description") or "").strip(),
                "tags": _normalize_tags(item.get("tags")),
                "clip_type": str(item.get("clip_type") or "").strip().lower(),
                "start_time": start,
                "end_time": end,
                "score": max(0, min(100, _coerce_int(item.get("score"), default=0))),
                "laugh_score": max(
                    0, min(5, _coerce_int(item.get("laugh_score"), default=0))
                ),
                "cringe_score": max(
                    0, min(5, _coerce_int(item.get("cringe_score"), default=0))
                ),
                "intrigue_score": max(
                    0, min(5, _coerce_int(item.get("intrigue_score"), default=0))
                ),
                "hook_sentence": str(item.get("hook_sentence") or "").strip(),
                "punchline": str(item.get("punchline") or "").strip(),
                "virality_reason": str(item.get("virality_reason") or "").strip(),
            }
        )

    return cleaned


def detect_content_type(transcript: Dict, llm_fn: LLMFn) -> Dict[str, str]:
    """Optional standalone content-type / density classifier.

    The main pipeline no longer calls this: :func:`call_highlight_api` now asks
    the model to report ``content_type`` / ``density`` inside the highlight
    response, so a short video costs a single LLM request instead of two. Kept
    as a utility for callers that want a bare classification on its own.
    """
    segments = transcript.get("segments", [])
    sample = " ".join(s["text"] for s in segments[:25])[:3000]
    prompt = f"{CONTENT_TYPE_PROMPT}\n\nОбразец транскрипта:\n{sample}"
    try:
        raw = llm_fn(prompt)
        return _parse_json_loose(raw)
    except Exception:
        return {"content_type": "other", "density": "medium"}


def _has_visuals(visuals: Optional[Dict]) -> bool:
    """True when a visual index with at least one described scene is present."""
    return bool(visuals and visuals.get("scenes"))


def build_transcript_log(transcript: Dict, visuals: Optional[Dict] = None) -> str:
    """Build the ranking log, enriched with visual context when available.

    Both paths keep the transcript's own granularity — one line per phrase,
    tagged with its exact ``[start - end]`` seconds — so the model can copy
    precise clip boundaries. Without visuals it is the plain transcript log;
    with visuals the merge adds a ``[Visuals: "..."]`` note to the phrases a
    described scene overlaps. (The earlier fixed 15-second windows hid the
    phrase boundaries and made the model cut whole windows, dragging in the
    neighbouring scenes.)
    """
    if _has_visuals(visuals):
        entries = merge_transcripts_and_visuals(transcript.get("segments", []), visuals)
        if entries:
            return format_merged_log(entries)
    return build_transcript_text(transcript)


def build_transcript_text(transcript: Dict) -> str:
    """Render the transcript as one line per phrase with exact seconds.

    Each line is ``[start - end] текст`` (two decimals), so the ranker sees
    where every phrase begins *and* ends and can copy those exact boundaries into
    ``start_time`` / ``end_time``. The old ``[12.3s] текст`` form only showed the
    start, so the model had to guess the phrase end — and usually overshot into
    the neighbouring phrase.
    """
    lines: List[str] = []
    for segment in transcript.get("segments", []) or []:
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(segment.get("text", "")).strip()
        lines.append(f"[{start:.2f} - {end:.2f}] {text}".rstrip())
    return "\n".join(lines)


def chunk_transcript(transcript: Dict, visuals: Optional[Dict] = None) -> List[Dict]:
    segments = transcript.get("segments", [])
    duration = transcript.get("duration", segments[-1]["end"] if segments else 0)
    chunks = []
    start = 0
    while start < duration:
        end = min(start + CHUNK_SIZE_SECONDS, duration)
        # Re-base segment timestamps to chunk-relative seconds. The prompt built
        # from these segments therefore shows local times, the model returns
        # local times, and `call_highlight_api` clamps them against the local
        # `duration`. `get_highlights` later re-adds `_offset` to map back to
        # absolute timeline positions.
        chunk_segs = [
            {
                "start": s["start"] - start,
                "end": s["end"] - start,
                "text": s["text"],
            }
            for s in segments
            if s["start"] >= start and s["end"] <= end + CHUNK_OVERLAP_SECONDS
        ]
        if chunk_segs:
            chunk = dict(transcript)
            chunk["segments"] = chunk_segs
            chunk["duration"] = end - start
            chunk["_offset"] = start
            # Visual scene descriptions are re-based to chunk-local seconds too,
            # so the merged ranking log lines up with the local segment times.
            if visuals and visuals.get("scenes"):
                rebased = [
                    {
                        "start": max(0.0, float(scene["start"]) - start),
                        "end": float(scene["end"]) - start,
                        "text": scene.get("text", ""),
                    }
                    for scene in visuals["scenes"]
                    if scene.get("end", 0) > start and scene.get("start", 0) < end
                ]
                if rebased:
                    chunk["_visuals"] = {"scenes": rebased}
            chunks.append(chunk)
        start += CHUNK_SIZE_SECONDS - CHUNK_OVERLAP_SECONDS
    return chunks


def call_highlight_api(
    transcript_text: str,
    content_info: Optional[Dict],
    duration: float,
    num_clips: int,
    llm_fn: LLMFn,
    is_chunk: bool = False,
    has_visuals: bool = False,
) -> Dict:
    # Ask for ~2× the user's target so dedupe has headroom, but cap so the model
    # doesn't have to generate a huge JSON payload (which can time out the model).
    target = max(num_clips * 2, 5)
    natural_max = max(2 if is_chunk else 3, int(duration / 90))
    min_clips = min(target, natural_max, 8)
    known = content_info or {}
    known_type = known.get("content_type")
    known_density = known.get("density")
    if known_type and known_density:
        content_hint = (
            f"Тип контента: {known_type} | Плотность: {known_density}. "
            f"Ориентируйся на это при выборе моментов и верни те же значения в ответе."
        )
    else:
        content_hint = (
            "Сначала определи тип контента (podcast, interview, tutorial, lecture, "
            "commentary, debate, vlog, anime, other) и плотность (low — в основном вода, "
            "medium, high — плотная информация/истории) и верни их в полях "
            '"content_type" и "density".'
        )
    if has_visuals:
        visual_hint = (
            'В логе у части реплик после текста стоит пометка [Visuals: "…"] — '
            "что происходит на экране в эти секунды. Оценивай синергию: момент, "
            "где слова подкрепляются ярким визуальным действием (эмоция, смех, "
            "фейл, необычный объект, графика), ценнее. Хук может быть и "
            "визуальным — резкое действие в кадре."
        )
    else:
        visual_hint = ""
    system = HIGHLIGHT_SYSTEM_PROMPT.format(
        virality_criteria=VIRALITY_CRITERIA,
        content_hint=content_hint,
        visual_hint=visual_hint,
        num_clips_instruction=(
            f"Верни НЕ БОЛЬШЕ {min_clips} клипов, лучшие — первыми. "
            f"Можно вернуть меньше — качество важнее количества. "
            f"Никогда не добивай список слабыми моментами."
        ),
    )
    base_prompt = f"{system}\n\nТранскрипт:\n{transcript_text}"
    prompt = base_prompt
    last_error = "unknown"

    for attempt in range(1, MAX_HIGHLIGHT_API_ATTEMPTS + 1):
        raw = llm_fn(prompt)
        try:
            parsed = _parse_json_loose(raw)
            highlights = _sanitize_highlights(
                parsed.get("highlights"), duration=duration
            )
            if highlights:
                return {
                    "highlights": highlights,
                    "content_type": str(
                        parsed.get("content_type") or known_type or "other"
                    ).strip(),
                    "density": str(
                        parsed.get("density") or known_density or "medium"
                    ).strip(),
                }
            last_error = "no valid highlights in response"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        print(
            f"[highlights] invalid model output on attempt {attempt}/{MAX_HIGHLIGHT_API_ATTEMPTS} ({last_error})",
            flush=True,
        )
        print(
            f"[highlights] raw response preview: {_preview(raw)}",
            flush=True,
        )

        if attempt < MAX_HIGHLIGHT_API_ATTEMPTS:
            prompt = (
                base_prompt
                + "\n\nВАЖНО: верни ТОЛЬКО валидный JSON: на верхнем уровне поля content_type, density и массив 'highlights'."
                + " Каждый элемент обязан содержать: clip_type, title, description, tags (массив коротких хэштегов без #),"
                + " start_time, end_time, score, laugh_score, cringe_score, intrigue_score, hook_sentence, punchline, virality_reason."
                + " Без markdown-ограждений, без комментариев."
            )

    raise RuntimeError(
        f"Highlight generator produced invalid output after {MAX_HIGHLIGHT_API_ATTEMPTS} attempts: {last_error}"
    )


def dedupe_highlights(highlights: List[Dict]) -> List[Dict]:
    """Drop a highlight if it overlaps >50% with a higher-scoring one already kept."""
    highlights = sorted(highlights, key=lambda x: int(x.get("score", 0)), reverse=True)
    kept: List[Dict] = []
    for h in highlights:
        h_start = float(h["start_time"])
        h_end = float(h["end_time"])
        h_dur = h_end - h_start
        overlapping = False
        for k in kept:
            latest_start = max(h_start, float(k["start_time"]))
            earliest_end = min(h_end, float(k["end_time"]))
            overlap = earliest_end - latest_start
            if overlap > 0 and overlap > 0.5 * h_dur:
                overlapping = True
                break
        if not overlapping:
            kept.append(h)
    return kept


def _nearest_boundary(
    boundaries: List[Tuple[float, bool]],
    target: float,
    *,
    limit: float,
    prefer_later: bool,
) -> Optional[float]:
    """Return the phrase boundary closest to ``target`` (within ``limit``).

    ``boundaries`` are the ``(time, closes_sentence)`` pairs from
    :func:`cues.phrase_boundaries`. A boundary that really closes a sentence is
    ranked ``_SENTENCE_BONUS`` seconds nearer than it is, so that a firm
    sentence end wins over a mid-thought pause that merely happens to sit right
    at the clip's edge (which is exactly how the dangling «…Вот и» / «…Всё-таки»
    tails appear). The ``limit`` still applies to the true distance.

    An exact tie is broken towards the outside of the clip: the later boundary
    for a clip's end (extend the phrase rather than trim it) and the earlier one
    for its start (keep the chosen hook). ``None`` means no boundary is close
    enough, so the caller keeps the plain cue edge.
    """
    best: Optional[float] = None
    best_key: Optional[Tuple[float, float]] = None
    for boundary, closes_sentence in boundaries:
        distance = abs(boundary - target)
        if distance > limit:
            continue
        score = distance - (_SENTENCE_BONUS if closes_sentence else 0.0)
        key = (score, -boundary if prefer_later else boundary)
        if best_key is None or key < best_key:
            best, best_key = boundary, key
    return best


def snap_highlights_to_transcript(
    highlights: List[Dict],
    transcript: Dict,
    *,
    start_padding: float = 0.0,
    end_padding: float = 0.0,
    pause_threshold: float = 0.6,
    max_end: Optional[float] = None,
) -> List[Dict]:
    """Привязать границы хайлайтов к целым фразам транскрипта.

    LLM выбирает ``start_time``/``end_time`` по тексту транскрипта, где видны
    только начала реплик, но не их концы, поэтому граница почти всегда падает
    внутрь фразы. Попавший в окно обрубок следующей реплики («…Вот и»,
    «…Всё-таки») — это и есть та «лишняя» концовка, из-за которой клип
    обрывается на полуфразе.

    Работает в два шага. Сначала окно расширяется до границ перекрывающихся
    кью (как и раньше): начало — до начала первого, конец — до конца последнего.
    Затем обе границы сдвигаются к ближайшей *настоящей* границе фразы — концу
    предложения или паузе в речи; границы, возникшие лишь из-за ограничений
    длины реплики, фразой не считаются (см. ``cues.phrase_boundaries``). Обрубок
    в концовке при этом либо отбрасывается, если он ближе, либо достраивается до
    целой фразы. Начало клипа двигается только назад, к началу фразы, чтобы
    никогда не срезать выбранный хук.

    Конец предложения (знак ``.!?…``) считается надёжной границей, а пауза —
    нет: пауза легко встаёт посреди мысли («… Всё-таки <пауза> она…»), и именно
    так появляется короткий обрубок в концовке. Поэтому граница с точкой
    получает фору ``_SENTENCE_BONUS`` и выигрывает у стоящей рядом паузы, но
    длинный обрубок, до конца которого LLM явно дотянулось, сохраняется.

    Сдвиг ограничен ``_MAX_PHRASE_SHIFT`` секундами: если рядом нет настоящей
    границы фразы (транскрипт без пунктуации и пауз), край остаётся на границе
    реплики — как и раньше. В самом конце добавляется запас
    ``start_padding``/``end_padding``, не заходя при этом в соседнюю реплику.

    Изменённые границы пишутся прямо в словари ``highlights``, поэтому нарезка
    и периклиповые субтитры используют их согласованно.
    """
    cues: List[Tuple[float, float]] = []
    for segment in transcript.get("segments", []) or []:
        try:
            cue_start = float(segment["start"])
            cue_end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if cue_end > cue_start:
            cues.append((cue_start, cue_end))
    cues.sort()

    # Настоящие границы фраз: концы предложений и паузы в речи. Начало (0 с)
    # и конец видео сюда не входят — иначе клип без пунктуации и пауз рядом с
    # ними растянуло бы до самого края дорожки.
    boundaries = phrase_boundaries(
        transcript.get("segments", []), pause_threshold=pause_threshold
    )

    for highlight in highlights:
        try:
            start = float(highlight["start_time"])
            end = float(highlight["end_time"])
        except (KeyError, TypeError, ValueError):
            continue

        overlapping = [c for c in cues if c[1] > start and c[0] < end]
        if not overlapping:
            continue
        covered_start = min(c[0] for c in overlapping)
        covered_end = max(c[1] for c in overlapping)

        # Начало — ближайшая граница фразы не позже текущего края окна: двигаем
        # только назад, чтобы не обрезать хук.
        phrase_start = _nearest_boundary(
            [b for b in boundaries if b[0] <= covered_start],
            start,
            limit=_MAX_PHRASE_SHIFT,
            prefer_later=False,
        )
        if phrase_start is None:
            phrase_start = covered_start

        # Конец — ближайшая граница фразы: если ближе обрубок в концовке, он
        # отбрасывается; если ближе конец начатой фразы — клип достраивается.
        phrase_end = _nearest_boundary(
            [b for b in boundaries if b[0] > phrase_start],
            end,
            limit=_MAX_PHRASE_SHIFT,
            prefer_later=True,
        )
        if phrase_end is None or phrase_end <= covered_start:
            phrase_end = covered_end

        prev_end = next((c[1] for c in reversed(cues) if c[1] <= phrase_start), None)
        next_start = next((c[0] for c in cues if c[0] >= phrase_end), None)

        new_start = phrase_start - start_padding
        if prev_end is not None:
            new_start = max(new_start, prev_end + 0.02)
        new_start = max(0.0, min(new_start, phrase_start))

        new_end = phrase_end + end_padding
        if next_start is not None:
            new_end = min(new_end, next_start - 0.02)
        new_end = max(new_end, phrase_end)
        if max_end is not None:
            new_end = min(new_end, max_end)

        if new_end <= new_start:
            new_end = new_start + 0.04

        delta_start = new_start - start
        delta_end = new_end - end
        if abs(delta_start) > 1e-3 or abs(delta_end) > 1e-3:
            print(
                f"[highlights]   границы привязаны к фразам: "
                f"[{start:.2f} → {end:.2f}] → [{new_start:.2f} → {new_end:.2f}] "
                f"(начало {delta_start:+.2f} с, конец {delta_end:+.2f} с)",
                flush=True,
            )

        highlight["start_time"] = round(new_start, 3)
        highlight["end_time"] = round(new_end, 3)

    return highlights


def _cue_windows(transcript: Dict) -> List[Tuple[float, float]]:
    """Кью транскрипта как отсортированные ``(start, end)``, без мусора."""
    cues: List[Tuple[float, float]] = []
    for segment in transcript.get("segments", []) or []:
        try:
            cue_start = float(segment["start"])
            cue_end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if cue_end > cue_start:
            cues.append((cue_start, cue_end))
    cues.sort()
    return cues


def _free_span(
    start: float,
    end: float,
    spans: List[Tuple[float, float]],
    video_end: float,
) -> Optional[Tuple[float, float]]:
    """Свободное место вокруг ``[start, end]`` с учётом занятых ``spans``.

    Возвращает ``(lower, upper)`` — до какой границы можно расширять окно влево
    и вправо, не заезжая в соседей и за пределы видео. ``None`` означает, что
    окно пересекается с занятым диапазоном: расширяться нельзя.
    """
    lower, upper = 0.0, video_end
    for other_start, other_end in spans:
        if other_end <= start:
            lower = max(lower, other_end)
        elif other_start >= end:
            upper = min(upper, other_start)
        else:
            return None
    return lower, upper


# Насколько окно может «перелететь» минимум, цепляясь за границу реплики. Если
# ближайшая граница дальше, чем осталось добрать, плюс этот запас — за неё не
# цепляемся: в транскрипте бывают дыры в десятки секунд (тишина, музыка, эндинг),
# и прыжок через такую дыру склеил бы в клип полминуты пустоты. Вместо прыжка
# окно добивается ровно до минимума.
_GROW_OVERSHOOT_TOLERANCE = 3.0


def _grow_window(
    start: float,
    end: float,
    cues: List[Tuple[float, float]],
    *,
    lower: float,
    upper: float,
    min_duration: float,
) -> Tuple[float, float]:
    """Расширить ``[start, end]`` до ``min_duration``, не выходя за пределы.

    Окно растёт целыми кью: на каждом шаге добавляется кью, ближайшая к текущему
    краю, — так клип остаётся на границах реплик, а не обрывается посреди слова.
    Границы, стоящие дальше, чем осталось добрать (плюс
    ``_GROW_OVERSHOOT_TOLERANCE``), игнорируются: в транскрипте бывают длинные
    дыры (тишина, музыка, эндинг), и цепляние за границу за такой дырой раздуло
    бы клип на десятки секунд пустоты. Если кью не заполняют окно, остаток
    добирается свободным местом между ``lower`` и ``upper``.
    """
    starts = [cue[0] for cue in cues if lower <= cue[0] <= start]
    ends = [cue[1] for cue in cues if end <= cue[1] <= upper]
    new_start, new_end = start, end
    next_start, next_end = len(starts) - 1, 0
    while new_end - new_start < min_duration and (
        next_start >= 0 or next_end < len(ends)
    ):
        # Сколько ещё нужно добрать, с запасом на «перелёт» до границы реплики.
        budget = min_duration - (new_end - new_start) + _GROW_OVERSHOOT_TOLERANCE
        grow_start = starts[next_start] if next_start >= 0 else None
        grow_end = ends[next_end] if next_end < len(ends) else None
        if grow_start is not None and new_start - grow_start > budget:
            grow_start = None
        if grow_end is not None and grow_end - new_end > budget:
            grow_end = None
        if grow_start is None and grow_end is None:
            # Остались только далёкие границы — не прыгаем, добиваем ниже.
            break
        if grow_start is None:
            new_end = grow_end
            next_end += 1
        elif grow_end is None:
            new_start = grow_start
            next_start -= 1
        elif grow_end - new_end <= new_start - grow_start:
            new_end = grow_end
            next_end += 1
        else:
            new_start = grow_start
            next_start -= 1

    if new_end - new_start < min_duration:
        deficit = min_duration - (new_end - new_start)
        take = min(new_start - lower, deficit)
        new_start -= take
        deficit -= take
        new_end += min(upper - new_end, deficit)

    return max(0.0, new_start), min(new_end, upper)


def enforce_min_clip_duration(
    highlights: List[Dict],
    transcript: Dict,
    *,
    min_duration: float,
    max_end: Optional[float] = None,
) -> List[Dict]:
    """Дотянуть слишком короткие клипы до ``min_duration`` (на месте).

    LLM нередко возвращает окно в одну-две реплики, и после привязки к фразам
    клип выходит в 2–3 секунды — для Shorts это мусор. Короткое окно
    достраивается целыми кью транскрипта в обе стороны, пока не наберётся
    ``min_duration``.

    Расширение не заходит в соседние клипы и за пределы видео, поэтому наложений
    не создаёт, но если места рядом нет, клип остаётся коротким — такие окна
    отбрасывает :func:`select_highlights`. ``min_duration <= 0`` отключает шаг.
    """
    if min_duration <= 0 or not highlights:
        return highlights

    cues = _cue_windows(transcript)
    if not cues:
        return highlights

    video_end = float(max_end) if max_end else cues[-1][1]
    spans = [(float(h["start_time"]), float(h["end_time"])) for h in highlights]

    for index, highlight in enumerate(highlights):
        start, end = spans[index]
        if end - start >= min_duration:
            continue

        room = _free_span(start, end, spans[:index] + spans[index + 1 :], video_end)
        if room is None:
            continue
        lower, upper = room
        if upper - lower <= end - start:
            continue

        new_start, new_end = _grow_window(
            start, end, cues, lower=lower, upper=upper, min_duration=min_duration
        )
        if new_end - new_start <= end - start:
            continue

        print(
            f"[highlights]   клип короче {min_duration:.0f}s — расширен "
            f"[{start:.2f} → {end:.2f}] → [{new_start:.2f} → {new_end:.2f}] "
            f"({new_end - new_start:.1f}s)",
            flush=True,
        )
        highlight["start_time"] = round(new_start, 3)
        highlight["end_time"] = round(new_end, 3)
        spans[index] = (new_start, new_end)

    return highlights


def select_highlights(
    highlights: List[Dict],
    transcript: Dict,
    *,
    num_clips: int,
    min_duration: float,
    max_end: Optional[float] = None,
) -> List[Dict]:
    """Выбрать до ``num_clips`` лучших моментов, каждый не короче ``min_duration``.

    Кандидаты перебираются по убыванию оценки. Короткий момент сначала
    достраивается :func:`enforce_min_clip_duration` с учётом уже принятых клипов;
    если места рядом не хватает, кандидат пропускается — так в нарезку не уходит
    обрубок в 0,15 с, а освободившееся место достаётся следующему кандидату по
    списку.

    Когда ни один кандидат не дотянул до минимума (``min_duration`` больше самого
    видео, нет транскрипта), возвращается обычный топ по оценке: лучше отдать
    короткие клипы, чем не отдать ничего.
    """
    if num_clips <= 0:
        return []

    ranked = sorted(highlights, key=lambda h: int(h.get("score", 0)), reverse=True)
    if min_duration <= 0:
        return ranked[:num_clips]

    accepted: List[Dict] = []
    for candidate in ranked:
        if len(accepted) >= num_clips:
            break
        # Принятые клипы уже не короче минимума, поэтому рост их не тронет: он
        # достраивает только кандидата — в свободном месте рядом с ними.
        group = accepted + [dict(candidate)]
        enforce_min_clip_duration(
            group, transcript, min_duration=min_duration, max_end=max_end
        )
        chosen = group[-1]
        if chosen["end_time"] - chosen["start_time"] < min_duration:
            # Рядом не нашлось места на целый клип — момент пропускается, а место
            # достаётся следующему кандидату.
            continue
        accepted.append(chosen)

    return accepted or ranked[:num_clips]


def _llm_label(settings: Settings) -> str:
    """Метка ``провайдер/модель`` для лога ранжирования."""
    provider = (settings.llm_provider or "openai").strip().lower()
    model = getattr(settings, provider + "_model", "?")
    return f"{provider}/{model}"


def get_highlights(
    transcript: Dict,
    settings: Settings,
    num_clips: int = 3,
    visuals: Optional[Dict] = None,
) -> Dict:
    """Main entry point — returns ``{highlights: [...]}`` sorted by score.

    The LLM provider and model come from ``settings`` (the single ``.env``).
    """
    llm_fn: LLMFn = lambda prompt: call_llm(prompt, settings)  # noqa: E731

    duration = transcript.get("duration", 0)
    print(
        f"[highlights] ranking {duration:.0f}s of transcript with "
        f"{_llm_label(settings)} — the LLM call, this step takes the longest",
        flush=True,
    )

    if duration >= LONG_VIDEO_THRESHOLD:
        chunks = chunk_transcript(transcript, visuals)
        print(
            f"[highlights] long video — splitting into {len(chunks)} chunks", flush=True
        )
        all_highlights: List[Dict] = []
        # The content type is detected by the model on the first chunk and then
        # reused for the rest, so classification costs no extra request.
        content_info: Optional[Dict] = None
        for i, chunk in enumerate(chunks):
            offset = chunk.get("_offset", 0)
            text = build_transcript_log(chunk, chunk.get("_visuals"))
            print(
                f"[highlights] chunk {i + 1}/{len(chunks)} (offset {offset:.0f}s)",
                flush=True,
            )
            result = call_highlight_api(
                text,
                content_info,
                chunk["duration"],
                num_clips=num_clips,
                llm_fn=llm_fn,
                is_chunk=True,
                has_visuals=_has_visuals(chunk.get("_visuals")),
            )
            if content_info is None:
                content_info = {
                    "content_type": result.get("content_type"),
                    "density": result.get("density"),
                }
                print(
                    f"[highlights] content={content_info['content_type']} density={content_info['density']}",
                    flush=True,
                )
            for h in result.get("highlights", []):
                h["start_time"] = float(h["start_time"]) + offset
                h["end_time"] = float(h["end_time"]) + offset
                all_highlights.append(h)
        highlights = dedupe_highlights(all_highlights)
    else:
        text = build_transcript_log(transcript, visuals)
        result = call_highlight_api(
            text,
            None,
            duration,
            num_clips=num_clips,
            llm_fn=llm_fn,
            has_visuals=_has_visuals(visuals),
        )
        print(
            f"[highlights] content={result.get('content_type')} density={result.get('density')} duration={duration:.0f}s",
            flush=True,
        )
        highlights = dedupe_highlights(result.get("highlights", []))

    return {"highlights": highlights}
