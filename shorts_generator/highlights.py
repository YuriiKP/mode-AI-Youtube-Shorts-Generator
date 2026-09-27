"""Find the most viral-worthy highlights in a transcript.

Logic ported from ViralVadoo's transcript_analysis/highlight_generator.py:
  - content-type / density detection
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

LLMFn = Callable[[str], str]


# --- English prompts (commented out; the Russian versions below are used) ---
#
# CONTENT_TYPE_PROMPT = """Analyze this video transcript sample and classify the content type.
# Choose one: podcast, interview, tutorial, lecture, commentary, debate, vlog, other.
# Also estimate content density: low (mostly filler/chit-chat), medium, or high (dense info/stories).
# Respond with JSON only: {"content_type": "...", "density": "..."}"""
#
#
# VIRALITY_CRITERIA = """
# Virality signals to prioritize (ranked by impact):
# 1. HOOK MOMENTS — statements that create immediate curiosity ("The secret is...", "Nobody talks about...", "I was completely wrong about...")
# 2. EMOTIONAL PEAKS — genuine surprise, laughter, anger, vulnerability, excitement; raw unscripted reactions
# 3. OPINION BOMBS — strong, polarizing or counter-intuitive statements that trigger agree/disagree
# 4. REVELATION MOMENTS — surprising facts, stats, or confessions that reframe how the viewer thinks
# 5. CONFLICT/TENSION — disagreement, pushback, or a problem being confronted head-on
# 6. QUOTABLE ONE-LINERS — a sentence that works as a standalone quote card
# 7. STORY PEAKS — the climax or twist of an anecdote; the payoff moment
# 8. PRACTICAL VALUE — a concrete tip, hack, or insight the viewer can immediately apply
# """
#
# HIGHLIGHT_SYSTEM_PROMPT = """You are an elite short-form video editor who has studied thousands of viral clips on TikTok, Instagram Reels, and YouTube Shorts. You know exactly what makes viewers stop scrolling, watch to the end, and share.
#
# {virality_criteria}
#
# Content type: {content_type} | Density: {density}
#
# Your task: identify the most viral-worthy highlights from the transcript.
#
# Rules:
# - Every highlight must open with a strong HOOK — a line that grabs attention within the first 3 seconds
# - Duration sweet spot: 45-90 seconds. Go shorter (20-44s) only for a perfect standalone one-liner. Go longer (91-180s) only when a story arc needs full context to land
# - Never cut mid-sentence or mid-thought — each clip must feel complete and self-contained
# - Clips must not overlap significantly with each other
# - Score 0-100 on viral potential (not general quality)
# - {num_clips_instruction}
# - For each highlight, identify the single best "hook_sentence" — the opening line that would make someone stop scrolling
# - Explain in one sentence why this clip is viral ("virality_reason")
# - Write a short "description" of 1-2 sentences for posting on platforms (YouTube Shorts, TikTok, Instagram Reels); no hashtags, emoji or markdown
#
# Respond ONLY with valid JSON (no markdown, no explanation):
# {{"highlights":[{{"title":"string","description":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","virality_reason":"string"}}]}}"""


# --- Russian prompts ---
CONTENT_TYPE_PROMPT = """Проанализируй этот образец транскрипта видео и определи тип контента.
Выбери один: podcast, interview, tutorial, lecture, commentary, debate, vlog, other.
Также оцени плотность контента: low (в основном вода и болтовня), medium или high (плотная информация/истории).
Отвечай ТОЛЬКО в формате JSON: {"content_type": "...", "density": "..."}"""


VIRALITY_CRITERIA = """
ГЛАВНОЕ ПРАВИЛО: Первые 1–3 секунды решают всё. Зритель решает смахнуть видео за доли секунды. Вступление должно сразу бросать в центр событий.

Сигналы виральности, которые нужно учитывать в первую очередь (по убыванию силы воздействия):
1. ПАРАДОКС И ИНТРИГА (ВОПРОС «ЗАЧЕМ?» / «ПОЧЕМУ?») — абсурдное, шокирующее или нестандартное действие персонажа + текстовый вопрос на экране.
2. ЭМОЦИОНАЛЬНЫЙ ПИК И «КРУТОСТЬ» — всплеск ярости, пафосный боевой момент, внезапный перелом в битве или трагедия.
3. ХУКИ И НЕДОСКАЗАННОСТЬ — завязка конфликта, диалог, обрывающийся перед ответом, фразы-триггеры («Он не должен был этого делать...», «Они не знали, с кем связались»).
4. ПИКИ ИСТОРИЙ / КУЛЬМИНАЦИЯ — кульминационная фраза или развязка противостояния.

КЛИП ДОЛЖЕН БЫТЬ ЦЕЛЬНЫМ, НЕ ОБРЫВАТЬЯ НА ПОЛУФРАЗЕ. ПРОВЕРЯЙ ЭТО!
"""

HIGHLIGHT_SYSTEM_PROMPT = """Ты элитный редактор коротких вертикальных видео, изучивший тысячи вирусных клипов в TikTok, Instagram Reels и YouTube Shorts. Ты точно знаешь, что заставляет зрителей прекратить листать, досматривать до конца и делиться.

{virality_criteria}

Тип контента: {content_type} | Плотность: {density}

Твоя задача: определить самые виральные моменты (хайлайты) в транскрипте.

Правила:
- Предпочтительная длительность: 30 - 60 секунд. Для крутого момента можно любую длину.
- Никогда не обрезай посреди предложения или мысли — каждый клип должен ощущаться завершённым и самодостаточным.
- Клипы не должны существенно пересекаться друг с другом.
- Бесшовность (Loop): По возможности делать зацикленный финал, плавно возвращающий к началу.
- Оценка 0-100 по виральному потенциалу (а не по общему качеству).
- {num_clips_instruction}
- Объясни одним предложением, почему этот клип виральный ("virality_reason").
- Дай "description" — короткое описание клипа на 1-2 предложения для заполнения при публикации на площадках (YouTube Shorts, TikTok, Instagram Reels). Пиши цепляюще и по сути, как подпись к видео, без хэштегов, эмодзи и markdown.

Отвечай ТОЛЬКО валидным JSON (без markdown, без пояснений):
{{"highlights":[{{"title":"string","description":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","virality_reason":"string"}}]}}"""


CHUNK_SIZE_SECONDS = 1200  # 20-min chunks for long videos
LONG_VIDEO_THRESHOLD = 1800  # chunk videos longer than 30 min
CHUNK_OVERLAP_SECONDS = 60
MAX_HIGHLIGHT_API_ATTEMPTS = 3

# Насколько далеко край клипа может уехать (в секундах) от границы реплики,
# выбранной LLM, чтобы попасть на настоящую границу фразы — конец предложения
# или паузу. Ограничение не даёт транскрипту без пунктуации и пауз растянуть
# или обрезать клип далеко от задуманного окна.
_MAX_PHRASE_SHIFT = 5.0

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
    segments = transcript.get("segments", [])
    sample = " ".join(s["text"] for s in segments[:25])[:3000]
    prompt = f"{CONTENT_TYPE_PROMPT}\n\nОбразец транскрипта:\n{sample}"
    try:
        raw = llm_fn(prompt)
        return _parse_json_loose(raw)
    except Exception:
        return {"content_type": "other", "density": "medium"}


def build_transcript_text(transcript: Dict) -> str:
    segments = transcript.get("segments", [])
    return "\n".join(f"[{s['start']:.1f}s] {s['text'].strip()}" for s in segments)


def chunk_transcript(transcript: Dict) -> List[Dict]:
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
            chunks.append(chunk)
        start += CHUNK_SIZE_SECONDS - CHUNK_OVERLAP_SECONDS
    return chunks


def call_highlight_api(
    transcript_text: str,
    content_info: Dict,
    duration: float,
    num_clips: int,
    llm_fn: LLMFn,
    is_chunk: bool = False,
) -> Dict:
    # Ask for ~2× the user's target so dedupe has headroom, but cap so the model
    # doesn't have to generate a huge JSON payload (which can time out the model).
    target = max(num_clips * 2, 5)
    natural_max = max(2 if is_chunk else 3, int(duration / 90))
    min_clips = min(target, natural_max, 8)
    system = HIGHLIGHT_SYSTEM_PROMPT.format(
        virality_criteria=VIRALITY_CRITERIA,
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
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
                return {"highlights": highlights}
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
                + "\n\nВАЖНО: верни ТОЛЬКО валидный JSON с массивом 'highlights' верхнего уровня."
                + " Каждый элемент обязан содержать: clip_type, title, description, start_time, end_time, score,"
                + " laugh_score, cringe_score, intrigue_score, hook_sentence, punchline, virality_reason."
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


def get_highlights(
    transcript: Dict,
    settings: Settings,
    num_clips: int = 3,
) -> Dict:
    """Main entry point — returns ``{highlights: [...]}`` sorted by score.

    The LLM provider and model come from ``settings`` (the single ``.env``).
    """
    llm_fn: LLMFn = lambda prompt: call_llm(prompt, settings)  # noqa: E731

    duration = transcript.get("duration", 0)
    content_info = detect_content_type(transcript, llm_fn=llm_fn)
    print(
        f"[highlights] content={content_info.get('content_type')} density={content_info.get('density')} duration={duration:.0f}s",
        flush=True,
    )

    if duration >= LONG_VIDEO_THRESHOLD:
        chunks = chunk_transcript(transcript)
        print(
            f"[highlights] long video — splitting into {len(chunks)} chunks", flush=True
        )
        all_highlights: List[Dict] = []
        for i, chunk in enumerate(chunks):
            offset = chunk.get("_offset", 0)
            text = build_transcript_text(chunk)
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
            )
            for h in result.get("highlights", []):
                h["start_time"] = float(h["start_time"]) + offset
                h["end_time"] = float(h["end_time"]) + offset
                all_highlights.append(h)
        highlights = dedupe_highlights(all_highlights)
    else:
        text = build_transcript_text(transcript)
        result = call_highlight_api(
            text, content_info, duration, num_clips=num_clips, llm_fn=llm_fn
        )
        highlights = dedupe_highlights(result.get("highlights", []))

    return {"highlights": highlights}
