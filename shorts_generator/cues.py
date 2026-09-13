"""Turn transcript segments into short, readable subtitle cues.

Whisper emits whole sentences — often several per segment — so a single subtitle
element can hang on screen for many seconds and blot out the video. That is what
makes long clips show up as "one big sentence" that never seems to leave.

This module re-chunks those segments into short cues — the individual phrases you
want to read one at a time. When faster-whisper provides word-level timestamps
the cue boundaries follow the actual speech (sentence ends, pauses and a maximum
length); when only plain segment text is available — for example when a
transcript is loaded back from its ``.srt`` cache — the time span is distributed
across the phrases in proportion to their length.

Both paths break on the same sentence/length rules, so a cue produced with word
timings round-trips through the ``.srt`` cache unchanged: splitting an already
split transcript yields the very same cues.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

# Punctuation that closes a sentence; a new cue starts right after it.
_SENTENCE_END = frozenset(".!?…")
# Characters that may trail a sentence/clause mark, e.g. the closing quote in
# ``он сказал: «да!».`` — they are skipped when inspecting the last character.
_TRAILING = "\"'”’»)]}»”’"
# Softer break points: a long-ish clause is cut after one of these when possible.
_SOFT_BREAK = frozenset(",;:—–")
# Punctuation that attaches to the preceding word without a space when word tokens
# are glued back together (``дело`` + ``.`` -> ``дело.``).
_NO_SPACE_BEFORE = frozenset(".,!?;:%)]}»”’…")

# A word that is longer than this is split at character level (defensive guard
# against a pathological single "word").
_MIN_MAX_CHARS = 8


def _normalize(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip the edges."""
    return " ".join(str(text).split())


def _join_words(words: List[Dict]) -> str:
    """Glue word tokens back into a phrase with correct spacing.

    faster-whisper usually keeps the leading space on each token (``" word"``),
    but not every model or language does. Rather than trusting that, a space is
    inserted between tokens unless the token already starts with whitespace or
    with punctuation that attaches to the previous word.
    """
    text = ""
    for word in words:
        piece = str(word.get("word", ""))
        if not text:
            text = piece.lstrip()
            continue
        if piece[:1].isspace() or piece[:1] in _NO_SPACE_BEFORE:
            text += piece
        else:
            text += " " + piece
    return _normalize(text)


def _ends_sentence(token: str) -> bool:
    """Return ``True`` when a single word ends a sentence."""
    token = token.strip()
    for char in reversed(token):
        if char in _TRAILING:
            continue
        return char in _SENTENCE_END
    return False


def _ends_soft(token: str) -> bool:
    """Return ``True`` when a word ends on a soft break (comma, dash, ...)."""
    token = token.strip()
    for char in reversed(token):
        if char in _TRAILING:
            continue
        return char in _SOFT_BREAK
    return False


def _cues_from_words(
    words: List[Dict],
    max_chars: int,
    max_words: int,
    max_duration: float,
    pause_threshold: float,
) -> List[Dict]:
    """Build cues from word-level timings, following the actual speech."""
    cues: List[Dict] = []
    current: List[Dict] = []

    def text_of(items: List[Dict]) -> str:
        return _join_words(items)

    def flush() -> None:
        if not current:
            return
        text = text_of(current)
        if text:
            cues.append(
                {
                    "start": float(current[0]["start"]),
                    "end": float(current[-1]["end"]),
                    "text": text,
                }
            )
        current.clear()

    for word in words:
        # Decide *before* extending the cue, so the character/word/duration caps
        # are respected instead of being overshot by the last appended word.
        if current:
            gap = float(word["start"]) - float(current[-1]["end"])
            prospective = _join_words(current + [word])
            duration = float(word["end"]) - float(current[0]["start"])
            if (
                _ends_sentence(current[-1]["word"])  # the previous word closed it
                or len(prospective) > max_chars
                or len(current) + 1 > max_words
                or duration > max_duration
                or gap >= pause_threshold  # a noticeable pause in speech
            ):
                flush()

        current.append(word)

        # Prefer to cut a *full-ish* clause at natural punctuation.
        if _ends_soft(word["word"]) and len(_join_words(current)) >= max_chars * 0.5:
            flush()

    flush()
    return cues


def _split_long_phrase(phrase: str, max_chars: int, max_words: int) -> List[str]:
    """Break a single over-long phrase at word boundaries."""
    parts: List[str] = []
    current: List[str] = []
    for word in phrase.split(" "):
        if current and (
            len(" ".join(current)) + 1 + len(word) > max_chars
            or len(current) + 1 > max_words
        ):
            parts.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        parts.append(" ".join(current))
    return parts


def _phrases_from_text(text: str, max_chars: int, max_words: int) -> List[str]:
    """Split plain text into short phrases (sentences, then length caps)."""
    text = _normalize(text)
    if not text:
        return []

    sentences: List[str] = []
    buffer = ""
    for token in text.split(" "):
        buffer = f"{buffer} {token}".strip()
        if _ends_sentence(token):
            sentences.append(buffer)
            buffer = ""
    if buffer:
        sentences.append(buffer)

    phrases: List[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars and len(sentence.split(" ")) <= max_words:
            phrases.append(sentence)
        else:
            phrases.extend(_split_long_phrase(sentence, max_chars, max_words))
    return phrases


def _cues_from_text(
    text: str,
    start: float,
    end: float,
    max_chars: int,
    max_words: int,
) -> List[Dict]:
    """Build cues from plain text, spreading the time span by phrase length."""
    phrases = _phrases_from_text(text, max_chars, max_words)
    if not phrases:
        return []
    if len(phrases) == 1:
        return [{"start": float(start), "end": float(end), "text": phrases[0]}]

    span = max(0.0, float(end) - float(start))
    total = sum(len(phrase) for phrase in phrases) or 1

    cues: List[Dict] = []
    cursor = float(start)
    for index, phrase in enumerate(phrases):
        if index == len(phrases) - 1:
            cue_end = float(end)
        else:
            cue_end = cursor + span * (len(phrase) / total)
        cues.append(
            {"start": round(cursor, 3), "end": round(cue_end, 3), "text": phrase}
        )
        cursor = cue_end
    return cues


def split_segments_into_cues(
    segments: Iterable[Dict],
    *,
    max_chars: int = 40,
    max_words: int = 9,
    max_duration: float = 3.5,
    pause_threshold: float = 0.6,
) -> List[Dict]:
    """Split transcript segments into short subtitle cues.

    Each returned cue is ``{"start": float, "end": float, "text": str}``. The
    function is idempotent: feeding it segments that are already cues (as stored
    in a ``.srt`` cache) returns them unchanged, so the same transcript can be
    re-processed any number of times without drifting.

    Args:
        segments: transcript segments, each ``{"start", "end", "text"}`` and
            optionally ``"words"`` (a list of ``{"start", "end", "word"}``).
        max_chars: maximum characters of a single cue.
        max_words: maximum words of a single cue.
        max_duration: maximum seconds a single cue is allowed to stay on screen.
        pause_threshold: silence (in seconds) between two words that forces a
            new cue when word timings are available.
    """
    max_chars = max(_MIN_MAX_CHARS, int(max_chars))
    max_words = max(1, int(max_words))
    max_duration = max(0.5, float(max_duration))

    cues: List[Dict] = []
    for segment in segments or []:
        text = _normalize(segment.get("text", ""))
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
        except (TypeError, ValueError):
            continue
        if end <= start:
            end = start + 0.04

        segment_cues: List[Dict] = []

        words = segment.get("words")
        if words:
            normalized: List[Dict] = []
            for word in words:
                raw = str(word.get("word", ""))
                if not raw.strip():
                    continue
                try:
                    normalized.append(
                        {
                            "start": float(word["start"]),
                            "end": float(word["end"]),
                            "word": raw,
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            if normalized:
                segment_cues = _cues_from_words(
                    normalized, max_chars, max_words, max_duration, pause_threshold
                )

        if not segment_cues and text:
            segment_cues = _cues_from_text(text, start, end, max_chars, max_words)

        cues.extend(segment_cues)

    return cues
