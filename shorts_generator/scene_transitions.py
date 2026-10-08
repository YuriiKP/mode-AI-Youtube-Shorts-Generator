"""Scene-transition detection and a "slide" (pan) plan for the vertical crop.

The vertical crop of a wide source keeps only a window of the frame; the sides
are thrown away. This module finds the scene transitions (cuts/fades) inside an
already-cut clip and turns them into a plan that slowly slides that window from
one side of the frame to the other, alternating the direction at every
transition. Sliding the window back and forth reveals the cropped-away parts of
the frame and adds a little montage movement.

Transitions are detected with PySceneDetect when it is installed and fall back
to a small OpenCV frame-difference detector otherwise, so the feature keeps
working without the extra dependency.
"""

from __future__ import annotations

from typing import List, NamedTuple, Sequence, Tuple, TypeVar

# A slide segment: ``(start_seconds, end_seconds, direction)`` where
# ``direction`` is ``+1`` towards the right edge of the frame, ``-1`` to the left.
SlideSegment = Tuple[float, float, int]

__all__ = [
    "SlideSegment",
    "CutEffect",
    "VALID_CUT_EFFECTS",
    "detect_transitions",
    "build_slide_segments",
    "slide_progress",
    "build_cut_effects",
    "parse_cut_effect_styles",
]

# Generic element type so ``_evenly_spaced`` keeps the type of what it samples.
_T = TypeVar("_T")


def detect_transitions(video_path: str, threshold: float = 27.0) -> List[float]:
    """Return the scene-change times (in seconds) found inside ``video_path``."""
    try:
        return _detect_with_scenedetect(video_path)
    except Exception:
        return _detect_with_opencv(video_path, threshold=threshold)


def _detect_with_scenedetect(video_path: str) -> List[float]:
    """Detect cuts (ContentDetector) and fades (ThresholdDetector)."""
    from scenedetect import SceneManager, open_video  # type: ignore
    from scenedetect.detectors import ContentDetector, ThresholdDetector  # type: ignore

    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(ContentDetector())
    manager.add_detector(ThresholdDetector())
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()
    # The first scene always begins at 0; every later start is a transition.
    return [start.get_seconds() for start, _ in scenes[1:]]


def _detect_with_opencv(video_path: str, threshold: float = 27.0) -> List[float]:
    """Fallback: flag frames whose mean pixel change exceeds ``threshold``."""
    import cv2  # type: ignore

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path} for transition detection")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    transitions: List[float] = []
    previous = None
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (64, 36))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if previous is not None:
            if float(cv2.absdiff(gray, previous).mean()) >= threshold:
                transitions.append(index / fps)
        previous = gray
        index += 1
    cap.release()
    return transitions


def _group_boundaries(
    transitions: Sequence[float], duration: float, gap: float
) -> List[float]:
    """Collapse transitions that are closer than ``gap`` seconds.

    Returns the segment boundaries as ``[0.0, ...cuts..., duration]``; clustered
    transitions contribute a single boundary so the slide spans the whole group.

    Every neighbouring pair of boundaries is kept at least ``gap`` seconds apart
    — including the distance to ``0.0`` and to ``duration`` — so each slide
    segment lasts at least ``gap`` seconds and the pan never whips across the
    frame in a fraction of a second. Transitions closer than ``gap`` to the start
    or to the end are dropped, so the slide simply continues straight through
    them. The only case a segment can be shorter than ``gap`` is a clip that is
    itself shorter than ``gap``, where a full-length slide simply does not fit.
    """
    boundaries: List[float] = [0.0]
    for time in sorted(transitions):
        if not 0.0 < time < duration:
            continue
        if time - boundaries[-1] < gap:
            continue  # too close to the previous boundary -> same group
        if duration - time < gap:
            continue  # too close to the end to fit a full-length slide
        boundaries.append(time)
    boundaries.append(duration)
    return boundaries


def build_slide_segments(
    transitions: Sequence[float], duration: float, gap: float = 3.0
) -> List[SlideSegment]:
    """Turn transition ``times`` into ``(start, end, direction)`` slide segments."""
    if duration <= 0:
        return []
    boundaries = _group_boundaries(transitions, duration, gap)
    segments: List[SlideSegment] = []
    direction = 1
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start <= 0:
            continue
        segments.append((start, end, direction))
        direction = -direction
    return segments


# --- Cut transitions (blend the source scene cuts inside one clip) ----------
# ``SLIDE_EFFECT`` above only *moves* the crop window when a scene changes; these
# cut effects instead blend the two shots together, softening the hard cut. The
# clip keeps its length (the frames are re-mixed, not overlapped) and the audio
# is untouched, so subtitles and music stay in sync.
class CutEffect(NamedTuple):
    """One cut-transition window, measured in output frames.

    ``cut_frame`` is the first frame of the *new* scene (the source cut);
    ``start_frame``..``end_frame`` is the whole window the effect may touch.
    Which frames are actually modified depends on ``style`` (see the applier in
    ``clipper``): a ``dissolve`` only blends the frames *after* the cut with the
    retained frames before it, while ``fade`` / ``flash`` / ``zoom`` use the
    entire window.
    """

    cut_frame: int
    start_frame: int
    end_frame: int
    style: str


# Styles the applier understands. ``dissolve`` blends old and new shots through
# each other, ``fade`` dips through black, ``flash`` through white and ``zoom``
# punches in and back out at the cut.
VALID_CUT_EFFECTS: Tuple[str, ...] = ("dissolve", "fade", "flash", "zoom")


def parse_cut_effect_styles(raw: "str | Sequence[str] | None") -> List[str]:
    """Normalise a ``CUT_EFFECT_TYPES`` value into a list of known styles.

    Accepts a comma-separated string or any sequence of names. Unknown names are
    dropped, and an empty result falls back to ``["dissolve"]`` so the feature
    always has something to apply — the caller can therefore pass the raw
    setting straight through without pre-validating it.
    """
    parts: List[str]
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = list(raw or [])
    styles = [str(part).strip().lower() for part in parts]
    styles = [style for style in styles if style in VALID_CUT_EFFECTS]
    return styles or ["dissolve"]


def _evenly_spaced(items: Sequence[_T], count: int) -> List[_T]:
    """Pick ``count`` items spread as evenly as possible across ``items``.

    Used to cap the number of transitions per clip: when more candidate cuts fit
    than ``CUT_EFFECT_MAX`` allows, keep an even sample rather than the first
    few, so the transitions are spread across the whole short instead of bunching
    up at the start.
    """
    if count >= len(items):
        return list(items)
    if count <= 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (count - 1)
    return [items[int(round(i * step))] for i in range(count)]


def build_cut_effects(
    transitions: Sequence[float],
    duration: float,
    fps: float,
    window_seconds: float = 0.25,
    styles: Sequence[str] = ("dissolve",),
    max_count: int = 3,
) -> List[CutEffect]:
    """Plan short transitions over the source scene cuts inside one clip.

    ``transitions`` are scene-change times (in seconds) found by
    :func:`detect_transitions`; they are turned into ``CutEffect`` windows in
    output frames. The window is centred on the cut and the styles are rotated
    across the kept cuts, so a clip with several transitions does not repeat the
    same effect every time.

    Only ``max_count`` transitions survive; when more cuts fit, an even sample is
    kept (see :func:`_evenly_spaced`). Windows are never allowed to overlap or to
    run off either end of the clip, so a transition always has real frames to
    work with on both sides.
    """
    if duration <= 0.0 or fps <= 0.0 or max_count <= 0:
        return []
    chosen_styles = [style for style in styles if style in VALID_CUT_EFFECTS]
    if not chosen_styles:
        chosen_styles = ["dissolve"]

    total_frames = int(round(duration * fps))
    window = max(2, int(round(window_seconds * fps)))
    before = max(1, window // 2)
    after = max(1, window - before)

    candidates: List[Tuple[int, int, int]] = []
    last_end = -1
    for time in sorted(transitions):
        cut = int(round(float(time) * fps))
        start = cut - before
        end = cut + after
        if start < 0 or end > total_frames:
            continue  # the window would run off the clip
        if start < last_end:
            continue  # closer than a full window to the previous transition
        candidates.append((cut, start, end))
        last_end = end

    effects: List[CutEffect] = []
    for index, (cut, start, end) in enumerate(_evenly_spaced(candidates, max_count)):
        style = chosen_styles[index % len(chosen_styles)]
        effects.append(CutEffect(cut, start, end, style))
    return effects


def _smoothstep(fraction: float) -> float:
    fraction = min(1.0, max(0.0, fraction))
    return fraction * fraction * (3.0 - 2.0 * fraction)


def slide_progress(time: float, segments: Sequence[SlideSegment]) -> float:
    """Horizontal position (0..1) of the crop window at ``time`` seconds."""
    if not segments:
        return 0.5
    chosen = segments[-1]
    for segment in segments:
        if time < segment[1]:
            chosen = segment
            break
    start, end, direction = chosen
    span = max(1e-6, end - start)
    eased = _smoothstep((time - start) / span)
    return eased if direction > 0 else 1.0 - eased
