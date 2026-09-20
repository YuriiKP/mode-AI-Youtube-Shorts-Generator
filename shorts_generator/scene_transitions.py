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

from typing import List, Sequence, Tuple

# A slide segment: ``(start_seconds, end_seconds, direction)`` where
# ``direction`` is ``+1`` towards the right edge of the frame, ``-1`` to the left.
SlideSegment = Tuple[float, float, int]

__all__ = [
    "SlideSegment",
    "detect_transitions",
    "build_slide_segments",
    "slide_progress",
]


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
    """
    boundaries: List[float] = [0.0]
    for time in sorted(transitions):
        if not 0.0 < time < duration:
            continue
        if time - boundaries[-1] < gap:
            continue  # too close to the previous boundary -> same group
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
