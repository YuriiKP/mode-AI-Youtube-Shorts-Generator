"""Clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
     window horizontally across the frame to keep faces centred (Haar
     cascade — same approach as the original repo, no external models). When
     SLIDE_EFFECT is enabled the window instead pans left/right between the
     scene transitions detected inside the clip (see scene_transitions.py),
     alternating the direction at every transition. When CUT_EFFECT is enabled
     those same cuts are instead (or additionally) blended together — a short
     dissolve / fade / flash / zoom / chroma / merge / whip / spin / shake /
     glitch that softens the hard cut without changing
     the clip length or its audio, so subtitles and music stay in sync.
"""

import collections
import math
import os
import random
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

from .scene_transitions import (
    CutEffect,
    build_cut_effects,
    build_slide_segments,
    detect_transitions,
    parse_cut_effect_styles,
    slide_progress,
)

# Default output folder, used only when the caller does not pass one.
DEFAULT_OUTPUT_DIR = "output"


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -to end → re-encoded mp4 with audio."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        source_path,
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        # Keep a single video + audio stream and drop chapters / subtitles /
        # data streams: leftover chapter metadata makes MoviePy's reader fail
        # later in the pipeline (its parser crashes on single-chapter files).
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-map_chapters",
        "-1",
        "-sn",
        "-dn",
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
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


# --- pixel helpers for the cut transitions ---------------------------------
# The cut effects re-mix the frames the crop pass already produced, so they only
# need a handful of cheap pixel operations (blend, shift, split, blur, rotate).
# They run inside the per-frame loop, hence the lazily imported cv2 / numpy.


def _as_u8(array, np):
    """Clamp a float image back into 8-bit and make it contiguous."""
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _mix(frame_a, frame_b, alpha, np):
    """Blend ``frame_a`` (weight ``1-alpha``) with ``frame_b`` (weight ``alpha``)."""
    if alpha >= 1.0:
        return np.ascontiguousarray(frame_b)
    if alpha <= 0.0:
        return np.ascontiguousarray(frame_a)
    a = frame_a.astype(np.float32)
    b = frame_b.astype(np.float32)
    return _as_u8(a * (1.0 - alpha) + b * alpha, np)


def _shift_frame(frame, delta, np, axis=1):
    """Shift a frame by ``delta`` pixels along ``axis``, replicating the edge."""
    if delta == 0:
        return frame
    out = frame.copy()
    size = frame.shape[axis]
    distance = abs(delta)
    if distance >= size:
        return out
    if axis == 1:
        if delta > 0:
            out[:, distance:] = frame[:, : size - distance]
        else:
            out[:, : size - distance] = frame[:, distance:]
    else:
        if delta > 0:
            out[distance:, :] = frame[: size - distance, :]
        else:
            out[: size - distance, :] = frame[distance:, :]
    return out


def _chroma_split(frame, shift, np):
    """Push the red and blue channels apart by ``shift`` px — colour fringing."""
    if shift == 0:
        return frame
    out = frame.copy()
    out[..., 2] = _shift_frame(frame, shift, np)[..., 2]
    out[..., 0] = _shift_frame(frame, -shift, np)[..., 0]
    return out


def _scale_center(frame, scale, cv2, np):
    """Zoom in by ``scale`` and crop back to size (edge content is kept)."""
    if scale <= 1.0 + 1e-3:
        return frame
    height, width = frame.shape[:2]
    new_w = max(2, int(round(width * scale)))
    new_h = max(2, int(round(height * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    return np.ascontiguousarray(resized[top : top + height, left : left + width])


def _gaussian(frame, sigma, cv2, np):
    """Gaussian blur with ``sigma``; a no-op for tiny values."""
    if sigma <= 0.05:
        return frame
    return np.ascontiguousarray(
        cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)
    )


def _motion_blur(frame, length, np):
    """Average ``length`` horizontal shifts — a directional (whip) blur."""
    if length <= 0:
        return frame
    acc = frame.astype(np.float32)
    for step in range(1, length + 1):
        acc += _shift_frame(frame, step, np).astype(np.float32)
    return acc / float(length + 1)


def _rotate(frame, angle, cv2, np):
    """Rotate by ``angle`` degrees, replicating edges instead of showing black."""
    if abs(angle) < 0.05:
        return frame
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return np.ascontiguousarray(rotated)


class _CutEffectRunner:
    """Applies the planned cut transitions while the crop pass streams frames.

    Instances are driven once per output frame, in order, by
    :func:`_reframe_vertical`. A short buffer of the *raw* cropped frames is kept
    so the cross-blend styles (``dissolve`` / ``merge``) can mix the frames before
    a cut with the ones after it; the punch styles (``zoom`` / ``chroma`` / ...)
    need no history. The clip keeps its length and its audio, so the caller's
    timings never shift.
    """

    _ZOOM_PEAK: float = 0.12  # ``zoom`` punch magnitude
    _CHROMA_ZOOM_PEAK: float = 0.16  # ``chroma`` zoom strength
    _CHROMA_SPLIT_PEAK: int = 7  # ``chroma`` red/blue separation, px
    _MERGE_BLUR_PEAK: float = 7.0  # ``merge`` blur at the cut, sigma
    _GLITCH_SPLIT_PEAK: int = 9  # ``glitch`` channel separation, px
    _GLITCH_SHIFT_PEAK: int = 14  # ``glitch`` slice displacement, px
    _GLITCH_SHAKE_PEAK: int = 7  # ``glitch`` whole-frame jitter, px
    _WHIP_BLUR_PEAK: int = 16  # ``whip`` motion-blur length, px
    _WHIP_SHIFT_PEAK: int = 10  # ``whip`` pan offset, px
    _SPIN_PEAK: float = 5.0  # ``spin`` rotation, degrees
    _SHAKE_PEAK: int = 12  # ``shake`` jitter, px

    # Styles that need the pre-cut frames buffered to cross-blend the shots.
    _BLEND_STYLES: Tuple[str, ...] = ("dissolve", "merge")

    def __init__(
        self, effects: "Sequence[CutEffect]", crop_w: int, crop_h: int
    ) -> None:
        self.effects = effects
        self.crop_w: int = crop_w
        self.crop_h: int = crop_h
        self.index: int = 0
        self._snapshot = None
        self._punch_dir: int = 1  # alternates the direction of directional styles
        # Only the cross-blend styles need history; size the ring buffer to the
        # longest such window so ``list(buffer)[-after:]`` always has enough.
        max_history = max(
            (
                effect.end_frame - effect.cut_frame
                for effect in effects
                if effect.style in self._BLEND_STYLES
            ),
            default=0,
        )
        self.buffer = collections.deque(maxlen=max_history) if max_history else None

    def apply(self, frame_index: int, frame):
        """Return ``frame`` with any transition for ``frame_index`` baked in."""
        # Retire effects that already ended (and drop their cross-blend snapshot).
        while (
            self.index < len(self.effects)
            and frame_index >= self.effects[self.index].end_frame
        ):
            self.index += 1
            self._snapshot = None
        self._punch_dir = 1 if self.index % 2 == 0 else -1

        result = frame
        if self.index < len(self.effects):
            effect = self.effects[self.index]
            if effect.start_frame <= frame_index < effect.end_frame:
                result = self._render(effect, frame_index, frame)

        # Buffer the *raw* frame: the cross-blend styles read history before the cut.
        if self.buffer is not None:
            self.buffer.append(frame)
        return result

    def _render(self, effect: CutEffect, frame_index: int, frame):
        import cv2
        import numpy as np

        cut, start, end, style = effect
        before = max(1, cut - start)
        after = max(1, end - cut)
        span = max(1, end - start)

        # Triangular ramp: 0 at the window edges, 1 exactly at the cut. The punch
        # styles scale their strength with it, so the effect peaks on the cut and
        # fades back to the untouched frame before the window ends.
        if frame_index < cut:
            punch = (frame_index - start + 1) / before
        else:
            punch = 1.0 - (frame_index - cut) / after

        if style in self._BLEND_STYLES:
            if frame_index < cut:
                if style == "merge":
                    return _gaussian(frame, self._MERGE_BLUR_PEAK * punch, cv2, np)
                return frame  # frames before the cut are written untouched
            if self._snapshot is None:
                history = list(self.buffer) if self.buffer is not None else []
                self._snapshot = history[-after:]
            step = frame_index - cut
            if step >= len(self._snapshot):
                return frame
            alpha = (step + 1) / after
            out = _mix(self._snapshot[step], frame, alpha, np)
            if style == "merge":
                out = _gaussian(out, self._MERGE_BLUR_PEAK * punch, cv2, np)
            return out

        if style in ("fade", "flash"):
            color = 0.0 if style == "fade" else 255.0
            mixed = frame.astype(np.float32) * (1.0 - punch) + color * punch
            return _as_u8(mixed, np)

        if style == "zoom":
            position = (frame_index - start) / span
            scale = 1.0 + self._ZOOM_PEAK * math.sin(math.pi * position)
            return _scale_center(frame, scale, cv2, np)

        if style == "chroma":
            # Zoom punch plus a red/blue split that peaks on the cut — the classic
            # "chromatic zoom" edit transition.
            out = _scale_center(frame, 1.0 + self._CHROMA_ZOOM_PEAK * punch, cv2, np)
            return _chroma_split(out, int(round(self._CHROMA_SPLIT_PEAK * punch)), np)

        if style == "whip":
            direction = self._punch_dir
            shifted = _shift_frame(
                frame, int(round(self._WHIP_SHIFT_PEAK * punch)) * direction, np
            )
            length = int(round(self._WHIP_BLUR_PEAK * punch))
            return _as_u8(_motion_blur(shifted, length, np), np)

        if style == "spin":
            angle = self._SPIN_PEAK * punch * self._punch_dir
            out = _rotate(frame, angle, cv2, np)
            return _scale_center(out, 1.0 + 0.05 * punch, cv2, np)

        if style == "shake":
            rng = random.Random(frame_index)
            amplitude = self._SHAKE_PEAK * punch
            out = _shift_frame(
                frame, int(round(rng.uniform(-1.0, 1.0) * amplitude)), np
            )
            out = _shift_frame(
                out, int(round(rng.uniform(-1.0, 1.0) * amplitude)), np, axis=0
            )
            return _scale_center(out, 1.0 + 0.04 * punch, cv2, np)

        if style == "glitch":
            rng = random.Random(frame_index)
            out = frame.copy()
            split = int(round(self._GLITCH_SPLIT_PEAK * punch))
            if split:
                out[..., 2] = _shift_frame(frame, split, np)[..., 2]
                out[..., 0] = _shift_frame(frame, -split, np)[..., 0]
            height = out.shape[0]
            max_shift = int(round(self._GLITCH_SHIFT_PEAK * punch))
            if max_shift > 0:
                for _ in range(3):
                    band_h = rng.randint(4, max(5, height // 8))
                    top = rng.randint(0, max(0, height - band_h))
                    delta = rng.randint(-max_shift, max_shift)
                    out[top : top + band_h] = _shift_frame(
                        out[top : top + band_h], delta, np
                    )
            jitter = int(round(self._GLITCH_SHAKE_PEAK * punch))
            if jitter > 0:
                out = _shift_frame(out, rng.randint(-jitter, jitter), np)
            return np.ascontiguousarray(out)

        return frame


def _reframe_vertical(
    in_path: str,
    out_path: str,
    aspect_ratio: str,
    face_tracking: bool = True,
    slide_effect: bool = False,
    slide_gap: float = 3.0,
    slide_range: float = 1.0,
    cut_effect: bool = False,
    cut_effect_duration: float = 0.25,
    cut_effect_types: Sequence[str] = ("dissolve",),
    cut_effect_max: int = 3,
) -> str:
    """Crop the cut clip to the target aspect ratio, tracking faces if possible."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required. Install it with:\n"
            "    pip install -r requirements.txt"
        ) from e

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Compute the largest crop that fits inside the frame at the target ratio.
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    # OpenCV <= 4.x ships Haar cascades via CascadeClassifier; OpenCV 5.x
    # removed them (only the DNN FaceDetectorYN remains, which needs a model
    # file). When disabled or unavailable we skip face tracking and fall back to
    # a static centre crop — the output still keeps the requested aspect ratio.
    face_cascade = None
    if face_tracking and hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                face_cascade = cascade
        except Exception:
            face_cascade = None
    if face_cascade is None:
        reason = (
            "disabled"
            if not face_tracking
            else "unavailable (no Haar cascades in this OpenCV build)"
        )
        print(
            f"[clip] face tracking {reason}; using static centre crop",
            flush=True,
        )

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    # --- optional slide between scene transitions -------------------------
    # Detect the cuts inside the cut clip and build a plan that slowly pans the
    # crop window from one side of the wide frame to the other, alternating the
    # direction at every (grouped) transition.
    max_x0 = max(0, src_w - crop_w)
    # ``slide_range`` (0..1) scales the full left/right travel. The reduced
    # range is centred, so ``1.0`` pans the window edge-to-edge and ``0.0``
    # freezes it in the middle, leaving an equal inset from both edges.
    range_fraction = min(1.0, max(0.0, slide_range))
    travel = int(round(max_x0 * range_fraction))
    slide_base = (max_x0 - travel) // 2

    # Scene cuts inside the clip are needed by both the slide (to know where to
    # pan) and the cut effects (to know where to blend), so detect them once.
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = (frame_count / fps) if fps else 0.0
    transitions: List[float] = []
    if (slide_effect and max_x0 > 0 and travel > 0) or cut_effect:
        try:
            transitions = detect_transitions(in_path)
        except Exception as exc:
            print(f"[clip] transition detection failed: {exc}", flush=True)
            transitions = []

    slide_segments = []
    if slide_effect and max_x0 > 0 and travel > 0:
        slide_segments = build_slide_segments(transitions, duration, gap=slide_gap)
        print(
            f"[clip] slide effect: {len(transitions)} transition(s) -> "
            f"{len(slide_segments)} slide segment(s)",
            flush=True,
        )

    cut_effects = []
    if cut_effect:
        cut_effects = build_cut_effects(
            transitions,
            duration,
            fps,
            window_seconds=cut_effect_duration,
            styles=parse_cut_effect_styles(cut_effect_types),
            max_count=cut_effect_max,
        )
        print(
            f"[clip] cut effect: {len(transitions)} transition(s) -> "
            f"{len(cut_effects)} blend(s)",
            flush=True,
        )
    cut_runner = _CutEffectRunner(cut_effects, crop_w, crop_h) if cut_effects else None

    frame_index = 0
    last_center: Optional[Tuple[int, int]] = None
    smoothing = 0.15  # how aggressively to chase a new face position
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
                )
                if len(faces) > 0:
                    # Pick the largest face — usually the speaker.
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    cx = x + w // 2
                    cy = y + h // 2
                    if last_center is None:
                        last_center = (cx, cy)
                    else:
                        lx, ly = last_center
                        last_center = (
                            int(lx + (cx - lx) * smoothing),
                            int(ly + (cy - ly) * smoothing),
                        )
            if last_center is None:
                last_center = (src_w // 2, src_h // 2)

            cx, cy = last_center
            y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))

            if slide_effect and max_x0 > 0:
                progress = slide_progress(
                    frame_index / fps if fps else 0.0, slide_segments
                )
                x0 = slide_base + int(round(progress * travel))
                x0 = max(0, min(max_x0, x0))
            else:
                x0 = max(0, min(max_x0, cx - crop_w // 2))

            # ``frame[...]`` is a strided *view* of the decoded frame. Some
            # OpenCV builds (4.14.x) raise "Unknown C++ exception from OpenCV
            # code" when a non-contiguous array reaches ``VideoWriter.write``,
            # so hand it a contiguous copy instead.
            cropped = np.ascontiguousarray(frame[y0 : y0 + crop_h, x0 : x0 + crop_w])
            if cut_runner is not None:
                cropped = cut_runner.apply(frame_index, cropped)
            writer.write(cropped)
            frame_index += 1
    finally:
        # Always release the reader and the writer: on Windows the reader keeps
        # an open handle on the cut clip, so if an error skips this the caller's
        # cleanup cannot delete that file (WinError 32) and the real error gets
        # masked by the failed removal.
        cap.release()
        writer.release()

    # Mux audio from the cut clip back onto the silent reframed video.
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        silent_path,
        "-i",
        in_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-map_chapters",
        "-1",
        "-sn",
        "-dn",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(silent_path)
    return out_path


def reframe_video(
    in_path: str,
    out_path: str,
    aspect_ratio: str = "9:16",
    *,
    face_tracking: bool = True,
    slide_effect: bool = False,
    slide_gap: float = 3.0,
    slide_range: float = 1.0,
    cut_effect: bool = False,
    cut_effect_duration: float = 0.25,
    cut_effect_types: Sequence[str] = ("dissolve",),
    cut_effect_max: int = 3,
) -> str:
    """Reframe an already-built video to ``aspect_ratio``, without re-cutting.

    This is the *crop* step of the pipeline — the same OpenCV pass ``crop_clip``
    runs — applied to a video that is already a single, finished file rather than
    to a time window of a source: nothing is seeked or cut, every frame of
    ``in_path`` is re-framed and the audio is carried over (see
    :func:`_reframe_vertical`). The caller keeps the file's timeline untouched,
    which is what lets timings computed for the input still line up after the
    call — subtitles and music therefore stay in sync.

    The montage pipeline uses it right after stitching its segments: those come
    out in the source aspect ratio, so the stitched file still has to be cut to
    the target shape before the rest of the pipeline (vertical fit, colour/lens
    effects, uniqueness, music, subtitles) runs, exactly as a highlight is
    cropped before it is enhanced.

    Unlike the private helper it wraps, the output path is required and the
    ``.silent.mp4`` intermediate is handled internally, so this is the intended
    entry point for callers outside this module.

    Returns:
        ``out_path`` — the reframed video, matching ``aspect_ratio``.
    """
    return _reframe_vertical(
        in_path,
        out_path,
        aspect_ratio,
        face_tracking=face_tracking,
        slide_effect=slide_effect,
        slide_gap=slide_gap,
        slide_range=slide_range,
        cut_effect=cut_effect,
        cut_effect_duration=cut_effect_duration,
        cut_effect_types=cut_effect_types,
        cut_effect_max=cut_effect_max,
    )


def crop_clip(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    face_tracking: bool = True,
    slide_effect: bool = False,
    slide_gap: float = 3.0,
    slide_range: float = 1.0,
    cut_effect: bool = False,
    cut_effect_duration: float = 0.25,
    cut_effect_types: Sequence[str] = ("dissolve",),
    cut_effect_max: int = 3,
) -> str:
    """Cut + reframe one highlight, returning the mp4 path."""
    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        _reframe_vertical(
            cut_path,
            out_path,
            aspect_ratio,
            face_tracking=face_tracking,
            slide_effect=slide_effect,
            slide_gap=slide_gap,
            slide_range=slide_range,
            cut_effect=cut_effect,
            cut_effect_duration=cut_effect_duration,
            cut_effect_types=cut_effect_types,
            cut_effect_max=cut_effect_max,
        )
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path


def crop_highlights(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    face_tracking: bool = True,
    slide_effect: bool = False,
    slide_gap: float = 3.0,
    slide_range: float = 1.0,
    cut_effect: bool = False,
    cut_effect_duration: float = 0.25,
    cut_effect_types: Sequence[str] = ("dissolve",),
    cut_effect_max: int = 3,
    start_index: int = 0,
) -> List[Dict]:
    out_dir = out_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    # Include the source file name in every clip so that processing several
    # videos into the same folder does not overwrite clips from earlier ones.
    # ``start_index`` keeps the clip number a single running order across all
    # source videos (first short ever = 1, regardless of which input it is).
    source_stem = os.path.splitext(os.path.basename(source_path))[0]
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        number = start_index + i
        out_path = os.path.join(out_dir, f"short_{number:02d}_{source_stem}.mp4")
        print(
            f"[clip] {i}/{len(highlights)}: {h.get('title', '(untitled)')}",
            flush=True,
        )
        try:
            crop_clip(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                face_tracking=face_tracking,
                slide_effect=slide_effect,
                slide_gap=slide_gap,
                slide_range=slide_range,
                cut_effect=cut_effect,
                cut_effect_duration=cut_effect_duration,
                cut_effect_types=cut_effect_types,
                cut_effect_max=cut_effect_max,
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
