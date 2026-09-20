"""Clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
     window horizontally across the frame to keep faces centred (Haar
     cascade — same approach as the original repo, no external models). When
     SLIDE_EFFECT is enabled the window instead pans left/right between the
     scene transitions detected inside the clip (see scene_transitions.py),
     alternating the direction at every transition.
"""

import os
import subprocess
from typing import Dict, List, Optional, Tuple

from .scene_transitions import (
    build_slide_segments,
    detect_transitions,
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


def _reframe_vertical(
    in_path: str,
    out_path: str,
    aspect_ratio: str,
    face_tracking: bool = True,
    slide_effect: bool = False,
    slide_gap: float = 3.0,
) -> str:
    """Crop the cut clip to the target aspect ratio, tracking faces if possible."""
    try:
        import cv2  # type: ignore
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
    slide_segments = []
    if slide_effect and max_x0 > 0:
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = (frame_count / fps) if fps else 0.0
        try:
            transitions = detect_transitions(in_path)
        except Exception as exc:
            print(f"[clip] transition detection failed: {exc}", flush=True)
            transitions = []
        slide_segments = build_slide_segments(transitions, duration, gap=slide_gap)
        print(
            f"[clip] slide effect: {len(transitions)} transition(s) -> "
            f"{len(slide_segments)} slide segment(s)",
            flush=True,
        )

    frame_index = 0
    last_center: Optional[Tuple[int, int]] = None
    smoothing = 0.15  # how aggressively to chase a new face position
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

        if slide_segments:
            progress = slide_progress(frame_index / fps if fps else 0.0, slide_segments)
            x0 = int(round(progress * max_x0))
            x0 = max(0, min(max_x0, x0))
        else:
            x0 = max(0, min(max_x0, cx - crop_w // 2))

        cropped = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
        writer.write(cropped)
        frame_index += 1

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


def crop_clip(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    face_tracking: bool = True,
    slide_effect: bool = False,
    slide_gap: float = 3.0,
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
) -> List[Dict]:
    out_dir = out_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    # Include the source file name in every clip so that processing several
    # videos into the same folder does not overwrite clips from earlier ones.
    source_stem = os.path.splitext(os.path.basename(source_path))[0]
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"short_{i:02d}_{source_stem}.mp4")
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
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
