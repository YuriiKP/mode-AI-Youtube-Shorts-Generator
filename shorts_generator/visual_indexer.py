"""Visual indexing — describe what happens on screen between transcription and ranking.

The visual indexer adds a step between Whisper transcription and LLM highlight
ranking: frames are sampled scene by scene and turned into short English
descriptions of the action (emotions, movements, objects, on-screen graphics).
Those descriptions are merged with the transcript so the ranking LLM can prefer
moments where the words are reinforced by a strong visual — a laugh, a reaction,
a fail, an unusual object.

The engine is chosen with ``VISUAL_INDEXER_TYPE`` in ``.env`` using the
*Strategy* pattern:

* ``florence`` — local, ``microsoft/Florence-2-large`` via ``transformers``
  (CUDA + float16 when available, CPU otherwise);
* ``gemini``   — cloud, ``google-genai`` (Gemini Flash); frames are sent in
  batches with the shared :data:`VISUAL_PROMPT`;
* ``gemini_video`` — cloud, the whole file is uploaded to the Gemini Files API
  (up to 2 GB) and described in a single request;
* ``qwen_video`` — local, ``Qwen2.5-VL`` via ``transformers`` + ``torch``; the
  video is cut into ``VISUAL_INDEXER_MAX_SCENE_SECONDS`` windows and each
  window's sampled frames are described in their own request, then the per-window
  timelines are concatenated into one index;
* ``none``     — disabled (same as ``VISUAL_INDEXER_ENABLED=false``).

Everything here is optional: the heavy libraries (``transformers``, ``torch``,
``google-genai``) are imported lazily inside the engines, so the rest of the
pipeline keeps working when visual indexing is switched off.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .config import ConfigError, Settings, require_gemini_key

__all__ = [
    "SceneSpan",
    "BaseVisualIndexer",
    "FlorenceIndexer",
    "GeminiFlashIndexer",
    "GeminiVideoIndexer",
    "QwenVideoIndexer",
    "build_visual_indexer",
    "index_video",
    "detect_scenes",
    "merge_transcripts_and_visuals",
    "format_merged_log",
    "VISUAL_PROMPT",
    "VisualIndexerSetupError",
]

# A scene window: ``(start_seconds, end_seconds)``.
SceneSpan = Tuple[float, float]

# Крупность лога ранжирования, в секундах. Используется ТОЛЬКО как запасной
# вариант, когда транскрипта нет вообще (немое видео): тогда визуальные сцены
# раскладываются по окнам такого размера. При наличии транскрипта лог строится
# по одной строке на реплику Whisper — с точностью до секунды, — чтобы модель
# могла скопировать границы выбранных фраз в start_time/end_time.
MERGE_WINDOW_SECONDS = 15.0

DEFAULT_FLORENCE_MODEL = "microsoft/Florence-2-large"
DEFAULT_SCENE_THRESHOLD = 27.0
DEFAULT_BATCH_SIZE = 8
GEMINI_MAX_ATTEMPTS = 3
GEMINI_RETRY_BACKOFF = 2.0
# Whole-video engine: how often to poll the Files API while Uploading, and how
# long to wait for the video to finish processing.
GEMINI_VIDEO_POLL_SECONDS = 2.0
GEMINI_UPLOAD_TIMEOUT_SECONDS = 900.0
# Qwen2.5-VL engine (``qwen_video``): fallback checkpoint used only when
# VISUAL_INDEXER_MODEL is empty, a fallback window length, and the generation cap.
# The engine carries no settings of its own — it reuses the shared knobs: the
# window length is VISUAL_INDEXER_MAX_SCENE_SECONDS and the frames per window are
# VISUAL_INDEXER_BATCH_SIZE; the model and device are VISUAL_INDEXER_MODEL /
# VISUAL_INDEXER_DEVICE, exactly like the local Florence engine.
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_QWEN_WINDOW_SECONDS = 15.0
DEFAULT_QWEN_MAX_NEW_TOKENS = 1024

# Задача Florence-2 для каждого взятого кадра. <MORE_DETAILED_CAPTION> возвращает
# целый абзац на кадр (до 1024 токенов); в логе ранжирования такие простыни
# занимали ~90% промпта и топили саму речь, из-за чего модель выбирала моменты
# почти вслепую. <DETAILED_CAPTION> даёт 1–2 предложения — этого достаточно,
# чтобы «увидеть» эмоцию или действие в кадре, не забивая транскрипт.
FLORENCE_TASK = "<DETAILED_CAPTION>"

# Сколько символов визуального описания окна доходит до лога ранжирования.
# Страховка от многословных движков: даже если модель выдала абзац, в промпт
# попадёт только его начало.
VISUAL_TEXT_MAX_CHARS = 300

# Bump when the on-disk visual-index layout changes; older files are ignored.
# v2: Florence переключён на FLORENCE_TASK (короткие подписи) — старый кэш
# хранит абзацы <MORE_DETAILED_CAPTION>, которые забивали промпт ранкера.
VISUAL_CACHE_VERSION = 2

# Shared prompt for the cloud engine (kept short and English, as required).
VISUAL_PROMPT = (
    "Describe what happens in these frames / this video. Focus on the dynamics, "
    "the speaker's emotions, sudden movements, visual humour and on-screen "
    "graphics. Ignore the static background. Keep it short and punchy, in "
    "English, at most 2-3 sentences."
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _chunks(items: Sequence[Any], size: int) -> Iterator[List[Any]]:
    """Yield ``items`` in lists of at most ``size`` elements."""
    size = max(1, int(size))
    for index in range(0, len(items), size):
        yield list(items[index : index + size])


def _strip_fences(raw: str) -> str:
    """Strip a fenced JSON block, if the model wrapped its JSON."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _timeline_end(segments: Sequence[Dict], scenes: Sequence[Dict]) -> float:
    """The last second covered by either the transcript or the visuals."""
    end = 0.0
    for segment in segments:
        try:
            end = max(end, float(segment["end"]))
        except (KeyError, TypeError, ValueError):
            continue
    for scene in scenes:
        try:
            end = max(end, float(scene["end"]))
        except (KeyError, TypeError, ValueError):
            continue
    return end


class _FrameExtractor:
    """Context manager that seeks an OpenCV capture and returns RGB frames.

    ``max_height`` (0 = keep the source resolution) downscales every returned
    frame so it is at most that many pixels tall. Frame-based cloud engines send
    each frame as an image, so a smaller frame means fewer tokens; the aspect
    ratio is preserved.
    """

    def __init__(self, video_path: str, max_height: int = 0) -> None:
        self.video_path = video_path
        self.max_height = int(max_height) if max_height else 0
        self._cv2 = None
        self._cap = None

    def __enter__(self) -> "_FrameExtractor":
        import cv2  # type: ignore

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"could not open {self.video_path!r} for frame extraction"
            )
        return self

    def frame_rgb(self, timestamp: float):
        """Return the frame at ``timestamp`` seconds as an RGB numpy array."""
        if self._cap is None or self._cv2 is None:
            return None
        self._cap.set(self._cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
        ok, frame = self._cap.read()
        if not ok:
            return None
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        return self._maybe_downscale(rgb)

    def _maybe_downscale(self, rgb):
        """Shrink a frame so its height is at most ``max_height`` (never grows)."""
        if self.max_height <= 0 or rgb is None:
            return rgb
        height, width = rgb.shape[:2]
        if height <= self.max_height:
            return rgb
        scale = self.max_height / float(height)
        new_width = max(1, int(round(width * scale)))
        return self._cv2.resize(
            rgb,
            (new_width, self.max_height),
            interpolation=self._cv2.INTER_AREA,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _probe_duration(video_path: str) -> float:
    """Best-effort video duration in seconds (0.0 when it cannot be read)."""
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        finally:
            cap.release()
        if fps > 0 and frames > 0:
            return frames / fps
    except Exception:
        pass
    return 0.0


def _probe_dimensions(video_path: str) -> Tuple[int, int]:
    """Best-effort ``(width, height)`` of a video (``(0, 0)`` when unreadable)."""
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0, 0
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
        finally:
            cap.release()
        return width, height
    except Exception:
        return 0, 0


def _downscale_for_upload(
    video_path: str, max_height: int, ffmpeg_path: str = ""
) -> Tuple[str, Optional[str]]:
    """Return ``(path_to_upload, temp_dir_or_None)`` shrunk to ``max_height``.

    The whole-video engine uploads the file to the Gemini Files API: a smaller
    file reaches Gemini sooner and finishes processing faster, so a 1080p source
    is re-encoded once to ``max_height`` before it is sent. The file is only ever
    *shrunk* — when it is already at or below ``max_height``, when its height
    cannot be read, or when FFmpeg is missing, the original path is returned
    unchanged and ``temp_dir`` is ``None`` (nothing to clean up).
    """
    if max_height <= 0:
        return video_path, None
    _, height = _probe_dimensions(video_path)
    if height and height <= max_height:
        return video_path, None

    import subprocess

    from .postprocess.ffmpeg import resolve_ffmpeg_binary

    ffmpeg = resolve_ffmpeg_binary(ffmpeg_path)
    temp_dir = tempfile.mkdtemp(prefix="vi_scale_")
    out_path = os.path.join(temp_dir, "video.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        f"scale=-2:{int(max_height)}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:  # noqa: BLE001 - fall back to the original file
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(
            f"[visual] could not downscale the video ({exc}); uploading the original",
            flush=True,
        )
        return video_path, None

    size = os.path.getsize(out_path)
    print(
        f"[visual] downscaled the upload to {int(max_height)}p "
        f"({size / (1024 * 1024):.1f} MB)",
        flush=True,
    )
    return out_path, temp_dir


def _ascii_upload_path(video_path: str):
    """Return ``(path_to_upload, temp_dir_or_None)``.

    The Gemini Files API puts the file name into an ASCII multipart header, so a
    file called e.g. "Ненасытный Берсерк.mp4" makes the upload fail with a
    ``UnicodeEncodeError``. When the name is not pure ASCII we upload through a
    hard link (or, failing that, a copy) that has an ASCII name, and hand the
    caller the temporary directory to clean up.
    """
    base = os.path.basename(video_path)
    try:
        base.encode("ascii")
        return video_path, None
    except UnicodeEncodeError:
        pass

    ext = os.path.splitext(base)[1] or ".mp4"
    parents = [os.path.dirname(os.path.abspath(video_path)), tempfile.gettempdir()]
    for parent in parents:
        try:
            temp_dir = tempfile.mkdtemp(prefix="vi_upload_", dir=parent)
        except OSError:
            continue
        temp_path = os.path.join(temp_dir, "video" + ext)
        try:
            os.link(video_path, temp_path)  # hard link first (no copy)
            return temp_path, temp_dir
        except OSError:
            try:
                shutil.copyfile(video_path, temp_path)
                return temp_path, temp_dir
            except OSError:
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue

    temp_dir = tempfile.mkdtemp(prefix="vi_upload_")
    temp_path = os.path.join(temp_dir, "video" + ext)
    shutil.copyfile(video_path, temp_path)
    return temp_path, temp_dir


# ---------------------------------------------------------------------------
# Scene detection
# ---------------------------------------------------------------------------


def detect_scenes(
    video_path: str,
    *,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    max_seconds: Optional[float] = None,
) -> List[SceneSpan]:
    """Return the scene windows ``(start, end)`` of a video.

    Uses PySceneDetect when available and falls back to a small OpenCV
    frame-difference detector otherwise. ``threshold`` tunes both paths: it is
    handed to PySceneDetect's ContentDetector when the library is installed and
    to the OpenCV difference detector otherwise. When ``max_seconds`` is set,
    scenes longer than that are split so sampled frames cover the whole clip (a
    talking-head video is often a single very long scene).
    """
    try:
        scenes = _scenes_with_scenedetect(video_path, threshold=threshold)
    except Exception:
        scenes = _scenes_with_opencv(video_path, threshold=threshold)
    if max_seconds and max_seconds > 0:
        scenes = _split_long_scenes(scenes, max_seconds)
    return scenes


def _scenes_with_scenedetect(
    video_path: str, *, threshold: float = DEFAULT_SCENE_THRESHOLD
) -> List[SceneSpan]:
    """Detect cuts with PySceneDetect, honoring ``threshold`` for content cuts."""
    from scenedetect import SceneManager, open_video  # type: ignore
    from scenedetect.detectors import ContentDetector, ThresholdDetector  # type: ignore

    video = open_video(video_path)
    manager = SceneManager()
    # Forward VISUAL_INDEXER_SCENE_THRESHOLD to the cut detector: ContentDetector
    # scores cuts on the same content_val scale as the setting (its own default
    # is ~27, the value the .env ships with), so raising the threshold now merges
    # nearby cuts instead of being silently ignored. ThresholdDetector keeps its
    # own default — its threshold is a luminance cut-off in a different range, so
    # the content threshold does not apply to it.
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.add_detector(ThresholdDetector())
    manager.detect_scenes(video)
    return [
        (start.get_seconds(), end.get_seconds())
        for start, end in manager.get_scene_list()
    ]


def _scenes_with_opencv(video_path: str, *, threshold: float = 27.0) -> List[SceneSpan]:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path!r} for scene detection")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration = frames / fps if fps > 0 and frames > 0 else 0.0
    cuts: List[float] = [0.0]
    previous = None
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (64, 36))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if (
                previous is not None
                and float(cv2.absdiff(gray, previous).mean()) >= threshold
            ):
                cuts.append(index / fps)
            previous = gray
            index += 1
    finally:
        cap.release()
    if duration <= 0 and cuts:
        duration = cuts[-1]
    cuts.append(duration)
    return [
        (cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]
    ]


def _split_long_scenes(
    scenes: Sequence[SceneSpan], max_seconds: float
) -> List[SceneSpan]:
    split: List[SceneSpan] = []
    for start, end in scenes:
        length = float(end) - float(start)
        if length <= max_seconds or max_seconds <= 0:
            split.append((float(start), float(end)))
            continue
        pieces = max(1, int(math.ceil(length / max_seconds)))
        step = length / pieces
        for i in range(pieces):
            piece_start = float(start) + i * step
            piece_end = min(float(end), piece_start + step)
            if piece_end > piece_start:
                split.append((piece_start, piece_end))
    return split


# ---------------------------------------------------------------------------
# Setup errors: raised before any expensive work when the selected engine
# cannot possibly run (missing dependency, missing API key). These are
# deterministic — re-running changes nothing — so they stop the pipeline with a
# fix-it hint instead of degrading silently.
# ---------------------------------------------------------------------------


class VisualIndexerSetupError(RuntimeError):
    """The selected visual-indexing engine cannot run as configured."""


def _install_hint(packages: str) -> str:
    """A ``pip install`` line bound to the interpreter running this script.

    Pinning ``sys.executable`` matters: a bare ``pip install`` may land in a
    different Python than the one executing the app (the usual cause of "but I
    installed it!"), so the hint names the exact interpreter to target.
    """
    return f'"{sys.executable}" -m pip install {packages}'


def _florence_dependency_message(exc: ImportError) -> str:
    return (
        "VISUAL_INDEXER_TYPE=florence needs 'transformers' and 'torch', but "
        f"importing them failed: {exc}\n"
        f"Interpreter: {sys.executable}\n"
        "Install the packages into THAT interpreter, then re-run:\n"
        f"    {_install_hint('transformers torch')}\n"
        "If you use a virtual environment, activate it first so 'pip' targets "
        "the same interpreter. Alternatively set VISUAL_INDEXER_TYPE=gemini or "
        "VISUAL_INDEXER_ENABLED=false."
    )


def _gemini_dependency_message(engine: str, exc: ImportError) -> str:
    return (
        f"{engine} needs the 'google-genai' SDK, but importing it failed: {exc}\n"
        f"Interpreter: {sys.executable}\n"
        "Install it into THAT interpreter, then re-run:\n"
        f"    {_install_hint('google-genai')}"
    )


def _qwen_dependency_message(exc: ImportError) -> str:
    return (
        "VISUAL_INDEXER_TYPE=qwen_video needs 'transformers', 'torch' and "
        f"'qwen-vl-utils', but importing them failed: {exc}\n"
        f"Interpreter: {sys.executable}\n"
        "Install the packages into THAT interpreter, then re-run:\n"
        f"    {_install_hint('transformers torch qwen-vl-utils')}\n"
        "Qwen2.5-VL needs transformers>=4.49, while Florence-2 in the same venv "
        "needs <4.54, so 'transformers>=4.49,<4.54' keeps both working."
    )


# ---------------------------------------------------------------------------
# Provider error classification
#
# Shared by the cloud engines so a permanent API error (a retired model name, a
# rejected key, a malformed request) is reported at once instead of being
# retried through the whole backoff — which is what hid a 404 "model no longer
# available" behind a "gave up" line and let the run continue without visuals.
# ---------------------------------------------------------------------------

# HTTP statuses worth retrying: request timeout, conflict and "too many
# requests". Any other 4xx is permanent; 5xx is transient overload.
_RETRYABLE_HTTP_STATUS = frozenset({408, 409, 429})

# Message-level fallback for SDKs that expose no usable status code.
_RETRYABLE_ERROR_MARKERS = (
    "resource_exhausted",
    "resource exhausted",
    "unavailable",
    "overloaded",
    "high demand",
    "rate limit",
    "rate_limit",
    "too many requests",
    "timed out",
    "timeout",
    "deadline",
    "temporarily",
    "try again",
)

# The "404 NOT_FOUND." / "503 UNAVAILABLE." prefix google-genai puts in str(exc).
_STATUS_IN_MESSAGE_RE = re.compile(r"\b([1-5]\d{2})\b\s+[A-Z][A-Z_]{2,}")


def _provider_http_status(exc: Exception) -> Optional[int]:
    """Best-effort HTTP status behind a provider exception, else ``None``.

    ``google-genai`` (and google-api-core) expose it as ``.code``; some SDKs use
    ``.status_code``. When neither is set the ``"404 NOT_FOUND. …"`` prefix of
    the message is parsed as a last resort.
    """
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, int):
            if 100 <= value <= 599:
                return value
            continue
        match = re.search(r"\b([1-5]\d{2})\b", str(value))
        if match:
            return int(match.group(1))
    match = _STATUS_IN_MESSAGE_RE.search(str(exc))
    return int(match.group(1)) if match else None


def _is_retryable_provider_error(exc: Exception) -> bool:
    """True only for errors a second attempt can plausibly fix.

    4xx other than 408/409/429 (a retired model name, a bad key, a malformed
    request) are permanent: they are surfaced immediately instead of being
    retried, so the real cause is not masked as a transient network blip.
    """
    status = _provider_http_status(exc)
    if status is not None:
        return status in _RETRYABLE_HTTP_STATUS or status >= 500
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _RETRYABLE_ERROR_MARKERS)


def _non_retryable_error(
    engine: str,
    model: str,
    exc: Exception,
    *,
    key_env: str = "GEMINI_API_KEY",
    model_env: str = "GEMINI_MODEL",
) -> RuntimeError:
    """Turn a permanent provider error into a clear, actionable ``RuntimeError``.

    ``key_env`` / ``model_env`` name the ``.env`` variables the caller's engine
    reads, so the fix-it hint points at the right key for Gemini and Qwen alike.
    """
    status = _provider_http_status(exc)
    what = f"HTTP {status}" if status is not None else "non-retryable error"
    message = f"{engine} request failed with a {what} that retrying cannot fix: {exc}"
    text = str(exc).lower()
    if status in (401, 403) or "api key" in text or "permission" in text:
        message += (
            f"\nCheck {key_env} in .env — the key looks rejected or lacks "
            "access to this API."
        )
    elif status == 404 or "not found" in text or "no longer available" in text:
        message += (
            f"\nThe model {model!r} ({model_env} in .env) is not available to "
            f"this key — it may be retired or renamed. Update {model_env} and "
            "re-run."
        )
    elif status == 400:
        message += (
            f"\nThe request was rejected as malformed; check {model_env} ({model!r})."
        )
    return RuntimeError(message)


# ---------------------------------------------------------------------------
# Strategy: base engine
# ---------------------------------------------------------------------------


class BaseVisualIndexer(ABC):
    """Strategy base for visual description engines.

    Subclasses receive the video path and the resolved settings, and turn a
    sequence of scene windows into short text descriptions keyed by timecode.
    """

    #: Human-readable engine name, used in logs and the result dict.
    name = "base"

    def __init__(self, video_path: str, settings: Settings) -> None:
        self.video_path = video_path
        self.settings = settings

    def preflight(self) -> None:
        """Validate that this engine can run, before any expensive work.

        Called by :func:`index_video` *before* scene detection so a missing
        dependency or an invalid configuration fails immediately instead of
        after minutes of frame analysis. The default is a no-op; engines
        override it to raise :class:`VisualIndexerSetupError`.
        """

    @abstractmethod
    def index(self, scenes: Sequence[SceneSpan]) -> Dict[str, Any]:
        """Return ``{"engine": str, "scenes": [{"start", "end", "text"}]}``."""
        raise NotImplementedError

    @staticmethod
    def midpoint(scene: SceneSpan) -> float:
        """The middle second of a scene — the frame that gets sampled."""
        start, end = float(scene[0]), float(scene[1])
        return start + (end - start) / 2.0

    def _frame_max_height(self) -> int:
        """Resolve ``VISUAL_INDEXER_MAX_HEIGHT`` (0 = keep the source size)."""
        try:
            return max(
                0, int(getattr(self.settings, "visual_indexer_max_height", 0) or 0)
            )
        except (TypeError, ValueError):
            return 0


# ---------------------------------------------------------------------------
# Local engine: Florence-2
# ---------------------------------------------------------------------------


class FlorenceIndexer(BaseVisualIndexer):
    """Local engine: Florence-2 (``transformers``) on CUDA when available."""

    name = "florence"

    def __init__(self, video_path: str, settings: Settings) -> None:
        super().__init__(video_path, settings)
        self._torch = None
        self._model = None
        self._processor = None
        self._device = "cpu"

    # -- model loading -----------------------------------------------------
    def preflight(self) -> None:
        """Check torch/transformers before scene detection touches the video."""
        try:
            import torch  # type: ignore  # noqa: F401
            from transformers import (  # type: ignore
                AutoModelForCausalLM,  # noqa: F401
                AutoProcessor,  # noqa: F401
            )
        except ImportError as exc:
            raise VisualIndexerSetupError(_florence_dependency_message(exc)) from exc

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoProcessor,
            )
        except ImportError as exc:
            # Reached only when preflight() was skipped (e.g. a direct API call).
            raise VisualIndexerSetupError(_florence_dependency_message(exc)) from exc

        model_name = self.settings.visual_indexer_model or DEFAULT_FLORENCE_MODEL
        device = self._resolve_device(torch)
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(
            f"[visual] loading Florence model={model_name} device={device}",
            flush=True,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=dtype
            )
            model = model.to(device)
        except RuntimeError as exc:
            if device == "cuda" and "out of memory" in str(exc).lower():
                print(
                    "[visual] CUDA out of memory loading Florence; retrying on CPU",
                    flush=True,
                )
                torch.cuda.empty_cache()
                device, dtype = "cpu", torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=True, torch_dtype=dtype
                ).to(device)
            else:
                raise RuntimeError(
                    f"could not load Florence model {model_name!r}: {exc}"
                ) from exc
        model.eval()
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        self._torch = torch
        self._model = model
        self._processor = processor
        self._device = device

    def _resolve_device(self, torch) -> str:
        requested = (self.settings.visual_indexer_device or "auto").strip().lower()
        if requested == "cpu":
            return "cpu"
        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            print("[visual] CUDA requested but unavailable; using CPU", flush=True)
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    # -- inference ---------------------------------------------------------
    def index(self, scenes: Sequence[SceneSpan]) -> Dict[str, Any]:
        scenes = list(scenes)
        if not scenes:
            return {"engine": self.name, "scenes": []}
        self._load()
        from PIL import Image  # type: ignore

        task = FLORENCE_TASK
        results: List[Dict[str, Any]] = []
        with _FrameExtractor(self.video_path, self._frame_max_height()) as extractor:
            for scene in scenes:
                rgb = extractor.frame_rgb(self.midpoint(scene))
                if rgb is None:
                    continue
                caption = self._describe(Image.fromarray(rgb), task)
                results.append(
                    {
                        "start": round(float(scene[0]), 3),
                        "end": round(float(scene[1]), 3),
                        "text": caption,
                    }
                )
        return {"engine": self.name, "scenes": results}

    def _describe(self, image, task: str) -> str:
        try:
            return self._generate(image, task)
        except RuntimeError as exc:
            torch = self._torch
            if self._device == "cuda" and "out of memory" in str(exc).lower():
                print(
                    "[visual] CUDA out of memory during inference; moving "
                    "Florence to CPU for the rest of the run",
                    flush=True,
                )
                torch.cuda.empty_cache()
                self._model = self._model.to("cpu")
                self._device = "cpu"
                return self._generate(image, task)
            raise

    def _generate(self, image, task: str) -> str:
        torch, model, processor, device = (
            self._torch,
            self._model,
            self._processor,
            self._device,
        )
        inputs = processor(text=task, images=image, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs["pixel_values"].to(
            device, dtype=torch.float16 if device == "cuda" else torch.float32
        )
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            text, task=task, image_size=(image.width, image.height)
        )
        return str(parsed.get(task, text)).strip()


# ---------------------------------------------------------------------------
# Cloud engine: Gemini Flash
# ---------------------------------------------------------------------------


class GeminiFlashIndexer(BaseVisualIndexer):
    """Cloud engine: ``google-genai`` (Gemini Flash), frames sent in batches."""

    name = "gemini"

    def __init__(self, video_path: str, settings: Settings) -> None:
        super().__init__(video_path, settings)
        self._genai = None
        self._client = None

    def preflight(self) -> None:
        """Check the SDK and the API key before scene detection runs."""
        try:
            from google import genai  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise VisualIndexerSetupError(
                _gemini_dependency_message("GeminiFlashIndexer", exc)
            ) from exc
        # Raises with a clear message when GEMINI_API_KEY is unset.
        require_gemini_key(self.settings)

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise VisualIndexerSetupError(
                _gemini_dependency_message("GeminiFlashIndexer", exc)
            ) from exc
        self._genai = genai
        self._client = genai.Client(api_key=require_gemini_key(self.settings))
        return self._client

    def index(self, scenes: Sequence[SceneSpan]) -> Dict[str, Any]:
        scenes = list(scenes)
        if not scenes:
            return {"engine": self.name, "scenes": []}
        client = self._get_client()
        from PIL import Image  # type: ignore

        batch_size = max(
            1, int(self.settings.visual_indexer_batch_size or DEFAULT_BATCH_SIZE)
        )
        model = self.settings.gemini_model or "gemini-2.5-flash"
        results: List[Dict[str, Any]] = []
        with _FrameExtractor(self.video_path, self._frame_max_height()) as extractor:
            for batch in _chunks(scenes, batch_size):
                prepared = []
                for scene in batch:
                    timestamp = self.midpoint(scene)
                    rgb = extractor.frame_rgb(timestamp)
                    if rgb is None:
                        continue
                    prepared.append((scene, Image.fromarray(rgb), timestamp))
                if prepared:
                    results.extend(self._describe_batch(client, model, prepared))
        return {"engine": self.name, "scenes": results}

    def _describe_batch(self, client, model: str, prepared) -> List[Dict[str, Any]]:
        timestamps = ", ".join(f"{ts:.1f}s" for _, _, ts in prepared)
        prompt = (
            f"{VISUAL_PROMPT}\n\n"
            "The frames are given in this order, one per timestamp (seconds): "
            f"{timestamps}.\n"
            "Return ONLY JSON of the form "
            '{"descriptions": [{"index": 1, "text": "..."}, ...]} where "index" '
            "is the 1-based frame number and text is the short English "
            "description of that frame."
        )
        contents = [prompt, *[image for _, image, _ in prepared]]

        last_error: Any = None
        for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config={
                        "temperature": 0.2,
                        "response_mime_type": "application/json",
                        "max_output_tokens": 2048,
                        # No tools/callables are passed, so automatic function
                        # calling (AFC) has nothing to run; disabling it makes
                        # the SDK take the plain generate_content path.
                        "automatic_function_calling": {"disable": True},
                    },
                )
                return self._map_response(response.text or "", prepared)
            except Exception as exc:  # noqa: BLE001 - network / timeout / quota
                last_error = exc
                # A permanent error (retired model name, rejected key, malformed
                # request) fails identically on every attempt: stop now with the
                # real cause instead of hiding it behind the backoff.
                if not _is_retryable_provider_error(exc):
                    raise _non_retryable_error("Gemini", model, exc) from exc
                print(
                    f"[visual] gemini request failed "
                    f"({type(exc).__name__}: {exc}) "
                    f"attempt {attempt}/{GEMINI_MAX_ATTEMPTS}",
                    flush=True,
                )
                if attempt < GEMINI_MAX_ATTEMPTS:
                    time.sleep(GEMINI_RETRY_BACKOFF * attempt)

        # Transient errors exhausted every attempt: surface them instead of
        # returning empty descriptions, so an enabled indexer either contributes
        # or stops the run (index_video adds the fix-it hint).
        raise RuntimeError(
            f"Gemini request failed after {GEMINI_MAX_ATTEMPTS} attempt(s) "
            f"({type(last_error).__name__}: {last_error})"
        ) from last_error

    def _map_response(self, raw: str, prepared) -> List[Dict[str, Any]]:
        texts: Dict[int, str] = {}
        try:
            data = json.loads(_strip_fences(raw))
            items = data.get("descriptions", []) if isinstance(data, dict) else data
            for item in items or []:
                index = int(item.get("index") or item.get("i") or 0)
                text = item.get("text") or item.get("description") or ""
                texts[index] = str(text).strip()
        except (ValueError, TypeError, AttributeError):
            texts = {}

        results: List[Dict[str, Any]] = []
        for index, (scene, _, _) in enumerate(prepared, start=1):
            results.append(
                {
                    "start": round(float(scene[0]), 3),
                    "end": round(float(scene[1]), 3),
                    "text": texts.get(index, ""),
                }
            )
        return results


# ---------------------------------------------------------------------------
# Cache: keep the described index in OUTPUT_DIR so re-cutting never pays for
# the same Gemini/Florence pass twice.
# ---------------------------------------------------------------------------


def _visual_cache_path(video_path: str, settings: Settings) -> str:
    """Return the ``.visual.json`` cache path for a video.

    Like the transcript ``.srt`` cache, the visual index lands in ``OUTPUT_DIR``
    under the video's stem (``video/talk.mkv`` → ``<OUTPUT_DIR>/talk.visual.json``)
    so re-cutting the same video with different clip settings reuses it instead
    of calling the indexer once more.
    """
    cache_dir = Path(settings.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / (Path(video_path).stem + ".visual.json"))


def _visual_cache_signature(settings: Settings) -> Dict[str, Any]:
    """Describe the settings that shape the visual index.

    Two runs share a cache only when this signature matches, so switching the
    engine, model or scene policy automatically invalidates the old file.
    """
    kind = (settings.visual_indexer_type or "florence").strip().lower()
    # Only gemini_video ignores the scene policy: it describes the whole file in
    # segments the model picks itself. Every other engine is shaped by the shared
    # visual-indexer knobs below.
    whole_video = kind in ("gemini_video", "gemini-video", "video")
    qwen = kind in ("qwen_video", "qwen-video", "qwen", "qwen2.5-vl", "qwen2_5_vl")
    if kind in (
        "gemini",
        "gemini_flash",
        "gemini-flash",
        "gemini_video",
        "gemini-video",
        "video",
    ):
        model = settings.gemini_model
    else:
        # florence and qwen_video share VISUAL_INDEXER_MODEL.
        model = settings.visual_indexer_model
    signature: Dict[str, Any] = {
        "engine": kind,
        "model": model,
        # The frame/upload height cap changes the sampled images, so a different
        # VISUAL_INDEXER_MAX_HEIGHT must invalidate the cached index. It applies
        # to every engine, gemini_video included (the upload is re-encoded).
        "max_height": int(getattr(settings, "visual_indexer_max_height", 0) or 0),
    }
    # Every engine but gemini_video is shaped by the scene-window length: florence
    # and gemini sample frames per scene, qwen_video uses it as its request window.
    if not whole_video:
        signature["max_scene_seconds"] = round(
            float(settings.visual_indexer_max_scene_seconds), 3
        )
    # The scene threshold only drives scene detection for the frame-based engines.
    if not whole_video and not qwen:
        signature["scene_threshold"] = round(
            float(settings.visual_indexer_scene_threshold), 3
        )
    # Frames per request: gemini batches frames, qwen_video samples frames per window.
    if qwen or kind in ("gemini", "gemini_flash", "gemini-flash"):
        signature["batch_size"] = int(settings.visual_indexer_batch_size)
    return signature


def _load_visual_cache(video_path: str, settings: Settings) -> Optional[Dict[str, Any]]:
    """Return a cached visual index when it is present, fresh and matching."""
    path = _visual_cache_path(video_path, settings)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"[visual] ignoring unreadable cache {path}: {exc}", flush=True)
        return None

    if not isinstance(payload, dict) or payload.get("version") != VISUAL_CACHE_VERSION:
        return None
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return None
    if payload.get("signature") != _visual_cache_signature(settings):
        print(
            f"[visual] indexer settings changed; re-indexing "
            f"{os.path.basename(video_path)}",
            flush=True,
        )
        return None
    try:
        source_mtime = os.path.getmtime(video_path)
    except OSError:
        source_mtime = 0.0
    if float(payload.get("source_mtime", 0.0)) < source_mtime:
        print(
            f"[visual] source video changed; re-indexing "
            f"{os.path.basename(video_path)}",
            flush=True,
        )
        return None

    print(
        f"[visual] reusing cached index ({len(scenes)} scene(s)): {path}",
        flush=True,
    )
    return {
        "engine": payload.get("engine") or settings.visual_indexer_type,
        "scenes": scenes,
        "cached": True,
    }


def _write_visual_cache(
    video_path: str, settings: Settings, result: Dict[str, Any]
) -> None:
    """Persist a visual index next to the video for later re-cuts."""
    scenes = result.get("scenes") or []
    if not scenes:
        return
    try:
        source_mtime = os.path.getmtime(video_path)
    except OSError:
        source_mtime = 0.0
    payload = {
        "version": VISUAL_CACHE_VERSION,
        "engine": result.get("engine"),
        "signature": _visual_cache_signature(settings),
        "source_mtime": source_mtime,
        "scenes": scenes,
    }
    path = _visual_cache_path(video_path, settings)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"[visual] could not write index cache {path}: {exc}", flush=True)
        return
    print(f"[visual] saved index: {path}", flush=True)


# ---------------------------------------------------------------------------
# Factory + high-level entry point
# ---------------------------------------------------------------------------


def build_visual_indexer(
    video_path: str, settings: Settings
) -> Optional[BaseVisualIndexer]:
    """Instantiate the engine chosen by ``VISUAL_INDEXER_TYPE`` (Strategy)."""
    if not settings.visual_indexer_enabled:
        return None
    kind = (settings.visual_indexer_type or "florence").strip().lower()
    if kind in ("", "none", "off", "disabled", "false"):
        return None
    if kind == "florence":
        return FlorenceIndexer(video_path, settings)
    if kind in ("gemini", "gemini_flash", "gemini-flash"):
        return GeminiFlashIndexer(video_path, settings)
    if kind in ("gemini_video", "gemini-video", "video"):
        return GeminiVideoIndexer(video_path, settings)
    if kind in ("qwen_video", "qwen-video", "qwen", "qwen2.5-vl", "qwen2_5_vl"):
        return QwenVideoIndexer(video_path, settings)
    raise ConfigError(
        f"Unknown VISUAL_INDEXER_TYPE={kind!r}; use 'florence', 'gemini', "
        "'gemini_video', 'qwen_video' or 'none'."
    )


def index_video(
    video_path: str,
    settings: Settings,
    *,
    scenes: Optional[Sequence[SceneSpan]] = None,
) -> Dict[str, Any]:
    """Describe the visuals of ``video_path`` using the configured engine.

    Returns ``{"engine": str, "scenes": [{"start", "end", "text"}]}``.

    Setup problems (missing dependencies or API key) raise
    :class:`VisualIndexerSetupError` *before* scene detection starts, so an
    enabled-but-unusable indexer fails loudly with a fix-it hint instead of
    burning minutes on frame analysis. Any other failure (an API timeout, out of
    VRAM, a broken model, ...) also raises: when visual indexing is enabled it
    either contributes or the run stops with the reason.

    With ``VISUAL_INDEXER_CACHE=true`` the described index is saved in
    ``OUTPUT_DIR`` (``<OUTPUT_DIR>/<video>.visual.json``, like the ``.srt``
    transcript cache) and reused on later runs while the source file and the
    indexer settings stay the same, so re-cutting the same video with different
    clip settings costs no extra Gemini/Florence requests.
    """
    indexer = build_visual_indexer(video_path, settings)
    if indexer is None:
        return {"engine": "none", "scenes": []}

    # Reuse a previously saved index when the source and the indexer settings
    # are unchanged (e.g. re-cutting the same video with a different NUM_CLIPS).
    # Only the default windowing is cacheable; explicit ``scenes`` re-index.
    use_cache = bool(getattr(settings, "visual_indexer_cache", True)) and scenes is None
    if use_cache:
        cached = _load_visual_cache(video_path, settings)
        if cached is not None:
            return cached

    # A missing dependency or an invalid configuration is deterministic: fail
    # now, before scene detection spends minutes scanning the whole video, and
    # point at the fix, instead of continuing without the visuals that
    # VISUAL_INDEXER_ENABLED=true asked for.
    indexer.preflight()

    if scenes is None:
        print(
            f"[visual] detecting scenes in {os.path.basename(video_path)} — "
            "frame-by-frame scan of the whole video; on a long clip this takes "
            "minutes, and the model is loaded only after it finishes",
            flush=True,
        )
    try:
        windows = (
            list(scenes)
            if scenes is not None
            else detect_scenes(
                video_path,
                threshold=settings.visual_indexer_scene_threshold,
                max_seconds=settings.visual_indexer_max_scene_seconds,
            )
        )
        if not windows:
            windows = [(0.0, _probe_duration(video_path))]
        print(
            f"[visual] {len(windows)} scene(s) to describe via {indexer.name}",
            flush=True,
        )
        result = indexer.index(windows)
    except VisualIndexerSetupError:
        # Deterministic setup problem (missing dependency / key): re-raise with
        # the fix-it hint rather than swallowing it.
        raise
    except Exception as exc:
        raise RuntimeError(
            "visual indexing is enabled (VISUAL_INDEXER_ENABLED=true, "
            f"VISUAL_INDEXER_TYPE={indexer.name}) but failed "
            f"({type(exc).__name__}: {exc}).\n"
            "Fix the cause above and re-run, or set "
            "VISUAL_INDEXER_ENABLED=false to continue without visual context."
        ) from exc

    described = [
        scene
        for scene in result.get("scenes", [])
        if str(scene.get("text", "")).strip()
    ]
    print(
        f"[visual] {len(described)} scene(s) described via {result.get('engine')}",
        flush=True,
    )
    if use_cache:
        _write_visual_cache(video_path, settings, result)
    return result


# ---------------------------------------------------------------------------
# Merging transcripts and visuals for the ranking log
# ---------------------------------------------------------------------------


def _as_segments(transcripts: Any) -> List[Dict]:
    """Accept either a transcript dict or a list of segment dicts."""
    if isinstance(transcripts, dict):
        return list(transcripts.get("segments", []) or [])
    return [s for s in (transcripts or []) if isinstance(s, dict)]


def _as_scenes(visuals: Any) -> List[Dict]:
    """Accept either a ``{"scenes": [...]}`` dict or a list of scene dicts."""
    if not visuals:
        return []
    if isinstance(visuals, dict):
        return [s for s in (visuals.get("scenes", []) or []) if isinstance(s, dict)]
    return [s for s in visuals if isinstance(s, dict)]


def _overlaps(item: Dict, start: float, end: float) -> bool:
    try:
        item_start = float(item["start"])
        item_end = float(item["end"])
    except (KeyError, TypeError, ValueError):
        return False
    return item_end > start and item_start < end


def _unique_scene_text(scenes: Sequence[Dict], start: float, end: float) -> str:
    """Join the distinct descriptions of the scenes overlapping a window."""
    seen: List[str] = []
    for scene in scenes:
        if not _overlaps(scene, start, end):
            continue
        text = str(scene.get("text", "")).strip()
        if text and text not in seen:
            seen.append(text)
    return " ".join(seen)


def _span_bounds(item: Dict) -> Optional[Tuple[float, float]]:
    """``(start, end)`` айтема транскрипта/сцены или ``None``, если он пустой."""
    try:
        start = float(item["start"])
        end = float(item["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end


def _merge_scene_windows(
    scenes: Sequence[Dict], *, window: float
) -> List[Dict[str, Any]]:
    """Разложить только визуальные сцены по окнам ``window`` секунд.

    Запасной путь для видео без транскрипта (немое видео): привязываться к
    репликам не к чему, поэтому описания кадров всё равно должны дойти до
    ранкера.
    """
    duration = _timeline_end([], scenes)
    if duration <= 0:
        return []
    step = float(window) if window and float(window) > 0 else MERGE_WINDOW_SECONDS
    entries: List[Dict[str, Any]] = []
    start = 0.0
    while start < duration - 1e-6:
        end = min(start + step, duration)
        visual_text = _unique_scene_text(scenes, start, end)
        if visual_text:
            entries.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "transcript": "",
                    "visuals": visual_text,
                }
            )
        start = end
    return entries


def merge_transcripts_and_visuals(
    transcripts: Any,
    visuals: Any,
    *,
    window: float = MERGE_WINDOW_SECONDS,
) -> List[Dict[str, Any]]:
    """Merge Whisper segments and visual descriptions into a single log.

    Both inputs share the same timeline. The merge keeps the transcript's own
    granularity — **one entry per Whisper segment** — so the ranker sees the
    exact start and end of every phrase and can copy them straight into
    ``start_time`` / ``end_time``::

        {"start": 12.3, "end": 15.1, "transcript": "...", "visuals": "..."}

    Flattening both inputs into fixed ``window``-second buckets (the earlier
    behaviour) threw that precision away: the model could only point at a whole
    bucket, so its windows landed on bucket edges and dragged the surrounding
    scenes into every clip. A visual description is attached to the first
    segment it overlaps and kept until it changes, so a long scene caption is
    not repeated on every line.

    When there is no transcript at all, the visuals are bucketed into
    ``window``-second entries so they still reach the ranker.

    Args:
        transcripts: a transcript dict (``{"segments": [...]}``) or a list of
            segment dicts (``{"start", "end", "text"}``).
        visuals: the :func:`index_video` result (``{"scenes": [...]}``) or a
            list of scene dicts (``{"start", "end", "text"}``).
        window: bucket length, in seconds, used only for the no-transcript
            fallback.

    Returns:
        A time-ordered list of merged entries (empty when there is nothing to
        merge).
    """
    segments = _as_segments(transcripts)
    scenes = _as_scenes(visuals)
    if not segments:
        return _merge_scene_windows(scenes, window=window) if scenes else []

    ordered = [segment for segment in segments if _span_bounds(segment) is not None]
    ordered.sort(key=lambda segment: float(segment["start"]))

    entries: List[Dict[str, Any]] = []
    shown_visual: Optional[str] = None
    for segment in ordered:
        bounds = _span_bounds(segment)
        if bounds is None:
            continue
        start, end = bounds
        transcript_text = str(segment.get("text", "")).strip()
        visual_text = _unique_scene_text(scenes, start, end)
        new_visual = ""
        if visual_text and visual_text != shown_visual:
            new_visual = visual_text
            shown_visual = visual_text
        if not transcript_text and not new_visual:
            continue
        entries.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "transcript": transcript_text,
                "visuals": new_visual,
            }
        )
    return entries


def format_merged_log(entries: Sequence[Dict[str, Any]]) -> str:
    """Render merged entries as the ranker's log lines.

    Each line carries the phrase's own second-precision window —
    ``[12.30 - 15.10] реплика`` — plus, when a scene description changes, a
    ``[Visuals: "..."]`` note. Seconds keep two decimals on purpose: the model
    can copy the exact start of the hook phrase and the exact end of the
    punchline phrase into ``start_time`` / ``end_time`` instead of rounding to a
    coarse window and swallowing the neighbouring scenes.
    """
    lines: List[str] = []
    for entry in entries:
        start, end = _span_bounds(entry) or (0.0, 0.0)
        head = f"[{start:.2f} - {end:.2f}]"
        parts: List[str] = []
        transcript = str(entry.get("transcript", "") or "").strip()
        if transcript:
            parts.append(transcript)
        visual = str(entry.get("visuals", "") or "").strip()
        if visual:
            # Длинное описание кадра — шум для ранкера: обрезаем окно, чтобы
            # речь не тонула в простыне визуального текста.
            if len(visual) > VISUAL_TEXT_MAX_CHARS:
                visual = visual[:VISUAL_TEXT_MAX_CHARS].rstrip() + "…"
            parts.append(f'[Visuals: "{visual}"]')
        lines.append(f"{head} {' '.join(parts)}".rstrip())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cloud engine: Gemini Flash on the whole video (Files API)
# ---------------------------------------------------------------------------


def _parse_timeline(raw: str, duration: float) -> List[Dict[str, Any]]:
    """Parse a model's ``{"scenes": [...]}`` reply into ``start/end/text`` scenes.

    Shared by the whole-video engines (Gemini and Qwen). Tolerates a bare list,
    missing keys and reversed bounds; when nothing parses but the model did return
    prose, that text is kept as a single segment covering ``duration`` (so it is
    not silently dropped).
    """
    data: Any = None
    try:
        data = json.loads(_strip_fences(raw))
    except (ValueError, TypeError):
        data = None

    items: Any = None
    if isinstance(data, dict):
        items = data.get("scenes") or data.get("moments") or data.get("segments")
    elif isinstance(data, list):
        items = data

    results: List[Dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("description") or "").strip()
        if not text:
            continue
        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
        except (TypeError, ValueError):
            start, end = 0.0, 0.0
        if end < start:
            start, end = end, start
        if end <= start:
            end = start + 1.0
        results.append({"start": round(start, 3), "end": round(end, 3), "text": text})

    if not results and raw.strip():
        whole = duration if duration and duration > 0 else 1e9
        results = [{"start": 0.0, "end": round(whole, 3), "text": raw.strip()[:1000]}]
    return results


class GeminiVideoIndexer(BaseVisualIndexer):
    """Cloud engine: upload the whole video (Files API) and describe it at once.

    Unlike :class:`GeminiFlashIndexer` (which sends sampled frames), this engine
    uploads the full file to the Gemini Files API (up to 2 GB) and asks the model
    to describe the requested time windows in a single request. It is closer to
    how Gemini is meant to be used with video — the model "sees" motion, cuts and
    audio-visual context — at the cost of uploading the file first.
    """

    name = "gemini_video"

    def __init__(self, video_path: str, settings: Settings) -> None:
        super().__init__(video_path, settings)
        self._genai = None
        self._client = None

    def preflight(self) -> None:
        """Check the SDK and the API key before scene detection runs."""
        try:
            from google import genai  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise VisualIndexerSetupError(
                _gemini_dependency_message("GeminiVideoIndexer", exc)
            ) from exc
        require_gemini_key(self.settings)

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise VisualIndexerSetupError(
                _gemini_dependency_message("GeminiVideoIndexer", exc)
            ) from exc
        self._genai = genai
        self._client = genai.Client(api_key=require_gemini_key(self.settings))
        return self._client

    def index(self, scenes: Sequence[SceneSpan]) -> Dict[str, Any]:
        client = self._get_client()
        model = self.settings.gemini_model or "gemini-2.5-flash"
        duration = _probe_duration(self.video_path) or max(
            (float(end) for _, end in scenes), default=0.0
        )
        uploaded = self._upload(client)
        try:
            described = self._describe(client, model, uploaded, duration)
        finally:
            self._delete(client, uploaded)
        return {"engine": self.name, "scenes": described}

    # -- Files API ---------------------------------------------------------
    def _upload(self, client):
        # Shrink the file first when VISUAL_INDEXER_MAX_HEIGHT is set: a smaller
        # upload reaches Gemini sooner and finishes processing faster. The
        # source itself is never touched — a downscaled copy lands in a temp dir.
        scaled_path, scale_dir = _downscale_for_upload(
            self.video_path,
            self._frame_max_height(),
            getattr(self.settings, "ffmpeg_path", ""),
        )
        upload_path, cleanup_dir = _ascii_upload_path(scaled_path)
        try:
            size = os.path.getsize(upload_path)
            print(
                f"[visual] uploading {os.path.basename(self.video_path)} to Gemini "
                f"Files ({size / (1024 * 1024):.1f} MB)",
                flush=True,
            )
            video_file = client.files.upload(file=upload_path)
            deadline = time.time() + GEMINI_UPLOAD_TIMEOUT_SECONDS
            state = _file_state(video_file)
            while state == "PROCESSING":
                if time.time() > deadline:
                    raise RuntimeError(
                        "Gemini timed out while processing the uploaded video"
                    )
                time.sleep(GEMINI_VIDEO_POLL_SECONDS)
                video_file = client.files.get(name=video_file.name)
                state = _file_state(video_file)
        finally:
            if cleanup_dir:
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            if scale_dir:
                shutil.rmtree(scale_dir, ignore_errors=True)
        if state == "FAILED":
            raise RuntimeError("Gemini failed to process the uploaded video")
        print(f"[visual] video uploaded and processed (state={state})", flush=True)
        return video_file

    def _delete(self, client, video_file) -> None:
        name = getattr(video_file, "name", None)
        if not name:
            return
        try:
            client.files.delete(name=name)
            print("[visual] removed the uploaded video from Gemini Files", flush=True)
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
            print(f"[visual] could not delete uploaded video: {exc}", flush=True)

    def _describe(
        self, client, model: str, video_file, duration: float
    ) -> List[Dict[str, Any]]:
        prompt = (
            f"{VISUAL_PROMPT}\n\n"
            "The full video is attached. Walk through the ENTIRE video from start "
            "to finish and split it into consecutive segments that cover the whole "
            "timeline. For each segment give start and end time in seconds and a "
            "short English description of what happens on screen.\n"
            "Return ONLY JSON of the form "
            '{"scenes": [{"start": <seconds>, "end": <seconds>, "text": "..."}, ...]} '
            "with about 10-40 segments, ordered by time, each at most 2-3 sentences."
        )

        attempts = max(
            1,
            int(getattr(self.settings, "llm_max_attempts", 0) or GEMINI_MAX_ATTEMPTS),
        )
        backoff = abs(
            float(
                getattr(self.settings, "llm_retry_backoff", GEMINI_RETRY_BACKOFF) or 0.0
            )
        )
        last_error: Any = None
        for attempt in range(1, attempts + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[video_file, prompt],
                    config={
                        "temperature": 0.2,
                        "response_mime_type": "application/json",
                        "max_output_tokens": 8192,
                        # No tools/callables are passed, so automatic function
                        # calling (AFC) has nothing to run. Disabling it makes
                        # the SDK take the plain generate_content path — no AFC
                        # loop and no "direct use of AFC is not recommended" log.
                        "automatic_function_calling": {"disable": True},
                    },
                )
                return self._parse_timeline(response.text or "", duration)
            except Exception as exc:  # noqa: BLE001 - network / timeout / quota
                last_error = exc
                # A permanent error (retired model name, rejected key, malformed
                # request) fails identically on every attempt: stop now with the
                # real cause instead of hiding it behind the backoff.
                if not _is_retryable_provider_error(exc):
                    raise _non_retryable_error("Gemini video", model, exc) from exc
                print(
                    f"[visual] gemini video request failed "
                    f"({type(exc).__name__}: {exc}) "
                    f"attempt {attempt}/{attempts}",
                    flush=True,
                )
                if attempt < attempts:
                    time.sleep(backoff * (2 ** (attempt - 1)))

        # Transient errors exhausted every attempt: surface them instead of
        # returning an empty index, so an enabled indexer either contributes or
        # stops the run (index_video adds the fix-it hint).
        raise RuntimeError(
            f"Gemini video request failed after {attempts} attempt(s) "
            f"({type(last_error).__name__}: {last_error})"
        ) from last_error

    def _parse_timeline(self, raw: str, duration: float) -> List[Dict[str, Any]]:
        return _parse_timeline(raw, duration)


def _file_state(video_file: Any) -> str:
    """Best-effort read of a Gemini File's processing state name."""
    state = getattr(video_file, "state", None)
    name = getattr(state, "name", None)
    if name is None and isinstance(state, str):
        return state
    return str(name or "")


# ---------------------------------------------------------------------------
# Local engine: Qwen2.5-VL, described chunk by chunk
# ---------------------------------------------------------------------------


def _qwen_windows(duration: float, chunk_seconds: float) -> List[SceneSpan]:
    """Cut ``[0, duration]`` into consecutive windows of ``chunk_seconds``."""
    size = max(1.0, float(chunk_seconds))
    windows: List[SceneSpan] = []
    start = 0.0
    while start < duration - 1e-6:
        end = min(duration, start + size)
        windows.append((start, end))
        start = end
    return windows or [(0.0, max(duration, 0.0))]


def _map_window_timeline(raw: str, start: float, end: float) -> List[Dict[str, Any]]:
    """Turn one window's model reply into absolute ``start/end/text`` scenes.

    The prompt asks for times relative to the window, but models occasionally
    answer with absolute seconds anyway; both are accepted (a reply whose times
    already sit inside ``[start, end]`` is read as absolute). Whatever comes back
    is clamped to the window so concatenated windows stay monotonic.
    """
    window_len = max(0.001, float(end) - float(start))
    segments = _parse_timeline(raw, window_len)
    if not segments:
        return []

    starts = [float(seg.get("start", 0.0)) for seg in segments]
    absolute = start > 0 and bool(starts) and min(starts) >= start - 1.0

    mapped: List[Dict[str, Any]] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        if not absolute:
            seg_start += start
            seg_end += start
        seg_start = min(max(seg_start, start), end)
        seg_end = min(max(seg_end, start), end)
        if seg_end <= seg_start:
            seg_end = min(end, seg_start + 1.0)
        if seg_end <= seg_start:
            continue
        mapped.append(
            {"start": round(seg_start, 3), "end": round(seg_end, 3), "text": text}
        )
    return mapped


class QwenVideoIndexer(BaseVisualIndexer):
    """Local engine: ``Qwen2.5-VL`` (``transformers``) on CUDA when available.

    Qwen2.5-VL understands video, but a whole long video in one request overflows
    the context window (and the GPU memory). So the timeline is cut into windows of
    ``VISUAL_INDEXER_MAX_SCENE_SECONDS`` seconds; each window's frames (up to
    ``VISUAL_INDEXER_BATCH_SIZE`` of them) are described in their own request,
    which returns a short ``{"scenes": ...}`` timeline for that window. The
    per-window timelines are then stitched into one index — the same shape
    :class:`GeminiVideoIndexer` produces, just split so the video never has to fit
    in a single call.

    The engine carries no settings of its own: the checkpoint is
    ``VISUAL_INDEXER_MODEL`` (a Hugging Face repo id), loaded on
    ``VISUAL_INDEXER_DEVICE`` (auto | cpu | cuda) like :class:`FlorenceIndexer`,
    and the window length / frames per window are
    ``VISUAL_INDEXER_MAX_SCENE_SECONDS`` / ``VISUAL_INDEXER_BATCH_SIZE``.
    """

    name = "qwen_video"

    def __init__(self, video_path: str, settings: Settings) -> None:
        super().__init__(video_path, settings)
        self._torch = None
        self._model = None
        self._processor = None
        self._device = "cpu"

    # -- model loading -----------------------------------------------------
    def preflight(self) -> None:
        """Check torch/transformers before scene detection touches the video."""
        try:
            import qwen_vl_utils  # type: ignore  # noqa: F401
            import torch  # type: ignore  # noqa: F401
            from transformers import (  # type: ignore
                AutoProcessor,  # noqa: F401
                Qwen2_5_VLForConditionalGeneration,  # noqa: F401
            )
        except ImportError as exc:
            raise VisualIndexerSetupError(_qwen_dependency_message(exc)) from exc

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:
            # Reached only when preflight() was skipped (e.g. a direct API call).
            raise VisualIndexerSetupError(_qwen_dependency_message(exc)) from exc

        model_name = self.settings.visual_indexer_model or DEFAULT_QWEN_MODEL
        device = self._resolve_device(torch)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(
            f"[visual] loading Qwen model={model_name} device={device}",
            flush=True,
        )
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=dtype
            )
            model = model.to(device)
        except RuntimeError as exc:
            if device == "cuda" and "out of memory" in str(exc).lower():
                print(
                    "[visual] CUDA out of memory loading Qwen; retrying on CPU",
                    flush=True,
                )
                torch.cuda.empty_cache()
                device, dtype = "cpu", torch.float32
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=dtype
                ).to(device)
            else:
                raise RuntimeError(
                    f"could not load Qwen model {model_name!r}: {exc}"
                ) from exc
        model.eval()
        processor = AutoProcessor.from_pretrained(model_name)

        self._torch = torch
        self._model = model
        self._processor = processor
        self._device = device

    def _resolve_device(self, torch) -> str:
        requested = (self.settings.visual_indexer_device or "auto").strip().lower()
        if requested == "cpu":
            return "cpu"
        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            print("[visual] CUDA requested but unavailable; using CPU", flush=True)
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    # -- inference ---------------------------------------------------------
    def index(self, scenes: Sequence[SceneSpan]) -> Dict[str, Any]:
        scenes = list(scenes)
        if not scenes:
            return {"engine": self.name, "scenes": []}
        self._load()

        duration = _probe_duration(self.video_path) or max(
            (float(end) for _, end in scenes), default=0.0
        )
        if duration > 0:
            windows = _qwen_windows(duration, self._window_seconds())
        else:
            # Unreadable duration: fall back to the windows handed over.
            windows = [(float(start), float(end)) for start, end in scenes]
        if not windows:
            return {"engine": self.name, "scenes": []}

        per_window = self._frames_per_window()
        described: List[Dict[str, Any]] = []
        with _FrameExtractor(self.video_path, self._frame_max_height()) as extractor:
            for start, end in windows:
                sampled = self._sample_frames(extractor, start, end, per_window)
                if not sampled:
                    continue
                described.extend(self._describe_window(start, end, sampled))
        return {"engine": self.name, "scenes": described}

    # -- tunables (shared with the other engines) --------------------------
    def _window_seconds(self) -> float:
        """Window length; reuses VISUAL_INDEXER_MAX_SCENE_SECONDS."""
        try:
            value = float(self.settings.visual_indexer_max_scene_seconds or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        return value if value > 0 else DEFAULT_QWEN_WINDOW_SECONDS

    def _frames_per_window(self) -> int:
        """Frames sampled per window; reuses VISUAL_INDEXER_BATCH_SIZE."""
        try:
            value = int(self.settings.visual_indexer_batch_size or 0)
        except (TypeError, ValueError):
            value = 0
        return max(1, value)

    def _sample_frames(self, extractor, start: float, end: float, count: int):
        """Return ``[(timestamp, rgb), ...]`` spread evenly across the window."""
        span = max(0.0, float(end) - float(start))
        if span <= 0:
            timestamps = [float(start)]
        else:
            step = span / count
            timestamps = [float(start) + step * (i + 0.5) for i in range(count)]
        frames: List[Tuple[float, Any]] = []
        for ts in timestamps:
            rgb = extractor.frame_rgb(ts)
            if rgb is not None:
                frames.append((ts, rgb))
        return frames

    def _describe_window(
        self, start: float, end: float, sampled
    ) -> List[Dict[str, Any]]:
        messages = self._build_messages(start, end, sampled)
        raw = self._generate(messages)
        return _map_window_timeline(raw, start, end)

    def _build_messages(self, start: float, end: float, sampled):
        from PIL import Image  # type: ignore

        content: List[Dict[str, Any]] = [
            {"type": "text", "text": self._window_prompt(start, end, len(sampled))}
        ]
        for ts, rgb in sampled:
            content.append({"type": "text", "text": f"Frame at {ts - start:.1f}s:"})
            content.append({"type": "image", "image": Image.fromarray(rgb)})
        return [{"role": "user", "content": content}]

    def _generate(self, messages) -> str:
        """Run the model, moving it to CPU and retrying once on CUDA OOM."""
        try:
            return self._run(messages)
        except RuntimeError as exc:
            torch = self._torch
            if self._device == "cuda" and "out of memory" in str(exc).lower():
                print(
                    "[visual] CUDA out of memory during Qwen inference; moving "
                    "Qwen to CPU for the rest of the run",
                    flush=True,
                )
                torch.cuda.empty_cache()
                self._model.to("cpu")
                self._device = "cpu"
                return self._run(messages)
            raise

    def _run(self, messages) -> str:
        from qwen_vl_utils import process_vision_info  # type: ignore

        processor = self._processor
        model = self._model
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)
        with self._torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=DEFAULT_QWEN_MAX_NEW_TOKENS
            )
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        decoded = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0] if decoded else ""

    @staticmethod
    def _window_prompt(start: float, end: float, frames: int) -> str:
        return (
            f"{VISUAL_PROMPT}\n\n"
            f"This is a clip from the full video, covering {start:.1f}s to "
            f"{end:.1f}s, shown as {frames} evenly spaced frames. The line before "
            "each frame gives its time from the START OF THIS CLIP.\n"
            "Split this clip into consecutive segments and describe what happens "
            "on screen in each. Give every start and end time in SECONDS RELATIVE "
            "TO THE START OF THIS CLIP (0 = the clip's first moment), NOT the "
            "absolute time in the full video.\n"
            'Return ONLY JSON of the form {"scenes": [{"start": <seconds>, '
            '"end": <seconds>, "text": "..."}, ...]} ordered by time, each text '
            "at most 2-3 sentences in English."
        )
