"""Post-processing engine: vertical re-frame, banner, subtitles and music.

This package is the internal engine behind the ``subtitles``, ``music`` and
``all`` commands. It is not meant to be driven directly — use the unified CLI
(``python main.py ...``) instead.

Every stage is optional and independent:

* **vertical fit** — re-frame the clip to ``FIT_ASPECT_RATIO`` (``9:16`` by
  default), filling the empty area with a blurred copy of the video;
* **banner** — overlay an image (``BANNER_IMAGE``) or a text band
  (``BANNER_TEXT``) on top of the frame;
* **subtitles** — burn an ``.srt`` onto the video;
* **music** — mix a background track *under* the existing audio.

The public entry point is :func:`shorts_generator.postprocess.pipeline.run`:

    from shorts_generator.postprocess.pipeline import run
    run("clip.mp4", "clip.done.mp4", settings, burn_subtitles=True, add_music=True)

It only needs MoviePy/Pillow/NumPy.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "2.0.0"
