"""Publish rendered shorts to social platforms via browser automation.

This package takes the clips produced by the clipping pipeline (see
:mod:`shorts_generator`) and uploads them to YouTube (via YouTube Studio) and
TikTok (via TikTok Studio), driving the browser through the ShardX anti-detect
engine (:mod:`publisher.browser`).

Design highlights
-----------------
* A *profile* is a named, persistent Chromium browser profile directory that
  keeps the cookies/logins for every platform (YouTube and TikTok) together.
* Uploads, login checks and the interactive "manual" mode all reuse the very
  same profile, so nothing has to be re-authenticated between steps.
* Everything is driven from the command line, wired into the project's single
  entry point (``python main.py publish ...``).

The heavy/optional browser dependencies (``httpx`` + ``patchright``, used to
drive the ShardX Launcher's automation API) are imported lazily inside the module
that actually launches a browser, so importing :mod:`publisher` never fails on a
machine where the browser stack is not installed yet.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
