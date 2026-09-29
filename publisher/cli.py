"""Command-line wiring for the publisher.

This module plugs the publisher into the project's single entry point::

    python main.py publish <action> [options]

Actions:

* ``manual``   — open one or more profiles in a visible browser so the user can
  log in / set accounts up by hand. Cookies are stored by the persistent profile
  automatically; this is the mode used to authorise accounts.
* ``check``    — report whether the stored logins are still valid, per profile and
  platform.
* ``upload``   — publish the rendered clips to the selected profiles/platforms
  (the ``profiles × platforms × clips`` matrix), with de-duplication and a dry
  run.
* ``profiles`` — list the known browser profiles and the recent upload history.

The module is deliberately self-contained: :func:`add_publish_actions` also adds
the shared ``--env`` / ``--quiet`` / ``--no-timing`` flags, so the caller only
has to create the ``publish`` sub-parser and hand it over.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import List, Optional

from shorts_generator.config import ConfigError

from .config import (
    VALID_PUBLISH_MODES,
    VALID_VISIBILITIES,
    load_publish_config,
)
from .log import setup_logging
from .platforms import ALL_PLATFORMS
from .publish import run_check, run_manual, run_profiles, run_upload

# ---------------------------------------------------------------------------
# Human-facing text
# ---------------------------------------------------------------------------

_PUBLISH_EPILOG = """\
examples:
  # log into YouTube and TikTok inside profile_1 (cookies saved automatically)
  python main.py publish manual --profile profile_1

  # open only the TikTok login page for a profile
  python main.py publish manual --profile anime_ru --platform tiktok

  # are the stored logins still valid?
  python main.py publish check --profile profile_1

  # list profiles and the recent upload history
  python main.py publish profiles

  # upload every clip to every profile on both platforms
  python main.py publish upload

  # only YouTube, only two profiles
  python main.py publish upload --profiles profile_1,anime_ru --platforms youtube

  # see what would happen, without opening a browser
  python main.py publish upload --dry-run

  # publish one file on its own, unlisted, with a 60s pause between uploads
  python main.py publish upload --video output/short_01.mkv --visibility unlisted --delay 60

Defaults come from the .env file (PUBLISH_* keys); any flag overrides them for
one run.
"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def add_publish_actions(publish: argparse.ArgumentParser) -> None:
    """Add the ``publish`` sub-actions (and shared flags) to ``publish``.

    ``publish`` is the ``argparse`` parser created for the ``publish`` command by
    the project's :func:`shorts_generator.cli.build_parser`. Besides the
    sub-actions this also adds the ``--env`` / ``--quiet`` / ``--no-timing``
    flags that :func:`shorts_generator.cli.main` expects to find on every
    command, so the caller does not need to add them separately.
    """
    publish.description = (
        "Publish rendered shorts to YouTube and/or TikTok using browser "
        "automation (the ShardX anti-detect engine) and persistent "
        "browser profiles."
    )
    publish.formatter_class = argparse.RawDescriptionHelpFormatter
    publish.epilog = _PUBLISH_EPILOG

    # Shared flags (mirror ``_add_common`` from the main CLI).
    publish.add_argument(
        "--env",
        dest="env_file",
        default=None,
        metavar="PATH",
        help="extra .env file to load (highest-priority file)",
    )
    publish.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="only print warnings and errors",
    )
    publish.add_argument(
        "--no-timing",
        dest="no_timing",
        action="store_true",
        help="accepted for parity with the other commands (publish has no stage timing)",
    )

    actions = publish.add_subparsers(
        dest="publish_action",
        metavar="ACTION",
        required=True,
    )

    # -- manual ---------------------------------------------------------------
    manual = actions.add_parser(
        "manual",
        help="open profiles in a browser to log in / set up by hand",
        description=(
            "Open each selected profile in a visible browser (on the login page "
            "of each selected platform), let you log in manually, and save the "
            "cookies automatically when the browser closes. This is how accounts "
            "are authorised."
        ),
    )
    manual.add_argument(
        "--profile",
        dest="profiles",
        default="all",
        metavar="NAME[,NAME...]",
        help="profile name, or a comma-separated list (default: all)",
    )
    manual.add_argument(
        "--platform",
        dest="platforms",
        default="all",
        metavar="NAME[,NAME...]",
        help="platform(s) to open: youtube, tiktok or all (default: all)",
    )
    manual.add_argument(
        "--url",
        dest="urls",
        action="append",
        default=[],
        metavar="URL",
        help="extra tab to open (repeatable); replaces the default login pages",
    )
    manual.add_argument(
        "--export-cookies",
        dest="export_cookies",
        action="store_true",
        help="also write a portable storage_state.json backup for each profile",
    )

    # -- check ----------------------------------------------------------------
    check = actions.add_parser(
        "check",
        help="verify the stored logins (no changes)",
        description="Report whether each profile is still logged into each platform.",
    )
    check.add_argument(
        "--profile",
        dest="profiles",
        default="all",
        metavar="NAME[,NAME...]",
        help="profile name, or a comma-separated list (default: all)",
    )
    check.add_argument(
        "--platform",
        dest="platforms",
        default="all",
        metavar="NAME[,NAME...]",
        help="platform(s) to check: youtube, tiktok or all (default: all)",
    )

    # -- profiles -------------------------------------------------------------
    actions.add_parser(
        "profiles",
        help="list browser profiles and recent upload history",
        description="List the known browser profiles and the most recent uploads.",
    )

    # -- upload ---------------------------------------------------------------
    upload = actions.add_parser(
        "upload",
        help="publish clips to the selected profiles/platforms",
        description=(
            "Upload the clips produced by the clipping pipeline to every selected "
            "profile × platform target, skipping clips that were already "
            "published there."
        ),
    )
    upload.add_argument(
        "--mode",
        dest="mode",
        choices=MODE_CHOICES,
        default=None,
        help="clip distribution: distribute or schedule (default: PUBLISH_MODE)",
    )
    upload.add_argument(
        "--schedule",
        dest="schedule",
        default=None,
        metavar="DATES",
        help="comma-separated ISO datetimes for schedule mode (default: PUBLISH_SCHEDULE)",
    )
    upload.add_argument(
        "--profiles",
        dest="profiles",
        default="all",
        metavar="NAME[,NAME...]",
        help="profile name(s) to publish with (default: all existing profiles)",
    )
    upload.add_argument(
        "--platforms",
        dest="platforms",
        default="all",
        metavar="NAME[,NAME...]",
        help="platform(s) to publish to: youtube, tiktok or all (default: all)",
    )
    upload.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        metavar="DIR",
        help="where the clips and shorts_info.json live (default: PUBLISH_OUTPUT_DIR)",
    )
    upload.add_argument(
        "--json",
        dest="json_path",
        default=None,
        metavar="PATH",
        help="explicit path to a shorts_info.json (overrides --output-dir)",
    )
    upload.add_argument(
        "--video",
        dest="video",
        default=None,
        metavar="PATH",
        help="publish a single video file instead of scanning the output directory",
    )
    upload.add_argument(
        "--only",
        dest="only",
        default=None,
        metavar="TOKENS",
        help="only these clips, e.g. '1,short_03' or a file-name substring",
    )
    upload.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=None,
        metavar="N",
        help="upload at most the first N clips",
    )
    upload.add_argument(
        "--visibility",
        dest="visibility",
        choices=VISIBILITY_CHOICES,
        default=None,
        help="YouTube visibility (default: PUBLISH_VISIBILITY)",
    )
    upload.add_argument(
        "--playlist",
        dest="playlist",
        default=None,
        metavar="NAME",
        help="YouTube playlist to add the videos to (default: PUBLISH_YT_PLAYLIST)",
    )
    upload.add_argument(
        "--delay",
        dest="delay",
        type=float,
        default=None,
        metavar="SECONDS",
        help="pause between two uploads (default: PUBLISH_DELAY)",
    )
    upload.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="show what would be uploaded without opening a browser",
    )
    upload.add_argument(
        "--rerun",
        dest="rerun",
        action="store_true",
        help="re-upload even clips that were already published",
    )
    upload.add_argument(
        "--rerun-failed",
        dest="rerun_failed",
        action="store_true",
        help="retry clips whose previous upload failed",
    )
    upload.add_argument(
        "--headless",
        dest="headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "run the browser headless (default: PUBLISH_HEADLESS; uploads are "
            "safer headed so the transfer is not cut short)"
        ),
    )
    upload.add_argument(
        "--export-cookies",
        dest="export_cookies",
        action="store_true",
        help="also write a portable storage_state.json backup after uploading",
    )


# ``--visibility`` choices mirror :data:`publisher.config.VALID_VISIBILITIES`.
VISIBILITY_CHOICES = tuple(VALID_VISIBILITIES)

# ``--mode`` choices mirror :data:`publisher.config.VALID_PUBLISH_MODES`.
MODE_CHOICES = tuple(VALID_PUBLISH_MODES)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _run(coro) -> int:
    """Run an async publisher coroutine, mapping errors to exit codes."""
    try:
        return int(asyncio.run(coro))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1


def run_publish(args: argparse.Namespace, base_dir: Optional[str] = None) -> int:
    """Entry point called by the main CLI for the ``publish`` command.

    Configures logging, loads the ``PUBLISH_*`` configuration and dispatches to
    the requested action. Returns a process exit code.
    """
    setup_logging(
        level=os.environ.get("LOG_LEVEL", "INFO"), quiet=getattr(args, "quiet", False)
    )

    try:
        cfg = load_publish_config(
            env_file=getattr(args, "env_file", None),
            # ``--mode`` / ``--schedule`` (when given) override the PUBLISH_*
            # values for this run.
            extra={
                "mode": getattr(args, "mode", None),
                "schedule": getattr(args, "schedule", None),
            },
            base_dir=base_dir,
        )
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    action = getattr(args, "publish_action", None)

    if action == "manual":
        return _run(
            run_manual(
                cfg,
                profiles=getattr(args, "profiles", "all"),
                platforms=getattr(args, "platforms", "all"),
                urls=getattr(args, "urls", []) or [],
                export_cookies=getattr(args, "export_cookies", False),
            )
        )

    if action == "check":
        return _run(
            run_check(
                cfg,
                profiles=getattr(args, "profiles", "all"),
                platforms=getattr(args, "platforms", "all"),
            )
        )

    if action == "profiles":
        return _run(run_profiles(cfg))

    if action == "upload":
        return _run(
            run_upload(
                cfg,
                profiles=getattr(args, "profiles", "all"),
                platforms=getattr(args, "platforms", "all"),
                output_dir=getattr(args, "output_dir", None),
                json_path=getattr(args, "json_path", None),
                video=getattr(args, "video", None),
                only=getattr(args, "only", None),
                limit=getattr(args, "limit", None),
                visibility=getattr(args, "visibility", None),
                playlist=getattr(args, "playlist", None),
                delay=getattr(args, "delay", None),
                dry_run=getattr(args, "dry_run", False),
                rerun=getattr(args, "rerun", False),
                rerun_failed=getattr(args, "rerun_failed", False),
                headless=getattr(args, "headless", None),
                export_cookies=getattr(args, "export_cookies", False),
                base_dir=base_dir,
            )
        )

    print(
        "publish: no action given. Use one of: manual, check, profiles, upload.",
        file=sys.stderr,
    )
    return 2


__all__: List[str] = ["add_publish_actions", "run_publish"]
