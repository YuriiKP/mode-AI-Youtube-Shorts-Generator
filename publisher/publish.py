"""High-level orchestration for the publisher.

This module ties everything together: it resolves the requested profiles and
platforms, loads the clips to publish, opens each profile once and drives the
selected platform uploaders through it, records the outcome in the SQLite state
store and prints a final report.

Three entry points are exposed, matching the three ``publish`` actions:

* :func:`run_manual` — open a profile in a visible browser so the user can log in
  and set accounts up by hand; cookies are saved automatically by the persistent
  profile.
* :func:`run_check` — report, per profile and platform, whether the stored login
  is still valid.
* :func:`run_upload` — the actual matrix upload: ``profiles × platforms ×
  clips``, with de-duplication, an optional dry run and a delay between uploads.

Nothing here talks to the browser engine directly; browser handling lives in
:mod:`publisher.session`, and per-site logic lives in
:mod:`publisher.platforms`.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

from . import human
from .config import PublishConfig
from .distribution import Job, build_jobs, jobs_by_profile, jobs_for_platform
from .log import log
from .model import Short, file_sha1
from .platforms import PlatformError, resolve_platforms
from .profile import (
    Profile,
    ProfileError,
    get_profile,
    list_profiles,
    resolve_profiles,
    unregistered_launcher_profiles,
)
from .session import (
    BrowserUnavailableError,
    export_storage_state,
    open_pages,
    open_profile_context,
    platform_urls,
)
from .source import load_shorts
from .state import StateDB, UploadStatus

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _effective_cfg(
    cfg: PublishConfig,
    *,
    visibility: Optional[str] = None,
    playlist: Optional[str] = None,
    delay: Optional[float] = None,
) -> PublishConfig:
    """Return ``cfg`` with per-run CLI overrides applied (via ``dataclasses``)."""
    changes: Dict[str, object] = {}
    if visibility:
        changes["visibility"] = visibility
    if playlist is not None:
        changes["playlist"] = playlist
    if delay is not None:
        changes["delay"] = delay
    return replace(cfg, **changes) if changes else cfg


@dataclass
class Outcome:
    """A single (profile, platform, clip) result, used to build the report."""

    profile: str
    platform: str
    clip: str
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == UploadStatus.OK


@dataclass
class Report:
    """Collects :class:`Outcome` rows and prints a summary."""

    rows: List[Outcome] = field(default_factory=list)

    def add(
        self, profile: str, platform: str, clip: str, status: str, detail: str = ""
    ) -> None:
        self.rows.append(Outcome(profile, platform, clip, status, detail))

    # -- statistics --------------------------------------------------------

    @property
    def ok_count(self) -> int:
        return sum(1 for row in self.rows if row.status == UploadStatus.OK)

    @property
    def failed_count(self) -> int:
        return sum(1 for row in self.rows if row.status == UploadStatus.FAILED)

    @property
    def skipped_count(self) -> int:
        return sum(1 for row in self.rows if row.status == UploadStatus.SKIPPED)

    def exit_code(self) -> int:
        """Non-zero when at least one upload failed."""
        return 1 if self.failed_count else 0

    # -- rendering ---------------------------------------------------------

    def render(self) -> str:
        if not self.rows:
            return "nothing to do"

        columns = ("Profile", "Platform", "Clip", "Result", "Detail")
        table = [
            (
                row.profile,
                row.platform,
                row.clip,
                row.status,
                row.detail,
            )
            for row in self.rows
        ]
        widths = [len(title) for title in columns]
        for line in table:
            for index, cell in enumerate(line):
                widths[index] = max(widths[index], len(str(cell)))

        def fmt(cells: Sequence[object]) -> str:
            return "  ".join(
                str(cell).ljust(widths[index]) for index, cell in enumerate(cells)
            )

        lines = [fmt(columns), fmt(["-" * width for width in widths])]
        lines.extend(fmt(line) for line in table)
        lines.append("")
        lines.append(
            f"total: {len(self.rows)}  ok: {self.ok_count}  "
            f"failed: {self.failed_count}  skipped: {self.skipped_count}"
        )
        return "\n".join(lines)


def _print_report(report: Report, title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(report.render())
    print()


# ---------------------------------------------------------------------------
# Manual mode
# ---------------------------------------------------------------------------


async def _wait_for_user(profile: Profile) -> None:
    """Block until the user presses Enter (so the browser stays open)."""
    prompt = (
        f"\n[{profile.name}] Браузер открыт. Войдите в аккаунты, настройте всё, "
        "что нужно.\nКуки сохранятся автоматически. Нажмите Enter здесь, чтобы "
        "закрыть браузер и продолжить...\n"
    )
    try:
        await asyncio.to_thread(input, prompt)
    except EOFError:  # pragma: no cover - no interactive stdin
        log.info("no interactive stdin; leaving the browser open for 5 minutes")
        await asyncio.sleep(300)


async def run_manual(
    cfg: PublishConfig,
    *,
    profiles: str = "all",
    platforms: str = "all",
    urls: Sequence[str] = (),
    export_cookies: bool = False,
) -> int:
    """Open the selected profiles in a visible browser for manual setup."""
    modules = resolve_platforms(platforms)
    try:
        selected = resolve_profiles(cfg, profiles, create=True)
    except ProfileError as exc:
        log.error("%s", exc)
        return 1

    start_urls = [url for url in urls if url]
    default_urls = platform_urls([module.NAME for module in modules])

    for profile in selected:
        log.info("opening profile '%s' for manual setup", profile.name)
        try:
            # Manual mode is always visible: the point is for the user to see and
            # use the browser.
            async with open_profile_context(cfg, profile, headless=False) as context:
                await open_pages(context, start_urls or default_urls)
                for module in modules:
                    profile.add_platform(module.NAME)
                await _wait_for_user(profile)
                if export_cookies:
                    await export_storage_state(context, profile)
        except (BrowserUnavailableError, ProfileError) as exc:
            log.error("%s", exc)
            return 1

    print(
        f"\nmanual session finished for {len(selected)} profile(s): "
        + ", ".join(profile.name for profile in selected)
    )
    return 0


# ---------------------------------------------------------------------------
# Login check
# ---------------------------------------------------------------------------


async def run_check(
    cfg: PublishConfig,
    *,
    profiles: str = "all",
    platforms: str = "all",
) -> int:
    """Report whether each profile is still logged into each platform."""
    modules = resolve_platforms(platforms)
    try:
        selected = resolve_profiles(cfg, profiles)
    except ProfileError as exc:
        log.error("%s", exc)
        return 1

    report = Report()
    all_ok = True

    for profile in selected:
        log.info("checking profile '%s'", profile.name)
        try:
            async with open_profile_context(cfg, profile) as context:
                for module in modules:
                    try:
                        ok = await module.check(context, cfg, log)
                    except Exception as exc:  # noqa: BLE001 - reported per target
                        ok = False
                        detail = f"error: {exc}"
                    else:
                        detail = "logged in" if ok else "not logged in"
                    if ok:
                        profile.add_platform(module.NAME)
                    else:
                        all_ok = False
                    report.add(
                        profile.name,
                        module.NAME,
                        "-",
                        UploadStatus.OK if ok else UploadStatus.FAILED,
                        detail,
                    )
        except (BrowserUnavailableError, ProfileError) as exc:
            log.error("%s", exc)
            for module in modules:
                report.add(
                    profile.name, module.NAME, "-", UploadStatus.FAILED, str(exc)
                )
            all_ok = False

    _print_report(report, "Login check")
    print("all logins valid" if all_ok else "some logins are missing or expired")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def _fingerprint(short: Short, cache: Dict[str, str]) -> str:
    """SHA-1 of a clip, memoised per file path."""
    if short.file not in cache:
        cache[short.file] = file_sha1(short.file)
    return cache[short.file]


def _warn_plan(cfg: PublishConfig, profiles, shorts, jobs) -> None:
    """Log the plan summary, and warn when there are fewer clips than needed."""
    profile_count = len(profiles)
    if profile_count == 0:
        return
    if cfg.mode == "schedule":
        capacity = profile_count * len(cfg.schedule)
        if len(shorts) < capacity:
            log.warning(
                "schedule: %d profile(s) x %d slot(s) = %d clips needed, but only "
                "%d available — some profiles/slots will have no clip",
                profile_count,
                len(cfg.schedule),
                capacity,
                len(shorts),
            )
    elif len(shorts) < profile_count:
        log.warning(
            "distribute: only %d clip(s) for %d profile(s) — some profiles will "
            "publish nothing this run",
            len(shorts),
            profile_count,
        )
    log.info(
        "plan: %d job(s) across %d profile(s) (mode=%s)",
        len(jobs),
        profile_count,
        cfg.mode,
    )


def _build_plan(
    jobs: List[Job],
    *,
    rerun: bool,
    rerun_failed: bool,
    state: StateDB,
) -> List[Outcome]:
    """Compute, without uploading, what would happen for every planned job."""
    plan: List[Outcome] = []
    cache: Dict[str, str] = {}
    for job in jobs:
        short = job.short
        platform_name = job.platform_name
        try:
            sha1 = _fingerprint(short, cache)
        except OSError as exc:
            plan.append(
                Outcome(
                    job.profile.name,
                    platform_name,
                    short.name,
                    UploadStatus.FAILED,
                    str(exc),
                )
            )
            continue
        reason = state.should_skip(
            job.profile.name,
            platform_name,
            sha1,
            rerun=rerun,
            rerun_failed=rerun_failed,
        )
        if reason:
            plan.append(
                Outcome(
                    job.profile.name,
                    platform_name,
                    short.name,
                    UploadStatus.SKIPPED,
                    reason,
                )
            )
        else:
            when = job.schedule_at.isoformat(sep=" ") if job.schedule_at else "now"
            plan.append(
                Outcome(
                    job.profile.name,
                    platform_name,
                    short.name,
                    UploadStatus.OK,
                    f"would upload ({when})",
                )
            )
    return plan


async def run_upload(
    cfg: PublishConfig,
    *,
    profiles: str = "all",
    platforms: str = "all",
    output_dir: Optional[str] = None,
    json_path: Optional[str] = None,
    video: Optional[str] = None,
    only=None,
    limit: Optional[int] = None,
    visibility: Optional[str] = None,
    playlist: Optional[str] = None,
    delay: Optional[float] = None,
    dry_run: bool = False,
    rerun: bool = False,
    rerun_failed: bool = False,
    headless: Optional[bool] = None,
    export_cookies: bool = False,
    base_dir: Optional[str] = None,
) -> int:
    """Upload the selected clips to the selected profiles/platforms."""
    cfg = _effective_cfg(cfg, visibility=visibility, playlist=playlist, delay=delay)
    modules = resolve_platforms(platforms)

    # 1) clips ------------------------------------------------------------
    try:
        shorts = load_shorts(
            output_dir or cfg.output_dir,
            json_path=json_path,
            video=video,
            only=only,
            limit=limit,
            base_dir=base_dir or cfg.base_dir,
            logger=log,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        log.error("could not load clips: %s", exc)
        return 1

    if not shorts:
        log.error("no clips to upload")
        return 1

    log.info(
        "loaded %d clip(s) for %d platform(s): %s",
        len(shorts),
        len(modules),
        ", ".join(module.NAME for module in modules),
    )

    # 2) profiles ---------------------------------------------------------
    try:
        selected = resolve_profiles(cfg, profiles)
    except ProfileError as exc:
        log.error("%s", exc)
        return 1

    # 3) plan (round-robin spread + optional schedule) --------------------
    jobs = build_jobs(selected, modules, shorts, mode=cfg.mode, slots=cfg.schedule)
    if not jobs:
        log.error("nothing to publish: no clips assigned to any profile")
        return 1
    _warn_plan(cfg, selected, shorts, jobs)

    # 4) dry run ----------------------------------------------------------
    if dry_run:
        with StateDB(cfg.db_full_path) as state:
            plan = _build_plan(
                jobs,
                rerun=rerun,
                rerun_failed=rerun_failed,
                state=state,
            )
        report = Report(rows=plan)
        _print_report(report, "Dry run — nothing will be uploaded")
        return 0

    # 5) real upload ------------------------------------------------------
    report = Report()
    digests: Dict[str, str] = {}
    target_delay = max(0.0, cfg.delay)
    grouped = jobs_by_profile(jobs, selected)

    with StateDB(cfg.db_full_path) as state:
        for profile in selected:
            profile_jobs = grouped.get(profile.name, [])
            if not profile_jobs:
                log.info("profile '%s': no clips assigned, skipping", profile.name)
                continue
            log.info("using profile '%s' (%d job(s))", profile.name, len(profile_jobs))
            try:
                async with open_profile_context(
                    cfg, profile, headless=headless
                ) as context:
                    for module in modules:
                        platform_jobs = jobs_for_platform(profile_jobs, module.NAME)
                        if not platform_jobs:
                            continue

                        profile.add_platform(module.NAME)

                        # The login is verified by the upload itself, on the very
                        # tab it drives, so no separate "check" tab is opened and
                        # closed first. When that first job reports an expired
                        # session, the platform's remaining clips are skipped
                        # with the same reason instead of being retried.
                        login_error = ""

                        for job in platform_jobs:
                            if login_error:
                                report.add(
                                    profile.name,
                                    module.NAME,
                                    job.short.name,
                                    UploadStatus.SKIPPED,
                                    login_error,
                                )
                                continue
                            short = job.short
                            if not short.exists:
                                report.add(
                                    profile.name,
                                    module.NAME,
                                    short.name,
                                    UploadStatus.FAILED,
                                    "file missing",
                                )
                                continue

                            try:
                                sha1 = _fingerprint(short, digests)
                            except OSError as exc:
                                report.add(
                                    profile.name,
                                    module.NAME,
                                    short.name,
                                    UploadStatus.FAILED,
                                    str(exc),
                                )
                                continue

                            reason = state.should_skip(
                                profile.name,
                                module.NAME,
                                sha1,
                                rerun=rerun,
                                rerun_failed=rerun_failed,
                            )
                            if reason:
                                log.info(
                                    "skip %s -> %s/%s (%s)",
                                    short.name,
                                    profile.name,
                                    module.NAME,
                                    reason,
                                )
                                report.add(
                                    profile.name,
                                    module.NAME,
                                    short.name,
                                    UploadStatus.SKIPPED,
                                    reason,
                                )
                                continue

                            when = (
                                job.schedule_at.isoformat(sep=" ")
                                if job.schedule_at
                                else "now"
                            )
                            log.info(
                                "uploading %s -> %s/%s (%s)",
                                short.name,
                                profile.name,
                                module.NAME,
                                when,
                            )
                            try:
                                result = await module.upload(
                                    context,
                                    short,
                                    cfg,
                                    log,
                                    schedule_at=job.schedule_at,
                                )
                            except Exception as exc:  # noqa: BLE001
                                result = None
                                error = str(exc)
                            else:
                                error = result.error

                            if result is not None and result.ok:
                                state.record(
                                    profile=profile.name,
                                    platform=module.NAME,
                                    short=short,
                                    video_sha1=sha1,
                                    status=UploadStatus.OK,
                                    remote_id=result.video_id,
                                    remote_url=result.url,
                                    visibility=cfg.visibility,
                                )
                                report.add(
                                    profile.name,
                                    module.NAME,
                                    short.name,
                                    UploadStatus.OK,
                                    result.url or result.video_id or "published",
                                )
                            elif result is not None and result.auth_required:
                                login_error = result.error or "not logged in"
                                log.warning(
                                    "%s: %s — skipping the remaining clips. "
                                    "Run: python main.py publish manual "
                                    "--profile %s",
                                    profile.name,
                                    login_error,
                                    profile.name,
                                )
                                report.add(
                                    profile.name,
                                    module.NAME,
                                    short.name,
                                    UploadStatus.SKIPPED,
                                    login_error,
                                )
                            else:
                                state.record(
                                    profile=profile.name,
                                    platform=module.NAME,
                                    short=short,
                                    video_sha1=sha1,
                                    status=UploadStatus.FAILED,
                                    error=error,
                                    visibility=cfg.visibility,
                                )
                                report.add(
                                    profile.name,
                                    module.NAME,
                                    short.name,
                                    UploadStatus.FAILED,
                                    error,
                                )

                            if target_delay:
                                # Space uploads apart with a little random
                                # jitter so the gaps are not perfectly even.
                                await human.delay(target_delay)

                    if export_cookies:
                        await export_storage_state(context, profile)
            except (BrowserUnavailableError, ProfileError) as exc:
                log.error("%s", exc)
                for job in profile_jobs:
                    report.add(
                        profile.name,
                        job.platform_name,
                        job.short.name,
                        UploadStatus.FAILED,
                        str(exc),
                    )

    _print_report(report, "Upload summary")
    return report.exit_code()


# ---------------------------------------------------------------------------
# Reporting commands
# ---------------------------------------------------------------------------


def render_profiles(cfg: PublishConfig) -> str:
    """A human-readable listing of the known browser profiles."""
    profiles = list_profiles(cfg)
    if not profiles:
        lines = [
            "no browser profiles yet.",
            "Create one with: python main.py publish manual --profile profile_1",
        ]
    else:
        lines = [f"profiles under {cfg.profiles_path}:", ""]
        for profile in profiles:
            lines.append(f"  - {profile.describe()}")
            if profile.created_at:
                lines.append(f"      created: {profile.created_at}")

    available = unregistered_launcher_profiles(cfg)
    if available:
        lines.append("")
        lines.append("also in the ShardX Launcher (name one to register it):")
        for name in available:
            lines.append(f"  - {name}")

    return "\n".join(lines)


def render_state(cfg: PublishConfig, limit: int = 50) -> str:
    """A human-readable listing of the recent upload history."""
    if not os.path.isfile(cfg.db_full_path):
        return "no upload history yet (the database does not exist)"
    with StateDB(cfg.db_full_path) as state:
        records = state.history(limit=limit)
        total = state.total()
    if not records:
        return "upload history is empty"
    lines = [f"last {len(records)} of {total} upload(s):", ""]
    for record in records:
        lines.append(
            f"  [{record.status}] {record.profile}/{record.platform}  "
            f"{record.video_file}  {record.remote_url or record.error}"
        )
    return "\n".join(lines)


async def run_profiles(cfg: PublishConfig) -> int:
    """Print the known profiles and the recent upload history."""
    print(render_profiles(cfg))
    print()
    print(render_state(cfg))
    return 0


__all__ = [
    "Outcome",
    "Report",
    "run_manual",
    "run_check",
    "run_upload",
    "run_profiles",
    "render_profiles",
    "render_state",
]
