"""Spread clips over profiles (and, optionally, over publish times).

The publisher no longer sends *every* clip to *every* profile: doing that means
posting identical videos to several accounts of the same platform, which is
exactly what gets those accounts flagged. Instead clips are handed out
**round-robin** and the result is a flat list of :class:`Job` objects — one per
``(profile, platform, clip)`` — that the upload loop simply executes.

Two modes (``PUBLISH_MODE``):

* ``distribute`` — clip ``i`` goes to profile ``i % M`` (one profile per round)
  and is published immediately;
* ``schedule`` — the same round-robin handout, but profile *p*'s *r*-th clip is
  scheduled for ``slots[r]``. The number of slots is therefore how many clips
  each profile publishes, and every profile uses the same slot list (a different
  clip each).

A clip handed to one profile is still paired with *every* selected platform, so
it is published there on YouTube *and* TikTok — but never to a second profile.

This module depends only on the model and profile types (no browser stack), so it
can be imported and unit-tested anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from .model import Short
from .profile import Profile

# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """One pending upload: a clip to one platform from one profile."""

    #: The profile whose account publishes the clip.
    profile: Profile
    #: The platform module (``publisher.platforms.youtube`` / ``tiktok``).
    platform: object
    #: The clip to publish.
    short: Short
    #: When to publish; ``None`` means "publish immediately".
    schedule_at: Optional[datetime] = None
    #: Index into the schedule slots (``-1`` when not scheduled).
    slot: int = -1

    @property
    def platform_name(self) -> str:
        """The platform's stable name (``youtube`` / ``tiktok``)."""
        return getattr(self.platform, "NAME", "")

    @property
    def label(self) -> str:
        """Human-friendly platform label for the report (falls back to the name)."""
        return getattr(self.platform, "LABEL", "") or self.platform_name


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def build_jobs(
    profiles: Sequence[Profile],
    platforms: Sequence[object],
    shorts: Sequence[Short],
    *,
    mode: str = "distribute",
    slots: Sequence[datetime] = (),
) -> List[Job]:
    """Assign clips to ``(profile, platform)`` targets and return the job list.

    The assignment is round-robin: clip ``i`` belongs to
    ``profiles[i % len(profiles)]``.

    * ``distribute`` — every clip becomes a job, published immediately;
    * ``schedule`` — clip ``i`` is profile ``i % M``'s ``i // M``-th clip and is
      scheduled for ``slots[i // M]``. Once there is no slot left for the next
      round, the remaining clips are dropped, so a profile never gets more than
      ``len(slots)`` clips.

    Args:
        profiles: profiles to spread the clips over (order is preserved and
            defines the round-robin position).
        platforms: platform modules to publish each clip to (e.g. YouTube and
            TikTok); each assignment is expanded to one job per platform.
        shorts: clips to publish, in pipeline order.
        mode: ``"distribute"`` or ``"schedule"`` (anything else behaves like
            ``"distribute"``).
        slots: publish times for ``schedule`` mode (ignored otherwise).

    Returns:
        The flat list of :class:`Job` objects, in execution order.
    """
    profiles = list(profiles)
    platforms = list(platforms)
    shorts = list(shorts)
    profile_count = len(profiles)

    if not profile_count or not platforms or not shorts:
        return []

    jobs: List[Job] = []

    if mode == "schedule":
        for index, short in enumerate(shorts):
            round_index, position = divmod(index, profile_count)
            if round_index >= len(slots):
                # No slot left for this round: stop handing clips out.
                break
            profile = profiles[position]
            slot_time = slots[round_index]
            for module in platforms:
                jobs.append(Job(profile, module, short, slot_time, round_index))
    else:  # distribute
        for index, short in enumerate(shorts):
            profile = profiles[index % profile_count]
            for module in platforms:
                jobs.append(Job(profile, module, short, None, -1))

    return jobs


def jobs_by_profile(
    jobs: Sequence[Job],
    profiles: Sequence[Profile],
) -> Dict[str, List[Job]]:
    """Group ``jobs`` by profile name, preserving the profile order.

    Profiles that were assigned no job still appear with an empty list, so the
    caller can report them (and keep the profile order stable).
    """
    grouped: Dict[str, List[Job]] = {profile.name: [] for profile in profiles}
    for job in jobs:
        grouped.setdefault(job.profile.name, []).append(job)
    return grouped


def jobs_for_platform(jobs: Sequence[Job], platform_name: str) -> List[Job]:
    """Return the subset of ``jobs`` that target ``platform_name``."""
    return [job for job in jobs if job.platform_name == platform_name]


def planned_counts(
    profiles: Sequence[Profile],
    shorts: Sequence[Short],
    *,
    mode: str = "distribute",
    slots: Sequence[datetime] = (),
) -> Dict[str, int]:
    """Return, per profile name, how many clips that profile will publish.

    Handy for a pre-flight summary or a shortfall warning without building the
    full job list.
    """
    profile_count = len(profiles)
    counts: Dict[str, int] = {profile.name: 0 for profile in profiles}
    if not profile_count or not shorts:
        return counts
    if mode == "schedule":
        limit = profile_count * len(slots)
        for index in range(min(len(shorts), limit)):
            counts[profiles[index % profile_count].name] += 1
    else:
        for index in range(len(shorts)):
            counts[profiles[index % profile_count].name] += 1
    return counts


__all__ = [
    "Job",
    "build_jobs",
    "jobs_by_profile",
    "jobs_for_platform",
    "planned_counts",
]
