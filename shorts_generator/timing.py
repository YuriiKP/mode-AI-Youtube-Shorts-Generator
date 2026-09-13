"""Wall-clock timing for the content-creation pipeline.

Creating a short runs through several distinct stages — downloading the source
video, transcribing it with Whisper, ranking highlights with an LLM, cropping
the vertical clips and enhancing them (music + subtitles). This module records
how long each stage takes so the CLI can show a per-stage summary at the end of
a run.

A single process-wide timer is shared by every stage through the convenience
helpers (:func:`start_timer`, :func:`time_stage`), so nested code can add its
own measurements without threading a timer object through every call. The
pipeline restarts the timer at the beginning of each top-level run.

Usage::

    from .timing import start_timer, time_stage, get_timer

    start_timer()
    with time_stage("transcribe"):
        transcript = transcribe(source, settings)
    print(get_timer().report("Time spent"))
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple


def format_duration(seconds: float) -> str:
    """Return a compact human-readable duration, e.g. ``1m 23.4s``.

    Sub-minute durations keep a tenth of a second (``12.3s``); longer ones drop
    to seconds (or whole minutes/hours) to stay readable in a summary table.
    """
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {secs:.0f}s"


class Timer:
    """Records how long named pipeline stages take.

    The same stage name may be recorded more than once (for example once per
    input video); :meth:`totals` sums those entries. Measurements use
    ``time.perf_counter`` so they are monotonic and unaffected by clock changes.
    """

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._records: List[Tuple[str, float]] = []

    def record(self, name: str, seconds: float) -> None:
        """Add a pre-measured ``seconds`` under the stage ``name``."""
        self._records.append((name, max(0.0, float(seconds))))

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the enclosed block and record it under ``name``.

        The measurement is written even when the block raises, so a failed
        stage still shows up in the summary.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - start)

    @property
    def elapsed(self) -> float:
        """Wall-clock seconds since the timer was created."""
        return time.perf_counter() - self._start

    @property
    def empty(self) -> bool:
        """``True`` when no stage has been recorded yet."""
        return not self._records

    def totals(self) -> Dict[str, float]:
        """Return ``{stage: total_seconds}``, summing repeated stage names."""
        totals: Dict[str, float] = {}
        for name, seconds in self._records:
            totals[name] = totals.get(name, 0.0) + seconds
        return totals

    def as_dict(self) -> Dict:
        """Return a JSON-serialisable snapshot of the measurements."""
        return {
            "total_seconds": round(self.elapsed, 3),
            "stages": [
                {"stage": name, "seconds": round(seconds, 3)}
                for name, seconds in self._records
            ],
            "stages_total_seconds": {
                name: round(seconds, 3) for name, seconds in self.totals().items()
            },
        }

    def report(self, title: str = "Timing") -> str:
        """Render a human-readable summary table of the recorded stages."""
        totals = self.totals()
        width = max((len(name) for name in totals), default=0)
        lines = [f"{title}:", ""]
        for name, seconds in totals.items():
            lines.append(f"  {name.ljust(width)}  {format_duration(seconds):>9}")
        lines.append("  " + "-" * (width + 11))
        lines.append(f"  {'total'.ljust(width)}  {format_duration(self.elapsed):>9}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Process-wide default timer
# ---------------------------------------------------------------------------

_default: Optional[Timer] = None


def start_timer() -> Timer:
    """Create a fresh timer and make it the process-wide default.

    Called at the start of each top-level run so measurements from a previous
    call (library use, tests) do not leak into the new summary.
    """
    global _default
    _default = Timer()
    return _default


def get_timer() -> Timer:
    """Return the process-wide default timer, creating one if necessary."""
    global _default
    if _default is None:
        _default = Timer()
    return _default


def time_stage(name: str):
    """Context manager timing a block on the process-wide default timer."""
    return get_timer().stage(name)
