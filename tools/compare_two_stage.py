"""Compare single-stage vs two-stage highlight generation on the same video.

The hypothesis behind ``TWO_STAGE_ANALYSIS`` is that asking the model for every
field at once (bounds, score, title, description, tags, three metrics, hook and
punchline) spreads its attention and hurts the *timing* it reports. This script
tests that directly: it transcribes one video once, then ranks the exact same
transcript twice —

* **single** — the original prompt, which returns the whole payload in one call;
* **two_stage** — the timing-only prompt, followed by one batched metadata call
  for the clips that survive selection.

For each mode it reports the model's RAW output (measured *before* the
boundaries are snapped to phrase boundaries) and the FINAL selected clips, so
the effect of the prompt split on the model's own timing accuracy is visible
separately from what the deterministic snapping already fixes.

Usage::

    python tools/compare_two_stage.py
    python tools/compare_two_stage.py --video input/clip.mp4 --clips 4
    python tools/compare_two_stage.py --no-transcribe   # reuse the cached .srt
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from shorts_generator.config import Settings, load_settings
from shorts_generator.cues import phrase_boundaries
from shorts_generator.highlights import (
    build_transcript_log,
    call_highlight_api,
    dedupe_highlights,
    enrich_highlights_metadata,
    select_highlights,
    snap_highlights_to_transcript,
)
from shorts_generator.llm import call_llm
from shorts_generator.subtitles import find_video_files
from shorts_generator.transcriber import transcribe

# All tolerances are in seconds.
_BOUNDARY_EPS = 0.05  # an edge this close to a phrase break counts as "on it"
_SENTENCE_TOL = 0.5  # an end this close to a sentence end counts as firm
_FAR_TOL = 1.5  # an end farther than this from any break is a raw mid-thought cut


class CallStats:
    """Wrap ``call_llm`` and record how many chars each phase sends/receives."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.phase = "ranking"
        self.calls = 0
        self.prompt_chars = 0
        self.response_chars = 0
        self.by_phase: Dict[str, int] = {}

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        self.prompt_chars += len(prompt)
        response = call_llm(prompt, self.settings) or ""
        self.response_chars += len(response)
        self.by_phase[self.phase] = self.by_phase.get(self.phase, 0) + len(response)
        return response


def nearest_boundary(
    boundaries: Sequence[Tuple[float, bool]], t: float
) -> Tuple[Optional[float], bool]:
    """Distance from ``t`` to the closest phrase boundary and whether it ends a sentence."""
    best_d: Optional[float] = None
    best_sentence = False
    for boundary, closes_sentence in boundaries:
        distance = abs(boundary - t)
        if best_d is None or distance < best_d:
            best_d = distance
            best_sentence = closes_sentence
    return best_d, best_sentence


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def raw_metrics(clips: List[Dict], boundaries: Sequence[Tuple[float, bool]]) -> Dict:
    """Metrics on the model's own output, before any snapping to phrases."""
    start_dists: List[float] = []
    end_dists: List[float] = []
    durations: List[float] = []
    ends_on_sentence = 0
    ends_far = 0
    for clip in clips:
        start = float(clip["start_time"])
        end = float(clip["end_time"])
        durations.append(end - start)
        start_dist, _ = nearest_boundary(boundaries, start)
        end_dist, end_closes = nearest_boundary(boundaries, end)
        if start_dist is not None:
            start_dists.append(start_dist)
        if end_dist is not None:
            end_dists.append(end_dist)
        if end_closes and end_dist is not None and end_dist <= _SENTENCE_TOL:
            ends_on_sentence += 1
        if end_dist is None or end_dist > _FAR_TOL:
            ends_far += 1
    return {
        "clips": len(clips),
        "start_dist": statistics.mean(start_dists) if start_dists else None,
        "end_dist": statistics.mean(end_dists) if end_dists else None,
        "ends_on_sentence": ends_on_sentence,
        "ends_far": ends_far,
        "dur_mean": statistics.mean(durations) if durations else None,
        "dur_min": min(durations) if durations else None,
        "dur_max": max(durations) if durations else None,
    }


def final_metrics(clips: List[Dict], boundaries: Sequence[Tuple[float, bool]]) -> Dict:
    """Metrics on the clips the pipeline would actually cut."""
    durations: List[float] = []
    spans: List[Tuple[float, float]] = []
    edges = 0
    edges_on = 0
    for clip in clips:
        start = float(clip["start_time"])
        end = float(clip["end_time"])
        durations.append(end - start)
        spans.append((start, end))
        for edge in (start, end):
            edges += 1
            distance, _ = nearest_boundary(boundaries, edge)
            if distance is not None and distance <= _BOUNDARY_EPS:
                edges_on += 1
    overlaps = 0
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a, b = spans[i], spans[j]
            if min(a[1], b[1]) - max(a[0], b[0]) > 0:
                overlaps += 1
    return {
        "clips": len(clips),
        "dur_mean": statistics.mean(durations) if durations else None,
        "dur_min": min(durations) if durations else None,
        "dur_max": max(durations) if durations else None,
        "overlaps": overlaps,
        "edges_on_boundary": f"{edges_on}/{edges}" if edges else "0/0",
    }


def run_mode(
    mode: str,
    transcript: Dict,
    settings: Settings,
    boundaries: Sequence[Tuple[float, bool]],
    num_clips: int,
) -> Dict:
    """Rank, snap, dedupe and select once for a single mode ('single'/'two_stage')."""
    stats = CallStats(settings)
    duration = float(transcript.get("duration", 0.0) or 0.0)
    text = build_transcript_log(transcript, None)

    result = call_highlight_api(
        text,
        None,
        duration,
        num_clips=num_clips,
        llm_fn=stats,
        is_chunk=False,
        has_visuals=False,
        two_stage=(mode == "two_stage"),
    )
    raw = list(result.get("highlights", []))
    raw_report = raw_metrics(raw, boundaries)

    # Copy before snapping so the RAW numbers keep describing the model output.
    candidates = dedupe_highlights([dict(clip) for clip in raw])
    if settings.clip_snap_to_transcript:
        snap_highlights_to_transcript(
            candidates,
            transcript,
            start_padding=settings.clip_start_padding,
            end_padding=settings.clip_end_padding,
            pause_threshold=settings.subtitle_pause_threshold,
            max_end=duration or None,
        )
    candidates = dedupe_highlights(candidates)
    top = select_highlights(
        candidates,
        transcript,
        num_clips=num_clips,
        min_duration=settings.clip_min_duration,
        max_end=duration or None,
    )

    if mode == "two_stage":
        stats.phase = "metadata"
        enrich_highlights_metadata(top, transcript, settings, llm_fn=stats)

    return {
        "mode": mode,
        "raw": raw_report,
        "final": final_metrics(top, boundaries),
        "candidates": len(candidates),
        "clips": top,
        "stats": stats,
    }


def print_report(
    video: str,
    transcript: Dict,
    boundaries: Sequence[Tuple[float, bool]],
    reports: List[Dict],
    num_clips: int,
) -> None:
    duration = float(transcript.get("duration", 0.0) or 0.0)
    print()
    print("=" * 72)
    print(f"video: {video}")
    print(
        f"duration: {duration:.0f}s   cues: {len(transcript.get('segments', []))}   "
        f"phrase boundaries: {len(boundaries)}   NUM_CLIPS: {num_clips}"
    )
    print("=" * 72)

    for report in reports:
        stats = report["stats"]
        raw = report["raw"]
        final = report["final"]
        print(f"\n--- {report['mode']} ---")
        print(
            f"  LLM calls: {stats.calls}   prompt: {stats.prompt_chars} chars   "
            f"response: {stats.response_chars} chars   "
            f"by phase: {stats.by_phase}"
        )
        print("  RAW model output (before snapping to phrases):")
        print(
            f"    clips={raw['clips']}  "
            f"mean |start-boundary|={_fmt(raw['start_dist'])}s  "
            f"mean |end-boundary|={_fmt(raw['end_dist'])}s"
        )
        print(
            f"    ends on a sentence: {raw['ends_on_sentence']}/{raw['clips']}   "
            f"ends far from any break (>{_FAR_TOL:g}s): {raw['ends_far']}"
        )
        print(
            f"    durations: mean={_fmt(raw['dur_mean'], 1)}s "
            f"min={_fmt(raw['dur_min'], 1)}s max={_fmt(raw['dur_max'], 1)}s"
        )
        print("  FINAL selected clips (after snap + dedupe + min-duration):")
        print(
            f"    from {report['candidates']} candidate(s) -> {final['clips']} clip(s)   "
            f"overlaps={final['overlaps']}   edges on a boundary={final['edges_on_boundary']}"
        )
        print(
            f"    durations: mean={_fmt(final['dur_mean'], 1)}s "
            f"min={_fmt(final['dur_min'], 1)}s max={_fmt(final['dur_max'], 1)}s"
        )
        for index, clip in enumerate(report["clips"], 1):
            start = float(clip["start_time"])
            end = float(clip["end_time"])
            title = str(clip.get("title") or "").strip()
            print(
                f"      #{index} [{start:7.2f} -> {end:7.2f}] {end - start:5.1f}s  "
                f"score={clip.get('score')}  «{title}»"
            )


def print_verdict(reports: List[Dict]) -> None:
    """A short before/after summary — the numbers the keep-or-drop call rests on."""
    by_mode = {report["mode"]: report for report in reports}
    single = by_mode.get("single")
    two_stage = by_mode.get("two_stage")
    if not single or not two_stage:
        return

    single_rank = single["stats"].by_phase.get("ranking", 0)
    two_rank = two_stage["stats"].by_phase.get("ranking", 0)

    print()
    print("-" * 72)
    print("VERDICT (lower is better for the distances):")
    print(
        f"  ranking output size: {single_rank} -> {two_rank} chars "
        f"(delta {two_rank - single_rank:+d})"
    )
    print(
        f"  mean |end-boundary|: {_fmt(single['raw']['end_dist'])}s -> "
        f"{_fmt(two_stage['raw']['end_dist'])}s"
    )
    print(
        f"  mean |start-boundary|: {_fmt(single['raw']['start_dist'])}s -> "
        f"{_fmt(two_stage['raw']['start_dist'])}s"
    )
    print(
        f"  ends far from any break: {single['raw']['ends_far']} -> "
        f"{two_stage['raw']['ends_far']}"
    )
    print(
        f"  final clips: {single['final']['clips']} -> {two_stage['final']['clips']}   "
        f"total LLM calls: {single['stats'].calls} -> {two_stage['stats'].calls}"
    )
    print("-" * 72)


def resolve_video(settings: Settings, explicit: Optional[str]) -> str:
    """First video to analyse: an explicit path, or ``INPUT`` from ``.env``."""
    source = explicit if explicit else settings.input
    resolved = settings.resolve(source)
    videos = find_video_files(resolved)
    if not videos:
        raise SystemExit(
            f"No video found at {resolved!r}. Pass --video PATH or set INPUT."
        )
    return videos[0]


def load_transcript(video: str, settings: Settings, no_transcribe: bool) -> Dict:
    """Transcribe the video, or load the cached ``.srt`` when Whisper is skipped."""
    if not no_transcribe:
        return transcribe(video, settings)

    from shorts_generator import transcriber as _transcriber

    cache_path = _transcriber._transcript_cache_path(video, settings)
    if not os.path.isfile(cache_path):
        raise SystemExit(
            f"--no-transcribe was given but there is no cached transcript at "
            f"{cache_path}. Run once without the flag to create it."
        )
    print(f"[compare] reusing cached transcript: {cache_path}", flush=True)
    cached = _transcriber._load_srt_cache(cache_path)
    return _transcriber._apply_cue_split(cached, settings)


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        help="video file to analyse (default: INPUT from .env)",
    )
    parser.add_argument(
        "--clips",
        type=int,
        default=None,
        help="NUM_CLIPS for this run (default: from .env)",
    )
    parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="reuse the cached .srt and never run Whisper",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    num_clips = args.clips or settings.num_clips

    video = resolve_video(settings, args.video)
    transcript = load_transcript(video, settings, args.no_transcribe)
    if not transcript.get("segments"):
        raise SystemExit(f"No transcript segments for {video}")

    boundaries = phrase_boundaries(
        transcript.get("segments", []),
        pause_threshold=settings.subtitle_pause_threshold,
    )

    reports: List[Dict] = []
    for mode in ("single", "two_stage"):
        print(f"[compare] running {mode} ranking ...", flush=True)
        reports.append(run_mode(mode, transcript, settings, boundaries, num_clips))

    print_report(video, transcript, boundaries, reports, num_clips)
    print_verdict(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
