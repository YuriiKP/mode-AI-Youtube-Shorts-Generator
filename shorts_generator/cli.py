"""Unified command-line interface — one entry point for the whole project.

    python main.py clip        # long video -> ranked vertical shorts
    python main.py transcribe  # Whisper -> .srt subtitles
    python main.py music       # add background music
    python main.py subtitles   # burn subtitles in
    python main.py all         # clip + music + subtitles in one go

Everything is configured through a single ``.env`` file (see ``.env.example``);
the flags below only override individual values for one run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

from .config import ConfigError, Settings, load_settings
from .pipeline import generate_shorts, generate_subtitles, resolve_input_videos
from .postprocess.log import setup_logging
from .timing import get_timer, start_timer

_EPILOG = """\
examples:
  python main.py clip                       # uses INPUT / OUTPUT_DIR from .env
  python main.py clip -i "video/talk.mkv" -n 5
  python main.py transcribe                 # writes <video>.srt next to the video
  python main.py music -m music/            # random track from a folder
  python main.py music -m song.mp3          # one specific track
  python main.py subtitles                  # same-named .srt, else Whisper
  python main.py all                        # everything, reading settings from .env

Configuration: copy .env.example to .env and edit it. Values there are the
defaults for every command; flags override them for a single run.
"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Options shared by every subcommand."""
    parser.add_argument(
        "--env",
        dest="env_file",
        default=None,
        metavar="PATH",
        help="extra .env file to load (highest-priority file)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="only print warnings and errors",
    )
    parser.add_argument(
        "--no-timing",
        dest="no_timing",
        action="store_true",
        help="do not print the summary of how long each stage took",
    )


def _add_io(parser: argparse.ArgumentParser) -> None:
    """Source and destination options."""
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        default=None,
        metavar="PATH|URL",
        help="video file, folder of videos or YouTube URL (default: INPUT)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=None,
        metavar="DIR",
        help="where to write results (default: OUTPUT_DIR)",
    )


def _add_clip_options(parser: argparse.ArgumentParser) -> None:
    """Highlight-ranking / cropping options."""
    parser.add_argument(
        "-n",
        "--num-clips",
        dest="num_clips",
        type=int,
        default=None,
        help="how many shorts to render (default: 3)",
    )
    parser.add_argument(
        "-a",
        "--aspect-ratio",
        dest="aspect_ratio",
        default=None,
        help="output aspect ratio (default: 9:16)",
    )
    parser.add_argument(
        "--format",
        dest="download_format",
        default=None,
        help="download resolution: 360 / 480 / 720 / 1080 (default: 720)",
    )
    parser.add_argument(
        "-l",
        "--language",
        dest="whisper_language",
        default=None,
        help="Whisper language code, e.g. ru or en (default: auto-detect)",
    )
    parser.add_argument(
        "--face-tracking",
        dest="face_tracking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="track faces for the vertical crop (default: on)",
    )


def _add_music_options(parser: argparse.ArgumentParser) -> None:
    """Background-music options."""
    parser.add_argument(
        "-m",
        "--music",
        dest="music",
        default=None,
        metavar="PATH",
        help="music file or folder (default: MUSIC)",
    )
    parser.add_argument(
        "--music-volume",
        dest="music_volume",
        type=float,
        default=None,
        help="music volume multiplier (default: 0.2)",
    )
    parser.add_argument(
        "--music-fade-out",
        dest="music_fade_out",
        type=float,
        default=None,
        metavar="SECONDS",
        help="fade the music out over the last N seconds (default: 3)",
    )


def _add_subtitle_options(parser: argparse.ArgumentParser) -> None:
    """Subtitle options."""
    parser.add_argument(
        "-s",
        "--source",
        dest="subtitle_source",
        choices=("auto", "file", "whisper", "none"),
        default=None,
        help="subtitle source: auto (same-named .srt, else Whisper), file, "
        "whisper or none (default: auto)",
    )
    parser.add_argument(
        "--subtitle-file",
        dest="subtitle_file",
        default=None,
        metavar="PATH",
        help="explicit .srt to burn in (overrides --source)",
    )


def _add_render_options(parser: argparse.ArgumentParser) -> None:
    """Vertical re-framing + banner options (shared by post-processing commands)."""
    parser.add_argument(
        "--fit-vertical",
        dest="fit_vertical",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="re-frame the output to a vertical frame, filling the empty space "
        "with a blurred copy of the video (default: on)",
    )
    parser.add_argument(
        "--fit-aspect-ratio",
        dest="fit_aspect_ratio",
        default=None,
        metavar="W:H",
        help="aspect ratio for the vertical frame (default: 9:16; empty uses "
        "ASPECT_RATIO)",
    )
    parser.add_argument(
        "--background-blur",
        dest="background_blur",
        type=int,
        default=None,
        metavar="N",
        help="strength of the blurred background fill, 0 disables the blur "
        "(default: 30)",
    )
    parser.add_argument(
        "--banner",
        dest="banner",
        default=None,
        metavar="TEXT|PATH",
        help="banner over the video: an existing image file (PNG/JPG) is drawn "
        "as an image, anything else as a text band (default: BANNER)",
    )
    parser.add_argument(
        "--banner-position",
        dest="banner_position",
        choices=("top", "bottom", "center"),
        default=None,
        help="where to place the banner (default: top)",
    )
    parser.add_argument(
        "--saturation",
        dest="saturation",
        type=float,
        default=None,
        metavar="N",
        help="saturation multiplier: 1.0 keeps the colours, 0 is greyscale and "
        "values above 1 boost colour (default: 1.0)",
    )
    parser.add_argument(
        "--sharpness",
        dest="sharpness",
        type=float,
        default=None,
        metavar="N",
        help="unsharp-mask strength, 0 disables the sharpening (default: 0)",
    )
    parser.add_argument(
        "--chromatic-aberration",
        dest="chromatic_aberration",
        type=float,
        default=None,
        metavar="PX",
        help="red/blue channel separation in pixels at the frame corner, "
        "0 disables the effect (default: 0)",
    )


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-json",
        dest="output_json",
        default=None,
        metavar="PATH",
        help="write the full result as JSON to this path",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "AI YouTube Shorts Generator — turn a long video into ranked vertical "
            "shorts, transcribe it, add music and burn in subtitles."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    # clip ------------------------------------------------------------------
    clip = sub.add_parser(
        "clip",
        help="cut the source into ranked vertical shorts",
        description="Download/transcribe/rank/crop the source into shorts.",
    )
    _add_io(clip)
    _add_clip_options(clip)
    _add_json(clip)
    _add_common(clip)

    # transcribe ------------------------------------------------------------
    transcribe = sub.add_parser(
        "transcribe",
        help="transcribe the source with Whisper into .srt",
        description=(
            "Transcribe a video file or every video in a folder with Whisper and "
            "write a .srt next to each one."
        ),
    )
    _add_io(transcribe)
    transcribe.add_argument(
        "-l",
        "--language",
        dest="whisper_language",
        default=None,
        help="Whisper language code, e.g. ru or en (default: auto-detect)",
    )
    _add_json(transcribe)
    _add_common(transcribe)

    # music -----------------------------------------------------------------
    music = sub.add_parser(
        "music",
        help="add background music to the source",
        description="Mix background music under the source audio.",
    )
    _add_io(music)
    _add_music_options(music)
    _add_render_options(music)
    _add_common(music)

    # subtitles -------------------------------------------------------------
    subtitles = sub.add_parser(
        "subtitles",
        help="burn subtitles into the source",
        description=(
            "Burn subtitles into the source. With --source auto (the default) a "
            "same-named .srt next to the video is used; if none is found, Whisper "
            "transcribes it first."
        ),
    )
    _add_io(subtitles)
    _add_subtitle_options(subtitles)
    subtitles.add_argument(
        "-l",
        "--language",
        dest="whisper_language",
        default=None,
        help="Whisper language code used when transcribing (default: auto)",
    )
    _add_render_options(subtitles)
    _add_common(subtitles)

    # all -------------------------------------------------------------------
    allp = sub.add_parser(
        "all",
        help="clip + music + subtitles in one run",
        description="Run the full pipeline: clip the source, then add music and subtitles.",
    )
    _add_io(allp)
    _add_clip_options(allp)
    _add_music_options(allp)
    _add_render_options(allp)
    allp.add_argument(
        "--music-enabled",
        dest="add_music",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="add background music to each short (default: on)",
    )
    allp.add_argument(
        "--subtitles-enabled",
        dest="add_subtitles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="burn subtitles into each short (default: on)",
    )
    _add_json(allp)
    _add_common(allp)

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_settings(args: argparse.Namespace) -> Settings:
    """Merge CLI overrides on top of the ``.env`` configuration."""
    from .config import settings_field_names

    overrides: Dict[str, object] = {}
    for name in settings_field_names():
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value

    return load_settings(
        env_file=getattr(args, "env_file", None),
        extra=overrides,
        base_dir=os.getcwd(),
    )


def _output_path_for(input_file: str, out_dir: str) -> str:
    """Pick the output path for a video, avoiding overwriting the source."""
    base = os.path.basename(input_file)
    candidate = os.path.join(out_dir, base)
    if os.path.abspath(candidate) == os.path.abspath(input_file):
        stem, ext = os.path.splitext(base)
        candidate = os.path.join(out_dir, f"{stem}_out{ext}")
    return candidate


def _maybe_write_json(path: Optional[str], payload: Dict) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"\nfull JSON written to {path}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _print_shorts(result: Dict, enhanced: bool) -> None:
    videos = result.get("videos") or [result]
    total_highlights = sum(len(video.get("highlights") or []) for video in videos)
    total_shorts = sum(len(video.get("shorts") or []) for video in videos)

    print("\n" + "=" * 72)
    if len(videos) == 1:
        print(f"Source video:  {videos[0].get('source_video_url')}")
    else:
        print(f"Source videos: {len(videos)}")
        for video in videos:
            print(f"  - {video.get('source_video_url')}")
    print(f"Highlights:    {total_highlights} candidates -> kept top {total_shorts}")
    print("=" * 72)

    index = 0
    for video in videos:
        shorts = video.get("shorts") or []
        if len(videos) > 1:
            print(f"\n=== {video.get('source_video_url')} ===")
        if not shorts:
            print("(no shorts rendered)")
            continue
        for short in shorts:
            index += 1
            ctype = short.get("clip_type") or "other"
            start = short.get("start_time", 0.0)
            end = short.get("end_time", 0.0)
            print(
                f"\n#{index}  score={short.get('score')}  [{ctype}]  {start:.1f}s -> {end:.1f}s"
            )
            print(f"     title:  {short.get('title')}")
            if short.get("hook_sentence"):
                print(f"     hook:   {short['hook_sentence']}")
            if short.get("punchline"):
                print(f"     punch:  {short['punchline']}")
            if short.get("clip_url"):
                print(f"     clip:   {short['clip_url']}")
            else:
                print(f"     clip:   FAILED ({short.get('error')})")
            if enhanced:
                if short.get("enhanced"):
                    print("     enhance: yes")
                    if short.get("subtitle_path"):
                        print(f"     srt:    {short['subtitle_path']}")
                elif short.get("enhance_error"):
                    print(f"     enhance: FAILED ({short['enhance_error']})")


def _print_timing() -> None:
    """Print the per-stage wall-clock summary for the run, if any was recorded."""
    timer = get_timer()
    if timer.empty:
        return
    print("\n" + timer.report("Time spent"))


def cmd_clip(settings: Settings, args: argparse.Namespace) -> int:
    result = generate_shorts(settings)
    _print_shorts(result, enhanced=False)
    _maybe_write_json(getattr(args, "output_json", None), result)
    return 0


def cmd_transcribe(settings: Settings, args: argparse.Namespace) -> int:
    result = generate_subtitles(settings)

    print("\n" + "=" * 72)
    print(f"Input:  {result['input']}")
    print(f"Files:  {len(result['results'])}")
    print("=" * 72)
    for i, entry in enumerate(result["results"], 1):
        print(f"\n#{i}  {entry.get('source_video')}")
        if entry.get("subtitle_path"):
            duration = entry.get("duration", 0) or 0
            print(
                f"     srt:    {entry['subtitle_path']} "
                f"({entry.get('segments')} segments, {duration:.0f}s)"
            )
        else:
            print(f"     srt:    FAILED ({entry.get('error')})")

    _maybe_write_json(getattr(args, "output_json", None), result)
    return 0


def _apply_to_inputs(
    settings: Settings,
    *,
    add_music: bool,
    burn_subtitles: bool,
) -> int:
    """Run the post-processing engine over every input video."""
    from .postprocess.pipeline import run as postprocess_run

    timer = start_timer()

    with timer.stage("download"):
        videos = resolve_input_videos(settings)
    out_dir = settings.resolve(settings.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    total = len(videos)
    print(f"[post] {total} video(s) -> {out_dir}", flush=True)
    for i, video in enumerate(videos, 1):
        out_path = _output_path_for(video, out_dir)
        print(f"[post] {i}/{total}: {os.path.basename(video)}", flush=True)
        with timer.stage("postprocess"):
            postprocess_run(
                video,
                out_path,
                settings,
                burn_subtitles=burn_subtitles,
                add_music=add_music,
            )
        print(f"[post]   -> {out_path}", flush=True)
    return 0


def cmd_music(settings: Settings, args: argparse.Namespace) -> int:
    if not (settings.music or "").strip():
        print(
            "no music configured: set MUSIC in .env or pass -m/--music PATH "
            "(a file or a folder).",
            file=sys.stderr,
        )
        return 2
    return _apply_to_inputs(settings, add_music=True, burn_subtitles=False)


def cmd_subtitles(settings: Settings, args: argparse.Namespace) -> int:
    return _apply_to_inputs(settings, add_music=False, burn_subtitles=True)


def cmd_all(settings: Settings, args: argparse.Namespace) -> int:
    result = generate_shorts(
        settings,
        enhance=True,
        add_music=getattr(args, "add_music", True),
        burn_subtitles=getattr(args, "add_subtitles", True),
    )
    _print_shorts(result, enhanced=True)
    _maybe_write_json(getattr(args, "output_json", None), result)
    return 0


_COMMANDS = {
    "clip": cmd_clip,
    "transcribe": cmd_transcribe,
    "music": cmd_music,
    "subtitles": cmd_subtitles,
    "all": cmd_all,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Windows uses 'charmap' by default, which cannot encode the arrows used in
    # the summaries; reconfigure the streams to UTF-8 so output works everywhere.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), quiet=args.quiet)

    try:
        settings = _build_settings(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    handler = _COMMANDS[args.command]
    try:
        code = handler(settings, args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        code = 130
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        print(f"\nFAILED: {exc}", file=sys.stderr)
        code = 1

    if not getattr(args, "no_timing", False):
        _print_timing()

    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
