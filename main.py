"""CLI entry point.

Usage:
    python main.py "https://www.youtube.com/watch?v=..." \
        --num-clips 3 --aspect-ratio 9:16
"""

import argparse
import json
import sys

# Windows uses 'charmap' by default, which can't encode Unicode characters
# like →. Reconfigure stdout/stderr to UTF-8 so output works on all platforms.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from shorts_generator import generate_shorts, generate_subtitles


def main() -> int:
    parser = argparse.ArgumentParser(description="AI YouTube Shorts Generator")
    parser.add_argument(
        "url",
        nargs="?",
        help="YouTube URL, file:// URL, or local file path. In "
        "--subtitles-only mode, a video file or a directory of videos.",
    )
    parser.add_argument(
        "--subtitles-only",
        action="store_true",
        help="Only generate .srt subtitles for a local video file or every "
        "video in a directory (passed as the positional path), skipping "
        "highlight ranking and rendering. Each video gets a .srt next to it "
        "with the same base name.",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=3,
        help="How many shorts to render (default: 3)",
    )
    parser.add_argument(
        "--aspect-ratio", default="9:16", help="Output aspect ratio (default: 9:16)"
    )
    parser.add_argument(
        "--format",
        default="720",
        help="Source download resolution: 360 / 480 / 720 / 1080 (default: 720)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force Whisper language code, e.g. 'en' (default: auto-detect)",
    )
    parser.add_argument(
        "--face-tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Track faces for the vertical crop (default). "
        "Use --no-face-tracking for a static centre crop.",
    )
    parser.add_argument(
        "--output-json", default=None, help="Write the full result JSON to this path"
    )
    args = parser.parse_args()

    # Subtitle-only mode: transcribe a file (or every video in a directory)
    # into .srt files, skipping highlight ranking and rendering entirely.
    if args.subtitles_only:
        if not args.url:
            parser.error("--subtitles-only requires a video file or directory path.")
        input_path = args.url

        try:
            result = generate_subtitles(input_path=input_path, language=args.language)
        except Exception as e:
            print(f"\nFAILED: {e}", file=sys.stderr)
            return 1

        print("\n" + "=" * 72)
        print("Mode:          subtitles-only")
        print(f"Input:         {result['input']}")
        print(f"Files:         {len(result['results'])}")
        print("=" * 72)
        for i, r in enumerate(result["results"], 1):
            print(f"\n#{i}  {r.get('source_video')}")
            if r.get("subtitle_path"):
                print(
                    f"     srt:    {r['subtitle_path']}  "
                    f"({r.get('segments')} segments, {r.get('duration', 0):.0f}s)"
                )
            else:
                print(f"     srt:    FAILED ({r.get('error')})")

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nFull JSON written to {args.output_json}")

        return 0

    if not args.url:
        parser.error("a YouTube URL, file:// URL, or local file path is required.")

    try:
        result = generate_shorts(
            youtube_url=args.url,
            num_clips=args.num_clips,
            aspect_ratio=args.aspect_ratio,
            download_format=args.format,
            language=args.language,
            face_tracking=args.face_tracking,
        )
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 72)
    print(f"Mode:          {result.get('mode', 'local')}")
    print(f"Source video:  {result['source_video_url']}")
    print(
        f"Highlights:    {len(result['highlights'])} candidates → kept top {len(result['shorts'])}"
    )
    print("=" * 72)
    for i, s in enumerate(result["shorts"], 1):
        ctype = s.get("clip_type") or "other"
        print(
            f"\n#{i}  score={s.get('score')}  [{ctype}]  {s.get('start_time'):.1f}s → {s.get('end_time'):.1f}s"
        )
        print(f"     title:  {s.get('title')}")
        print(f"     hook:   {s.get('hook_sentence')}")
        if s.get("punchline"):
            print(f"     punch:  {s.get('punchline')}")
        if s.get("clip_url"):
            print(f"     clip:   {s['clip_url']}")
        else:
            print(f"     clip:   FAILED ({s.get('error')})")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
