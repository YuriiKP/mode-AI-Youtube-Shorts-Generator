"""Unified command-line interface — one entry point for the whole project.

    python main.py clip        # long video -> ranked vertical shorts
    python main.py montage     # one video -> one stitched hook clip
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
from .pipeline import (
    generate_montage,
    generate_shorts,
    generate_subtitles,
    resolve_input_videos,
)
from .postprocess.log import setup_logging
from .timing import get_timer, start_timer

_EPILOG = """\
examples:
  python main.py clip                       # uses INPUT / OUTPUT_DIR from .env
  python main.py clip -i "video/talk.mkv" -n 5
  python main.py clip --slide --slide-transition-gap 3   # slide crop across cuts
  python main.py montage -i "video/talk.mkv"   # one stitched 50-90s hook clip
  python main.py transcribe                 # writes <video>.srt next to the video
  python main.py music -m music/            # random track from a folder
  python main.py music -m song.mp3          # one specific track
  python main.py subtitles                  # same-named .srt, else Whisper
  python main.py all                        # everything, reading settings from .env
  python main.py preview                    # one random frame with the picture look

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
    parser.add_argument(
        "--slide",
        dest="slide_effect",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="slide the crop window across scene transitions inside each clip "
        "(default: off)",
    )
    parser.add_argument(
        "--slide-transition-gap",
        dest="slide_transition_gap",
        type=float,
        default=None,
        metavar="SECONDS",
        help="group transitions closer than this into one slide (default: 3)",
    )
    parser.add_argument(
        "--slide-range",
        dest="slide_range",
        type=float,
        default=None,
        metavar="0..1",
        help="fraction of the full crop travel used by the slide: 1 = edge to "
        "edge, 0 = no movement (default: 1)",
    )
    parser.add_argument(
        "--cut-effect",
        dest="cut_effect",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="blend a short transition over the source scene cuts inside each "
        "clip, softening the hard cuts (default: off)",
    )
    parser.add_argument(
        "--cut-effect-duration",
        dest="cut_effect_duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="length of each cut transition in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--cut-effect-types",
        dest="cut_effect_types",
        default=None,
        metavar="LIST",
        help="comma-separated transition styles rotated across cuts: "
        "dissolve, merge, fade, flash, zoom, chroma, whip, spin, shake, glitch "
        "(default: dissolve)",
    )
    parser.add_argument(
        "--cut-effect-max",
        dest="cut_effect_max",
        type=int,
        default=None,
        metavar="N",
        help="maximum transitions per clip (default: 3)",
    )
    parser.add_argument(
        "--visual-indexing",
        dest="visual_indexer_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="index the on-screen visuals and feed them to highlight ranking "
        "(default: off)",
    )
    parser.add_argument(
        "--visual-indexer-type",
        dest="visual_indexer_type",
        choices=("florence", "gemini_video", "qwen_video", "none"),
        default=None,
        help="visual indexer engine: florence / qwen_video (local, "
        "transformers/torch) or gemini_video (cloud, google-genai) "
        "(default: florence)",
    )
    parser.add_argument(
        "--visual-indexer-model",
        dest="visual_indexer_model",
        default=None,
        metavar="MODEL",
        help="model id for the local engines: a Florence-2 checkpoint for "
        "florence (default: microsoft/Florence-2-large) or a Qwen3.5 repo id "
        "for qwen_video (default: Qwen/Qwen3.5-9B)",
    )
    parser.add_argument(
        "--visual-indexer-cache",
        dest="visual_indexer_cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="save the visual index next to the video and reuse it on later "
        "runs, so re-cutting the same video skips the Gemini/Florence requests "
        "(default: on)",
    )
    parser.add_argument(
        "--visual-indexer-max-height",
        dest="visual_indexer_max_height",
        type=int,
        default=None,
        metavar="PX",
        help="downscale the frames (and the gemini_video upload) to at most "
        "this height before the model sees them (applies to qwen_video frames "
        "too): 0 keeps the source resolution (default: 0)",
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
    parser.add_argument(
        "--speed",
        dest="speed",
        type=float,
        default=None,
        metavar="N",
        help="playback-speed multiplier: 1.0 keeps the original speed, values "
        "above 1 speed the clip up (1.5 is 50%% faster) and values below 1 slow "
        "it down (0.5 is half speed); the audio is re-timed to match "
        "(default: 1.0)",
    )


def _add_uniqueness_options(parser: argparse.ArgumentParser) -> None:
    """Anti-duplicate (uniqueness) options shared by the picture commands."""
    parser.add_argument(
        "--unique-mirror",
        dest="unique_mirror",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mirror the frame horizontally to shift its perceptual hash "
        "(default: off; also flips any text baked into the video)",
    )
    parser.add_argument(
        "--unique-crop",
        dest="unique_crop",
        type=int,
        default=None,
        metavar="PX",
        help="trim PX pixels off every edge and stretch the frame back to its "
        "original size — the strongest frame-hash mover (default: 0)",
    )
    parser.add_argument(
        "--unique-noise",
        dest="unique_noise",
        type=float,
        default=None,
        metavar="N",
        help="temporal grain strength, 0..100 (default: 0)",
    )
    parser.add_argument(
        "--unique-brightness",
        dest="unique_brightness",
        type=float,
        default=None,
        metavar="N",
        help="brightness shift, -1..1 (default: 0)",
    )
    parser.add_argument(
        "--unique-contrast",
        dest="unique_contrast",
        type=float,
        default=None,
        metavar="N",
        help="contrast multiplier, 1.0 keeps the source (default: 1.0)",
    )
    parser.add_argument(
        "--unique-gamma",
        dest="unique_gamma",
        type=float,
        default=None,
        metavar="N",
        help="gamma adjustment, 1.0 keeps the source (default: 1.0)",
    )
    parser.add_argument(
        "--unique-hue",
        dest="unique_hue",
        type=float,
        default=None,
        metavar="DEG",
        help="hue rotation in degrees, -180..180 (default: 0)",
    )
    parser.add_argument(
        "--unique-pitch",
        dest="unique_pitch",
        type=float,
        default=None,
        metavar="PERCENT",
        help="micro pitch shift in percent (e.g. 0.5); changes the audio "
        "fingerprint without changing the clip length (default: 0)",
    )
    parser.add_argument(
        "--unique-loudness",
        dest="unique_loudness",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="normalise loudness to a platform-standard level, rewriting the "
        "waveform (default: off)",
    )
    parser.add_argument(
        "--unique-gain",
        dest="unique_gain",
        type=float,
        default=None,
        metavar="DB",
        help="extra audio gain in dB (default: 0)",
    )
    parser.add_argument(
        "--unique-metadata",
        dest="unique_metadata",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="strip the container tags of the result and stamp a fresh unique "
        "comment so the file bytes differ (default: on)",
    )
    parser.add_argument(
        "--unique-randomize",
        dest="unique_randomize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="jitter every uniqueness value per render so each export is a "
        "distinct variant (default: off)",
    )
    parser.add_argument(
        "--unique-jitter",
        dest="unique_jitter",
        type=float,
        default=None,
        metavar="0..1",
        help="spread used by --unique-randomize (default: 0.5)",
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

    # montage ---------------------------------------------------------------
    montage = sub.add_parser(
        "montage",
        help="stitch the most dynamic connected moments into one hook clip",
        description=(
            "Pick the most dynamic, semantically connected moments of the source "
            "and cut + join them into one vertical hook clip (~50-90s), then add "
            "music and subtitles. The montage length and per-segment bounds are "
            "hardcoded next to the prompt in montage.py, not configurable here."
        ),
    )
    _add_io(montage)
    # Reused for the transcription-side flags (language, download format,
    # visual indexing): the per-clip ranking/crop flags are accepted for
    # consistency with ``clip`` but ignored by the montage path.
    _add_clip_options(montage)
    _add_render_options(montage)
    _add_uniqueness_options(montage)
    montage.add_argument(
        "--music-enabled",
        dest="add_music",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="add background music to the montage (default: on)",
    )
    montage.add_argument(
        "--subtitles-enabled",
        dest="add_subtitles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="burn subtitles into the montage (default: on)",
    )
    _add_json(montage)
    _add_common(montage)

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
    # No picture render options here: the ``music`` command is audio-only and
    # never recolours, re-frames or re-times the video, so advertising effects,
    # banner or fit flags would be misleading (they would be silently ignored).
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
    _add_uniqueness_options(subtitles)
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
    _add_uniqueness_options(allp)
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

    # preview ---------------------------------------------------------------
    preview = sub.add_parser(
        "preview",
        help="render one random frame with the current picture settings",
        description=(
            "Render a single random frame through the same picture stages as a "
            "real short (vertical frame + blurred background, colour/lens "
            "effects, banner and the subtitle look) and save it as a PNG. No "
            "transcription happens: when subtitles are on the placeholder text "
            '"тестовый кадр" is drawn, so the look can be tuned in seconds.'
        ),
    )
    _add_io(preview)
    _add_render_options(preview)
    _add_uniqueness_options(preview)
    preview.add_argument(
        "--subtitles-enabled",
        dest="add_subtitles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw a sample subtitle cue in the test frame (default: on)",
    )
    preview.add_argument(
        "--count",
        dest="preview_count",
        type=int,
        default=1,
        metavar="N",
        help="how many random frames to render (default: 1)",
    )
    preview.add_argument(
        "--time",
        dest="preview_time",
        type=float,
        default=None,
        metavar="SECONDS",
        help="sample this second instead of a random one (handy to compare "
        "settings on the very same frame)",
    )
    _add_common(preview)

    # publish ---------------------------------------------------------------
    publish = sub.add_parser(
        "publish",
        help="upload the rendered shorts to YouTube / TikTok",
    )
    from publisher.cli import add_publish_actions

    add_publish_actions(publish)

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


def _write_shorts_info(result: Dict, output_dir: str) -> Optional[str]:
    """Write the shorts sheet: a text copy-paste file plus a JSON sidecar.

    Two files are written next to the rendered clips:

    * ``shorts_info.txt`` — a human-readable copy-paste sheet for the publishing
      fields of the platforms (YouTube, TikTok, Instagram, ...): one block per
      rendered short with its title, description and tags.
    * ``shorts_info.json`` — the same data in a machine-readable form, consumed
      by ``python main.py publish`` so the uploader never has to re-enter
      anything by hand.
    """
    videos = result.get("videos") or [result]
    blocks: List[str] = []
    entries: List[Dict] = []
    total = 0
    for video in videos:
        shorts = video.get("shorts") or []
        if not shorts:
            continue
        source = video.get("source_video_url") or result.get("source_video_url")
        source_name = os.path.basename(str(source)) if source else ""
        lines = [f"Источник: {source_name}", ""]
        # ``total`` is never reset between videos: the #N numbering is a single
        # running order across every input, matching the numbers in the clip
        # file names (short_01_..., short_02_...).
        for short in shorts:
            total += 1
            clip = short.get("clip_url")
            clip_name = os.path.basename(str(clip)) if clip else ""
            tags = [str(t) for t in (short.get("tags") or []) if str(t).strip()]
            lines.append(f"#{total}")
            lines.append(f"Название:    {short.get('title') or '(без названия)'}")
            lines.append(f"Описание:    {short.get('description') or '(без описания)'}")
            if tags:
                lines.append(f"Теги:        {' '.join('#' + tag for tag in tags)}")
            if clip:
                lines.append(f"Файл:        {clip_name}")
            lines.append("")
            entries.append(
                {
                    "number": total,
                    "title": short.get("title") or "",
                    "description": short.get("description") or "",
                    "tags": tags,
                    "file": clip_name,
                    "source": source_name,
                    "clip_type": short.get("clip_type") or "",
                    "score": short.get("score") or 0,
                    "hook_sentence": short.get("hook_sentence") or "",
                    "punchline": short.get("punchline") or "",
                }
            )
        blocks.append("\n".join(lines).rstrip())

    if total == 0:
        return None

    header = "Информация о шортсах (для заполнения на площадках)"
    text = "\n\n".join([header, "=" * len(header), *blocks]) + "\n"

    resolved_dir = output_dir
    if not os.path.isabs(resolved_dir):
        resolved_dir = os.path.abspath(resolved_dir)
    os.makedirs(resolved_dir, exist_ok=True)
    path = os.path.join(resolved_dir, "shorts_info.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)

    json_path = os.path.join(resolved_dir, "shorts_info.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"shorts": entries}, handle, ensure_ascii=False, indent=2)

    print(f"\nshorts info written to {path}")
    print(f"shorts info JSON written to {json_path}")
    return path


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

    _print_failures(result)


def _print_failures(result: Dict) -> None:
    """List the source videos that failed and say what happened to the rest.

    A run that carried on past a per-video problem (bad audio, a broken ffmpeg
    pass, a window with no usable material) processed every remaining input, so
    the header reports that. A run that stopped early did so on a failure that
    was not about one input — the provider refusing every request the same way —
    and never touched the sources that came after it; saying "the rest were
    processed" there would be a lie, so the header names the early stop instead.
    """
    failures = result.get("failures") or []
    if not failures:
        return
    print("\n" + "-" * 72)
    if result.get("aborted"):
        print(f"Failed videos: {len(failures)} (the run stopped at the first one)")
    else:
        print(f"Failed videos: {len(failures)} (the rest were processed)")
    for failure in failures:
        source = failure.get("source_video_url")
        print(
            f"  - {os.path.basename(str(source)) if source else '?'}: {failure.get('error')}"
        )
    print("-" * 72)


def _print_timing() -> None:
    """Print the per-stage wall-clock summary for the run, if any was recorded."""
    timer = get_timer()
    if timer.empty:
        return
    print("\n" + timer.report("Time spent"))


def _exit_code_for(result: Dict) -> int:
    """0 on a clean run, 1 when some videos failed but the rest completed.

    Per-video failures no longer abort the whole run (see
    ``pipeline._run``), so the sheet and clips from the sources that *did*
    succeed are kept. The non-zero code keeps that partial failure visible to
    scripts and shells without throwing the good output away.
    """
    failures = result.get("failures") or []
    if failures:
        if result.get("aborted"):
            print(
                f"\n{len(failures)} video(s) failed; the run stopped there because "
                "the failure was not about one input — the remaining inputs were "
                "left untouched and can be processed by re-running.",
                file=sys.stderr,
            )
        else:
            print(
                f"\n{len(failures)} video(s) failed; the clips and shorts_info from "
                "the other inputs were kept.",
                file=sys.stderr,
            )
        return 1
    return 0


def cmd_clip(settings: Settings, args: argparse.Namespace) -> int:
    # Write the shorts sheet after every video (not just at the end): if a later
    # source aborts — exhausted LLM quota, bad audio, a broken cut — the clips
    # already rendered earlier still get their info saved instead of losing it.
    def _save_info(partial: Dict) -> None:
        _write_shorts_info(partial, settings.output_dir)

    # MONTAGE_MODE=true switches ``clip`` to the montage pipeline: one long video
    # becomes a single stitched hook clip instead of several separate shorts.
    # No enhance here — ``clip`` never adds music or burned-in subtitles.
    if getattr(settings, "montage_mode", False):
        print(
            "[clip] MONTAGE_MODE is on — routing to the montage pipeline",
            flush=True,
        )
        result = generate_montage(settings, enhance=False, on_video_done=_save_info)
        _print_shorts(result, enhanced=False)
        _write_shorts_info(result, settings.output_dir)
        _maybe_write_json(getattr(args, "output_json", None), result)
        return _exit_code_for(result)

    result = generate_shorts(settings, on_video_done=_save_info)
    _print_shorts(result, enhanced=False)
    _write_shorts_info(result, settings.output_dir)
    _maybe_write_json(getattr(args, "output_json", None), result)
    return _exit_code_for(result)


def cmd_montage(settings: Settings, args: argparse.Namespace) -> int:
    # Same incremental write as ``clip``/``all``: the info sheet is refreshed
    # after each video so a later failure does not cost the info already saved.
    def _save_info(partial: Dict) -> None:
        _write_shorts_info(partial, settings.output_dir)

    result = generate_montage(
        settings,
        enhance=True,
        add_music=getattr(args, "add_music", True),
        burn_subtitles=getattr(args, "add_subtitles", True),
        on_video_done=_save_info,
    )
    _print_shorts(result, enhanced=True)
    _write_shorts_info(result, settings.output_dir)
    _maybe_write_json(getattr(args, "output_json", None), result)
    return _exit_code_for(result)


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
    apply_picture: bool = True,
) -> int:
    """Run the post-processing engine over every input video.

    ``apply_picture`` is forwarded to the engine; when ``False`` it leaves the
    source video untouched and performs only the requested subtitle/music stage,
    which is what keeps the dedicated commands modular.
    """
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
                apply_picture=apply_picture,
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
    # The music command is audio-only: it must not recolour, re-frame or re-time
    # the picture, so the picture stages are switched off.
    return _apply_to_inputs(
        settings, add_music=True, burn_subtitles=False, apply_picture=False
    )


def cmd_subtitles(settings: Settings, args: argparse.Namespace) -> int:
    return _apply_to_inputs(settings, add_music=False, burn_subtitles=True)


def cmd_all(settings: Settings, args: argparse.Namespace) -> int:
    # Same incremental write as ``clip``: the sheet is refreshed after each video
    # so a failure partway through a batch does not cost the info of the shorts
    # already rendered (and enhanced) before it.
    def _save_info(partial: Dict) -> None:
        _write_shorts_info(partial, settings.output_dir)

    # MONTAGE_MODE=true switches ``all`` to the montage pipeline as well, keeping
    # the music/subtitle stages that ``all`` normally runs: one long video
    # becomes a single stitched hook clip instead of several separate shorts.
    if getattr(settings, "montage_mode", False):
        print(
            "[all] MONTAGE_MODE is on — routing to the montage pipeline",
            flush=True,
        )
        result = generate_montage(
            settings,
            enhance=True,
            add_music=getattr(args, "add_music", True),
            burn_subtitles=getattr(args, "add_subtitles", True),
            on_video_done=_save_info,
        )
        _print_shorts(result, enhanced=True)
        _write_shorts_info(result, settings.output_dir)
        _maybe_write_json(getattr(args, "output_json", None), result)
        return _exit_code_for(result)

    result = generate_shorts(
        settings,
        enhance=True,
        add_music=getattr(args, "add_music", True),
        burn_subtitles=getattr(args, "add_subtitles", True),
        on_video_done=_save_info,
    )
    _print_shorts(result, enhanced=True)
    _write_shorts_info(result, settings.output_dir)
    _maybe_write_json(getattr(args, "output_json", None), result)
    return _exit_code_for(result)


def cmd_preview(settings: Settings, args: argparse.Namespace) -> int:
    from .preview import render_preview_frames

    paths = render_preview_frames(
        settings,
        count=int(getattr(args, "preview_count", 1) or 1),
        subtitles=bool(getattr(args, "add_subtitles", True)),
        timestamp=getattr(args, "preview_time", None),
    )

    print("\n" + "=" * 72)
    print(f"preview frame(s) written: {len(paths)}")
    for path in paths:
        print(f"  - {path}")
    print("=" * 72)
    return 0


def cmd_publish(settings: Settings, args: argparse.Namespace) -> int:
    """Delegate the ``publish`` command to the :mod:`publisher` package."""
    from publisher.cli import run_publish

    return run_publish(args, base_dir=settings.base_dir)


_COMMANDS = {
    "clip": cmd_clip,
    "montage": cmd_montage,
    "transcribe": cmd_transcribe,
    "music": cmd_music,
    "subtitles": cmd_subtitles,
    "all": cmd_all,
    "preview": cmd_preview,
    "publish": cmd_publish,
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
