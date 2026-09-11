# AI YouTube Shorts Generator

**The open-source alternative to Opus Clip, Vidyo.ai, Klap, SubMagic, 2short.ai, and other AI clipping tools.** Drop in any long-form YouTube video (or a local file) and get back ranked, viral-ready 9:16 shorts — for free, with no per-clip credits, no watermarks, and full control over the highlight algorithm.

Built for creators, agencies, and developers who don't want to pay $20–$300/month or be capped on minutes processed. Uses GPT-class LLM highlight detection and local Whisper transcription to extract the most viral-worthy moments and auto-crop them vertically for TikTok, Reels, and Shorts.

<p align="center"><a href="https://www.youtube.com/watch?v=kT1CO4BYV3A"><img src="https://i.ytimg.com/vi/kT1CO4BYV3A/maxresdefault.jpg" width="720"></a></p>
<p align="center"><a href="https://www.youtube.com/watch?v=kT1CO4BYV3A"><b>▶ Watch: Free Unlimited AI Image Generator (Truly no limits, Open Source, No Watermark) </b></a></p>

![longshorts](https://github.com/user-attachments/assets/3f5d1abf-bf3b-475f-8abf-5e253003453a)

<p align="center">
  <a href="https://github.com/Anil-matcha/awesome-generative-ai-apps">
    <img src="https://img.shields.io/badge/Part%20of-Awesome%20Generative%20AI%20Apps-FFD700?style=for-the-badge&logo=github&logoColor=black" alt="Awesome Generative AI Apps">
  </a>
</p>

> 🎨 **[Explore 50+ more open-source AI apps →](https://github.com/Anil-matcha/awesome-generative-ai-apps)**

## Why Use This Instead of Opus Clip / Vidyo.ai / Klap?

| | This repo | Opus Clip / Vidyo.ai / Klap / SubMagic |
|---|---|---|
| **Price** | Free + open source (pay only for LLM usage) | $20–$300/month subscriptions |
| **Per-clip credits** | None — process unlimited videos | Monthly minute caps, overage fees |
| **Watermarks** | Never | On free tiers |
| **Highlight algorithm** | Fully editable virality framework | Black box |
| **Output format** | Any aspect ratio, any resolution | Locked presets |
| **Batch processing** | `xargs` an entire URL list | Manual upload one-by-one |
| **JSON / API output** | Built-in (`--output-json`) | Limited or paid tier only |
| **Self-hostable** | Yes — runs on your machine or server | SaaS only, your videos sit on their servers |
| **White-label / embeddable** | Yes — MIT licensed, import as Python lib | No |

## Features

- **🎬 YouTube (or Local File) In, Vertical Out**: Hand it any YouTube URL or local video — get back N viral-ready 9:16 mp4s
- **🔌 Fully Local Pipeline**: Runs entirely on your machine with `yt-dlp`, `faster-whisper`, and `ffmpeg`/`opencv`, and lets you pick OpenAI, DeepSeek, or Gemini for highlight ranking
- **🤖 Virality-Aware Highlight Selection**: Clips ranked on hooks, emotional peaks, opinion bombs, revelation moments, conflict, quotable lines, story peaks, and practical value — not just generic "interesting"
- **📈 Score + Hook + Reason for Every Clip**: Each highlight comes with a viral score, an opening hook line, and a one-sentence explanation of why it works
- **🎤 Local Whisper Transcription**: `faster-whisper` runs on CPU or CUDA — no cloud transcription service needed
- **🧩 Long-Video Aware**: Videos over 30 minutes are auto-chunked with overlap so nothing gets missed
- **♻️ Smart Dedupe**: Overlapping highlights are collapsed by score so you never get two near-duplicate clips
- **🎯 Smart Vertical Crop**: OpenCV face tracking with motion smoothing — disable it with `--no-face-tracking` for a static centre crop when a clip has no faces in frame
- **📱 Any Aspect Ratio**: 9:16 for TikTok/Reels/Shorts, 1:1 for square, anything else by flag
- **🧰 CLI + Python Library**: Use it from the shell or import `generate_shorts(...)` into your own pipeline
- **📝 Subtitles Only**: Add `--subtitles-only` and pass a video file or folder as the positional path to write `.srt` subtitles next to each video (same base name) for a single file or a whole folder — skipping highlight ranking and rendering
- **📦 JSON Output**: `--output-json` dumps the full result (transcript + every candidate highlight + final clip paths) for downstream automation

---

## Installation

### Prerequisites

- Python 3.10+
- `ffmpeg` on your PATH
- An LLM API key — `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, or `GEMINI_API_KEY` (only the highlight-ranking step is remote)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator.git
   cd AI-Youtube-Shorts-Generator
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the project root:
   ```bash
   LLM_PROVIDER=openai         # openai, deepseek, or gemini
   OPENAI_API_KEY=your_openai_key_here
   OPENAI_MODEL=gpt-4o-mini          # optional, default gpt-4o-mini
   DEEPSEEK_API_KEY=your_deepseek_key_here
   DEEPSEEK_MODEL=deepseek-chat      # optional, default deepseek-chat
   DEEPSEEK_BASE_URL=https://api.deepseek.com   # optional, default https://api.deepseek.com
   GEMINI_API_KEY=your_gemini_key_here
   GEMINI_MODEL=gemini-2.5-flash      # optional, default gemini-2.5-flash
   WHISPER_MODEL=base                # tiny / base / small / medium / large-v3-turbo / large-v3
   WHISPER_DEVICE=auto               # auto / cpu / cuda
   OUTPUT_DIR=output                 # where the rendered mp4s land
   ```

## Usage

### Single video

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

The rendered shorts land in `./output/short_01.mp4`, `short_02.mp4`, … (override with `OUTPUT_DIR`).

### With options

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" \
    --num-clips 5 \
    --aspect-ratio 9:16 \
    --output-json result.json
```

### Local file or path

You can pass a `file://` URL or a direct filesystem path and skip YouTube entirely:

```bash
python main.py "/Users/you/Videos/input.mp4"
python main.py "file:///Users/you/Videos/input.mp4"
```

The Python API works the same way:

```python
from shorts_generator import generate_shorts

result = generate_shorts(
    "/Users/you/Videos/input.mp4",
    num_clips=5,
    aspect_ratio="9:16",
)
for short in result["shorts"]:
    print(short["score"], short["title"], short["clip_url"])
```

Transcription is cached as an `.srt` file in `OUTPUT_DIR` using the
video's base name. If the cache already exists and is newer than the source
file, the app reuses it instead of running Whisper again.

Downloads are also cached in `OUTPUT_DIR` as
`source_<youtube_id>.mp4` when the input is a YouTube URL. If that file already
exists, the app skips `yt-dlp` and reuses the cached video.

### Generate subtitles only (no clips)

Need just the transcript? Add `--subtitles-only` and pass a video file or a
directory of videos as the positional path. Each video gets its own `.srt`
written next to it with the same base name (e.g. `video/talk.mkv` →
`video/talk.srt`) — no highlight ranking and no rendering.

```bash
# One video file
python main.py --subtitles-only "video/talk.mkv"

# Every video in a folder
python main.py --subtitles-only "video/"
```

Pass `--language ru` (or another ISO-639-1 code) to lock the recognition
language, and `--output-json result.json` to dump a small summary of the run.
Subtitles are produced locally with `faster-whisper`, and each `.srt` doubles as
the transcription cache — re-running skips Whisper while the `.srt` is still
newer than the source video.

The Python API mirrors this:

```python
from shorts_generator import generate_subtitles

result = generate_subtitles("video/", language="ru")
for item in result["results"]:
    print(item["source_video"], "->", item["subtitle_path"])
```

### Batch processing

Create a `urls.txt` file with one URL per line, then:

```bash
xargs -a urls.txt -I{} python main.py "{}"
```

### CLI flags

| Flag | Default | Notes |
|------|---------|-------|
| `--subtitles-only` | off | Only generate `.srt` subtitles (one file or a whole folder), skipping highlight ranking and rendering. Pass the video file or folder as the positional path |
| `--num-clips` | `3` | How many shorts to render |
| `--aspect-ratio` | `9:16` | Any ratio; `9:16` for TikTok/Reels, `1:1` for square |
| `--format` | `720` | Source download resolution: `360` / `480` / `720` / `1080` |
| `--language` | auto | Force Whisper language code (e.g. `en`) |
| `--face-tracking` / `--no-face-tracking` | on | `--no-face-tracking` uses a static centre crop instead of OpenCV face tracking |
| `--output-json` | — | Dump the full result (transcript + all candidates) to a file |

### Pipeline backends

| Step | Implementation |
|---|---|
| Download | `yt-dlp` for remote URLs, direct file path for local inputs |
| Transcription | `faster-whisper` (CPU or CUDA) |
| Highlight LLM | `LLM_PROVIDER=openai` uses OpenAI (`gpt-4o-mini` by default), `LLM_PROVIDER=deepseek` uses DeepSeek (`deepseek-chat` by default, OpenAI-compatible client with a custom base URL), `LLM_PROVIDER=gemini` uses Gemini (`gemini-2.5-flash` by default) |
| Vertical crop | `ffmpeg` cut + OpenCV face tracking (disable with `--no-face-tracking`) |
| Output | local mp4 paths |
| Required keys | `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, or `GEMINI_API_KEY` (+ `ffmpeg` on PATH) |

## How It Works

1. **Download**: Fetches the source video with `yt-dlp` (or uses a local file directly)
2. **Transcribe**: `faster-whisper` produces a timestamped transcript
3. **Detect content type**: An LLM classifies the video (podcast, interview, tutorial, vlog, etc.) and density, so the prompt can be tuned per content style
4. **Long-video chunking**: Videos > 30 min are split into 20-min overlapping chunks
5. **Highlight ranking**: An LLM scans the transcript through a virality framework — hook moments, emotional peaks, opinion bombs, revelations, conflict, quotables, story peaks, practical value — and emits ranked candidates with scores 0–100
6. **Dedupe**: Overlapping candidates are collapsed by score (>50% overlap → keep the higher score)
7. **Top-N selection**: The top `--num-clips` candidates are selected
8. **Cut + vertical crop**: Each highlight is cut with `ffmpeg` and reframed to the requested aspect ratio with OpenCV face tracking

**Output**: a list of local mp4 paths plus, for each clip, its title, viral score, hook sentence, and a one-line reason explaining why it should perform.

## Output

Console output looks like:

```
========================================================================
Highlights:    7 candidates → kept top 3
========================================================================

#1  score=92  124.3s → 187.6s
     title:  The one mistake that cost me $50K
     hook:   "Nobody talks about this, but it killed my first startup..."
     clip:   output/short_01.mp4

#2  score=88  ...
```

`--output-json result.json` produces:

```json
{
  "source_video_url": "output/source_abc123.mp4",
  "transcript": { "duration": 1873.4, "segments": [...] },
  "highlights": [ {...}, {...}, ... ],
  "shorts": [
    {
      "title": "...",
      "start_time": 124.3,
      "end_time": 187.6,
      "score": 92,
      "hook_sentence": "...",
      "virality_reason": "...",
      "clip_url": "output/short_01.mp4"
    }
  ]
}
```

## Configuration

### Highlight selection criteria
Edit `shorts_generator/highlights.py`:
- **Virality framework**: `VIRALITY_CRITERIA` — the ranked list of signals the LLM optimizes for
- **System prompt**: `HIGHLIGHT_SYSTEM_PROMPT` — duration sweet spot, hook rules, JSON schema
- **Chunk size**: `CHUNK_SIZE_SECONDS` (default 1200) — chunk length for long videos
- **Long-video threshold**: `LONG_VIDEO_THRESHOLD` (default 1800) — videos longer than this are chunked
- **Chunk overlap**: `CHUNK_OVERLAP_SECONDS` (default 60) — overlap between chunks so cross-boundary clips aren't missed

### Whisper transcription
Audio is transcribed locally by `faster-whisper` (CPU or CUDA). Set the model and
device with `WHISPER_MODEL` / `WHISPER_DEVICE`, and pass
`--language <code>` to lock the recognition to a specific language; otherwise it
auto-detects.

## Project Structure

```
AI-Youtube-Shorts-Generator/
├── main.py                       CLI entry point
├── requirements.txt              all dependencies (yt-dlp, faster-whisper, LLM clients, opencv)
├── .env.example
└── shorts_generator/
    ├── config.py                 env / settings (LLM provider + Whisper)
    ├── downloader.py             yt-dlp download
    ├── transcriber.py            faster-whisper transcription (+ .srt cache)
    ├── highlights.py             LLM virality ranking
    ├── llm.py                    OpenAI / DeepSeek / Gemini client selector
    ├── clipper.py                ffmpeg cut + OpenCV vertical crop
    ├── subtitles.py              subtitle-only: .srt generation (no ranking/rendering)
    └── pipeline.py               end-to-end orchestrator
```

## Troubleshooting

### Whisper produced no segments
The video may have no detectable speech, or it may be in a language Whisper struggles with. Try passing `--language en` (or the correct ISO-639-1 code) to skip auto-detection.

### Looking for better results?
Tune `VIRALITY_CRITERIA` and `HIGHLIGHT_SYSTEM_PROMPT` in `shorts_generator/highlights.py`, or switch `LLM_PROVIDER` to a stronger model for the highlight-ranking step.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Related Projects

- [AI Influencer Generator](https://github.com/SamurAIGPT/AI-Influencer-Generator)
- [Text to Video AI](https://github.com/SamurAIGPT/Text-To-Video-AI)
- [Faceless Video Generator](https://github.com/SamurAIGPT/Faceless-Video-Generator)
- [AI B-roll Generator](https://github.com/Anil-matcha/AI-B-roll)
- [No-code YouTube Shorts Generator](https://www.vadoo.tv/clip-youtube-video)
- [ai-creator-academy](https://github.com/Anil-matcha/ai-creator-academy) — free curriculum teaching creators how to monetize AI-generated shorts and video content