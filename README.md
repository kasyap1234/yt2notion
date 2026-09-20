# YouTube → illustrated Notion notes

Watches a YouTube **playlist or single video** with a video-native Gemini
model (not a transcript), and writes **interview-prep study notes** — decision
points, architecture evolution, glossary, speakable interview lines — plus
clean Mermaid flowcharts (alongside screenshots when the whiteboard changes), and one
Notion page per video under a parent "course" page.

Uses **Gemini 3.8 Flash** with **agentic video understanding** by default —
the model navigates the timeline and loads only what it needs, cutting token
use sharply on long videos (vs a fixed 1 FPS sweep).

Tuned for screen-recorded/whiteboard content (e.g. system design mock
interviews) where the diagram is built up incrementally and the labels on it
carry real information a transcript would miss — but works for general
educational videos too.

## 1. Install

You need Python 3.9+, plus **ffmpeg** on your system PATH (not a Python
package). Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
# macOS
brew install ffmpeg uv
# Ubuntu/Debian
sudo apt install ffmpeg
# install uv: https://docs.astral.sh/uv/getting-started/installation/

cd youtube-notion-videos
uv sync
```

This creates a local `.venv` and installs locked dependencies from
`pyproject.toml` / `uv.lock`.

## 2. Get your API keys

**Gemini API key** — https://aistudio.google.com/apikey

**Notion integration + parent page:**
1. Create an internal integration at https://www.notion.so/my-integrations
   and copy its secret.
2. In Notion, open (or create) the page you want the course to live under,
   click `...` → `Add connections`, and connect your integration to it.
3. Copy that page's ID from its URL — the 32-character string right before
   any `?` in `https://notion.so/Your-Page-Title-<PAGE_ID>`.

## 3. Configure

Create a `.env` file in the project root (loaded automatically):

```bash
GEMINI_API_KEY="..."
NOTION_API_KEY="..."
NOTION_PARENT_PAGE_ID="..."
```

Or export the same variables in your shell.

Optional overrides (defaults shown):

```bash
export GEMINI_MODEL="gemini-3.8-flash"
export GEMINI_MEDIA_PROCESSING="AGENTIC"   # or STATIC for short clips
export GEMINI_API_MODE="flex"              # flex | batch | standard
export GEMINI_BATCH_POLL_SECONDS="30"      # poll interval for batch jobs
export PIPELINE_CONCURRENCY="4"            # parallel workers for playlist runs
export MAX_VIDEO_HEIGHT="720"
export FRAME_WINDOW_SECONDS="2.0"          # how far around a flagged timestamp to search
export FRAME_STEP_SECONDS="0.5"            # how densely to sample that window
export PIPELINE_WORKDIR="./pipeline_data"
```

## 4. Run

```bash
# Playlist
uv run main.py "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"

# Single video
uv run main.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Regenerate notes for a video already marked done (creates a new page)
uv run main.py --force "https://www.youtube.com/watch?v=VIDEO_ID"

# Full playlist with Batch API (50% cheaper, resumes via progress.json)
uv run main.py --api-mode batch --concurrency 4 "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"
```

It will:
1. Resolve the URL (playlist listing or single-video metadata) without
   downloading yet.
2. Create one Notion page for the course / video set.
3. For each video: ask Gemini (preferring a direct public YouTube URL +
   agentic processing; falling back to download + Files API upload), get
   structured notes (summary, architecture evolution, glossary, takeaways,
   checkpoints with concept/trade-offs/interview lines, optional Mermaid,
   screenshot flags), download only if screenshots are needed, extract the sharpest
   nearby frame for each flagged moment, and write a rich Notion sub-page.
4. Save progress to `pipeline_data/progress.json` after every video, so if
   it crashes or gets rate-limited partway through a long playlist, re-running
   the same command skips everything already marked `"done"` instead of
   re-spending Gemini calls on videos you already have.

**Output style:** notes are written as interview-prep material (decision
points, glossary, architecture evolution, speakable lines) — not a lecture
transcript or verbose paraphrase. Use `--force` to regenerate a page for
A/B comparison; the old Notion page is kept.

Downloaded video files are deleted after each video is processed; only the
extracted screenshots get kept (and only until they're uploaded to Notion).

## Things worth knowing before a big run

- **Cost tiers (default: Batch).** Set `GEMINI_API_MODE` or pass `--api-mode`:
  - `batch` — **50% off**, async, up to 24h turnaround (best for playlists)
  - `flex` — **50% off**, sync, ~1–15 min latency, best-effort
  - `standard` — full price, fastest (original chat/streaming path)
- **Concurrency (default: 4 workers).** Playlist runs use `PIPELINE_CONCURRENCY`
  or `--concurrency`. In **batch** mode: downloads/uploads run in parallel, then
  all pending videos go into **one multi-video batch job** (Google processes them
  concurrently). In **flex/standard** mode: Gemini calls run in parallel.
  **Notion pages are always created in playlist order** so child pages appear
  in the right sequence under the course page.
- **Agentic video is the default.** Set `GEMINI_MEDIA_PROCESSING=STATIC` only
  if you want a fixed 1 FPS sweep (sometimes lower TTFT on clips under ~5
  minutes). Agentic mode is supported on Gemini 3.8 / 3.7 / 3.6 Flash.
- **Public YouTube URLs** can be passed straight to Gemini (preview feature).
  Private or restricted videos fall back to download + File API upload.
- **Timestamps can drift a couple of seconds**, especially on longer videos.
  That's why frames are grabbed as a small burst around each flagged moment
  and the sharpest one is kept, rather than trusting a single exact frame.
- **Cost scales with video length and count.** Agentic mode helps a lot, but
  a whole playlist is still one Gemini call per video — check current pricing
  before pointing this at a very long course.
- **yt-dlp and YouTube's terms of service**: downloading videos this way
  sits in a legal gray area for redistribution; using it to make personal
  study notes is generally the kind of use people don't run into trouble
  over, but it's worth knowing.

## Project layout

- `main.py` — orchestrates the whole run, tracks progress, lazy downloads
- `youtube_pipeline.py` — resolve single/playlist URL, download, extract frames
- `gemini_notes.py` — agentic Gemini prompt + YouTube URI / File API paths
- `notion_writer.py` — Notion API client (pages, rich blocks, image upload)
- `config.py` — environment variables and tunables (auto-loads `.env`)
- `utils.py` — timestamp parsing
- `pyproject.toml` / `uv.lock` — dependencies (managed with `uv`)
