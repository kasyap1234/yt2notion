"""Configuration and environment variables for the pipeline."""
import os
import shutil
from pathlib import Path

# Load .env from the project root when present (no extra dependency).
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.is_file():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value

# --- Required ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
NOTION_PARENT_PAGE_ID = os.environ.get("NOTION_PARENT_PAGE_ID")

# --- Optional, with sensible defaults ---
# Model names move fast -- check https://ai.google.dev/gemini-api/docs/models
# Agentic video understanding (media_processing=AGENTIC) is supported on
# Gemini 3.8 / 3.7 / 3.6 Flash and cuts token use sharply on long videos.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

# AGENTIC = model navigates the timeline and loads only what it needs.
# STATIC = fixed 1 FPS sweep (better TTFT on very short clips).
GEMINI_MEDIA_PROCESSING = os.environ.get(
    "GEMINI_MEDIA_PROCESSING", "AGENTIC"
).upper()

# Inference tier for Gemini calls (all offer ~50% savings vs standard):
#   batch    — async Batch API, up to 24h turnaround (bulk playlists)
#   flex     — sync Flex tier, ~1–15 min latency, best-effort (default)
#   standard — full-price chat/streaming (fastest, original behavior)
GEMINI_API_MODE = os.environ.get("GEMINI_API_MODE", "flex").lower()

# How often to poll a Batch job for completion.
GEMINI_BATCH_POLL_SECONDS = int(
    os.environ.get("GEMINI_BATCH_POLL_SECONDS", "30")
)

# Parallel workers for playlist runs (downloads, flex/standard Gemini calls).
PIPELINE_CONCURRENCY = max(1, int(os.environ.get("PIPELINE_CONCURRENCY", "1")))

# Max wait per Gemini request (ms). Prevents hung Flex calls from blocking forever.
GEMINI_REQUEST_TIMEOUT_MS = int(
    os.environ.get("GEMINI_REQUEST_TIMEOUT_MS", str(20 * 60 * 1000))
)

# Notion's Direct Upload (image) API needs a version that supports file
# uploads. Check https://developers.notion.com/reference/versioning if this
# stops working -- Notion revs this string periodically.
NOTION_VERSION = os.environ.get("NOTION_VERSION", "2026-03-11")
NOTION_API_BASE = "https://api.notion.com/v1"

# Max video height to download. 720p keeps storage/bandwidth reasonable
# while still being sharp enough to read whiteboard text in screenshots.
MAX_VIDEO_HEIGHT = int(os.environ.get("MAX_VIDEO_HEIGHT", "720"))

# When a moment is flagged for a screenshot, grab several frames in this
# window around the timestamp and keep the sharpest one. This absorbs
# Gemini's timestamp drift and avoids landing on a motion-blurred frame.
FRAME_WINDOW_SECONDS = float(os.environ.get("FRAME_WINDOW_SECONDS", "2.0"))
FRAME_STEP_SECONDS = float(os.environ.get("FRAME_STEP_SECONDS", "0.5"))

WORKDIR = os.environ.get("PIPELINE_WORKDIR", "./pipeline_data")
PROGRESS_FILE = os.path.join(WORKDIR, "progress.json")


def validate():
    missing = [
        name
        for name, val in [
            ("GEMINI_API_KEY", GEMINI_API_KEY),
            ("NOTION_API_KEY", NOTION_API_KEY),
            ("NOTION_PARENT_PAGE_ID", NOTION_PARENT_PAGE_ID),
        ]
        if not val
    ]
    if missing:
        raise SystemExit(
            "Missing required environment variables: " + ", ".join(missing)
        )
    if GEMINI_MEDIA_PROCESSING not in ("AGENTIC", "STATIC"):
        raise SystemExit(
            "GEMINI_MEDIA_PROCESSING must be AGENTIC or STATIC, "
            f"got {GEMINI_MEDIA_PROCESSING!r}"
        )
    if GEMINI_API_MODE not in ("standard", "flex", "batch"):
        raise SystemExit(
            "GEMINI_API_MODE must be standard, flex, or batch, "
            f"got {GEMINI_API_MODE!r}"
        )
    if not shutil.which("ffmpeg"):
        raise SystemExit(
            "ffmpeg is required for video merge and screenshot frames but was "
            "not found on PATH. Install it with: brew install ffmpeg"
        )
