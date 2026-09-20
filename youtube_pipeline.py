"""Helpers for listing a YouTube playlist or single video, downloading
videos, and pulling sharp frames out of them with ffmpeg."""
import json
import os
import re
import shutil
import subprocess
import tempfile
from urllib.parse import parse_qs, urlparse

import cv2
import requests

from config import FRAME_STEP_SECONDS, FRAME_WINDOW_SECONDS, MAX_VIDEO_HEIGHT

_WATCH_HOSTS = {"www.youtube.com", "youtube.com", "m.youtube.com", "music.youtube.com"}
_SHORT_HOSTS = {"youtu.be"}


def _yt_dlp_cmd(*args):
    """Build a yt-dlp command, enabling Node as JS runtime when available."""
    cmd = ["yt-dlp", "--no-warnings", "--socket-timeout", "15"]
    if shutil.which("node"):
        cmd.extend(["--js-runtimes", "node"])
    cmd.extend(args)
    return cmd


def _fetch_title_oembed(watch_url, timeout=5):
    """Fast title lookup via YouTube oEmbed (~300ms). No JS runtime needed."""
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": watch_url, "format": "json"},
            timeout=timeout,
        )
        if resp.ok:
            title = resp.json().get("title")
            if title:
                return title.strip()
    except requests.RequestException:
        pass
    return None


def enrich_entry_title(entry):
    """Fill in a human title when resolve only had the video id."""
    if entry.get("title") and entry["title"] != entry["id"]:
        return entry
    title = _fetch_title_oembed(entry["url"])
    if title:
        return {**entry, "title": title}
    return entry


def canonical_watch_url(video_id):
    return f"https://www.youtube.com/watch?v={video_id}"


def is_playlist_url(url):
    """True for playlist pages. watch?v=...&list=... counts as a single video."""
    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/")
    qs = parse_qs(parsed.query)
    if path.endswith("/playlist") and "list" in qs:
        return True
    if "list" in qs and "v" not in qs and _extract_video_id(url) is None:
        return True
    return False


def _extract_video_id(url):
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if host in _SHORT_HOSTS:
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid or None
    if host in _WATCH_HOSTS or host.endswith(".youtube.com"):
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
        # /shorts/ID or /embed/ID or /live/ID
        m = re.match(r"^/(?:shorts|embed|live)/([^/?#]+)", parsed.path or "")
        if m:
            return m.group(1)
    return None


def resolve_youtube_url(url):
    """Return (title, [ {id, title, url}, ... ]) for a playlist or single video.

    Does not download any video content.
    """
    if is_playlist_url(url):
        return list_playlist(url)
    return get_single_video(url)


def list_playlist(playlist_url):
    """Return (playlist_title, [ {id, title, url}, ... ]) without downloading
    any video content."""
    result = subprocess.run(
        _yt_dlp_cmd("--flat-playlist", "-J", playlist_url),
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    entries = []
    for entry in data.get("entries", []):
        if not entry:
            continue
        video_id = entry.get("id")
        entries.append({
            "id": video_id,
            "title": entry.get("title") or video_id,
            "url": canonical_watch_url(video_id),
        })
    playlist_title = data.get("title") or "YouTube playlist"
    return playlist_title, entries


def get_single_video(video_url):
    """Return (title, [single entry]) for a watch / youtu.be / shorts URL.

    Parses the video id locally (instant). Title is filled via oEmbed in
    enrich_entry_title() before processing — avoids a slow yt-dlp metadata
    round-trip at resolve time.
    """
    video_id = _extract_video_id(video_url)
    if not video_id:
        raise ValueError(f"Could not extract a video ID from URL: {video_url}")

    watch_url = canonical_watch_url(video_id)
    return video_id, [{"id": video_id, "title": video_id, "url": watch_url}]


def find_local_video(video_id, video_dir):
    """Return path to a previously downloaded mp4 if still on disk."""
    direct = os.path.join(video_dir, f"{video_id}.mp4")
    if os.path.isfile(direct):
        return direct
    if os.path.isdir(video_dir):
        for fname in os.listdir(video_dir):
            if fname.startswith(video_id) and fname.endswith(".mp4"):
                return os.path.join(video_dir, fname)
    return None


def local_video_duration(local_path):
    """Best-effort duration in seconds for an existing mp4."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                local_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(float(result.stdout.strip()))
    except (subprocess.CalledProcessError, ValueError, OSError):
        return None


def resolve_local_video(entry, video_dir):
    """Use cached download if present; otherwise download from YouTube."""
    cached = find_local_video(entry["id"], video_dir)
    if cached:
        duration = local_video_duration(cached)
        return cached, duration or 0
    return download_video(entry["url"], video_dir)


def download_video(video_url, out_dir):
    """Download a single video (video+audio, capped resolution) and return
    (local_path, duration_seconds)."""
    video_id = _extract_video_id(video_url)
    if video_id:
        video_url = canonical_watch_url(video_id)

    os.makedirs(out_dir, exist_ok=True)
    out_template = os.path.join(out_dir, "%(id)s.%(ext)s")
    fmt = f"bestvideo[height<={MAX_VIDEO_HEIGHT}]+bestaudio/best[height<={MAX_VIDEO_HEIGHT}]"

    subprocess.run(
        _yt_dlp_cmd(
            "-f", fmt, "--merge-output-format", "mp4",
            "-o", out_template, video_url,
        ),
        check=True,
    )

    info = subprocess.run(
        _yt_dlp_cmd("-J", "--skip-download", video_url),
        capture_output=True, text=True, check=True,
    )
    meta = json.loads(info.stdout)
    video_id = meta["id"]
    duration = meta.get("duration", 0)

    local_path = os.path.join(out_dir, f"{video_id}.mp4")
    if not os.path.exists(local_path):
        # yt-dlp may have picked a different container; find whatever it made.
        for fname in os.listdir(out_dir):
            if fname.startswith(video_id):
                local_path = os.path.join(out_dir, fname)
                break
    return local_path, duration


def _ffmpeg_path():
    return shutil.which("ffmpeg") or "ffmpeg"


def _extract_frame(video_path, timestamp_s, out_path):
    """Grab a single frame at timestamp_s with ffmpeg. Returns True on success."""
    result = subprocess.run(
        [
            _ffmpeg_path(), "-y", "-ss", f"{max(timestamp_s, 0):.3f}",
            "-i", video_path, "-frames:v", "1", "-q:v", "2", out_path,
        ],
        capture_output=True,
    )
    return result.returncode == 0 and os.path.exists(out_path)


def _sharpness(image_path):
    """Higher = sharper (variance of the Laplacian). Used to pick the best
    candidate frame out of a small burst."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0
    return cv2.Laplacian(img, cv2.CV_64F).var()


def grab_best_frame(video_path, timestamp_s, out_path, duration=None):
    """Extract several candidate frames around timestamp_s and keep the
    sharpest. This absorbs Gemini's timestamp drift (the flagged moment can
    be a second or two off) and skips motion-blurred grabs. Returns True if a
    frame was written to out_path."""
    offsets = []
    t = -FRAME_WINDOW_SECONDS
    while t <= FRAME_WINDOW_SECONDS + 1e-6:
        offsets.append(t)
        t += FRAME_STEP_SECONDS

    best_path, best_score = None, -1.0
    with tempfile.TemporaryDirectory() as tmp:
        for i, offset in enumerate(offsets):
            candidate_ts = timestamp_s + offset
            if duration is not None and candidate_ts > duration:
                continue
            candidate_path = os.path.join(tmp, f"candidate_{i}.png")
            if _extract_frame(video_path, candidate_ts, candidate_path):
                score = _sharpness(candidate_path)
                if score > best_score:
                    best_score, best_path = score, candidate_path

        if best_path is None:
            return False

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        shutil.move(best_path, out_path)
    return True
