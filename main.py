"""Turn a YouTube playlist or single video into illustrated Notion pages.

Usage:
    # .env in project root is loaded automatically (see README)
    uv run main.py "https://www.youtube.com/playlist?list=..."
    uv run main.py "https://www.youtube.com/watch?v=..."
    uv run main.py --force "https://www.youtube.com/watch?v=..."

Safe to re-run: progress is saved to pipeline_data/progress.json, so an
interrupted run picks up where it left off instead of reprocessing videos
(and re-spending Gemini calls) that already succeeded. Use --force to
regenerate a video that is already marked done (creates a new Notion page).
"""
import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import notion_writer
import youtube_pipeline
from gemini_notes import get_notes_for_video, run_batch_playlist
from utils import timestamp_to_seconds

_progress_lock = threading.Lock()


def _load_progress():
    if os.path.exists(config.PROGRESS_FILE):
        with open(config.PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def _save_progress(progress):
    os.makedirs(config.WORKDIR, exist_ok=True)
    with open(config.PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def _update_video_progress(progress, video_id, data):
    with _progress_lock:
        progress.setdefault("videos", {})[video_id] = data
        _save_progress(progress)


def _needs_frames(notes):
    return any(cp.get("capture_frame") for cp in notes.get("checkpoints", []))


def _strip_leading_number(text):
    return re.sub(r"^\d+\.\s*", "", str(text).strip())


def _build_page_children(notes, entry, video_dir, local_path, duration):
    """Assemble interview-prep Notion blocks from structured notes + frames."""
    children = []

    summary = (notes.get("summary") or "").strip()
    if summary:
        children.append(notion_writer.callout(summary, emoji="📝"))

    evolution = [
        stage.strip()
        for stage in (notes.get("architecture_evolution") or [])
        if stage and str(stage).strip()
    ]
    if evolution:
        children.append(notion_writer.heading2("Architecture evolution"))
        for stage in evolution:
            children.append(
                notion_writer.numbered_item(_strip_leading_number(stage))
            )
        children.append(notion_writer.divider())

    takeaways = (notes.get("key_takeaways") or [])[:5]
    if takeaways:
        children.append(notion_writer.heading2("Key takeaways"))
        for item in takeaways:
            if item and str(item).strip():
                children.append(notion_writer.bulleted_item(item))
        children.append(notion_writer.divider())

    glossary = notes.get("glossary") or []
    glossary_items = [
        g for g in glossary
        if g.get("term") and g.get("definition")
    ]
    if glossary_items:
        glossary_children = [
            notion_writer.bulleted_item(
                f"{g['term']}: {g['definition']}"
            )
            for g in glossary_items
        ]
        children.append(
            notion_writer.toggle("Glossary", glossary_children)
        )
        children.append(notion_writer.divider())

    frames_dir = os.path.join(video_dir, "frames")
    for i, cp in enumerate(notes.get("checkpoints", [])):
        children.append(
            notion_writer.heading2(cp.get("heading") or f"Checkpoint {i + 1}")
        )

        ts = (cp.get("timestamp") or "").strip()
        if ts:
            children.append(notion_writer.paragraph(ts))

        concept = (cp.get("concept") or "").strip()
        why = (cp.get("why_it_matters") or "").strip()
        if concept or why:
            callout_parts = []
            if concept:
                callout_parts.append(f"Concept: {concept}")
            if why:
                callout_parts.append(f"Why it matters: {why}")
            children.append(
                notion_writer.callout("\n".join(callout_parts), emoji="💡")
            )

        for bullet in cp.get("bullets") or []:
            if bullet and str(bullet).strip():
                children.append(notion_writer.bulleted_item(bullet))

        notes_text = (cp.get("notes") or "").strip()
        if notes_text:
            children.append(notion_writer.paragraph(notes_text))

        interview_line = (cp.get("interview_line") or "").strip()
        if interview_line:
            children.append(
                notion_writer.quote(f"Interview line: {interview_line}")
            )

        tradeoff = (cp.get("tradeoff") or "").strip()
        if tradeoff:
            children.append(notion_writer.quote(f"Trade-off: {tradeoff}"))

        mermaid = cp.get("mermaid")
        if mermaid and str(mermaid).strip():
            children.append(notion_writer.mermaid_block(mermaid))

        if cp.get("capture_frame") and local_path:
            ts_seconds = timestamp_to_seconds(cp.get("timestamp", "0:00"))
            frame_path = os.path.join(frames_dir, f"{entry['id']}_{i}.png")
            try:
                got_frame = youtube_pipeline.grab_best_frame(
                    local_path, ts_seconds, frame_path, duration=duration
                )
            except FileNotFoundError as e:
                print(f"    [{entry['id']}] Screenshot skipped: {e}", flush=True)
                got_frame = False
            if got_frame:
                print(
                    f"    [{entry['id']}] Uploading screenshot {i + 1}...",
                    flush=True,
                )
                upload_id = notion_writer.upload_image(frame_path)
                children.append(
                    notion_writer.image_block(
                        upload_id, caption=cp.get("capture_reason")
                    )
                )

        self_check = [
            q.strip()
            for q in (cp.get("self_check") or [])
            if q and str(q).strip()
        ]
        if self_check:
            quiz_children = [
                notion_writer.bulleted_item(q) for q in self_check
            ]
            children.append(
                notion_writer.toggle("Check yourself", quiz_children)
            )

        children.append(notion_writer.divider())

    return children


def _cleanup_local_video(local_path, video_dir=None):
    """Remove a one-off download; keep files in the shared video_tmp cache."""
    if not local_path:
        return
    if video_dir:
        cache_dir = os.path.abspath(video_dir)
        if os.path.abspath(os.path.dirname(local_path)) == cache_dir:
            return
    try:
        os.remove(local_path)
    except OSError:
        pass


def write_notion_page(entry, notes, course_page_id, video_dir, local_path, duration):
    """Build blocks and create the Notion page from structured notes."""
    if _needs_frames(notes) and local_path is None:
        print(f"  [{entry['id']}] Resolving local video for screenshots...", flush=True)
        local_path, duration = youtube_pipeline.resolve_local_video(
            entry, video_dir
        )

    children = _build_page_children(notes, entry, video_dir, local_path, duration)
    print(f"  [{entry['id']}] Writing Notion page...", flush=True)
    return notion_writer.create_page(course_page_id, entry["title"], children)


def fetch_notes_for_entry(entry, video_dir):
    """Run Gemini for one video. Returns (notes, local_path, duration)."""
    local_path, duration = None, None
    video_id = entry["id"]

    print(f"  [{video_id}] Asking Gemini to watch it and write notes...", flush=True)
    if config.GEMINI_API_MODE == "batch":
        print(f"  [{video_id}] Downloading for Batch API...", flush=True)
        local_path, duration = youtube_pipeline.resolve_local_video(
            entry, video_dir
        )
        notes, _ = get_notes_for_video(
            local_video_path=local_path,
            video_id=video_id,
        )
    else:
        local_path = youtube_pipeline.find_local_video(video_id, video_dir)
        if local_path:
            duration = youtube_pipeline.local_video_duration(local_path) or 0
            print(
                f"  [{video_id}] Using cached mp4 (skipping YouTube URI).",
                flush=True,
            )
            notes, _ = get_notes_for_video(
                local_video_path=local_path,
                video_id=video_id,
            )
        else:
            try:
                notes, _ = get_notes_for_video(
                    youtube_url=entry["url"],
                    video_id=video_id,
                )
            except Exception as first_err:
                print(
                    f"  [{video_id}] YouTube URI failed ({first_err}); "
                    "resolving local video...",
                    flush=True,
                )
                local_path, duration = youtube_pipeline.resolve_local_video(
                    entry, video_dir
                )
                notes, _ = get_notes_for_video(
                    local_video_path=local_path,
                    video_id=video_id,
                )

    checkpoints = notes.get("checkpoints", [])
    print(f"  [{video_id}] Got {len(checkpoints)} checkpoints.", flush=True)
    return notes, local_path, duration


def process_video(entry, course_page_id, video_dir):
    notes, local_path, duration = fetch_notes_for_entry(entry, video_dir)
    print(
        f"  [{entry['id']}] Building Notion page...",
        flush=True,
    )
    page_id = write_notion_page(
        entry, notes, course_page_id, video_dir, local_path, duration
    )
    _cleanup_local_video(local_path, video_dir)
    return page_id


def _handle_video_success(entry, page_id, progress, prior, force):
    video_id = entry["id"]
    _update_video_progress(
        progress,
        video_id,
        {"status": "done", "notion_page_id": page_id},
    )
    print(f"  [{video_id}] Notion page: {page_id}", flush=True)
    if force and prior.get("notion_page_id") and prior["notion_page_id"] != page_id:
        print(
            f"  [{video_id}] Old Notion page (kept): {prior['notion_page_id']}",
            flush=True,
        )


def _handle_video_failure(entry, exc, progress):
    video_id = entry["id"]
    print(f"  [{video_id}] FAILED: {exc}", flush=True)
    _update_video_progress(
        progress,
        video_id,
        {"status": "failed", "error": str(exc)},
    )


def _process_one(entry, course_page_id, video_dir, progress, prior, force):
    print(f"\nProcessing: {entry['title']}", flush=True)
    if prior.get("status") == "done" and force and prior.get("notion_page_id"):
        print(f"  Previous Notion page: {prior['notion_page_id']}", flush=True)
    page_id = process_video(entry, course_page_id, video_dir)
    _handle_video_success(entry, page_id, progress, prior, force)


def _process_batch_playlist(pending, course_page_id, video_dir, progress, force, concurrency):
    print(
        f"\nBatch mode: preparing {len(pending)} video(s) with "
        f"{concurrency} parallel workers, then one multi-video batch job.",
        flush=True,
    )
    prepared = run_batch_playlist(pending, video_dir, concurrency)

    for prep in prepared:
        entry = prep["entry"]
        video_id = entry["id"]
        prior = progress.get("videos", {}).get(video_id, {})
        result = prep.get("notes_result", {})
        local_path = prep.get("local_path")
        duration = prep.get("duration")
        try:
            if result.get("error"):
                raise RuntimeError(result["error"])
            notes = result["notes"]
            print(f"\nWriting Notion page: {entry['title']}", flush=True)
            page_id = write_notion_page(
                entry, notes, course_page_id, video_dir, local_path, duration
            )
            _handle_video_success(entry, page_id, progress, prior, force)
        except Exception as exc:
            _handle_video_failure(entry, exc, progress)
        finally:
            _cleanup_local_video(local_path, video_dir)


def _process_concurrent(pending, course_page_id, video_dir, progress, force, concurrency):
    print(
        f"\nFetching notes for {len(pending)} video(s) with Flex "
        f"({concurrency} workers); Notion pages written in playlist order "
        f"as each slot completes.",
        flush=True,
    )
    results = {}
    results_lock = threading.Lock()
    write_lock = threading.Lock()
    next_index = 0

    def try_write_ready():
        nonlocal next_index
        with write_lock:
            while next_index < len(pending):
                entry = pending[next_index]
                video_id = entry["id"]
                if video_id not in results:
                    break
                status, payload = results.pop(video_id)
                prior = progress.get("videos", {}).get(video_id, {})
                local_path = None
                try:
                    if status == "err":
                        raise payload
                    notes, local_path, duration = payload
                    print(f"\nWriting Notion page: {entry['title']}", flush=True)
                    page_id = write_notion_page(
                        entry, notes, course_page_id, video_dir, local_path, duration
                    )
                    _handle_video_success(entry, page_id, progress, prior, force)
                except Exception as exc:
                    _handle_video_failure(entry, exc, progress)
                finally:
                    if status == "ok":
                        _cleanup_local_video(local_path, video_dir)
                next_index += 1

    def fetch_one(entry):
        try:
            payload = fetch_notes_for_entry(entry, video_dir)
            with results_lock:
                results[entry["id"]] = ("ok", payload)
            try_write_ready()
        except Exception as exc:
            with results_lock:
                results[entry["id"]] = ("err", exc)
            try_write_ready()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(fetch_one, entry) for entry in pending]
        for future in as_completed(futures):
            future.result()


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Turn a YouTube playlist or video into Notion study notes."
    )
    parser.add_argument(
        "url",
        help="YouTube playlist or video URL",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-process videos even if already marked done in progress.json. "
            "Creates a new Notion page (does not overwrite the old one)."
        ),
    )
    parser.add_argument(
        "--api-mode",
        choices=["standard", "flex", "batch"],
        help=(
            "Gemini inference tier: batch (50%% off, async up to 24h), "
            "flex (50%% off, sync ~1-15 min), or standard (full price). "
            "Default: GEMINI_API_MODE env or batch."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help=(
            "Parallel workers for playlist runs (downloads + Gemini calls). "
            "Default: PIPELINE_CONCURRENCY env or 4."
        ),
    )
    return parser.parse_args(argv)


def main():
    args = _parse_args(sys.argv[1:])

    config.validate()
    source_url = args.url
    force = args.force
    if args.api_mode:
        config.GEMINI_API_MODE = args.api_mode
    concurrency = max(1, args.concurrency or config.PIPELINE_CONCURRENCY)

    print("Resolving YouTube URL...")
    course_title, entries = youtube_pipeline.resolve_youtube_url(source_url)
    kind = "playlist" if youtube_pipeline.is_playlist_url(source_url) else "video"

    if kind == "video" and entries:
        print("  Fetching title...")
        entries = [youtube_pipeline.enrich_entry_title(e) for e in entries]
        course_title = entries[0]["title"]

    print(f"Found {len(entries)} video(s) ({kind}): '{course_title}'")
    print(
        f"Gemini model={config.GEMINI_MODEL} "
        f"media_processing={config.GEMINI_MEDIA_PROCESSING} "
        f"api_mode={config.GEMINI_API_MODE} "
        f"concurrency={concurrency}"
    )
    if force:
        print("Force mode: will re-process videos already marked done.")

    progress = _load_progress()

    if "course_page_id" not in progress:
        progress["course_page_id"] = notion_writer.create_page(
            config.NOTION_PARENT_PAGE_ID, course_title
        )
        progress.setdefault("videos", {})
        _save_progress(progress)
    course_page_id = progress["course_page_id"]
    progress.setdefault("videos", {})

    video_dir = os.path.join(config.WORKDIR, "video_tmp")

    pending = []
    for entry in entries:
        video_id = entry["id"]
        prior = progress["videos"].get(video_id, {})
        if prior.get("status") == "done" and not force:
            print(f"Skipping (already done): {entry['title']}")
            continue
        pending.append(entry)

    if not pending:
        print("\nNothing to do.")
        return

    use_batch_playlist = (
        config.GEMINI_API_MODE == "batch"
        and len(pending) > 1
    )
    use_concurrent = concurrency > 1 and len(pending) > 1 and not use_batch_playlist

    if use_batch_playlist:
        _process_batch_playlist(
            pending, course_page_id, video_dir, progress, force, concurrency
        )
    elif use_concurrent:
        _process_concurrent(
            pending, course_page_id, video_dir, progress, force, concurrency
        )
    else:
        for entry in pending:
            video_id = entry["id"]
            prior = progress["videos"].get(video_id, {})
            try:
                _process_one(
                    entry, course_page_id, video_dir, progress, prior, force
                )
            except Exception as exc:
                _handle_video_failure(entry, exc, progress)

    print("\nDone. Progress saved to", config.PROGRESS_FILE)


if __name__ == "__main__":
    main()
