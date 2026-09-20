"""Send a YouTube URL or local video to Gemini and get structured,
timestamped interview-prep notes. Uses agentic video understanding by default."""
import json
import logging
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    GEMINI_API_MODE,
    GEMINI_BATCH_POLL_SECONDS,
    GEMINI_MEDIA_PROCESSING,
    GEMINI_MODEL,
    GEMINI_REQUEST_TIMEOUT_MS,
    PIPELINE_CONCURRENCY,
)

logging.getLogger("google_genai").setLevel(logging.ERROR)

_client = None


def _get_client():
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=GEMINI_REQUEST_TIMEOUT_MS),
        )
    return _client


PROMPT = """You are a senior system-design interview coach. Watch the video \
(including whiteboard/diagram on screen and audio). Write interview-prep study \
notes for a candidate with ~1–3 years of experience.

Output JSON only. No markdown fences. No commentary outside JSON.

VOICE RULES:
- Use direct technical voice ("Decouple compute from storage because…").
- Forbidden phrases: "the instructor", "the speaker", "in this video", \
"kicks off", "sets the stage", "as mentioned", "the video shows".
- Bullets are the primary content. Keep "notes" empty unless you must add \
whiteboard labels, numbers, or multi-step flows bullets cannot carry.
- Bullets must NOT repeat each other or restate "notes".
- key_takeaways: max 5 items; each starts with a verb or principle.

CHECKPOINT TRIGGERS — add a checkpoint when:
- The whiteboard diagram changes (new/removed component, arrow, label), OR
- A new design decision, trade-off, or core concept deserves its own section.

MERMAID (required on most checkpoints):
- Include a crisp flowchart on every checkpoint where the whiteboard shows \
components, arrows, or data flow — even when capture_frame=true.
- Turn messy whiteboard sketches into clean, readable diagrams with \
meaningful node labels (Client, AppServer, Database, LoadBalancer — not A/B).
- Use flowchart TD or LR; show the exact topology at that moment (boxes + \
directed edges matching the whiteboard).
- Omit mermaid only for purely conceptual sections with no diagram \
(e.g. definitions, trade-off discussion with no new boxes).
- Screenshots and Mermaid complement each other: screenshot = raw frame; \
Mermaid = clean reference you can redraw in an interview.

SCREENSHOTS: capture_frame=true only when the whiteboard meaningfully changed \
vs the previous checkpoint.

SELF-CHECK: 1–2 short questions per checkpoint that test understanding (not \
video title recall).

GOOD checkpoint example:
{
  "timestamp": "02:11",
  "heading": "Client–server baseline",
  "concept": "A client sends requests; one app server handles them and returns data.",
  "why_it_matters": "Every system design interview starts here before you add scale.",
  "bullets": [
    "Client lives outside the data center; app server runs inside it.",
    "At low traffic, one server is enough — no need to over-engineer.",
    "Breaks when concurrent load or persistent state grows."
  ],
  "notes": "",
  "interview_line": "I'd start with a single app server and only split tiers when load or state forces it.",
  "tradeoff": "Simple to build, but cannot scale horizontally once user state lives on the server.",
  "self_check": ["What breaks first when you add a second app server with local state?"],
  "mermaid": "flowchart LR\\n  Client --> AppServer",
  "capture_frame": true,
  "capture_reason": "Initial client-server diagram"
}

BAD (do NOT write like this):
- "The instructor sets the stage using Facebook…"
- bullets that repeat the same facts as "notes"
- empty mermaid on checkpoints that show a whiteboard diagram
- generic nodes like A --> B with no labels

Return a JSON object:
  summary — 2–3 sentences; no meta about "this video series"
  key_takeaways — max 5 decision-shaped strings
  architecture_evolution — ordered list of architecture stages, e.g. \
"1. Single app server"
  glossary — array of {term, definition}; min 3 terms when the video \
introduces vocabulary
  checkpoints — array of checkpoint objects with:
    timestamp, heading, concept, why_it_matters, bullets (3–6 strings),
    notes (optional, often empty), interview_line, tradeoff (when a design \
choice exists), self_check (0–2 questions), mermaid (optional), \
capture_frame, capture_reason (if capture_frame)
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_takeaways": {
            "type": "array",
            "items": {"type": "string"},
        },
        "architecture_evolution": {
            "type": "array",
            "items": {"type": "string"},
        },
        "glossary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "definition": {"type": "string"},
                },
                "required": ["term", "definition"],
            },
        },
        "checkpoints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "heading": {"type": "string"},
                    "concept": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "notes": {"type": "string"},
                    "bullets": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "interview_line": {"type": "string"},
                    "tradeoff": {"type": "string"},
                    "self_check": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "mermaid": {"type": "string"},
                    "capture_frame": {"type": "boolean"},
                    "capture_reason": {"type": "string"},
                },
                "required": [
                    "timestamp",
                    "heading",
                    "concept",
                    "why_it_matters",
                    "bullets",
                    "capture_frame",
                ],
            },
        },
    },
    "required": [
        "summary",
        "key_takeaways",
        "architecture_evolution",
        "glossary",
        "checkpoints",
    ],
}


_BATCH_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
}


def _generation_config(service_tier=None):
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=_RESPONSE_SCHEMA,
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.HIGH,
        ),
    )
    if service_tier is not None:
        config.service_tier = service_tier
    return config


def _generation_config_dict(service_tier=None):
    config = {
        "response_mime_type": "application/json",
        "response_schema": _RESPONSE_SCHEMA,
        "thinking_config": {"thinking_level": "HIGH"},
    }
    if service_tier is not None:
        config["service_tier"] = (
            service_tier.value
            if isinstance(service_tier, types.ServiceTier)
            else str(service_tier).lower()
        )
    return config


def _media_processing_enum():
    if GEMINI_MEDIA_PROCESSING == "STATIC":
        return types.MediaProcessing.STATIC
    return types.MediaProcessing.AGENTIC


def _video_part(file_uri, mime_type="video/*"):
    return types.Part(
        file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
        media_processing=_media_processing_enum(),
    )


def _build_contents(file_uri, mime_type="video/*"):
    return [_video_part(file_uri, mime_type=mime_type), PROMPT]


def _build_batch_request(file_uri, mime_type="video/*", video_id=None):
    part = {
        "file_data": {
            "file_uri": file_uri,
            "mime_type": mime_type,
        },
        "media_processing": GEMINI_MEDIA_PROCESSING,
    }
    request = {
        "contents": [{
            "parts": [part, {"text": PROMPT}],
            "role": "user",
        }],
        "config": _generation_config_dict(),
    }
    if video_id:
        request["metadata"] = {"video_id": video_id}
    return request


def _response_text(response):
    if response is None:
        raise RuntimeError("Gemini returned an empty response.")
    text = getattr(response, "text", None)
    if text and str(text).strip():
        return text
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                return part_text
    raise RuntimeError("Gemini returned an empty response.")


def _strip_code_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lower().startswith("json"):
            text = text[text.find("\n") + 1:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _parse_notes(raw_text):
    raw = _strip_code_fences(raw_text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Gemini did not return valid JSON: {e}\n---\n{raw[:1000]}"
        )

    checkpoints = data.get("checkpoints") or []
    if not checkpoints:
        raise RuntimeError("Gemini returned no checkpoints for this video.")

    return {
        "summary": data.get("summary") or "",
        "key_takeaways": (data.get("key_takeaways") or [])[:5],
        "architecture_evolution": data.get("architecture_evolution") or [],
        "glossary": data.get("glossary") or [],
        "checkpoints": checkpoints,
    }


def _text_from_chunk(chunk):
    parts = []
    for candidate in getattr(chunk, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    return "".join(parts)


def _generate_standard(contents):
    """Stream via Chat API (recommended for agentic video AFC)."""
    config = _generation_config()
    client = _get_client()
    message = contents if isinstance(contents, list) else [contents]
    last_error = None

    for attempt in range(3):
        try:
            chat = client.chats.create(model=GEMINI_MODEL, config=config)
            print(
                "  Gemini standard tier — analyzing video "
                "(agentic mode — often takes a few minutes)...",
                flush=True,
            )
            parts = []
            steps = 0
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*non-text parts in the response.*",
                )
                for chunk in chat.send_message_stream(message):
                    steps += 1
                    if steps % 20 == 0:
                        print(f"  ... still analyzing ({steps} steps)", flush=True)
                    text = _text_from_chunk(chunk)
                    if text:
                        parts.append(text)
            raw = "".join(parts)
            if raw.strip():
                print(f"  Gemini finished ({steps} steps).", flush=True)
                return raw
            raise RuntimeError("Gemini returned an empty response.")
        except Exception as e:
            last_error = e
            msg = str(e)
            retryable = any(
                token in msg
                for token in ("500", "503", "UNKNOWN", "empty response")
            )
            if attempt < 2 and retryable:
                wait = 2 ** attempt
                print(f"  Gemini error, retrying in {wait}s...", flush=True)
                time.sleep(wait)
                continue
            raise
    raise last_error


def _generate_flex(contents):
    """Sync Flex tier — ~50% cheaper, ~1–15 min latency."""
    client = _get_client()
    config = _generation_config(service_tier=types.ServiceTier.FLEX)
    print(
        "  Gemini Flex tier (50% discount, ~1–15 min latency)...",
        flush=True,
    )
    last_error = None
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )
            return _response_text(response)
        except Exception as e:
            last_error = e
            msg = str(e)
            retryable = any(
                token in msg for token in ("500", "503", "UNKNOWN", "429")
            )
            if attempt < 4 and retryable:
                wait = min(2 ** attempt, 16)
                short = msg.split("\n", 1)[0][:120]
                print(
                    f"  Gemini Flex error ({short}), retrying in {wait}s...",
                    flush=True,
                )
                time.sleep(wait)
                continue
            raise
    raise last_error


def _poll_batch_job(batch_job):
    client = _get_client()
    state = getattr(batch_job, "state", None) or "UNKNOWN"
    print(
        f"  Batch job {batch_job.name} submitted "
        f"(50% discount, target turnaround up to 24h).",
        flush=True,
    )
    while state not in _BATCH_TERMINAL_STATES:
        print(f"  ... batch status: {state}", flush=True)
        time.sleep(GEMINI_BATCH_POLL_SECONDS)
        batch_job = client.batches.get(name=batch_job.name)
        state = getattr(batch_job, "state", None) or "UNKNOWN"

    if state != "JOB_STATE_SUCCEEDED":
        error = getattr(batch_job, "error", None)
        raise RuntimeError(
            f"Batch job ended with state {state}"
            + (f": {error}" if error else "")
        )
    return batch_job


def _generate_batch(file_uri, mime_type="video/*", video_id=None):
    """Async Batch API — ~50% cheaper, up to 24h turnaround."""
    client = _get_client()
    request = _build_batch_request(file_uri, mime_type, video_id=video_id)
    display_name = f"notes-{video_id or 'video'}"
    batch_job = client.batches.create(
        model=GEMINI_MODEL,
        src=[request],
        config={"display_name": display_name},
    )
    batch_job = _poll_batch_job(batch_job)

    dest = getattr(batch_job, "dest", None)
    responses = getattr(dest, "inlined_responses", None) if dest else None
    if not responses:
        raise RuntimeError("Batch job succeeded but returned no responses.")

    item = responses[0]
    if getattr(item, "error", None):
        err = item.error
        message = getattr(err, "message", None) or str(err)
        raise RuntimeError(f"Batch request failed: {message}")

    raw = _response_text(getattr(item, "response", None))
    print("  Batch request finished.", flush=True)
    return raw


def _generate(contents, file_uri=None, mime_type="video/*", video_id=None):
    mode = GEMINI_API_MODE
    if mode == "batch":
        if not file_uri:
            raise ValueError("Batch mode requires a file_uri")
        return _generate_batch(file_uri, mime_type, video_id=video_id)
    if mode == "flex":
        try:
            return _generate_flex(contents)
        except Exception as e:
            msg = str(e)
            if any(token in msg for token in ("503", "429", "UNAVAILABLE")):
                print(
                    "  Flex tier unavailable; falling back to standard tier...",
                    flush=True,
                )
                return _generate_standard(contents)
            raise
    return _generate_standard(contents)


def _run_generation(file_uri, mime_type="video/*", video_id=None):
    contents = _build_contents(file_uri, mime_type)
    raw = _generate(
        contents,
        file_uri=file_uri,
        mime_type=mime_type,
        video_id=video_id,
    )
    return _parse_notes(raw)


def _upload_local_video(local_video_path):
    client = _get_client()
    uploaded = client.files.upload(file=local_video_path)
    while uploaded.state.name == "PROCESSING":
        time.sleep(3)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini file upload failed: {uploaded.state.name}")
    return uploaded


def _delete_file(uploaded):
    if not uploaded or not getattr(uploaded, "name", None):
        return
    try:
        _get_client().files.delete(name=uploaded.name)
    except Exception:
        pass


def get_notes_from_youtube_url(youtube_url, video_id=None):
    return _run_generation(youtube_url, mime_type="video/*", video_id=video_id)


def get_notes_from_local_video(local_video_path, video_id=None):
    uploaded = _upload_local_video(local_video_path)
    try:
        mime = getattr(uploaded, "mime_type", None) or "video/mp4"
        return _run_generation(uploaded.uri, mime_type=mime, video_id=video_id)
    finally:
        _delete_file(uploaded)


def get_notes_for_video(youtube_url=None, local_video_path=None, video_id=None):
    if GEMINI_API_MODE == "batch":
        if youtube_url and not local_video_path:
            try:
                return get_notes_from_youtube_url(youtube_url, video_id), False
            except Exception as err:
                print(
                    f"  Batch YouTube URI failed ({err}); "
                    "downloading for file upload..."
                )
        if not local_video_path:
            raise ValueError("Batch mode requires a local video file")
        return get_notes_from_local_video(local_video_path, video_id), True

    if youtube_url and not local_video_path:
        try:
            return get_notes_from_youtube_url(youtube_url, video_id), False
        except Exception as first_err:
            print(
                f"  YouTube URI failed ({first_err}); "
                "need local video for file upload...",
                flush=True,
            )
            raise

    if youtube_url and local_video_path:
        try:
            return get_notes_from_youtube_url(youtube_url, video_id), False
        except Exception as e:
            print(f"  YouTube URI path failed ({e}); falling back to file upload...")

    if not local_video_path:
        raise ValueError("Need youtube_url and/or local_video_path")

    return get_notes_from_local_video(local_video_path, video_id), True


def prepare_batch_video(entry, video_dir):
    """Download + upload one video and return a Batch inline request."""
    import youtube_pipeline

    video_id = entry["id"]
    print(f"  [{video_id}] Downloading for batch...", flush=True)
    local_path, duration = youtube_pipeline.resolve_local_video(entry, video_dir)
    print(f"  [{video_id}] Uploading to Gemini Files API...", flush=True)
    uploaded = _upload_local_video(local_path)
    mime = getattr(uploaded, "mime_type", None) or "video/mp4"
    request = _build_batch_request(uploaded.uri, mime, video_id=video_id)
    return {
        "entry": entry,
        "local_path": local_path,
        "duration": duration,
        "uploaded": uploaded,
        "request": request,
    }


def create_batch_job(requests, display_name="youtube-notes"):
    """Submit a multi-video batch job and poll until it finishes."""
    client = _get_client()
    print(
        f"  Submitting batch job with {len(requests)} video(s) "
        f"(50% discount, processed concurrently by Google)...",
        flush=True,
    )
    batch_job = client.batches.create(
        model=GEMINI_MODEL,
        src=requests,
        config={"display_name": display_name},
    )
    return _poll_batch_job(batch_job)


def parse_batch_job_notes(batch_job):
    """Map video_id -> parsed notes or error string from a finished batch job."""
    dest = getattr(batch_job, "dest", None)
    responses = getattr(dest, "inlined_responses", None) if dest else None
    if not responses:
        raise RuntimeError("Batch job succeeded but returned no responses.")

    results = {}
    for item in responses:
        metadata = getattr(item, "metadata", None) or {}
        video_id = metadata.get("video_id")
        if getattr(item, "error", None):
            err = item.error
            message = getattr(err, "message", None) or str(err)
            if video_id:
                results[video_id] = {"error": message}
            continue
        try:
            raw = _response_text(getattr(item, "response", None))
            notes = _parse_notes(raw)
            if video_id:
                results[video_id] = {"notes": notes}
        except Exception as exc:
            if video_id:
                results[video_id] = {"error": str(exc)}
    return results


def run_batch_playlist(entries, video_dir, concurrency=None):
    """Prepare videos in parallel, submit one batch job, return prep in playlist order."""
    workers = concurrency or PIPELINE_CONCURRENCY
    prepared_by_id = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(prepare_batch_video, entry, video_dir): entry
            for entry in entries
        }
        for future in as_completed(futures):
            entry = futures[future]
            try:
                prepared_by_id[entry["id"]] = future.result()
            except Exception as exc:
                prepared_by_id[entry["id"]] = {"entry": entry, "error": str(exc)}

    prepared = [prepared_by_id[entry["id"]] for entry in entries]

    ok = [p for p in prepared if "request" in p]
    notes_by_id = {}
    if ok:
        batch_job = create_batch_job(
            [p["request"] for p in ok],
            display_name=f"notes-{len(ok)}-videos",
        )
        notes_by_id = parse_batch_job_notes(batch_job)

    for prep in prepared:
        video_id = prep["entry"]["id"]
        if "error" in prep and "request" not in prep:
            prep["notes_result"] = {"error": prep["error"]}
        elif video_id in notes_by_id:
            prep["notes_result"] = notes_by_id[video_id]
        else:
            prep["notes_result"] = {"error": "No batch response for this video"}

        if "uploaded" in prep:
            _delete_file(prep["uploaded"])

    return prepared
