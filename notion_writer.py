"""Minimal Notion API client: create pages, append blocks, and upload images
via Notion's Direct Upload flow."""
import os
import time


import requests

from config import NOTION_API_BASE, NOTION_API_KEY, NOTION_VERSION

_HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": NOTION_VERSION,
}


def _request(method, path, retries=4, **kwargs):
    url = f"{NOTION_API_BASE}{path}"
    extra_headers = kwargs.pop("headers", {})
    for attempt in range(retries):
        resp = requests.request(
            method, url, headers={**_HEADERS, **extra_headers}, **kwargs
        )
        if resp.status_code == 429:
            time.sleep(int(resp.headers.get("Retry-After", "2")))
            continue
        if resp.status_code >= 500 and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        if not resp.ok:
            raise RuntimeError(
                f"Notion API {method} {path} failed: {resp.status_code} {resp.text}"
            )
        return resp.json() if resp.text else {}
    raise RuntimeError(f"Notion API {method} {path} kept getting rate-limited")


def create_page(parent_page_id, title, children=None):
    """Create a page under parent_page_id. Notion caps children at 100 per
    request, so extra blocks are appended in follow-up calls."""
    payload = {
        "parent": {"page_id": parent_page_id},
        "properties": {"title": {"title": [{"text": {"content": title[:2000]}}]}},
    }
    if children:
        payload["children"] = children[:100]
    page = _request("POST", "/pages", json=payload)
    page_id = page["id"]
    if children and len(children) > 100:
        append_children(page_id, children[100:])
    return page_id


def append_children(block_id, children):
    for i in range(0, len(children), 100):
        batch = children[i:i + 100]
        _request("PATCH", f"/blocks/{block_id}/children", json={"children": batch})


def upload_image(file_path, content_type="image/png"):
    """Direct-upload an image to Notion and return a file_upload id that can
    be attached to an image block."""
    created = _request(
        "POST", "/file_uploads",
        json={"filename": os.path.basename(file_path), "content_type": content_type},
    )
    upload_id = created["id"]
    upload_url = created["upload_url"]
    with open(file_path, "rb") as f:
        resp = requests.post(
            upload_url,
            headers=_HEADERS,
            files={"file": (os.path.basename(file_path), f, content_type)},
        )
    if not resp.ok:
        raise RuntimeError(f"Notion file upload send failed: {resp.status_code} {resp.text}")
    return upload_id


def _rich_text(text):
    """Notion caps each rich_text item at 2000 chars; split longer content."""
    content = text or ""
    if not content:
        return [{"type": "text", "text": {"content": ""}}]
    chunks = []
    for i in range(0, len(content), 2000):
        chunks.append({
            "type": "text",
            "text": {"content": content[i:i + 2000]},
        })
    return chunks


# --- Block builders ---

def heading2(text):
    return {
        "object": "block", "type": "heading_2",
        "heading_2": {"rich_text": _rich_text(text)},
    }


def heading3(text):
    return {
        "object": "block", "type": "heading_3",
        "heading_3": {"rich_text": _rich_text(text)},
    }


def paragraph(text):
    return {
        "object": "block", "type": "paragraph",
        "paragraph": {"rich_text": _rich_text(text)},
    }


def bulleted_item(text):
    return {
        "object": "block", "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": _rich_text(text)},
    }


def numbered_item(text):
    return {
        "object": "block", "type": "numbered_list_item",
        "numbered_list_item": {"rich_text": _rich_text(text)},
    }


def toggle(title, children=None):
    block = {
        "object": "block",
        "type": "toggle",
        "toggle": {
            "rich_text": _rich_text(title),
        },
    }
    if children:
        block["toggle"]["children"] = children
    return block


def callout(text, emoji="💡"):
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": _rich_text(text),
            "icon": {"type": "emoji", "emoji": emoji},
        },
    }


def quote(text):
    return {
        "object": "block", "type": "quote",
        "quote": {"rich_text": _rich_text(text)},
    }


def mermaid_block(mermaid_source):
    """Notion renders Mermaid when language is set to mermaid on a code block."""
    source = (mermaid_source or "").strip()
    if source.startswith("```"):
        source = source.strip("`")
        if source.lower().startswith("mermaid"):
            source = source[len("mermaid"):].lstrip("\n")
    return {
        "object": "block",
        "type": "code",
        "code": {
            "rich_text": _rich_text(source),
            "language": "mermaid",
        },
    }


def image_block(file_upload_id, caption=None):
    block = {
        "object": "block", "type": "image",
        "image": {"type": "file_upload", "file_upload": {"id": file_upload_id}},
    }
    if caption:
        block["image"]["caption"] = _rich_text(caption)
    return block


def divider():
    return {"object": "block", "type": "divider", "divider": {}}
