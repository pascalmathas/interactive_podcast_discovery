"""Utilities for downloading audio files from RSS feeds."""

from __future__ import annotations

import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

__all__ = [
    "sanitize_for_path",
    "download_audio_entry",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_for_path(text: str) -> str:
    """Make a string safe as a file or directory name."""
    s = str(text)
    s = re.sub(r"\s+", "_", s)  # Replace whitespace with underscores
    s = re.sub(r"[^a-zA-Z0-9_-]", "", s)  # Remove invalid characters
    s = re.sub(r"__+", "_", s)  # Replace multiple underscores
    s = s.strip("_-")  # Remove leading/trailing underscores
    return s or "unnamed_episode"


def _get_head_response(url: str) -> Optional[requests.Response]:
    """Perform a HEAD request to retrieve headers, with error handling."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        print(f"Warning: HEAD request to {url} failed: {e}")
    except Exception as e:
        print(f"Unexpected error during HEAD request to {url}: {e}")
    return None


def _get_audio_enclosure(entry: Dict[str, Any]) -> Optional[str]:
    """Extract the URL of the first audio enclosure from an RSS entry."""
    audio_enclosures = [
        enc
        for enc in entry.get("enclosures", [])
        if "audio" in enc.get("type", "")
    ]
    return audio_enclosures[0].get("href") if audio_enclosures else None


def _derive_filename(entry: Dict[str, Any], episode_url: str) -> str:
    """Derive a safe and informative filename for an episode."""
    head_resp = _get_head_response(episode_url)
    original_filename = ""

    # Try to get filename from Content-Disposition header
    if head_resp:
        cd = head_resp.headers.get("content-disposition")
        if cd:
            fname_match = re.findall('filename="?([^";]+)"?', cd)
            if fname_match:
                original_filename = fname_match[0]

    # Fallback to URL if header doesn't provide filename
    if not original_filename:
        original_filename = os.path.basename(episode_url.split("?", 1)[0])

    _, ext = os.path.splitext(original_filename)

    # Try to guess extension from MIME type if it's missing
    if not ext and head_resp:
        mime_type = head_resp.headers.get("content-type")
        if mime_type:
            ext = mimetypes.guess_extension(mime_type.split(";", 1)[0]) or ""

    # Final fallback for extension
    if not ext:
        ext = ".mp3"

    # Determine base name: this should be based on the episode title.
    base_name_part = sanitize_for_path(entry.get("title") or "unnamed_episode")

    # Add publication date to the filename
    date_prefix = ""
    if "published_parsed" in entry and entry["published_parsed"]:
        try:
            date_prefix = time.strftime("%Y-%m-%d", entry["published_parsed"])
        except (ValueError, TypeError):
            pass  # Ignore if parsing fails

    return f"{date_prefix}_{base_name_part}{ext}" if date_prefix else f"{base_name_part}{ext}"


def _download_file(url: str, filepath: Path, title: Optional[str]):
    """Download a file if it doesn't already exist."""
    if filepath.exists():
        print(f"File already exists, download skipped: {filepath}")
        return

    print(f"Downloading episode: {title or 'No title'} -> {filepath}")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download completed.")
    except requests.RequestException as e:
        print(f"Error downloading from {url}: {e}")
        if filepath.exists():
            filepath.unlink()  # Remove incomplete file
    except Exception as e:
        print(f"Unexpected error during download: {e}")
        if filepath.exists():
            filepath.unlink()


def _create_metadata(entry: Dict[str, Any], feed_url: str, episode_url: str) -> Dict[str, Any]:
    """Assemble a metadata dictionary for the downloaded episode."""
    return {
        "title": entry.get("title"),
        "link": entry.get("link"),
        "guid": entry.get("id"),
        "description": entry.get("description"),
        "published": entry.get("published"),
        "date": entry.get("published"),
        "summary": entry.get("summary"),
        "tags": [t.get("term") for t in entry.get("tags", [])] if entry.get("tags") else None,
        "rss_feed_url": feed_url,
        "audio_url": episode_url,
    }


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

def download_audio_entry(entry: Dict[str, Any], feed_url: str, output_dir: str) -> Optional[Dict[str, Any]]:
    """Download the audio enclosure from one RSS entry.

    Parameters
    ----------
    entry
        An RSS item as returned by *feedparser*.
    feed_url
        The original RSS feed URL (for metadata).
    output_dir
        Target directory.

    Returns
    -------
    dict | None
        ``{"filepath": str, "metadata": dict}`` or *None* if there's no audio.
    """
    episode_url = _get_audio_enclosure(entry)
    if not episode_url:
        return None

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = _derive_filename(entry, episode_url)
    filepath = output_path / filename

    _download_file(episode_url, filepath, entry.get("title", "Unnamed episode"))

    if not filepath.exists():
        return None  # Download failed

    metadata = _create_metadata(entry, feed_url, episode_url)

    return {"filepath": str(filepath), "metadata": metadata} 