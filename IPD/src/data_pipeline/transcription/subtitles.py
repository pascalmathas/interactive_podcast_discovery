"""Helper functions for writing WhisperX transcripts.

Extracted from ``file2transcript.py`` to keep that core logic concise.
"""

from __future__ import annotations

import json
from typing import Any, Dict

__all__ = [
    "save_transcripts",
]


def save_transcripts(
    result: Dict[str, Any],
    base_filename: str,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Write transcript result to a JSON file.

    Parameters
    ----------
    result
        WhisperX output conforming to the standard JSON schema.
    base_filename
        Path **without** extension where the JSON file will be saved.
    metadata
        Optional episode/file metadata to include in the JSON.
    """
    # --- JSON ---
    with open(f"{base_filename}.json", "w", encoding="utf-8") as fh:
        if metadata is None:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        else:
            json.dump({"metadata": metadata, "result": result}, fh, indent=2, ensure_ascii=False)

    print(f"Transcript saved: {base_filename}.json") 