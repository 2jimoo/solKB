from __future__ import annotations
import json
from typing import Any, Dict


class JsonlKB:
    """Append-only JSONL KB/logger."""

    def __init__(self, path: str):
        self.path = path

    def append(self, event: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
