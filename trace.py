"""JSONL event trace."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


class JsonlTrace:
    def __init__(self, path: str, *, echo: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo

    def emit(self, event: str, **data: object) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.echo:
            visible = {
                key: value
                for key, value in data.items()
                if key not in {"message", "reply"}
            }
            suffix = f" {json.dumps(visible, ensure_ascii=False)}" if visible else ""
            print(f"[{event}]{suffix}", file=sys.stderr)
