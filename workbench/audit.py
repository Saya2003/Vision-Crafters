from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from workbench.config import AUDIT_JSONL_PATH


def append_audit(payload: dict[str, Any]) -> None:
    if not AUDIT_JSONL_PATH:
        return

    row = {"ts_iso": datetime.now(timezone.utc).isoformat(), **payload}
    path = AUDIT_JSONL_PATH

    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str, ensure_ascii=False))
        fh.write("\n")
