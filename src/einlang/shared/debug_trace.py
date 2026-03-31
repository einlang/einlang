from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


_TRUTHY = {"1", "true", "yes", "on", "debug", "verbose", "all", "*"}


def _is_truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in _TRUTHY


def _topic_tokens() -> list[str]:
    raw = (os.environ.get("EINLANG_DEBUG_TOPICS") or "").strip().lower()
    if not raw:
        return []
    return [token.strip() for token in raw.replace(";", ",").split(",") if token.strip()]


def debug_topic_enabled(topic: str) -> bool:
    topic_norm = (topic or "").strip().lower()
    if _is_truthy(os.environ.get("EINLANG_DEBUG_MODE")) or _is_truthy(os.environ.get("EINLANG_DEBUG_ALL")):
        return True
    if _is_truthy(os.environ.get("EINLANG_DEBUG_AUTODIFF")) and topic_norm.startswith(
        ("autodiff", "runtime.quotient")
    ):
        return True
    for token in _topic_tokens():
        if token in ("all", "*"):
            return True
        if topic_norm == token or topic_norm.startswith(token + "."):
            return True
    return False


def debug_log_path() -> Path:
    configured = (os.environ.get("EINLANG_DEBUG_LOG_FILE") or "").strip()
    if configured:
        return Path(configured).expanduser()
    debug_dir = (os.environ.get("EINLANG_DEBUG_DIR") or "").strip()
    if debug_dir:
        return Path(debug_dir).expanduser() / "einlang-debug.ndjson"
    return Path.cwd() / ".cursor" / "einlang-debug.ndjson"


def emit_debug_log(
    topic: str,
    location: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    if not debug_topic_enabled(topic):
        return
    path = debug_log_path()
    payload = {
        "timestamp_ms": int(time.time() * 1000),
        "pid": os.getpid(),
        "topic": topic,
        "location": location,
        "message": message,
        "data": data or {},
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
    except Exception:
        pass
