from __future__ import annotations

import json
import os
import sys
import time
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


def emit_debug_log(
    topic: str,
    location: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    if not debug_topic_enabled(topic):
        return
    payload = {
        "timestamp_ms": int(time.time() * 1000),
        "pid": os.getpid(),
        "topic": topic,
        "location": location,
        "message": message,
        "data": data or {},
    }
    try:
        sys.stderr.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
    except Exception:
        pass
