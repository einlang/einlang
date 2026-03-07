"""Python bridge helpers for Whisper-tiny demo.

Called from main.ein via  python::whisper_helpers::decode_tokens
"""

import os
import json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _build_byte_decoder():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _build_byte_decoder()
_ID2TOK = None


def _load_tokenizer():
    global _ID2TOK
    if _ID2TOK is not None:
        return
    tok_path = os.path.join(_SCRIPT_DIR, "tokenizer.json")
    with open(tok_path) as f:
        data = json.load(f)
    _ID2TOK = {}
    for k, v in data.get("model", {}).get("vocab", {}).items():
        _ID2TOK[v] = k
    for entry in data.get("added_tokens", []):
        _ID2TOK[entry["id"]] = entry["content"]


def decode_tokens(token_ids):
    """Decode a list/array of Whisper token IDs to a UTF-8 string.

    Strips special tokens (SOT, EOT, language, task, timestamps).
    """
    _load_tokenizer()

    if isinstance(token_ids, np.ndarray):
        token_ids = token_ids.flatten().tolist()
    token_ids = [int(t) for t in token_ids]

    EOT = 50257
    SPECIAL = {50257, 50258, 50259, 50359, 50363}
    pieces = []
    for tid in token_ids:
        if tid == EOT:
            break
        if tid in SPECIAL or tid >= 50364:
            continue
        pieces.append(_ID2TOK.get(tid, ""))

    text = "".join(pieces)
    raw = bytearray(_BYTE_DECODER.get(c, ord(c)) for c in text)
    return raw.decode("utf-8", errors="replace").strip()


_t0 = None


def tic():
    global _t0
    import time
    _t0 = time.perf_counter()


def toc(block_name: str):
    global _t0
    import time
    if _t0 is None:
        return
    elapsed = time.perf_counter() - _t0
    print(f"[block] {block_name}: {elapsed:.2f}s", flush=True)
    _t0 = None
