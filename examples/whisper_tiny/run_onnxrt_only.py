#!/usr/bin/env python3
"""Run Whisper-tiny with ONNX Runtime only (no PyTorch/transformers).

Uses PINTO_model_zoo encoder/decoder ONNX. Requires: onnxruntime, numpy.
Downloads ONNX files on first run. Reads samples/jfk.npy (mel) and tokenizer.json.

Usage: python3 run_onnxrt_only.py
"""

import os
import sys
import time
import json
import urllib.request
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(SCRIPT_DIR, "samples")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
ONNX_DIR = os.path.join(SCRIPT_DIR, "onnx")
PINTO_BASE = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/onnx"

EOT = 50257
SOT, LANG_EN, TRANSCRIBE, NO_TS = 50258, 50259, 50359, 50363


def download_onnx(name: str) -> str:
    os.makedirs(ONNX_DIR, exist_ok=True)
    path = os.path.join(ONNX_DIR, f"{name}_11.onnx")
    if os.path.isfile(path):
        return path
    url = f"{PINTO_BASE}/{name}_11.onnx"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, path)
    return path


def load_tokenizer():
    path = os.path.join(SCRIPT_DIR, "tokenizer.json")
    with open(path) as f:
        data = json.load(f)
    id2tok = {}
    for k, v in data.get("model", {}).get("vocab", {}).items():
        id2tok[v] = k
    for entry in data.get("added_tokens", []):
        id2tok[entry["id"]] = entry["content"]
    return id2tok


def decode_tokens(token_ids, id2tok):
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_decoder = {chr(c): b for b, c in zip(bs, cs)}
    special = {50257, 50258, 50259, 50359, 50363}
    pieces = []
    for tid in token_ids:
        if tid == EOT:
            break
        if tid in special or tid >= 50364:
            continue
        pieces.append(id2tok.get(tid, ""))
    text = "".join(pieces)
    raw = bytearray(byte_decoder.get(c, ord(c)) for c in text)
    return raw.decode("utf-8", errors="replace").strip()


def main():
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. pip install onnxruntime", file=sys.stderr)
        sys.exit(1)

    mel_path = os.path.join(SAMPLES_DIR, "jfk.npy")
    if not os.path.isfile(mel_path):
        print(f"Missing {mel_path}. Run download_weights.py first.", file=sys.stderr)
        sys.exit(1)
    tok_path = os.path.join(SCRIPT_DIR, "tokenizer.json")
    if not os.path.isfile(tok_path):
        print("Missing tokenizer.json. Run download_weights.py first.", file=sys.stderr)
        sys.exit(1)

    mel = np.load(mel_path).astype(np.float32)
    if mel.ndim == 2:
        mel = mel[np.newaxis, ...]  # (1, 80, 3000)

    # Load encoder (input: mel, output: encoder hidden state)
    enc_path = download_onnx("tiny_encoder")
    enc_sess = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    enc_in_name = enc_sess.get_inputs()[0].name
    enc_out_name = enc_sess.get_outputs()[0].name

    # Load decoder (inputs: tokens, audio_features, kv_cache, offset)
    dec_path = download_onnx("tiny_decoder")
    dec_sess = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    dec_in_names = [inp.name for inp in dec_sess.get_inputs()]
    dec_out_names = [o.name for o in dec_sess.get_outputs()]

    id2tok = load_tokenizer()

    # Encoder
    t0 = time.perf_counter()
    encoder_out = enc_sess.run(
        [enc_out_name],
        {enc_in_name: mel},
    )[0]
    t_enc = time.perf_counter() - t0

    # Greedy decode loop
    tokens = [SOT, LANG_EN, TRANSCRIBE, NO_TS]
    n_group = 1
    # PINTO tiny kv_cache shape: [8, n_group, length, 384]
    kv_cache = np.zeros((8, n_group, len(tokens), 384), dtype=np.float32)
    t_dec = 0.0
    for _ in range(36):
        tokens_np = np.array([tokens], dtype=np.int64)
        offset = len(tokens) - 1
        # Map common names to possible ONNX input names
        feed = {}
        for name in dec_in_names:
            if "token" in name.lower():
                feed[name] = tokens_np
            elif "audio" in name.lower() or "feature" in name.lower() or "encoder" in name.lower():
                feed[name] = encoder_out
            elif "kv" in name.lower() or "cache" in name.lower():
                feed[name] = kv_cache
            elif "offset" in name.lower():
                feed[name] = np.array([offset], dtype=np.int64)
        t0 = time.perf_counter()
        outs = dec_sess.run(dec_out_names, feed)
        t_dec += time.perf_counter() - t0
        logits = outs[0]  # (1, seq, vocab)
        next_id = int(np.argmax(logits[0, -1]))
        tokens.append(next_id)
        if len(outs) >= 2 and outs[1].size > 0:
            kv_cache = outs[1].astype(np.float32)
        if next_id == EOT:
            break

    text = decode_tokens(tokens, id2tok)
    total = t_enc + t_dec
    print(f"[ONNX Runtime] encoder: {t_enc:.2f}s  decoder: {t_dec:.2f}s  total: {total:.2f}s")
    print(f"[ONNX Runtime] tokens: {len(tokens)}  text: {text!r}")
    print()
    print("--- ONNX Runtime output ---")
    print(text)


if __name__ == "__main__":
    main()
