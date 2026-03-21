#!/usr/bin/env python3
"""Fast NumPy port of examples/deit_tiny/main.ein (einsum conv + vectorized attention)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def patch_conv(X: np.ndarray, W: np.ndarray, B: np.ndarray, stride: int = 16) -> np.ndarray:
    """X [1,3,H,W], W [Cout,3,kh,kw] -> [1,Cout,Oh,Ow]."""
    kh, kw = W.shape[2], W.shape[3]
    x = X[0]
    H, Wi = x.shape[1], x.shape[2]
    h_out = (H - kh) // stride + 1
    w_out = (Wi - kw) // stride + 1
    p = np.stack(
        [
            [x[:, ph * stride : ph * stride + kh, pw * stride : pw * stride + kw] for pw in range(w_out)]
            for ph in range(h_out)
        ]
    )
    out = np.einsum("pqcmn,ocmn->opq", p.astype(np.float32), W.astype(np.float32))
    out = out + B.reshape(-1, 1, 1).astype(np.float32)
    return out[np.newaxis, ...]


def layer_norm_last(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return gamma * (x - m) / np.sqrt(v + eps) + beta


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def transformer_block(
    x: np.ndarray,
    L: int,
    blk_ln1_w: np.ndarray,
    blk_ln1_b: np.ndarray,
    blk_qkv_w: np.ndarray,
    blk_qkv_b: np.ndarray,
    blk_proj_w: np.ndarray,
    blk_proj_b: np.ndarray,
    blk_ln2_w: np.ndarray,
    blk_ln2_b: np.ndarray,
    blk_fc1_w: np.ndarray,
    blk_fc1_b: np.ndarray,
    blk_fc2_w: np.ndarray,
    blk_fc2_b: np.ndarray,
) -> np.ndarray:
    ln1 = layer_norm_last(x, blk_ln1_w[L], blk_ln1_b[L])
    qkv = ln1[0] @ blk_qkv_w[L] + blk_qkv_b[L]
    # main.ein: Q from qkv[s,h*64+d], K from 192+h*64+d, V from 384+h*64+d (all Q heads, then K, then V)
    q_part = qkv[:, :192].reshape(197, 3, 64)
    k_part = qkv[:, 192:384].reshape(197, 3, 64)
    v_part = qkv[:, 384:576].reshape(197, 3, 64)
    Q = np.swapaxes(q_part, 0, 1)
    K = np.swapaxes(k_part, 0, 1)
    V = np.swapaxes(v_part, 0, 1)
    scale = 1.0 / np.sqrt(64.0)
    score = scale * np.matmul(Q, np.swapaxes(K, -1, -2))
    smax = score.max(axis=-1, keepdims=True)
    expv = np.exp(score - smax)
    attn = expv / expv.sum(axis=-1, keepdims=True)
    ctx = np.matmul(attn, V)
    cat = ctx.transpose(1, 0, 2).reshape(197, 192)
    proj = cat @ blk_proj_w[L] + blk_proj_b[L]
    res1 = x.copy()
    res1[0] = x[0] + proj
    ln2 = layer_norm_last(res1, blk_ln2_w[L], blk_ln2_b[L])
    fc1 = ln2[0] @ blk_fc1_w[L] + blk_fc1_b[L]
    act = gelu_tanh(fc1)
    fc2 = act @ blk_fc2_w[L] + blk_fc2_b[L]
    out = res1.copy()
    out[0] = res1[0] + fc2
    return out


def infer_numpy(img: np.ndarray, w: dict[str, np.ndarray]) -> int:
    patches = patch_conv(img, w["patch_proj_w"], w["patch_proj_b"], 16)
    flat = np.zeros((196, 192), dtype=np.float32)
    for s in range(196):
        ph, pw = s // 14, s % 14
        flat[s] = patches[0, :, ph, pw]
    tokens = np.zeros((1, 197, 192), dtype=np.float32)
    tokens[0, 0] = w["cls_token"][0, 0]
    tokens[0, 1:] = flat
    embedded = tokens + w["pos_embed"]
    x = embedded
    for L in range(12):
        x = transformer_block(
            x,
            L,
            w["blk_ln1_w"],
            w["blk_ln1_b"],
            w["blk_qkv_w"],
            w["blk_qkv_b"],
            w["blk_proj_w"],
            w["blk_proj_b"],
            w["blk_ln2_w"],
            w["blk_ln2_b"],
            w["blk_fc1_w"],
            w["blk_fc1_b"],
            w["blk_fc2_w"],
            w["blk_fc2_b"],
        )
    final_ln = layer_norm_last(x, w["norm_w"], w["norm_b"])
    cls = final_ln[0, 0]
    logits = cls @ w["head_w"].T + w["head_b"]
    return int(np.argmax(logits))


def main() -> None:
    wdir = ROOT / "weights"
    w = {k: np.load(wdir / f"{k}.npy") for k in [
        "patch_proj_w", "patch_proj_b", "cls_token", "pos_embed", "norm_w", "norm_b",
        "head_w", "head_b", "blk_ln1_w", "blk_ln1_b", "blk_qkv_w", "blk_qkv_b",
        "blk_proj_w", "blk_proj_b", "blk_ln2_w", "blk_ln2_b", "blk_fc1_w", "blk_fc1_b",
        "blk_fc2_w", "blk_fc2_b",
    ]}
    preds = [infer_numpy(np.load(ROOT / "samples" / f"{i}.npy"), w) for i in range(3)]
    print("numpy ref preds", preds, "expected [285, 207, 949]")


if __name__ == "__main__":
    main()
