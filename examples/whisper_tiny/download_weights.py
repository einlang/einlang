#!/usr/bin/env python3
"""Download Whisper-tiny weights and prepare a sample audio clip.

No torch/transformers required.  Only needs: numpy, scipy, onnx, onnxruntime.

Usage:
    python3 download_weights.py          # download + extract + verify
    python3 download_weights.py --skip-verify   # skip onnxruntime check
"""

import os
import sys
import json
import struct
import urllib.request
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
SAMPLES_DIR = os.path.join(SCRIPT_DIR, "samples")
HF_BASE = "https://huggingface.co/openai/whisper-tiny/resolve/main"
N_LAYERS = 4

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
CHUNK_LENGTH = 30  # seconds


# ---------------------------------------------------------------------------
# Safetensors parser (pure Python + numpy, no library needed)
# ---------------------------------------------------------------------------
def load_safetensors(path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        data = f.read()

    _DT = {"F32": np.float32, "F16": np.float16, "BF16": "bf16",
           "I32": np.int32, "I64": np.int64}

    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt = _DT[meta["dtype"]]
        shape = tuple(meta["shape"])
        start, end = meta["data_offsets"]
        raw = data[start:end]
        if dt == "bf16":
            u16 = np.frombuffer(raw, dtype=np.uint16).copy()
            arr = np.zeros(len(u16), dtype=np.float32)
            arr.view(np.uint32)[:] = u16.astype(np.uint32) << 16
            arr = arr.reshape(shape)
        else:
            arr = np.frombuffer(raw, dtype=dt).reshape(shape).astype(np.float32)
        tensors[name] = arr
    return tensors


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def download(url, dest):
    if os.path.exists(dest):
        print(f"  [cached] {dest}")
        return
    print(f"  Downloading {url} ...")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def save_npy(name, arr):
    path = os.path.join(WEIGHTS_DIR, name)
    np.save(path, arr.astype(np.float32))
    print(f"  {name}: {tuple(arr.shape)}")


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------
def download_and_extract_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    st_path = os.path.join(SCRIPT_DIR, "model.safetensors")
    download(f"{HF_BASE}/model.safetensors", st_path)

    print("Parsing safetensors ...")
    sd = load_safetensors(st_path)

    # Encoder convolutions
    save_npy("enc_conv1_w.npy", sd["model.encoder.conv1.weight"])
    save_npy("enc_conv1_b.npy", sd["model.encoder.conv1.bias"])
    save_npy("enc_conv2_w.npy", sd["model.encoder.conv2.weight"])
    save_npy("enc_conv2_b.npy", sd["model.encoder.conv2.bias"])
    save_npy("enc_pos_emb.npy", sd["model.encoder.embed_positions.weight"])

    def _stack(prefix, pattern, *, transpose=False):
        parts = []
        for i in range(N_LAYERS):
            key = f"model.{prefix}.layers.{i}.{pattern}"
            a = sd[key]
            if transpose:
                a = a.T
            parts.append(a)
        return np.stack(parts, axis=0)

    # Encoder blocks
    save_npy("enc_blk_sa_ln_w.npy", _stack("encoder", "self_attn_layer_norm.weight"))
    save_npy("enc_blk_sa_ln_b.npy", _stack("encoder", "self_attn_layer_norm.bias"))
    save_npy("enc_blk_sa_q_w.npy",  _stack("encoder", "self_attn.q_proj.weight", transpose=True))
    save_npy("enc_blk_sa_q_b.npy",  _stack("encoder", "self_attn.q_proj.bias"))
    save_npy("enc_blk_sa_k_w.npy",  _stack("encoder", "self_attn.k_proj.weight", transpose=True))
    save_npy("enc_blk_sa_v_w.npy",  _stack("encoder", "self_attn.v_proj.weight", transpose=True))
    save_npy("enc_blk_sa_v_b.npy",  _stack("encoder", "self_attn.v_proj.bias"))
    save_npy("enc_blk_sa_o_w.npy",  _stack("encoder", "self_attn.out_proj.weight", transpose=True))
    save_npy("enc_blk_sa_o_b.npy",  _stack("encoder", "self_attn.out_proj.bias"))
    save_npy("enc_blk_mlp_ln_w.npy", _stack("encoder", "final_layer_norm.weight"))
    save_npy("enc_blk_mlp_ln_b.npy", _stack("encoder", "final_layer_norm.bias"))
    save_npy("enc_blk_fc1_w.npy",   _stack("encoder", "fc1.weight", transpose=True))
    save_npy("enc_blk_fc1_b.npy",   _stack("encoder", "fc1.bias"))
    save_npy("enc_blk_fc2_w.npy",   _stack("encoder", "fc2.weight", transpose=True))
    save_npy("enc_blk_fc2_b.npy",   _stack("encoder", "fc2.bias"))

    save_npy("enc_ln_w.npy", sd["model.encoder.layer_norm.weight"])
    save_npy("enc_ln_b.npy", sd["model.encoder.layer_norm.bias"])

    # Decoder embeddings
    save_npy("dec_tok_emb.npy", sd["model.decoder.embed_tokens.weight"])
    save_npy("dec_pos_emb.npy", sd["model.decoder.embed_positions.weight"])

    # Decoder blocks — self-attention
    save_npy("dec_blk_sa_ln_w.npy", _stack("decoder", "self_attn_layer_norm.weight"))
    save_npy("dec_blk_sa_ln_b.npy", _stack("decoder", "self_attn_layer_norm.bias"))
    save_npy("dec_blk_sa_q_w.npy",  _stack("decoder", "self_attn.q_proj.weight", transpose=True))
    save_npy("dec_blk_sa_q_b.npy",  _stack("decoder", "self_attn.q_proj.bias"))
    save_npy("dec_blk_sa_k_w.npy",  _stack("decoder", "self_attn.k_proj.weight", transpose=True))
    save_npy("dec_blk_sa_v_w.npy",  _stack("decoder", "self_attn.v_proj.weight", transpose=True))
    save_npy("dec_blk_sa_v_b.npy",  _stack("decoder", "self_attn.v_proj.bias"))
    save_npy("dec_blk_sa_o_w.npy",  _stack("decoder", "self_attn.out_proj.weight", transpose=True))
    save_npy("dec_blk_sa_o_b.npy",  _stack("decoder", "self_attn.out_proj.bias"))

    # Decoder blocks — cross-attention
    save_npy("dec_blk_ca_ln_w.npy", _stack("decoder", "encoder_attn_layer_norm.weight"))
    save_npy("dec_blk_ca_ln_b.npy", _stack("decoder", "encoder_attn_layer_norm.bias"))
    save_npy("dec_blk_ca_q_w.npy",  _stack("decoder", "encoder_attn.q_proj.weight", transpose=True))
    save_npy("dec_blk_ca_q_b.npy",  _stack("decoder", "encoder_attn.q_proj.bias"))
    save_npy("dec_blk_ca_k_w.npy",  _stack("decoder", "encoder_attn.k_proj.weight", transpose=True))
    save_npy("dec_blk_ca_v_w.npy",  _stack("decoder", "encoder_attn.v_proj.weight", transpose=True))
    save_npy("dec_blk_ca_v_b.npy",  _stack("decoder", "encoder_attn.v_proj.bias"))
    save_npy("dec_blk_ca_o_w.npy",  _stack("decoder", "encoder_attn.out_proj.weight", transpose=True))
    save_npy("dec_blk_ca_o_b.npy",  _stack("decoder", "encoder_attn.out_proj.bias"))

    # Decoder blocks — MLP
    save_npy("dec_blk_mlp_ln_w.npy", _stack("decoder", "final_layer_norm.weight"))
    save_npy("dec_blk_mlp_ln_b.npy", _stack("decoder", "final_layer_norm.bias"))
    save_npy("dec_blk_fc1_w.npy",   _stack("decoder", "fc1.weight", transpose=True))
    save_npy("dec_blk_fc1_b.npy",   _stack("decoder", "fc1.bias"))
    save_npy("dec_blk_fc2_w.npy",   _stack("decoder", "fc2.weight", transpose=True))
    save_npy("dec_blk_fc2_b.npy",   _stack("decoder", "fc2.bias"))

    save_npy("dec_ln_w.npy", sd["model.decoder.layer_norm.weight"])
    save_npy("dec_ln_b.npy", sd["model.decoder.layer_norm.bias"])

    os.remove(st_path)
    print(f"\nWeights saved to {WEIGHTS_DIR}")


# ---------------------------------------------------------------------------
# Mel spectrogram (matches OpenAI whisper preprocessing)
# ---------------------------------------------------------------------------
def _mel_frequencies(n_mels, fmin=0.0, fmax=8000.0):
    def _hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def _mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    return _mel_to_hz(mels)


def _mel_filter_bank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS):
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mel_f = _mel_frequencies(n_mels, fmin=0.0, fmax=sr / 2.0)
    fb = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        lo, mid, hi = mel_f[i], mel_f[i + 1], mel_f[i + 2]
        up = (fft_freqs - lo) / max(mid - lo, 1e-10)
        down = (hi - fft_freqs) / max(hi - mid, 1e-10)
        fb[i] = np.maximum(0, np.minimum(up, down))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    fb *= enorm[:, np.newaxis]
    return fb


def compute_log_mel(audio_16k):
    audio = np.zeros(CHUNK_LENGTH * SAMPLE_RATE, dtype=np.float32)
    audio[: min(len(audio_16k), len(audio))] = audio_16k[: len(audio)]

    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2))
    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    idx = np.arange(N_FFT)[None, :] + np.arange(n_frames)[:, None] * HOP_LENGTH
    frames = padded[idx] * window
    spec = np.fft.rfft(frames, n=N_FFT)
    magnitudes = np.abs(spec[:-1, :]) ** 2

    filters = _mel_filter_bank()
    mel = filters @ magnitudes.T
    log_mel = np.log10(np.maximum(mel, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    return log_mel.astype(np.float32)


# ---------------------------------------------------------------------------
# Audio sample (JFK "ask not what your country can do for you" — speech, not music)
# ---------------------------------------------------------------------------
JFK_FLAC_URL = "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"


def _load_audio_from_file(path, sr_target=SAMPLE_RATE):
    """Load audio from path (flac or wav), return (audio_float32, sr)."""
    import wave
    path_lower = path.lower()
    if path_lower.endswith(".flac"):
        try:
            import soundfile as sf
            audio, sr = sf.read(path)
            audio = audio.astype(np.float32)
        except ImportError:
            import subprocess
            tmp_wav = path + ".wav.tmp"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", path, "-ar", str(sr_target), "-ac", "1", tmp_wav],
                    check=True, capture_output=True,
                )
                with wave.open(tmp_wav, "rb") as wf:
                    sr = wf.getframerate()
                    raw = wf.readframes(wf.getnframes())
                    sw = wf.getsampwidth()
                dt = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
                audio = np.frombuffer(raw, dtype=dt).astype(np.float32)
                if dt != np.uint8:
                    audio = audio / (2 ** (sw * 8 - 1))
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)
            except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
                raise RuntimeError("For .flac install soundfile (pip install soundfile) or ffmpeg: " + str(e))
    else:
        import wave as _wave
        with _wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            sw = wf.getsampwidth()
        dt = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
        audio = np.frombuffer(raw, dtype=dt).astype(np.float32)
        if dt == np.uint8:
            audio = (audio - 128.0) / 128.0
        else:
            audio = audio / (2 ** (sw * 8 - 1))
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != sr_target:
        from scipy.signal import resample
        n_out = int(len(audio) * sr_target / sr)
        audio = resample(audio, n_out).astype(np.float32)
    return audio


def prepare_audio():
    import wave

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    mel_path = os.path.join(SAMPLES_DIR, "jfk.npy")
    if os.path.exists(mel_path):
        print(f"[cached] {mel_path}")
        return

    # Prefer local JFK speech file (so golden_ref.txt transcript matches).
    for name in ("jfk.flac", "jfk.wav"):
        local = os.path.join(SAMPLES_DIR, name)
        if os.path.exists(local):
            print(f"  Using local {name}")
            audio = _load_audio_from_file(local)
            break
    else:
        # Download JFK sample to samples/jfk.flac (kept for future runs).
        jfk_flac = os.path.join(SAMPLES_DIR, "jfk.flac")
        try:
            download(JFK_FLAC_URL, jfk_flac)
            print(f"  Using {jfk_flac}")
            audio = _load_audio_from_file(jfk_flac)
        except Exception as e:
            raise RuntimeError(
                "Failed to get JFK speech sample (needed for golden_ref.txt transcript). "
                "Install: pip install soundfile (or ffmpeg for .flac). "
                "Or download manually: curl -L -o samples/jfk.flac " + JFK_FLAC_URL + " ; error: " + str(e)
            ) from e

    print(f"  Audio: {len(audio)} samples ({len(audio)/SAMPLE_RATE:.1f}s)")
    mel = compute_log_mel(audio)
    np.save(mel_path, mel)
    print(f"  Mel spectrogram: {mel.shape} -> {mel_path}")
    # Save audio for ONNX/pipeline comparison (16k mono)
    try:
        import wave as _wave
        wav_path = os.path.join(SAMPLES_DIR, "audio.wav")
        with _wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
        print(f"  Audio WAV: {wav_path}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def download_tokenizer():
    tok_path = os.path.join(SCRIPT_DIR, "tokenizer.json")
    if os.path.exists(tok_path):
        print(f"[cached] {tok_path}")
        return
    download(f"{HF_BASE}/tokenizer.json", tok_path)
    print(f"  Tokenizer saved to {tok_path}")


# ---------------------------------------------------------------------------
# Reference transcription (numpy only, no onnx) for accuracy tests
# ---------------------------------------------------------------------------
def get_reference_transcription():
    """Run numpy encoder + decoder on jfk.npy and return decoded text.
    Returns (text, tokens) or (None, None) if weights/samples/tokenizer missing.
    """
    mel_path = os.path.join(SAMPLES_DIR, "jfk.npy")
    tok_path = os.path.join(SCRIPT_DIR, "tokenizer.json")
    if not os.path.exists(mel_path) or not os.path.exists(tok_path):
        return None, None
    if not os.path.isdir(WEIGHTS_DIR):
        return None, None
    mel = np.load(mel_path).astype(np.float32)
    sd = {}
    for f in os.listdir(WEIGHTS_DIR):
        if f.endswith(".npy"):
            sd[f[:-4]] = np.load(os.path.join(WEIGHTS_DIR, f))
    enc_out = _numpy_encoder(mel, sd)
    SOT, LANG_EN, TRANSCRIBE, NO_TS, EOT = 50258, 50259, 50359, 50363, 50257
    tokens = [SOT, LANG_EN, TRANSCRIBE, NO_TS]
    for _ in range(36):
        logits = _numpy_decoder_step(tokens, enc_out, sd)
        nxt = int(np.argmax(logits))
        tokens.append(nxt)
        if nxt == EOT:
            break
    text = _decode_tokens(tokens, tok_path)
    return text, tokens


# ---------------------------------------------------------------------------
# Optional: verify reference output with onnxruntime
# ---------------------------------------------------------------------------
def verify_reference():
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not available, skipping verification.")
        return

    text, tokens = get_reference_transcription()
    if text is None:
        print("No mel spectrogram or tokenizer found, skipping verification.")
        return

    print("\nReference greedy decode using numpy weights ...")
    print(f"  Token IDs ({len(tokens)}): {tokens}")
    print(f"  Text: {text!r}")
    print(f"\n=== Expected output for main.ein assertion ===")
    print(f'  "{text}"')


def _numpy_encoder(mel, sd):
    D, S = 384, 1500

    x = mel[np.newaxis, :, :]  # (1, 80, 3000)

    # Conv1: stride=1, pad=1
    x = _conv1d(x, sd["enc_conv1_w"], sd["enc_conv1_b"], stride=1, pad=1)
    x = _gelu(x)
    # Conv2: stride=2, pad=1
    x = _conv1d(x, sd["enc_conv2_w"], sd["enc_conv2_b"], stride=2, pad=1)
    x = _gelu(x)

    x = x[0].T  # (1500, 384)
    x = x + sd["enc_pos_emb"][:S]

    for L in range(N_LAYERS):
        x = _encoder_block(x, L, sd)

    x = _layer_norm(x, sd["enc_ln_w"], sd["enc_ln_b"])
    return x  # (1500, 384)


def _encoder_block(x, L, sd):
    D = 384
    n = _layer_norm(x, sd["enc_blk_sa_ln_w"][L], sd["enc_blk_sa_ln_b"][L])
    attn = _self_attention(n, L, sd, prefix="enc_blk_sa")
    x = x + attn
    n = _layer_norm(x, sd["enc_blk_mlp_ln_w"][L], sd["enc_blk_mlp_ln_b"][L])
    h = n @ sd["enc_blk_fc1_w"][L] + sd["enc_blk_fc1_b"][L]
    h = _gelu(h)
    h = h @ sd["enc_blk_fc2_w"][L] + sd["enc_blk_fc2_b"][L]
    return x + h


def _numpy_decoder_step(tokens, enc_out, sd):
    T, D = len(tokens), 384

    tok_emb = sd["dec_tok_emb"][tokens]  # (T, 384)
    pos_emb = sd["dec_pos_emb"][:T]
    x = tok_emb + pos_emb

    causal = np.full((T, T), -1e9, dtype=np.float32)
    for i in range(T):
        causal[i, :i + 1] = 0.0

    for L in range(N_LAYERS):
        x = _decoder_block(x, enc_out, L, sd, causal)

    x = _layer_norm(x, sd["dec_ln_w"], sd["dec_ln_b"])
    logits = x[-1] @ sd["dec_tok_emb"].T  # (51865,)
    return logits


def _decoder_block(x, enc, L, sd, mask):
    n = _layer_norm(x, sd["dec_blk_sa_ln_w"][L], sd["dec_blk_sa_ln_b"][L])
    attn = _self_attention(n, L, sd, prefix="dec_blk_sa", mask=mask)
    x = x + attn
    n = _layer_norm(x, sd["dec_blk_ca_ln_w"][L], sd["dec_blk_ca_ln_b"][L])
    attn = _cross_attention(n, enc, L, sd)
    x = x + attn
    n = _layer_norm(x, sd["dec_blk_mlp_ln_w"][L], sd["dec_blk_mlp_ln_b"][L])
    h = n @ sd["dec_blk_fc1_w"][L] + sd["dec_blk_fc1_b"][L]
    h = _gelu(h)
    h = h @ sd["dec_blk_fc2_w"][L] + sd["dec_blk_fc2_b"][L]
    return x + h


def _self_attention(x, L, sd, prefix, mask=None):
    n_heads, head_dim = 6, 64
    S, D = x.shape

    Q = x @ sd[f"{prefix}_q_w"][L] + sd[f"{prefix}_q_b"][L]
    K = x @ sd[f"{prefix}_k_w"][L]
    V = x @ sd[f"{prefix}_v_w"][L] + sd[f"{prefix}_v_b"][L]

    Q = Q.reshape(S, n_heads, head_dim).transpose(1, 0, 2)
    K = K.reshape(S, n_heads, head_dim).transpose(1, 0, 2)
    V = V.reshape(S, n_heads, head_dim).transpose(1, 0, 2)

    score = Q @ K.transpose(0, 2, 1) / 8.0
    if mask is not None:
        score = score + mask[np.newaxis, :, :]
    score = score - score.max(axis=-1, keepdims=True)
    w = np.exp(score)
    w = w / w.sum(axis=-1, keepdims=True)
    ctx = (w @ V).transpose(1, 0, 2).reshape(S, D)
    return ctx @ sd[f"{prefix}_o_w"][L] + sd[f"{prefix}_o_b"][L]


def _cross_attention(x, enc, L, sd):
    n_heads, head_dim = 6, 64
    Sq, D = x.shape
    Se = enc.shape[0]

    Q = x @ sd["dec_blk_ca_q_w"][L] + sd["dec_blk_ca_q_b"][L]
    K = enc @ sd["dec_blk_ca_k_w"][L]
    V = enc @ sd["dec_blk_ca_v_w"][L] + sd["dec_blk_ca_v_b"][L]

    Q = Q.reshape(Sq, n_heads, head_dim).transpose(1, 0, 2)
    K = K.reshape(Se, n_heads, head_dim).transpose(1, 0, 2)
    V = V.reshape(Se, n_heads, head_dim).transpose(1, 0, 2)

    score = Q @ K.transpose(0, 2, 1) / 8.0
    score = score - score.max(axis=-1, keepdims=True)
    w = np.exp(score)
    w = w / w.sum(axis=-1, keepdims=True)
    ctx = (w @ V).transpose(1, 0, 2).reshape(Sq, D)
    return ctx @ sd["dec_blk_ca_o_w"][L] + sd["dec_blk_ca_o_b"][L]


def _conv1d(x, w, b, stride=1, pad=0):
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    N, Ci, Li = x.shape
    Co, Cig, K = w.shape
    Lo = (Li - K) // stride + 1
    out = np.zeros((N, Co, Lo), dtype=np.float32)
    for co in range(Co):
        for k in range(K):
            out[:, co, :] += np.sum(
                x[:, :Cig, k : k + Lo * stride : stride] * w[co, :, k : k + 1],
                axis=1,
            )
        out[:, co, :] += b[co]
    return out


def _layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _decode_tokens(token_ids, tokenizer_path):
    if not os.path.exists(tokenizer_path):
        return f"<no tokenizer: {token_ids}>"
    with open(tokenizer_path) as f:
        tok_data = json.load(f)

    id2tok = {}
    vocab = tok_data.get("model", {}).get("vocab", {})
    for k, v in vocab.items():
        id2tok[v] = k

    added = tok_data.get("added_tokens", [])
    for entry in added:
        id2tok[entry["id"]] = entry["content"]

    special = {50257, 50258, 50259, 50359, 50363}
    byte_pieces = []
    for tid in token_ids:
        if tid in special:
            continue
        tok_str = id2tok.get(tid, "")
        byte_pieces.append(tok_str)

    text = "".join(byte_pieces)
    text = bytearray(
        [_BYTE_DECODER.get(c, ord(c)) for c in text]
    ).decode("utf-8", errors="replace")
    return text.strip()


def _build_byte_maps():
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

_BYTE_DECODER = _build_byte_maps()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    download_and_extract_weights()
    prepare_audio()
    download_tokenizer()
    if "--skip-verify" not in sys.argv:
        verify_reference()
