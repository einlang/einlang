# 6 — Whisper-tiny: Speech-to-Text

> **Previous**: [`deit_tiny/`](../deit_tiny/) · **Next**: [`units/`](../units/) (reference)

**Speech to text**: transcribes audio (speech) into text. End-to-end transcription using OpenAI's [Whisper-tiny](https://huggingface.co/openai/whisper-tiny) model. This is the most complex example — it combines everything from the previous models and adds an encoder-decoder architecture with autoregressive generation.

## Architecture

| Component | Details |
|-----------|---------|
| Encoder | 2x Conv1D (80->384, stride 1/2) + 4 transformer blocks |
| Decoder | 4 transformer blocks with causal self-attention + cross-attention |
| d_model | 384 |
| n_heads | 6 (head_dim = 64) |
| FFN dim | 1536 |
| Vocab | 51865 tokens (GPT-2 BPE) |
| Params | ~39M |

## What's new here

Building on the self-attention from [deit_tiny/](../deit_tiny/), this example introduces:

- **1D convolution** — `conv` on rank-3 tensors dispatches to `conv1d`, processing mel spectrogram frames instead of 2D image patches.
- **Encoder-decoder architecture** — the encoder processes audio into a fixed representation; the decoder generates text tokens conditioned on that representation.
- **Cross-attention** — decoder queries attend to encoder keys/values (`Kc[s in 0..1500, d in 0..384]` uses `enc[s, k]`), bridging the two halves of the model.
- **Causal self-attention** — the decoder uses a lower-triangular mask (`+ if j <= i { 0.0 } else { -10000.0 }`) so each token can only attend to previous tokens.
- **Autoregressive decoding** — `let tokens[t in 4..40] = { ... argmax(logits) }` uses Einlang's recurrence mechanism: each token depends on all previously generated tokens.
- **Python bridge for BPE** — `python::whisper_helpers::decode_tokens(tokens)` converts token IDs back to text using the GPT-2 tokenizer.

## Files

| File | Description |
|------|-------------|
| `main.ein` | Full Whisper-tiny encoder + decoder in Einlang |
| `download_weights.py` | Downloads model weights from HuggingFace, prepares sample audio |
| `whisper_helpers.py` | Python bridge for BPE token decoding |
| `tokenizer.json` | GPT-2 BPE vocabulary (downloaded by setup script) |

## Setup

```bash
cd examples/whisper_tiny
python3 download_weights.py
```

This downloads ~75 MB of weights (safetensors format, parsed without torch), downloads a **real speech sample** (JFK “ask not what your country can do for you” from OpenAI’s Whisper tests), computes the mel spectrogram, and optionally verifies the output using a pure-numpy reference decoder.

## Run

From the repo root (or with `cwd` in `examples/whisper_tiny`):

```bash
python3 -m einlang examples/whisper_tiny/main.ein
```

**Sample input**: mel spectrogram from the JFK clip (real speech). Saved as `samples/jfk.npy`. To switch to the JFK sample after using another clip, remove `samples/jfk.npy` and run `download_weights.py` again. **Output**: transcribed text (speech-to-text).

## Compare with golden reference

From `examples/whisper_tiny`:

```bash
python3 compare_with_golden.py
```

This runs the NumPy reference and Einlang `main.ein` and compares their output to `golden_ref.txt` (expected transcript for the JFK sample). Use `--no-numpy` or `--no-einlang` to skip one run.

## How it works

1. **Preprocessing**: Mel spectrogram (80 bins x 3000 frames) is precomputed from 16 kHz audio by the download script using scipy FFT.
2. **Encoder**: Two 1D convolutions (with GELU) downsample the mel to 1500 frames, add positional embeddings, then process through 4 transformer blocks with self-attention.
3. **Decoder**: Greedy autoregressive loop using Einlang's recurrence:
   - Start tokens `[SOT, en, transcribe, notimestamps]` seed the sequence.
   - Each step embeds all tokens so far, runs 4 decoder blocks (causal self-attention + cross-attention to encoder output + MLP), projects to vocabulary logits, and takes argmax.
   - Generates up to 36 new tokens.
4. **Output**: Token IDs are decoded to text via the Python bridge (`whisper_helpers.py`).

This completes the model examples. For a comprehensive reference of every language feature, see [`units/`](../units/).
