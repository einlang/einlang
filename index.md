---
layout: default
title: Einlang
---

# Einlang

You write the math. Indices, sums, the shape of things—right there in the code. The compiler reads them too. Wrong shape? It tells you before you run a single line.

No string indices. No guessing. Just the notation you'd use on a whiteboard, with a type system that's actually paying attention.

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);   // matrix multiply — shapes checked
```

**Where clauses** put index algebra next to the math: e.g. `where ih = oh + kh, iw = ow + kw` for a conv—no magic offsets, the compiler checks the mapping. **Recurrences** are declarations: base cases and a step in the bracket; the compiler figures out the order (RNNs, dynamic programming, stencils). **Rest patterns** in match destructure arrays: `[first, ..rest]` or `[..rest]`—head and tail in one pattern. Same language for ODEs, PDEs, and models like MNIST, ViT, or Whisper-style—one guarantee everywhere: if it type-checks, the shapes are correct.

**First move:** clone the repo, then run  
`pip install -e . && python3 -m einlang -c "let C[i,j] = sum[k](A[i,k]*B[k,j]);"`

[Open the repo →](https://github.com/einlang/einlang) · [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md)
