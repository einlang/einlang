---
layout: book
title: "Chapter 16 · The Night Before the Run"
---

# Chapter 16 · The Night Before the Run

> "The purpose of computing is insight, not numbers."
>
> — Richard Hamming

*Graduation · A transformer audit, cross-attention, and the attention that survived a code review*

---

It is 9:47 PM on a Friday. The cluster has 64 GPUs reserved for Saturday at 8 AM. The model is a 6-layer transformer with 8 attention heads, a hidden dimension of 512, and a vocabulary of 32,000 tokens. The training data is prepared. The hyperparameters are tuned. The team believes the code is correct.

Dana is the engineer who wrote it. Jesse is the engineer who has to approve the merge before the GPUs wake up. They are sitting in front of the same monitor, scrolling through the forward pass.

"Shapes all check out," Dana says. "Loss curve looks normal on the small-scale test."

"Show me the source," Jesse says. "Not the shapes. The names."

---

## The Code, As Written

Dana opens the file. It is 160 lines. The structure is familiar: embeddings, per-layer attention, LayerNorm, feedforward, output projection, loss.

```rust
// === Dimensions ===
let d_model = 512;
let n_heads = 8;
let d_head = 64;
let d_ff = 2048;
let n_layers = 6;
let vocab_size = 32000;

// === Embedding weights ===
let tok_emb[vocab, d_model] = init_embedding(vocab, d_model);
let pos_emb[max_len, d_model] = init_positional(max_len, d_model);

// === Per-layer attention weights ===
let Wq[layer, head, d_model, d_head] = init_weight(n_layers, n_heads, d_model, d_head);
let Wk[layer, head, d_model, d_head] = init_weight(n_layers, n_heads, d_model, d_head);
let Wv[layer, head, d_model, d_head] = init_weight(n_layers, n_heads, d_model, d_head);
let Wo[layer, head, d_head, d_model] = init_weight(n_layers, n_heads, d_head, d_model);

// === LayerNorm weights ===
let ln_scale[layer, d_model] = ones(d_model);
let ln_bias[layer, d_model] = zeros(d_model);

// === Feedforward weights ===
let ff_w1[layer, d_ff, d_model] = init_weight(n_layers, d_ff, d_model);
let ff_w2[layer, d_model, d_ff] = init_weight(n_layers, d_model, d_ff);

// === Output projection ===
let out_w[vocab, d_model] = init_weight(vocab, d_model);

// === Embed tokens ===
let embedded[b, seq, d_model] =
    tok_emb[input[b, seq], d_model] + pos_emb[seq, d_model];

// === Transformer layer recurrence ===
let x[0, b, seq, d_model] = embedded[b, seq, d_model];

```

![Q, K, V coordinate flow: names distinguish self-attention from cross-attention](figures/attention_flow.svg)

The figure traces the coordinate flow through attention. On the left, the input $\mathbf{x}$ carries `seq` and `d_model`. Three projections branch off: $\mathbf{Q}$ (with coordinate `seq_q` and `d_head`), $\mathbf{K}$ (with `seq_k` and `d_head`), and $\mathbf{V}$ (with `seq_k` and `d_head`). The score matrix $\mathbf{QK}^\top$ sits in the center with axes `seq_q` and `seq_k`—these are the two coordinates whose names encode the architecture. After softmax over `seq_k`, the output $\mathbf{O}$ carries `seq_q` and `d_head`. The bottom panels spell out the distinction: in self-attention, `seq_q` and `seq_k` range over the same sequence; in cross-attention, `seq_q` is the decoder positions and `seq_k` is the encoder positions. The coordinate names make the architecture decision visible in the indices themselves.

```rust
// --- Multi-head self-attention ---
//
// Attention is a communication protocol with named roles:
//   seq_q = "who is asking" (the query position)
//   seq_k = "who is answering" (the key position)
//   head  = "which conversation" (isolated communication channel)
//   d_head = "the vocabulary" (what is being communicated)
//
// These four coordinates are not arbitrary. Each carries a distinct
// contract. seq_q and seq_k may be the same domain (self-attention) or
// different (cross-attention). head isolates gradient flow—heads do not
// compete. d_head is the inner dimension that gets contracted in the
// score computation. Every shape-compatible attention bug is a violation
// of one of these contracts.
let Q[layer in 0..n_layers, b, head, seq, d_head] =
    sum[d_model](x[layer, b, seq, d_model] *
                  Wq[layer, head, d_model, d_head]);

let K[layer in 0..n_layers, b, head, seq, d_head] =
    sum[d_model](x[layer, b, seq, d_model] *
                  Wk[layer, head, d_model, d_head]);

let V[layer in 0..n_layers, b, head, seq, d_head] =
    sum[d_model](x[layer, b, seq, d_model] *
                  Wv[layer, head, d_model, d_head]);

let scores[layer in 0..n_layers, b, head, seq_q, seq_k] =
    sum[d_head](Q[layer, b, head, seq_q, d_head] *
                 K[layer, b, head, seq_k, d_head])
    / (d_head as f32) ** 0.5;

let masked_scores[layer in 0..n_layers, b, head, seq_q, seq_k] =
    if seq_k <= seq_q
        { scores[layer, b, head, seq_q, seq_k] }
        else { -1e9 };

let attn_weights[layer in 0..n_layers, b, head, seq_q, seq_k] =
    softmax[seq_k](masked_scores[layer, b, head, seq_q, seq_k]);

let head_out[layer in 0..n_layers, b, head, seq_q, d_head] =
    sum[seq_k](attn_weights[layer, b, head, seq_q, seq_k] *
                V[layer, b, head, seq_k, d_head]);

let attn_out[layer in 0..n_layers, b, seq_q, d_model] =
    sum[head, d_head](head_out[layer, b, head, seq_q, d_head] *
                       Wo[layer, head, d_head, d_model]);

// --- Residual + LayerNorm ---
let x_post_attn[layer in 0..n_layers, b, seq, d_model] =
    x[layer, b, seq, d_model] + attn_out[layer, b, seq, d_model];

let mu[layer in 0..n_layers, b, seq] =
    mean[d_model](x_post_attn[layer, b, seq, d_model]);

let sigma[layer in 0..n_layers, b, seq] =
    mean[d_model]((x_post_attn[layer, b, seq, d_model] -
                    mu[layer, b, seq]) ** 2.0) ** 0.5;

let normed[layer in 0..n_layers, b, seq, d_model] =
    (x_post_attn[layer, b, seq, d_model] - mu[layer, b, seq])
    / (sigma[layer, b, seq] + 1e-5);

let ln_out[layer in 0..n_layers, b, seq, d_model] =
    normed[layer, b, seq, d_model] * ln_scale[layer, d_model]
    + ln_bias[layer, d_model];

// --- Feedforward ---
let ff_hidden[layer in 0..n_layers, b, seq, d_ff] =
    sum[d_model](ln_out[layer, b, seq, d_model] *
                  ff_w1[layer, d_ff, d_model]);

let ff_activated[layer in 0..n_layers, b, seq, d_ff] =
    if ff_hidden[layer, b, seq, d_ff] > 0.0
        { ff_hidden[layer, b, seq, d_ff] }
        else { 0.0 };

let ff_out[layer in 0..n_layers, b, seq, d_model] =
    sum[d_ff](ff_activated[layer, b, seq, d_ff] *
               ff_w2[layer, d_model, d_ff]);

// --- Residual connection to next layer ---
let x[layer+1, b, seq, d_model] =
    ln_out[layer, b, seq, d_model] + ff_out[layer, b, seq, d_model];

// === Output ===
let logits[b, seq, vocab] =
    sum[d_model](x[n_layers, b, seq, d_model] * out_w[vocab, d_model]);

let probs[b, seq, vocab] = softmax[vocab](logits[b, seq, vocab]);
let loss = cross_entropy[vocab](logits[b, seq, vocab], labels[b, seq]);

// === Gradients ===
let dWq = @loss / @Wq;
let dWk = @loss / @Wk;
let dWv = @loss / @Wv;
let dWo = @loss / @Wo;
let d_ln_scale = @loss / @ln_scale;
let d_ln_bias = @loss / @ln_bias;
let d_ff_w1 = @loss / @ff_w1;
let d_ff_w2 = @loss / @ff_w2;
let d_out_w = @loss / @out_w;
```

"Looks clean," Jesse says. "But we're going to read it with four questions. Same four questions we'd ask of any tensor program. Ready?"

Dana nods. Jesse scrolls to the top.

---

## First Question: Which Coordinate Is Being Eliminated?

"The first thing I look at," Jesse says, "is every reduction bracket. What does the bracket claim to consume, and is that claim consistent with the coordinates that survive?"

They scan the reductions together:

- `sum[d_model]` in the Q, K, V projections. Consumes `d_model`. Surviving: `[layer, b, head, seq, d_head]`. Correct—`d_model` is the inner dimension being contracted.
- `sum[d_head]` in the attention scores. Consumes `d_head`. Surviving: `[layer, b, head, seq_q, seq_k]`. Correct.
- `softmax[seq_k]` in the attention weights. Consumes `seq_k`. Surviving: `[layer, b, head, seq_q, seq_k]`. Wait.

"Softmax is a reduction," Jesse says. "It consumes the coordinate in its bracket—`seq_k`—and reconstructs it with normalized values. The coordinate `seq_k` should survive with the same domain. Does it?"

Dana checks the declaration. `attn_weights` has coordinates `[layer, b, head, seq_q, seq_k]`. Yes, `seq_k` survives. Good.

- `sum[seq_k]` in the output computation. Consumes `seq_k`. Surviving: `[layer, b, head, seq_q, d_head]`. Correct—`seq_k` is the key dimension being weighted and summed.
- `sum[head, d_head]` in the output projection. Consumes both `head` and `d_head`. Surviving: `[layer, b, seq_q, d_model]`. Correct—the heads are merged.

"The reductions are consistent," Jesse says. "Every bracket names the coordinate it consumes. No coordinate is consumed implicitly. Good."

He scrolls down to the LayerNorm.

- `mean[d_model]` for mu. Consumes `d_model`. Surviving: `[layer, b, seq]`. Correct.
- `mean[d_model]` for sigma. Same.

"The reductions are clean. But there's a second question."

---

## Second Question: Which Coordinate Is Being Copied Along?

Jesse points to the LayerNorm application:

```rust
let ln_out[layer in 0..n_layers, b, seq, d_model] =
    normed[layer, b, seq, d_model] * ln_scale[layer, d_model]
    + ln_bias[layer, d_model];
```

"`normed` has four coordinates: `[layer, b, seq, d_model]`. `ln_scale` has two: `[layer, d_model]`. These are different ranks."

Dana frowns. "In einlang, broadcasting is only automatic for same-rank tensors. A 4D tensor and a 2D tensor—the compiler won't broadcast them. It requires explicit indexing when ranks differ."

"So this line doesn't compile."

Dana scrolls through the test logs. "It compiled on the small-scale test. But we were using a different broadcasting mode—the team added a flag to allow cross-rank broadcasting during prototyping."

"And that flag hides the fact that `b` and `seq` are being silently replicated in the `ln_scale` multiplication," Jesse says. "The replication is mathematically correct—LayerNorm *should* be independent of batch and sequence position. But the independence is invisible. A reader has to compare coordinate lists to discover it."

"The fix?" Dana asks.

"You have two options. One: use a coordinate-aware function that makes the independence part of its contract."

```rust
fn layer_norm[feature](
    x: [f32; ..batch, feature],
    scale: [f32; feature],
    bias: [f32; feature]
) -> [f32; ..batch, feature] {
    let mu[..batch] = mean[feature](x[..batch, feature]);
    let sigma[..batch] = mean[feature]((x[..batch, feature] - mu[..batch]) ** 2.0) ** 0.5;
    let normed[..batch, feature] = (x[..batch, feature] - mu[..batch]) / (sigma[..batch] + 1e-5);
    normed[..batch, feature] * scale[feature] + bias[feature]
}
```

"Now the fact that `scale` and `bias` are independent of everything except `feature` is visible in their type signatures. The rest packs `..batch` absorb whatever leading coordinates the caller provides. No implicit broadcasting. No cross-rank surprises."

"Option two," Dana says, "is to make the broadcasting explicit inline by indexing `ln_scale` at the full coordinate set and letting the compiler verify the omission:"

```rust
// Explicit: ln_scale omits b and seq by not mentioning them
let ln_out[layer, b, seq, d_model] =
    normed[layer, b, seq, d_model] * ln_scale[layer, d_model]
    + ln_bias[layer, d_model];
```

"But this still relies on implicit broadcasting for different ranks," Jesse says. "The function approach is safer. Either way, the point is: the independence of the normalization parameters from the batch dimensions should be a visible fact in the code, not a consequence of the broadcasting engine's behavior."

Dana makes a note. "Use the function. It's one more abstraction, but the contract is checkable."

---

## Third Question: Can You Trace a Coordinate from Source to Destination?

Jesse scrolls to the attention output and the residual connection.

```rust
let attn_out[layer in 0..n_layers, b, seq_q, d_model] =
    sum[head, d_head](head_out[layer, b, head, seq_q, d_head] *
                       Wo[layer, head, d_head, d_model]);

let x_post_attn[layer in 0..n_layers, b, seq, d_model] =
    x[layer, b, seq, d_model] + attn_out[layer, b, seq, d_model];
```

"Look at the coordinate names," Jesse says. "The attention output declares its sequence coordinate as `seq_q`. The residual connection indexes it as `seq`. Are these the same coordinate?"

Dana studies the two lines. "In self-attention, they're the same domain. The query sequence and the key sequence are the same tokens. So `seq_q` and `seq` refer to the same thing."

"Then why do they have different names?"

Dana is quiet for a moment. "Because I wrote the attention block first, using `seq_q` and `seq_k` to make the roles clear inside the attention computation. Then I wrote the residual connection and used `seq` because the role distinction wasn't needed outside attention."

"So the same domain has two names. The compiler sees `seq_q` and `seq` as different coordinates. If they have the same extent, the residual addition might compile—einlang allows addition of tensors with different coordinate names if the extents match and the compiler can verify the correspondence. But there's a flag for that too. If the extents differ—say, in a cross-attention scenario where the query length and key length are different—this residual connection would produce a shape mismatch."

"Or worse," Dana says, "if the extents happen to be equal but the domains are actually different, the addition would silently produce a valid tensor with semantically wrong values. The Square Matrix Test."

"Exactly. The fix is simple: use `seq` consistently for self-attention. If you need `seq_q` and `seq_k` inside the attention computation for clarity, rename at the boundary."

Dana rewrites:

```rust
let attn_out[layer in 0..n_layers, b, seq, d_model] =
    sum[head, d_head](head_out[layer, b, head, seq, d_head] *
                       Wo[layer, head, d_head, d_model]);
```

"Now `seq` flows from the input embedding, through every layer, to the output projection. One name, one identity, traceable from source to destination."

"That's the third habit," Jesse says. "Permute with a source. But it applies even when there's no permutation—just a coordinate renaming. If a coordinate has the same identity throughout the computation, use the same name. If it doesn't, the place where the name changes is the place where the identity changes. That place should be visible."

---

## Fourth Question: The Declaration Bracket

Jesse points to the recurrence that connects layers:

```rust
let x[layer+1, b, seq, d_model] =
    ln_out[layer, b, seq, d_model] + ff_out[layer, b, seq, d_model];
```

"Read the declaration bracket."

Dana reads: `x[layer+1, ...]`. She closes her eyes. "Ah."

"The declaration bracket can only contain identifiers, literals, and named rests. No expressions. `layer+1` is an expression."

"I put the computation in the wrong place," Dana says. "The bracket is for naming *what* is being defined—the element at index `layer+1` over some domain. The body is for saying *how* to compute it, using backward references like `layer-1`."

She rewrites:

```rust
let x[layer in 1..=n_layers, b, seq, d_model] =
    ln_out[layer-1, b, seq, d_model] + ff_out[layer-1, b, seq, d_model];
```

"Now the declaration bracket says `layer in 1..=n_layers`—a domain, not an expression. The body uses `layer-1`—a backward reference, which is allowed because `layer-1` is strictly smaller than `layer`. The rule is consistent: the left side names the element being defined. The right side computes it from earlier elements."

"This is the same rule from Chapter 3," Jesse says. "The index slot in the declaration bracket is for naming, not computing. It's not an arbitrary syntactic restriction. It's what makes the backward-only constraint checkable. If `layer+1` were allowed in the bracket, the compiler couldn't statically verify that the recurrence only reads from the past."

---

## The Gradient Question

Jesse scrolls to the bottom.

```rust
let dWq = @loss / @Wq;
let dWk = @loss / @Wk;
let dWv = @loss / @Wv;
let dWo = @loss / @Wo;
let d_out_w = @loss / @out_w;
```

"What are the coordinates of `d_out_w`?"

Dana thinks. "`out_w` has coordinates `[vocab, d_model]`. The forward pass uses it in `sum[d_model](x[...] * out_w[vocab, d_model])`. The gradient with respect to `out_w` must sum over all coordinates that `out_w` omits—which are `b` and `seq`. So `d_out_w` should have coordinates `[vocab, d_model]`."

"And if the compiler accidentally summed over `vocab` instead, producing `[b, seq, d_model]`?"

"Then the gradient would have the wrong shape. Except—if `vocab` happened to equal `d_model`—"

"Square Matrix Test again. The shapes would match by coincidence. The optimizer would apply an invalid gradient update. The loss would still go down—just for the wrong reasons."

Dana is silent for a moment. "How do we verify this?"

"You don't need to. The compiler does. The autodiff rules from Chapter 8 guarantee that `@loss / @out_w` sums over exactly the coordinates that `out_w` omits. The coordinate structure of the gradient is determined by the coordinate structure of the forward pass. But the question—'does the backward reduction match the forward omission?'—is exactly the fourth habit. Forward and backward, symmetric. You ask it to verify that you trust the compiler's answer."

---

## The Conversation Continues: Cross-Attention

Jesse leans back. "The code is correct for self-attention. But the architecture document mentions cross-attention for the second phase—the decoder attending to the encoder output. How would you extend this?"

Dana opens a fresh buffer. "Cross-attention means the queries come from one sequence and the keys and values come from another. The two sequences may have different lengths and different coordinate identities."

She sketches:

```rust
// Encoder output: sequence of source tokens
let enc_out[enc_seq, d_model] = encoder(source);

// Decoder hidden state: sequence of target tokens (so far)
let dec_hidden[dec_seq, d_model] = decoder_embedding(target_prefix);

// Cross-attention: decoder queries attend to encoder keys/values
let Q_cross[b, head, dec_seq, d_head] =
    sum[d_model](dec_hidden[dec_seq, d_model] *
                  Wq_cross[head, d_model, d_head]);

let K_cross[b, head, enc_seq, d_head] =
    sum[d_model](enc_out[enc_seq, d_model] *
                  Wk_cross[head, d_model, d_head]);

let V_cross[b, head, enc_seq, d_head] =
    sum[d_model](enc_out[enc_seq, d_model] *
                  Wv_cross[head, d_model, d_head]);

let scores_cross[b, head, dec_seq, enc_seq] =
    sum[d_head](Q_cross[b, head, dec_seq, d_head] *
                 K_cross[b, head, enc_seq, d_head])
    / (d_head as f32) ** 0.5;

let attn_cross[b, head, dec_seq, enc_seq] =
    softmax[enc_seq](scores_cross[b, head, dec_seq, enc_seq]);

let cross_out[b, head, dec_seq, d_head] =
    sum[enc_seq](attn_cross[b, head, dec_seq, enc_seq] *
                  V_cross[b, head, enc_seq, d_head]);
```

Jesse studies the coordinate names. "`dec_seq` and `enc_seq` are different. The score matrix is `[..., dec_seq, enc_seq]`—query positions on one axis, key positions on the other. If someone accidentally used `seq` for both, the compiler would see a square matrix and the attention would silently become self-attention."

"If the lengths are different, the shape mismatch would catch it," Dana says. "But if the lengths happen to be equal—the Square Matrix Test, again—the bug survives. Distinct names make it survive the compiler instead."

"This is exactly why the attention block in the original code used `seq_q` and `seq_k`—the convention anticipates cross-attention. In self-attention, both happen to be `seq`. In cross-attention, one is `dec_seq` and the other is `enc_seq`. The naming convention makes the architecture visible in the coordinate names."

Dana nods. "And the compiler enforces it. If I try to compute `scores_cross` with `Q[dec_seq]` and `K[enc_seq]` but label the result `[seq, seq]`, the compiler will report that `dec_seq` and `enc_seq` don't match the declared coordinate. The names are part of the type."

---

## The Conversation Deepens: Multi-Query Attention

"While we're on attention variants," Jesse says, "the team has been talking about multi-query attention for the next iteration. Fewer key-value heads than query heads, to reduce the KV cache size. How does that change the coordinate structure?"

Dana thinks for a moment. "In multi-head attention, every query head has its own key head and value head. The coordinate `head` spans query, key, and value equally. In multi-query attention, the query has `n_heads` heads but the keys and values share a single head—or a smaller number of heads."

"So the `head` coordinate has different extents for Q vs. K/V."

"Exactly. And if you use the same coordinate name `head` for both, you're claiming they're the same domain—which they aren't, or aren't entirely. The notation should make that visible."

She sketches the multi-query variant:

```rust
// Multi-query attention: many query heads, fewer key-value heads
let n_q_heads = 8;
let n_kv_heads = 2;   // shared across query heads

let Wq[head_q, d_model, d_head] = init_weight(n_q_heads, d_model, d_head);
let Wk[head_kv, d_model, d_head] = init_weight(n_kv_heads, d_model, d_head);
let Wv[head_kv, d_model, d_head] = init_weight(n_kv_heads, d_model, d_head);

let Q[b, head_q, seq, d_head] = sum[d_model](x[b, seq, d_model] * Wq[head_q, d_model, d_head]);
let K[b, head_kv, seq, d_head] = sum[d_model](x[b, seq, d_model] * Wk[head_kv, d_model, d_head]);
let V[b, head_kv, seq, d_head] = sum[d_model](x[b, seq, d_model] * Wv[head_kv, d_model, d_head]);

// The score: for each head_q, use the corresponding head_kv group
let scores[b, head_q, seq_q, seq_k] =
    sum[d_head](Q[b, head_q, seq_q, d_head] *
                 K[b, head_kv_of[head_q], seq_k, d_head])
    / (d_head as f32) ** 0.5;
```

Jesse points at the last line. "`head_kv_of[head_q]`—what's that?"

"A mapping from query heads to key-value heads. If `n_q_heads = 8` and `n_kv_heads = 2`, then `head_kv_of` maps query heads 0-3 to KV head 0, and query heads 4-7 to KV head 1. It's an explicit function, not an implicit broadcast. The coordinate names—`head_q` vs. `head_kv`—make the structural asymmetry visible. You can't accidentally treat them as the same coordinate because they have different names."

"And the shapes?"

"Q has `[b, head_q, seq, d_head]`. K has `[b, head_kv, seq, d_head]`. The score computation needs to align `head_q` with `head_kv` via the mapping. A standard `sum[d_head]` won't work because the `head` coordinates don't match. The compiler would reject a naive contraction attempt. The mapping `head_kv_of[head_q]` is required—and checked."

"This is the coordinate habit applied to architecture design," Jesse says. "The notation doesn't just record the architecture. It constrains it. A shape-compatible bug—using the wrong KV head for a query head—is a coordinate mismatch. The compiler catches it."

---

## What Survives the Audit

Dana commits the changes:

1. **LayerNorm** is wrapped in a coordinate-aware function with rest packs. The independence of scale and bias from batch coordinates is visible in the type signature.

2. **The `seq_q` coordinate** in `attn_out` is renamed to `seq`. One name, one identity, from embedding to output.

3. **The recurrence** uses `layer in 1..=n_layers` in the declaration bracket and `layer-1` in the body. The left side names. The right side computes.

4. **The gradient coordinates** are verified: each `@loss / @param` has the same coordinates as `param`. The compiler enforces this. The audit confirms it.

5. **Cross-attention** uses distinct coordinate names for decoder sequence (`dec_seq`) and encoder sequence (`enc_seq`). The architecture is visible in the names.

6. **Multi-query attention** uses distinct coordinate names for query heads (`head_q`) and key-value heads (`head_kv`), with an explicit mapping between them. The structural asymmetry is a coordinate-level fact.

It is 11:14 PM. The GPUs are still reserved for 8 AM. The code is different than it was two hours ago—not in what it computes, but in what it says. The shapes are the same. The names are clearer. The compiler has more to check, and the next reader—Dana, three months from now, after the model architecture has changed twice—has more to read.

"That's the whole thing, isn't it?" Dana says. "The coordinate habit isn't about writing correct code. It's about writing code that stays correct when you're not the one reading it."

Jesse closes the laptop. "See you Saturday."

---

This is the graduation. You can now read a tensor program and see not just the shapes, but the coordinates—which survive, which are consumed, which are broadcast, which are reduced. The four habits are not something you apply deliberately, one at a time, with effort. They are how you read.

Every reduction bracket is a claim about which coordinate is eliminated. Every broadcast is a claim about which coordinate is independent. Every coordinate name is a claim about identity—where the data came from, where it is going, whether the name at the destination matches the name at the source. The gradient of a loss with respect to a parameter is a claim about which coordinates the parameter omitted in the forward pass.

The compiler checks these claims. The habits check the compiler. And the notation makes both possible.

In the cross-attention sketch, the softmax is over `enc_seq`. If someone changed it to `softmax[dec_seq]`, the shapes would be identical—a probability distribution over the decoder positions instead of the encoder positions. The compiler would not catch this because the shapes are the same. But in code review, the coordinate name `dec_seq` would stand out: the reader knows cross-attention should attend over the encoder sequence, not the decoder sequence. The name is the signal. Multi-query attention with `head_kv_of[head_q]` requires a mapping function: if `n_q_heads = 8` and `n_kv_heads = 4`, the mapping `head_kv_of[head_q] = floor(head_q / 2)` makes the grouping explicit, and the compiler checks that every query head maps to a valid KV head index. For mixture of experts, the routing operation `let route[b, t] = argmax[e](gate_prob[b, t, e])` sends each token to its top expert, but tokens beyond capacity are silently dropped and replaced by a fallback. The mask `keep[b, t]` has the same shape as the output, invisible to every shape check. To make this visible, attach the mask to a coordinate: `let expert_mask[t, e] = ...` and require the downstream computation to explicitly broadcast over `e` with the mask as a guard. The coordinate name `e` in the mask tells the reader: some expert slots may be empty, and the code accounted for it.
