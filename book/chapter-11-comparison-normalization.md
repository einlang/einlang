---
layout: book
title: "Chapter 11 · Comparison: Normalization"
---

# Chapter 11 · Comparison: Normalization

> "A notation is a tool for thought."
>
> — Kenneth Iverson

*Comparisons · LayerNorm, RMSNorm, GroupNorm in two notations*

---

You are two weeks into a Transformer project. LayerNorm is working. You swap in RMSNorm for the memory-efficient run—same `dim=-1`, same shape, loss looks fine. Then you try GroupNorm on the convolutional front-end. `dim=-1` again. The shapes align. The loss descends. Three days later you notice the GroupNorm is normalizing over `channels_per_group` instead of `spatial`. The `dim=-1` that was `feature` in LayerNorm became `channel-group-index` in GroupNorm, silently.

Each normalization normalizes over different coordinates. Each uses a position number to say which one. Switch from one to another, and every `dim` must be audited—because `dim=-1` means `feature` in LayerNorm, `channel` in GroupNorm, and nothing at all in RMSNorm.

Part III built the compiler—the proof that names can be checked mechanically. Part IV asks the practical question: does any of this matter for real code? The next three chapters demonstrate the answer across three domains, each escalating the stakes.

Chapter 11 asks: **does the pattern hold?** Normalization — the simplest skeleton, the fewest coordinates. If names don't earn their keep here, they don't earn it anywhere.

Chapter 12 asks: **what does the pattern reveal?** Attention — where self-attention and cross-attention have identical positional code, and the distinction is in runtime shapes, not in source.

Chapter 13 asks: **what does the pattern prevent?** Physics — the oldest domain, where integer indices have been silently swapping coordinates since Fortran, and the bugs produce plausible-but-wrong results that only a physicist's eye catches.

This chapter takes the first question. Three normalization functions appear below in both PyTorch and Einlang, side by side.

---

## The Normalization Skeleton

LayerNorm, RMSNorm, GroupNorm, and InstanceNorm share a single skeleton: reduce to get statistics, broadcast them back, apply elementwise. The four functions differ only in *which coordinates* the reduction consumes. The PyTorch and Einlang versions are laid side by side, and the coordinate names tell the story.

---

## LayerNorm

Given an input of shape `(batch, seq, feature)`, LayerNorm normalizes across the `feature` dimension for each `(batch, seq)` position independently.

**PyTorch:**

```python
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta
```

`dim=-1` is correct as long as `feature` is the last dimension. It always is—until a refactoring makes it not. `keepdim=True` is needed so the broadcast aligns; forgetting it produces a silent shape mismatch in the gradient.

**Einlang:**

```rust
fn layer_norm[feature](x: [f32; ..b, feature], gamma: [f32; feature], beta: [f32; feature])
    -> [f32; ..b, feature]
{
    let mean[..b] = mean[feature](x[..b, feature]);
    let centered[..b, feature] = x[..b, feature] - mean[..b];
    let var[..b] = mean[feature](centered[..b, feature] ** 2.0);
    (centered[..b, feature] / (var[..b] ** 0.5 + 1e-5)) * gamma[feature] + beta[feature]
}
```

What the Einlang version makes visible:
- `mean[feature]` says "I am reducing over `feature`." The name is in the bracket.
- `mean[..b]` says "`mean` only has batch dimensions." The broadcast over `feature` is explicit in the subtraction `x[..b, feature] - mean[..b]`—`mean` omits `feature`, so it broadcasts along it.
- `gamma[feature]` says "gamma aligns with the feature dimension." The pack `..b` absorbs whatever batch structure exists.

If the input changes from `(batch, seq, feature)` to `(batch, feature, seq)`, the Einlang code still works—`..b` now absorbs `(batch,)` and `feature` is at position 1 instead of 2. The PyTorch code silently normalizes over `seq` instead of `feature`.

---

## RMSNorm

RMSNorm is simpler than LayerNorm: no mean subtraction, just scaling by the root-mean-square.

**PyTorch:**

```python
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma
```

**Einlang:**

```rust
fn rms_norm[feature](x: [f32; ..b, feature], gamma: [f32; feature])
    -> [f32; ..b, feature]
{
    let sq[..b, feature] = x[..b, feature] ** 2.0;
    let ms[..b] = mean[feature](sq[..b, feature]);
    x[..b, feature] / (ms[..b] ** 0.5 + 1e-5) * gamma[feature]
}
```

The skeleton is identical to LayerNorm—reduce over `feature`, broadcast back along it, apply elementwise. The difference is only *which statistics* are computed. In PyTorch, LayerNorm and RMSNorm are different classes with different internal logic but identical `dim=-1` interfaces. The fact that they share a skeleton is invisible in the code. In Einlang, you can overlay the two functions and see that only the body differs—the coordinate contract is the same.

---

## GroupNorm

GroupNorm divides channels into groups and normalizes within each group. This requires splitting the `channel` dimension into `(group, channel_per_group)` and reducing over both `channel_per_group` and the spatial dimensions.

**PyTorch:**

```python
class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        N, C, H, W = x.shape
        G = self.num_groups
        x = x.reshape(N, G, C // G, H, W)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.reshape(N, C, H, W)
        return x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
```

The reshape-permute-reshape dance is the positional price of grouping. `dim=(2, 3, 4)` means "reduce over channel_per_group, height, and width"—but those positions are only correct after the reshape. The reader must mentally compile the grouping semantics from the reshape chain: `reshape` splits channels into groups, `mean(dim=(2,3,4))` reduces within each group, `reshape` merges them back. The grouping is a manual compilation step, performed by the programmer, invisible in the source.

Before you read the Einlang version, stop and ask: from the PyTorch code alone, which coordinates does `dim=(2, 3, 4)` reduce over? You know because the comment says `N, C, H, W` and you counted positions after the reshape. Now ask: if a temporal dimension is prepended next month, what does `dim=(2, 3, 4)` reduce over? You can't know without redoing the positional arithmetic. The answer is in the positions. The positions change. The code doesn't tell you.

**Einlang:**

```rust
fn group_norm[group, c_in_group, ..s](
    x: [f32; ..b, group, c_in_group, ..s],
    gamma: [f32; group, c_in_group],
    beta: [f32; group, c_in_group]
) -> [f32; ..b, group, c_in_group, ..s]
{
    let mean[..b, group] = mean[c_in_group, ..s](
        x[..b, group, c_in_group, ..s]
    );
    let centered[..b, group, c_in_group, ..s] =
        x[..b, group, c_in_group, ..s] - mean[..b, group];
    let var[..b, group] = mean[c_in_group, ..s](
        centered[..b, group, c_in_group, ..s] ** 2.0
    );
    (centered[..b, group, c_in_group, ..s]
        / (var[..b, group] ** 0.5 + 1e-5))
        * gamma[group, c_in_group] + beta[group, c_in_group]
}
```

What the Einlang version makes visible:
- `mean[c_in_group, ..s]` names exactly which coordinates are being reduced. No `dim=(2, 3, 4)` whose meaning depends on a reshape.
- `gamma[group, c_in_group]` aligns with two coordinates. No `.view(1, -1, 1, 1)` to manually position the broadcast. The omission of batch and spatial from gamma's brackets is the megaphone at work—gamma speaks on group and c_in_group, stays silent on batch and spatial, and the silence is the broadcast.
- No reshape is needed because the coordinates `group` and `c_in_group` are separate from the start. The function signature declares the grouped layout directly.

---

## InstanceNorm: The Fourth Variant

LayerNorm normalizes over `feature`. GroupNorm normalizes over `c_in_group + spatial`. InstanceNorm normalizes over `spatial` alone—one statistic per channel per sample. It is used in style transfer, where the "style" of an image is captured by per-channel statistics.

**PyTorch:**

```python
class InstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        # x: (N, C, H, W)
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
```

`dim=(2, 3)` means "reduce over H and W." But how do you know H and W are at positions 2 and 3? Because the comment says `# x: (N, C, H, W)`. If the input has shape `(N, C, L)` for 1D, `dim=(2,)`. If it has shape `(N, T, C, H, W)` for video, `dim=(3, 4)`. The `dim` tuple is a function of the input layout. Change the layout, and every normalization call must be audited.

**Einlang:**

```rust
fn instance_norm[..s, channel](x: [f32; ..b, channel, ..s],
    gamma: [f32; channel], beta: [f32; channel])
    -> [f32; ..b, channel, ..s]
{
    let mean[..b, channel] = mean[..s](x[..b, channel, ..s]);
    let centered[..b, channel, ..s] =
        x[..b, channel, ..s] - mean[..b, channel];
    let var[..b, channel] = mean[..s](
        centered[..b, channel, ..s] ** 2.0
    );
    (centered[..b, channel, ..s] / (var[..b, channel] ** 0.5 + 1e-5))
        * gamma[channel] + beta[channel]
}
```

`..s` absorbs however many spatial dimensions there are—1, 2, 3. The reduction bracket `mean[..s]` doesn't change. The skeleton is the same as LayerNorm, RMSNorm, and GroupNorm. Only the coordinate in the bracket differs.

Now overlay all four normalization functions. The differences are exactly which coordinates appear in the reduction bracket:

| Function | Reduction bracket | Broadcast params |
|---|---|---|
| LayerNorm | `mean[feature]` | `gamma[feature]` |
| RMSNorm | `mean[feature]` | `gamma[feature]` |
| InstanceNorm | `mean[..s]` | `gamma[channel]` |
| GroupNorm | `mean[c_in_group, ..s]` | `gamma[group, c_in_group]` |

The body of every function is: reduce to get statistics, subtract-and-divide, scale-and-shift. The reduction bracket is the only structural difference. In the PyTorch versions, this unity is invisible—each function has its own `dim` argument, its own view-reshape chain, its own parameter shape conventions. The skeleton is scattered across four classes.

---

## BatchNorm: Where the Skeleton Breaks

BatchNorm normalizes over the batch dimension. In training, it computes per-feature statistics across the batch. In inference, it uses running averages. The coordinate structure—reduce over `..b`, broadcast back over `feature`—is the same in both modes. The semantic difference is *where the statistics come from*: the current tensor or an accumulated buffer. That distinction is invisible in the positional `dim=0` and in the named `mean[..b]` alike.

```rust
// Both paths share the same coordinate structure:
let batch_mean[feature] = mean[..b](x[..b, feature]);     // training: fresh
let infer_mean[feature] = running_mean[feature];                  // inference: accumulated
```

The reduction bracket names what is consumed. It does not name whether the statistics are fresh or accumulated. That distinction lives in the data dependency graph, not in the coordinate structure. The coordinate skeleton can only carry so much. The rest is in the code's semantics—and being honest about that boundary is as important as celebrating what names can check.


---

## Tracing a Reshape Bug

Let's trace a reshape bug through its entire life, in both notations. This is the bug that the coordinate audit is designed to catch before it reaches production.

**Day 0.** A programmer writes GroupNorm for 4D input `(N, C, H, W)`:

```python
def group_norm(x, num_groups, gamma, beta, eps=1e-5):
    N, C, H, W = x.shape
    x = x.reshape(N, num_groups, C // num_groups, H, W)
    mean = x.mean(dim=(2, 3, 4), keepdim=True)
    var = x.var(dim=(2, 3, 4), keepdim=True)
    return ((x - mean) / (var + eps).sqrt()).reshape(N, C, H, W) * gamma + beta
```

The `dim=(2,3,4)` tuple means: normalizing over `C//num_groups`, `H`, and `W`. The programmer knows this because they can count: dimension 2 is `C//num_groups`, dimension 3 is `H`, dimension 4 is `W`. The code is correct.

**Day 60.** A colleague adds temporal dimension for video input. The tensor is now `(N, T, C, H, W)`. The colleague updates the reshape:

```python
N, T, C, H, W = x.shape
x = x.reshape(N, T, num_groups, C // num_groups, H, W)
mean = x.mean(dim=(2, 3, 4), keepdim=True)  # BUG
```

`dim=(2,3,4)` now means: normalizing over `num_groups`, `C//num_groups`, and `H`. Wait — `num_groups` at position 2, `C//num_groups` at position 3, `H` at position 4. But `W` is at position 5. And `T` is at position 1. The tuple `(2,3,4)` needs to be `(3,4,5)`. The programmer forgets. The code runs. The shapes match because `keepdim=True` preserves the reduced dimensions. The bug is: GroupNorm is now normalizing over `(num_groups, c_in_group, H)` instead of `(c_in_group, H, W)`. Width is not normalized. Group is normalized — collapsing the grouped structure.

The loss still descends. The model still produces video outputs. But the normalization is wrong. The bug will surface as "the model performs worse on wide videos."

Now replay in Einlang:

```rust
// Original
fn group_norm[g, c_in_group, ..s](x: [f32; ..b, g, c_in_group, ..s], ...)

// With temporal dimension added — same signature
fn group_norm[g, c_in_group, ..s](x: [f32; ..b, t, g, c_in_group, ..s], ...)
```

The signature absorbs `t` into `..b` (if it's a leading dimension) or `..s` (if temporal is treated as spatial). The reduction bracket `mean[c_in_group, ..s]` doesn't change. The coordinates being reduced are still named `c_in_group` and `..s`. The position of those coordinates in the tensor layout doesn't matter — the names find them.

The bug doesn't happen. Not because the programmer is smarter. Because the notation doesn't require positional arithmetic. The coordinate names abstract over positions. Adding a dimension changes which positions the coordinates map to, but the reduction bracket still names the same coordinates. No `dim` tuple to update. No reshape chain to re-align.

Pause here. You just watched a real bug trace across two notations. In the PyTorch version, the bug takes three days to surface, survives integration tests, and produces a model that trains but performs worse on wide videos. In the Einlang version, the bug cannot be written—`mean[c_in_group, ..s]` still means all channel-group and spatial dimensions regardless of where `t` is inserted. The difference is not that the Einlang programmer is more careful. The difference is that the notation has a place for the fact "I am normalizing over channel groups and spatial dimensions," and the PyTorch notation encodes that fact as a tuple of positions that silently rot when the layout changes.

What facts in your own codebase are encoded as positional tuples?

---

## The Coordinate Audit

Every normalization function can be audited with three questions, each a specialization of the broadcast self-audit from Chapter 4:

1. **Which coordinates does the reduction consume?** In `mean[feature]`, the consumed coordinate is `feature`. In `mean[c_in_group, ..s]`, the consumed coordinates are `c_in_group` and all spatial dimensions. The reduction bracket names them directly. In a positional `dim=-1` or `dim=(2,3,4)`, the consumed coordinates must be inferred from the layout and the reshape chain.

2. **Which coordinates do the broadcast parameters align with?** In `gamma[feature]`, gamma aligns with `feature`—the same coordinate that was consumed by the reduction. This is the Inversion Rule from Chapter 4: the reduction consumes `feature`, then `gamma` broadcasts back along `feature`. In a positional `.view(1, -1, 1, 1)`, the alignment is encoded in the view shape, which must be reconstructed by the reader.

3. **Does the normalization axis change meaning if the layout changes?** If the input changes from `(batch, feature)` to `(batch, time, feature)`, does `dim=-1` still mean `feature`? In LayerNorm, yes—`feature` is conventionally the last axis. In GroupNorm after a reshape, no—the positions shift and `dim` must be updated. The Einlang versions are stable under layout changes because the coordinate names don't change, only the positions they map to.

Before leaving normalization, read this function — one you have never seen:

```rust
fn normalize[j, k](x: [f32; ..b, j, k], gamma: [f32; j, k], beta: [f32; j, k]) -> [f32; ..b, j, k] {
    let m[..b] = mean[j, k](x[..b, j, k]);
    let v[..b] = mean[j, k]((x[..b, j, k] - m[..b]) ** 2.0);
    (x[..b, j, k] - m[..b]) / (v[..b] + 1e-5) ** 0.5 * gamma[j, k] + beta[j, k]
}
```

The reduction bracket says `mean[j, k]` — so it consumes `j` and `k`. The output has `{..b, j, k}` and `gamma` has `{j, k}` — so `gamma` broadcasts over `..b`, the difference of those two sets. If `x` changes from `(batch, j, k)` to `(batch, time, j, k)`, the reduction bracket doesn't change. `j` and `k` are found by name. The positional equivalent `dim=(-2, -1)` would survive if `time` is prepended — but not if `time` lands between `j` and `k`. The named bracket doesn't care where `time` lands.

Three questions, three answers, all visible in the signature without reading the body. That is the audit.

Normalization established the baseline: the skeleton holds across four variants, and the coordinate name absorbs layout changes that would silently corrupt a positional `dim=`. Chapter 12 raises the stakes. Normalization has one reduction axis. Attention has five coordinates, three architectural variants, and a runtime cache whose correctness depends on which coordinate is concatenated. The question shifts from *does the pattern hold?* to *what does the pattern reveal that positional code cannot say?*
