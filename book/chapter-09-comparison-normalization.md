---
layout: book
title: "Chapter 9 · Comparison: Normalization"
---

# Chapter 9 · Comparison: Normalization

> "You don't understand a notation until you've seen what it hides."
>
> — The author

*Comparisons · LayerNorm, RMSNorm, GroupNorm in two notations*

---

You are two weeks into a Transformer project. LayerNorm is working. You swap in RMSNorm for the memory-efficient run—same `dim=-1`, same shape, loss looks fine. Then you try GroupNorm on the convolutional front-end. `dim=-1` again. The shapes align. The loss descends. Three days later you notice the GroupNorm is normalizing over `channels_per_group` instead of `spatial`. The `dim=-1` that was `feature` in LayerNorm became `channel-group-index` in GroupNorm, silently.

Each normalization normalizes over different coordinates. Each uses a position number to say which one. Switch from one to another, and every `dim` must be audited—because `dim=-1` means `feature` in LayerNorm, `channel` in GroupNorm, and nothing at all in RMSNorm.

Every chapter so far has been in Einlang. We built the primitives, composed them into functions, traced their gradients, and gave the language a name.

This chapter changes the lens. We take three real normalization functions—LayerNorm, RMSNorm, GroupNorm—and write each in both PyTorch and Einlang, side by side. The question is not "which is better." The question is: **what does each notation make visible, and what does each notation hide?**

---

## The Normalization Skeleton

Chapter 5 showed that LayerNorm, RMSNorm, GroupNorm, and InstanceNorm share a single skeleton: reduce to get statistics, broadcast them back, apply elementwise. The four functions differ only in *which coordinates* the reduction consumes. This chapter does not repeat that skeleton table. Instead, we lay the PyTorch and Einlang versions side by side and let the coordinate names tell the story.

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
fn layer_norm[feature](x: [f32; ..batch, feature], gamma: [f32; feature], beta: [f32; feature])
    -> [f32; ..batch, feature]
{
    let mean[..batch] = mean[feature](x[..batch, feature]);
    let centered[..batch, feature] = x[..batch, feature] - mean[..batch];
    let var[..batch] = mean[feature](centered[..batch, feature] ** 2.0);
    (centered[..batch, feature] / (var[..batch] ** 0.5 + 1e-5)) * gamma[feature] + beta[feature]
}
```

What the Einlang version makes visible:
- `mean[feature]` says "I am reducing over `feature`." The name is in the bracket.
- `mean[..batch]` says "`mean` only has batch dimensions." The broadcast over `feature` is explicit in the subtraction `x[..batch, feature] - mean[..batch]`—`mean` omits `feature`, so it broadcasts along it.
- `gamma[feature]` says "gamma aligns with the feature dimension." The pack `..batch` absorbs whatever batch structure exists.

If the input changes from `(batch, seq, feature)` to `(batch, feature, seq)`, the Einlang code still works—`..batch` now absorbs `(batch,)` and `feature` is at position 1 instead of 2. The PyTorch code silently normalizes over `seq` instead of `feature`.

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
fn rms_norm[feature](x: [f32; ..batch, feature], gamma: [f32; feature])
    -> [f32; ..batch, feature]
{
    let sq[..batch, feature] = x[..batch, feature] ** 2.0;
    let ms[..batch] = mean[feature](sq[..batch, feature]);
    x[..batch, feature] / (ms[..batch] ** 0.5 + 1e-5) * gamma[feature]
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

**Einlang:**

```rust
fn group_norm[group, c_in_group, ..spatial](
    x: [f32; ..batch, group, c_in_group, ..spatial],
    gamma: [f32; group, c_in_group],
    beta: [f32; group, c_in_group]
) -> [f32; ..batch, group, c_in_group, ..spatial]
{
    let mean[..batch, group] = mean[c_in_group, ..spatial](
        x[..batch, group, c_in_group, ..spatial]
    );
    let centered[..batch, group, c_in_group, ..spatial] =
        x[..batch, group, c_in_group, ..spatial] - mean[..batch, group];
    let var[..batch, group] = mean[c_in_group, ..spatial](
        centered[..batch, group, c_in_group, ..spatial] ** 2.0
    );
    (centered[..batch, group, c_in_group, ..spatial]
        / (var[..batch, group] ** 0.5 + 1e-5))
        * gamma[group, c_in_group] + beta[group, c_in_group]
}
```

What the Einlang version makes visible:
- `mean[c_in_group, ..spatial]` names exactly which coordinates are being reduced. No `dim=(2, 3, 4)` whose meaning depends on a reshape.
- `gamma[group, c_in_group]` aligns with two coordinates. No `.view(1, -1, 1, 1)` to manually position the broadcast.
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
fn instance_norm[..spatial, channel](x: [f32; ..batch, channel, ..spatial],
    gamma: [f32; channel], beta: [f32; channel])
    -> [f32; ..batch, channel, ..spatial]
{
    let mean[..batch, channel] = mean[..spatial](x[..batch, channel, ..spatial]);
    let centered[..batch, channel, ..spatial] =
        x[..batch, channel, ..spatial] - mean[..batch, channel];
    let var[..batch, channel] = mean[..spatial](
        centered[..batch, channel, ..spatial] ** 2.0
    );
    (centered[..batch, channel, ..spatial] / (var[..batch, channel] ** 0.5 + 1e-5))
        * gamma[channel] + beta[channel]
}
```

`..spatial` absorbs however many spatial dimensions there are—1, 2, 3. The reduction bracket `mean[..spatial]` doesn't change. The skeleton is the same as LayerNorm, RMSNorm, and GroupNorm. Only the coordinate in the bracket differs.

Now overlay all four normalization functions. The differences are exactly which coordinates appear in the reduction bracket:

| Function | Reduction bracket | Broadcast params |
|---|---|---|
| LayerNorm | `mean[feature]` | `gamma[feature]` |
| RMSNorm | `mean[feature]` | `gamma[feature]` |
| InstanceNorm | `mean[..spatial]` | `gamma[channel]` |
| GroupNorm | `mean[c_in_group, ..spatial]` | `gamma[group, c_in_group]` |

The body of every function is: reduce to get statistics, subtract-and-divide, scale-and-shift. The reduction bracket is the only structural difference. In the PyTorch versions, this unity is invisible—each function has its own `dim` argument, its own view-reshape chain, its own parameter shape conventions. The skeleton is scattered across four classes.

---

## The Reshape Bug, Revisited

Chapter 5 showed the GroupNorm reshape chain breaking when a temporal dimension is added: the `dim=(2,3,4)` tuple encodes positions that are only correct after the reshape, and adding `T` at position 1 shifts every subsequent position. The programmer must recalculate `(2,3,4)` → `(3,4,5)`. If they forget—and they do—the normalization collapses across groups, the loss still descends, and the model ships with blurrier segmentations.

In the Einlang version, adding `T` changes nothing. `mean[c_in_group, ..spatial]` names the reduced coordinates directly. No positions to shift. No tuples to renumber. The coordinate names abstract over layout—the compiler maps them to whatever positions they occupy.

If the input shape changes from `(batch, feature)` to `(batch, seq, feature)`, which line of code changes?

In the PyTorch LayerNorm: nothing—`dim=-1` still refers to the last axis, and if `feature` is still last, it still works. In the PyTorch GroupNorm: the reshape chain breaks. In the Einlang versions: nothing changes. The coordinate names in the brackets don't change, only the positions they map to, and the compiler handles that mapping.

Now ask the question in reverse. If the code *doesn't* change but the layout does—is that safety, or is it silence?

## BatchNorm: Where the Skeleton Breaks

BatchNorm normalizes over the batch dimension, not the feature dimension. In training, it computes per-feature statistics across the batch. In inference, it uses running averages. This introduces a complication that none of the previous normalizations had: the normalization depends on the data distribution, and the data distribution changes between training and inference.

```python
# PyTorch BatchNorm
class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)  # over batch
            var = x.var(dim=0, unbiased=False)
            # update running stats...
        else:
            mean = self.running_mean
            var = self.running_var
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta
```

The coordinate story: `dim=0` reduces over `batch`. The running statistics are shape `(feature,)`. In inference, they broadcast over the batch. The code works. But the distinction between "statistics computed from the current batch" (training) and "statistics accumulated from many batches" (inference) is captured in an `if self.training` branch and a `register_buffer` call. The coordinate structure is identical in both modes. The semantic difference—fresh vs. accumulated statistics—is a runtime flag, not a structural distinction.

In Einlang, the training/inference distinction could be captured in the type system by whether the batch statistic carries a time coordinate:

```rust
fn batch_norm[feature, ..batch](x: [f32; ..batch, feature],
    gamma: [f32; feature], beta: [f32; feature],
    running_mean: [f32; feature], running_var: [f32; feature])
    -> [f32; ..batch, feature]
{
    // Training: compute fresh statistics from current batch
    let batch_mean[feature] = mean[..batch](x[..batch, feature]);
    let batch_var[feature] = mean[..batch](
        (x[..batch, feature] - batch_mean[feature]) ** 2.0
    );
    // Training path
    let train_out[..batch, feature] =
        (x[..batch, feature] - batch_mean[feature])
        / (batch_var[feature] ** 0.5 + 1e-5)
        * gamma[feature] + beta[feature];
    // Inference path
    let infer_out[..batch, feature] =
        (x[..batch, feature] - running_mean[feature])
        / (running_var[feature] ** 0.5 + 1e-5)
        * gamma[feature] + beta[feature];
    // ...
}
```

Both paths reduce over `..batch` to produce `{feature}`-shaped statistics. The difference is whether those statistics came from the current tensor or from a running accumulator. The reduction bracket is the same. The source of the statistics is different. The naming makes explicit what `self.training` hides: in training, `batch_mean` is derived from `x` (creating a data dependency for the backward pass); in inference, `running_mean` is a constant (no gradient flows through it).

This is the boundary where the coordinate notation meets the runtime. The reduction bracket names what is consumed. It does not name whether the statistics are fresh or accumulated. That distinction lives in the data dependency graph, not in the coordinate structure. The coordinate skeleton can only carry so much. The rest is in the code's semantics.


---

*Open your LayerNorm or RMSNorm implementation. Find `dim=-1`. Ask: is `-1` still the feature dimension? How do you know? The answer should be in the code, not in your memory of what the tensor shape was when you wrote it.*

---

### Stop and Think: Audit Your Normalization

Find the normalization code in your current project. It might be a `LayerNorm` call, an `RMSNorm`, a `GroupNorm`, or a custom `(x - mean) / std * gamma + beta`. For each one:

1. **What coordinate does the reduction consume?** Not "which position" — which coordinate. If the code says `x.mean(dim=-1)`, the answer depends on what's at position -1. Trace it back to the data loader or the declaration. Write the coordinate name down.

2. **What would break if the dimension order changed?** If someone added a `head` dimension or a `time` dimension, would the reduction still consume the right coordinate? For LayerNorm with `dim=-1`, the answer might be "yes, feature is conventionally last." For GroupNorm with `dim=(2,3,4)`, the answer is almost certainly "no — those positions correspond to specific dimensions in the reshape chain."

3. **Is the reduction semantically correct?** LayerNorm normalizes over `feature`. GroupNorm normalizes over `c_in_group` and `..spatial`. Are you normalizing over the coordinates you intend to normalize over? If the answer requires running the code to check shapes, the notation isn't carrying the intent.

Now imagine each normalization written with coordinate names. `mean[feature](x)` instead of `x.mean(dim=-1)`. `mean[c_in_group, ..spatial](x)` instead of `x.reshape(...).mean(dim=(2,3,4))`. For each one, the coordinate name answers question 1. The name's independence from position answers question 2. The semantic meaning of the name answers question 3.

The normalization audit is a specialization of the broadcast self-audit from Chapter 4. Every normalization is a reduce-broadcast-elementwise pattern. The reduction bracket names the consumed coordinates. The broadcast parameters name the alignment. The skeleton is the same. The names change. The audit is the same. The questions change.

---

## Tracing a Reshape Bug

Let's trace a reshape bug through its entire life, in both notations. This is the bug that the coordinate audit is designed to catch before it reaches production.

**Day 0.** A programmer writes GroupNorm for 4D input `(N, C, H, W)`:

```python
def group_norm(x, num_groups, gamma, beta):
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
fn group_norm[g, c_in_group, ..spatial](x: [f32; ..batch, g, c_in_group, ..spatial], ...)

// With temporal dimension added — same signature
fn group_norm[g, c_in_group, ..spatial](x: [f32; ..batch, t, g, c_in_group, ..spatial], ...)
```

The signature absorbs `t` into `..batch` (if it's a leading dimension) or `..spatial` (if temporal is treated as spatial). The reduction bracket `mean[c_in_group, ..spatial]` doesn't change. The coordinates being reduced are still named `c_in_group` and `..spatial`. The position of those coordinates in the tensor layout doesn't matter — the names find them.

The bug doesn't happen. Not because the programmer is smarter. Because the notation doesn't require positional arithmetic. The coordinate names abstract over positions. Adding a dimension changes which positions the coordinates map to, but the reduction bracket still names the same coordinates. No `dim` tuple to update. No reshape chain to re-align.

---

## The Coordinate Audit for Normalization

Every normalization function can be audited with three questions. They are the same questions as the broadcast self-audit from Chapter 4, specialized for normalization's three-step skeleton:

1. **Which coordinates does the reduction consume?** In `mean[feature]`, the consumed coordinate is `feature`. In `mean[c_in_group, ..spatial]`, the consumed coordinates are `c_in_group` and all spatial dimensions. The reduction bracket names them directly. In a positional `dim=-1` or `dim=(2,3,4)`, the consumed coordinates must be inferred from the layout and the reshape chain.

2. **Which coordinates do the broadcast parameters align with?** In `gamma[feature]`, gamma aligns with `feature`—the same coordinate that was consumed by the reduction. This is the Inversion Rule from Chapter 4: the reduction consumes `feature`, then `gamma` broadcasts back along `feature`. In a positional `.view(1, -1, 1, 1)`, the alignment is encoded in the view shape, which must be reconstructed by the reader.

3. **Does the normalization axis change meaning if the layout changes?** If the input changes from `(batch, feature)` to `(batch, time, feature)`, does `dim=-1` still mean `feature`? In LayerNorm, yes—`feature` is conventionally the last axis. In GroupNorm after a reshape, no—the positions shift and `dim` must be updated. The Einlang versions are stable under layout changes because the coordinate names don't change, only the positions they map to.

Three questions. The next chapter applies them to a harder case: attention, where self-attention and cross-attention have identical PyTorch code—and where the Square Matrix Test returns.
