---
layout: book
title: "Chapter 8 · Comparison: Normalization"
---

# Chapter 8 · Comparison: Normalization

> "You don't understand a notation until you've seen what it hides."
>
> — The author

*Comparisons · LayerNorm, RMSNorm, GroupNorm in two notations*

---

You are two weeks into a Transformer project. LayerNorm is working. You swap in RMSNorm for the memory-efficient run—same `dim=-1`, same shape, loss looks fine. Then you try GroupNorm on the convolutional front-end. `dim=-1` again. The shapes align. The loss descends. Three days later you notice the GroupNorm is normalizing over `channels_per_group` instead of `spatial`. The `dim=-1` that was `feature` in LayerNorm became `channel-group-index` in GroupNorm, silently.

Each normalization normalizes over different coordinates. Each uses a position number to say which one. Switch from one to another, and every `dim` must be audited—because `dim=-1` means `feature` in LayerNorm, `channel` in GroupNorm, and nothing at all in RMSNorm.

Every chapter so far has been in einlang. We built the primitives, composed them into functions, traced their gradients, and gave the language a name.

This chapter changes the lens. We take three real normalization functions—LayerNorm, RMSNorm, GroupNorm—and write each in both PyTorch and einlang, side by side. The question is not "which is better." The question is: **what does each notation make visible, and what does each notation hide?**

---

## The Normalization Skeleton

Every normalization follows the same three-step skeleton:

1. **Reduce** to get statistics (mean, variance, max).
2. **Broadcast** the statistics back to the original shape.
3. **Apply** elementwise (subtract, divide, scale, shift).

The difference between LayerNorm, RMSNorm, and GroupNorm is only *which coordinates* are reduced over and *which coordinates* the broadcast parameters align with. In a positional notation, these differences are buried in `dim` arguments and reshape chains. In a named notation, they are visible in the coordinate names.

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

What the einlang version makes visible:
- `mean[feature]` says "I am reducing over `feature`." The name is in the bracket.
- `mean[..batch]` says "`mean` only has batch dimensions." The broadcast over `feature` is explicit in the subtraction `x[..batch, feature] - mean[..batch]`—`mean` omits `feature`, so it broadcasts along it.
- `gamma[feature]` says "gamma aligns with the feature dimension." The pack `..batch` absorbs whatever batch structure exists.

If the input changes from `(batch, seq, feature)` to `(batch, feature, seq)`, the einlang code still works—`..batch` now absorbs `(batch,)` and `feature` is at position 1 instead of 2. The PyTorch code silently normalizes over `seq` instead of `feature`.

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

The skeleton is identical to LayerNorm—reduce over `feature`, broadcast back along it, apply elementwise. The difference is only *which statistics* are computed. In PyTorch, LayerNorm and RMSNorm are different classes with different internal logic but identical `dim=-1` interfaces. The fact that they share a skeleton is invisible in the code. In einlang, you can overlay the two functions and see that only the body differs—the coordinate contract is the same.

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

What the einlang version makes visible:
- `mean[c_in_group, ..spatial]` names exactly which coordinates are being reduced. No `dim=(2, 3, 4)` whose meaning depends on a reshape.
- `gamma[group, c_in_group]` aligns with two coordinates. No `.view(1, -1, 1, 1)` to manually position the broadcast.
- No reshape is needed because the coordinates `group` and `c_in_group` are separate from the start. The function signature declares the grouped layout directly.

---

## What the Comparison Reveals

The three PyTorch implementations are correct. They run. They're fast. The problem is not that they don't work—it's that the coordinate story is stored in the programmer's head, not in the code.

- `dim=-1` in LayerNorm means `feature`. Until it doesn't.
- `dim=(2, 3, 4)` in GroupNorm means `c_in_group, H, W`. But only after the reshape.
- `.view(1, -1, 1, 1)` means "gamma is silent on batch, height, and width." The code doesn't say that. You deduce it from the shape.
- `keepdim=True` means "I need this for broadcasting later." The forward intent and backward consequence are connected only by convention.

In the einlang versions, these facts are in the brackets. The coordinate that is reduced is named. The coordinates that are broadcast are visible by their absence. The skeleton—reduce, broadcast, apply—is the same across all three functions, and the similarity is visible in the code.

The next chapter compares attention mechanisms—where the gap between what the code says and what the code means grows wider still.
