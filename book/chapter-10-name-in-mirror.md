---
layout: book
title: "Chapter 10 · The Name in the Mirror"
---

# Chapter 10 · The Name in the Mirror

> "A problem well stated is a problem half solved."
>
> — Charles Kettering

*Construction · Range inference, shape analysis, type propagation, constraint solving, and execution strategies*

---

Your program passed all five rules. You run it. It crashes.

`index out of bounds: 32`

The index `i` was declared. The checker saw it. But the checker only asks "is this name declared?"—not "what values can it take?" The declaration said `i in 0..N`, and the compiler never computed `N`. It had no machinery to compute `N`. The name was verified. The range was not.

Now the compiler has to actually answer the question it skipped: what does each name *imply*? Before it can lower `class` into `axis=1` or `batch` into a loop, it must answer three questions about every expression in the tree:

1. **Range.** `i` was declared — but what values can it take? 0 to 10? 0 to `batch_size`?
2. **Shape.** Which coordinates survive? What is the output shape?
3. **Type.** Is this expression `f32` or `f64`? Integer or float?

The 15-line checker asks "is this name declared?" These three questions ask "what does this name imply?" The checker verifies. The analyzer infers. Together they form the complete compiler frontend—one that can take a program without a single runtime value and determine the shape of every tensor in it, so `i in 0..32` is known before `data[i]` is ever evaluated.

---

## Range Inference

You write:

```rust
let result[i] = data[i] * 2.0;
```

`data` has shape `[N]`. You never wrote `i`'s range. The compiler infers it.

Before reading how the compiler does it, do it by hand. `data` has shape `[N]`. What values can `i` take? You know instantly: `0` to `N-1`. You didn't run the program. You looked at `data[i]` and `data`'s shape. The compiler does the same thing.

The algorithm: find every array access where the variable appears, back-compute a bound from each array's declared shape, and intersect. Here, `data[i]` is the only access. `data` has shape `[N]`, so `i < N`. Range: `i in [0, N)`.

Now a two-access case:

```rust
let result[i] = arr[i] + arr[i + 1];
```

`arr` has shape `[N]`. Two accesses:

- `arr[i]` → `i < N` → `i in [0, N)`
- `arr[i + 1]` → `i + 1 < N` → `i in [0, N - 1)`

Intersection: `i in [0, N - 1)`. The compiler inferred that `i` cannot reach `N - 1` because `arr[i + 1]` would be out of bounds. No annotation. No `range` declaration. The name `i`, appearing in two positions, carries enough information.

Now a case with two different arrays:

```rust
let result[i] = x[i] + y[2 * i];
```

`x` has shape `[N]`. `y` has shape `[M]`.

- `x[i]` → `i < N` → `i in [0, N)`
- `y[2 * i]` → `2 * i < M` → `i < ceil(M / 2)` → `i in [0, ceil(M / 2))`

Intersection: `i in [0, min(N, ceil(M / 2)))`.

The compiler doesn't know `N` or `M` at compile time — they may be runtime values. But it knows the *relationship*: `i`'s range is bounded by both. The lowering pass will emit `for i in range(min(N, ceil(M / 2)))`. The bound is symbolic. The check is structural.

### When Inference Fails

Inference needs at least one array access where the variable appears as a simple index (not inside a pack, not in an arithmetic expression that the constraint solver can't invert). If you write:

```rust
let result[i] = f(i);
```

where `f` is an opaque function, the compiler has no array shape to back-compute from. The variable `i` is declared but unconstrained. The compiler reports:

```
error: cannot infer range for index variable `i`
  → no array access found where `i` appears as a direct index
  → declare the range explicitly: `i in 0..N`
```

This is not a limitation of the inference algorithm. It is a boundary of the notation: index variables exist to index into arrays. If a variable never indexes into an array, the compiler asks what it *is* indexing into.

### Packs and Ranges

Packs don't change the inference. When the compiler sees:

```rust
let result[..b, i] = x[..b, i] + 1.0;
```

it resolves `..b` first — anchoring on `i` to determine which dimensions `..b` absorbs. Then it infers `i`'s range from `x`'s shape at the position where `i` appears. The pack is a group of dimensions. The index variable is a single position. They don't interfere.

### The Principle

Range inference works because of the coordinate-name philosophy. Every index variable appears literally in array index expressions. Every array has a declared shape. The compiler traces the variable through the accesses, reads the shapes, and intersects. No data needed. No execution needed. The names carry the constraints.

You can do this by hand. When you write `let result[i] = arr[i] + arr[i + 1]`, you know `i` can't reach `N - 1`. You don't run the program to know it. You look at `arr[i + 1]` and mentally subtract one from the bound. The compiler does the same thing — systematically, for every variable, without getting tired.

---

## Shape Analysis

Once every index variable has a range, shapes follow mechanically. The shape of a tensor is the list of its surviving coordinates, each replaced by its range length.

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

`A` has shape `[N, K]`. `B` has shape `[K, M]`. The reduction `sum[k]` consumes `k`. Surviving coordinates: `{i, j}`. Ranges: `i in [0, N)`, `j in [0, M)`. Output shape: `[N, M]`.

The compiler traces this through the IR tree. Each node has a coordinate set. Index nodes introduce coordinates. Reductions subtract them. The output shape is the outermost node's coordinate set, with each coordinate replaced by its range.

A shape mismatch is caught here. If the user declares:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);  // C declared with [i, j]
```

but the body somehow loses `j` — say, through a mistaken reduction — the compiler reports:

```
error: declared output coordinates {i, j} do not match body coordinates {i}
  → missing coordinate: j
```

This is Rule 5 from Chapter 9, applied to shapes. The declaration says one thing. The body says another. The compiler reports the difference.

---

## Type Propagation

Types flow from leaves to root. It is the simplest pass in the compiler — one traversal of the tree, bottom-up, no backtracking.

Consider a fragment of a linear layer:

```rust
let x: [f32; batch, in] = input();
let W: [f32; out, in] = param();
let scaled[batch, out] = sum[in](x[batch, in] * W[out, in]) * 0.5;
```

Walk the tree of `sum[in](x[batch, in] * W[out, in]) * 0.5`:

```
        * (f32)
       / \
      /   \ 
     /    0.5 (f32)
    /
sum[in] (f32)
    |
    * (f32)
   / \
  /   \
x     W
(f32) (f32)
```

`x` is `f32` — declared. `W` is `f32` — declared. `x * W` — both `f32`, result `f32`. `sum[in](...)` — reduces over `in`, element type unchanged, `f32`. `0.5` — literal, `f32`. `sum_result * 0.5` — both `f32`, result `f32`. The entire expression is `f32`. Done.

Every literal has a type. `3.14` is `f32`. `5` is `i32`. Every variable inherits its type from its declaration. Every operation propagates the types of its operands upward. The only check: index variables used in arithmetic (`i + 1`) must be integer-typed. If a float variable appears as an index, the compiler reports it.

Types and shapes are orthogonal. A tensor's type says what the elements are. Its shape says how many elements there are and how they are arranged. The type propagates through the element values. The shape propagates through the coordinate structure. They meet only at the leaves — the array declarations that specify both.

The pass is twelve lines of code. But without it, the compiler cannot generate correct code — it wouldn't know whether to emit `np.float32` or `np.float64`. The simplest pass. Also the one that would silently corrupt every number in the program if it were wrong.

---

## The Constraint Solver

You write a convolution:

```rust
let output[b, oc, oh, ow] = sum[ic, kh, kw](
    input[b, ic, oh + kh, ow + kw] * weight[oc, ic, kh, kw]
);
```

The ranges are known: `oh` up to `output_height`, `kh` up to `kernel_height`. But `oh + kh` indexes into `input` — and `input` has `input_height`. Is `oh + kh` always less than `input_height`?

You check by hand. The worst case: `oh = output_height - 1`, `kh = kernel_height - 1`. Sum: `output_height + kernel_height - 2`. If that's less than `input_height`, safe. If not, out of bounds. Simple arithmetic.

The compiler does the same check — automatically, for every index expression in the program. It is a miniature theorem prover, but it only knows one trick: pattern-matching on the expression tree.

Take `(i * 2 + offset) < N`. The solver walks the tree:

1. **Top level is addition.** `(i * 2) + offset < N`. Subtract offset: `i * 2 < N - offset`.
2. **Recurse into multiplication.** `i * 2 < N - offset`. Divide by 2 (with ceiling): `i < ceil((N - offset) / 2)`.
3. **Base case.** `i < bound`. Range is `[0, ceil(bound))`.

Three rewrites. The solver chains them mechanically. Addition → subtract. Multiplication → divide. Division → multiply. The ceiling is `(a + b - 1) / b` at the IR level, so it works for any input size.

Now ask: what can the solver *not* prove?

**Negative coefficients.** The solver reads `k - target < bound`. To isolate `target`, it would need to add `target` to both sides and subtract `bound` — producing a lower bound. The solver only computes upper bounds. `k - target < bound` is recognized as unsolvable by the current pass. The pattern is there. The solver sees it. It says: *I can't.*

**Modulo.** `i % 2`. The solver stares at it. Modulo is not addition, not multiplication, not division. The solver has no rewrite rule for it. The range of `i % 2` is `{0, 1}`, but plugging that into `data[i % 2]` requires reasoning about value sets, not arithmetic bounds. Different kind of solver. Different pass. Not this one.

**Nonlinear.** `i * j`, `i ** 2`. Products of two variables. The solver doesn't handle them — and almost nothing in a tensor program asks it to. When these appear, the programmer is doing something unusual, and the compiler's silence is the right response: *declare the range yourself.*

When the solver fails:

```
error: cannot prove index expression `oh + kh` stays within bounds of `input`
  → constraint: oh + kh < input_height
  → oh in [0, output_height), kh in [0, kernel_height)
  → try declaring the range explicitly or simplifying the index expression
```

The error names every variable. Shows every range. States the constraint. The programmer reads it and knows exactly what the compiler couldn't prove — and exactly what to fix.

This is the difference between a named error and a positional one. The positional equivalent is `IndexError: dimension 3 out of bounds`. Which dimension is 3? What was supposed to be in bounds? The number answers neither question. The names answer both.

The solver is not a general-purpose theorem prover. It is a pattern matcher for the expressions that appear in real tensor indices — linear combinations with positive coefficients, the arithmetic of stencils and convolutions and pooling windows. Every expression it proves safe is one less silent bug. Every expression it cannot prove is flagged, with names, before the program runs.

---

Before moving on, trace the inference for three more cases. `data` has shape `[N]`.

`let result[i] = data[i] + data[i + 2];`

Two accesses. `data[i]` says `i < N`. `data[i + 2]` says `i + 2 < N`, so `i < N - 2`. Intersection: `i` stops at `N - 2`. The compiler subtracts two from the bound without being asked. You did the same thing in your head just now.

`let result[i] = data[2 * i] + data[2 * i + 1];`

`2*i < N` and `2*i + 1 < N`. The second is tighter. Intersection: `i < ceil((N-1)/2)`. The compiler divides by two, takes the ceiling, and infers the bound. You didn't run the program. You looked at the index expressions and the shape.

`let result[i] = data[i] + other[3 * i];` — `other` has shape `[M]`.

Two different arrays, two different shapes. `data[i]` gives `i < N`. `other[3*i]` gives `3*i < M` → `i < ceil(M/3)`. Intersection: `i < min(N, ceil(M/3))`. The bound is the tighter of two constraints from two different tensors. The compiler traces the variable through every access, reads every shape, intersects. No annotation. No execution. The names and shapes carry the constraints.

---

## Runtime-Dependent Coordinates

Everything so far assumes ranges are known at compile time. Not all are.

You have a batch of variable-length sequences. `input[batch, seq]`. For sample 0, `seq` has 15 tokens. For sample 1, 23 tokens. For sample 2, 8. The coordinate `seq` means something different in every row.

Try handing this to `np.softmax(logits, axis=1)`. It runs — NumPy doesn't care that row 0 has 15 elements and row 2 has 8. But `axis=1` normalizes over the whole axis, including padding zeros. Your model now attends to nothing. The loss drops — it learned to predict padding. You find the bug three days later.

The compiler saw this coming. `seq` is *ragged* — its extent depends on `batch`. No single `axis=` integer captures that. The compiler cannot emit `np.softmax(logits, axis=1)`. It must emit a loop:

```python
for b in range(batch):
    seq_len = lengths[b]
    row = logits[b, :seq_len]
    m = np.max(row)
    e = np.exp(row - m)
    result[b, :seq_len] = e / np.sum(e)
```

Same coordinate. Different strategy. The compiler chose it — not the programmer.

Three kinds of runtime-dependence trigger this:

**Ragged dimensions.** A coordinate's extent varies with another. Sequence lengths. Graph node counts. Image resolutions in a batch. The compiler verifies the coordinate exists and is used consistently. It cannot lower it to a constant. It emits a dynamic loop, per element, with the correct bound each time.

**Sparse coordinates.** Not every position carries data. A `(batch, class)` logit matrix where only a subset of classes are valid per sample. The `where` clause from Chapter 2 records the mask; the compiler emits it alongside the loop. The coordinate is named. The mask is named. The relationship between them is visible in the source.

**Data-dependent shapes.** The output shape depends on the data, not just the input shapes. `top_k` returns a variable number of elements. Beam search expands dynamically. The compiler cannot know the output shape at compile time because the output shape is the answer.

This is not a flaw. Every tensor compiler faces this boundary, named or positional. The positional compiler says: "axis 1 has dynamic extent." Axis 1 might be `seq`, `class`, `head`, or `feature`. The number doesn't know. The name does. `seq` is ragged. The error names the coordinate. The strategy names the coordinate. The loop bound reads the coordinate's runtime length. The name survives analysis and lowering — it is the thread connecting the static check to the dynamic execution.

---

## Execution Strategies

The compiler has ranges. It has shapes. It knows which coordinates are static and which are ragged. Now it must decide: how to execute.

Three doors.

**Door 1: Vectorized.** Every coordinate has a static range. No ragged dimensions. No data-dependent shapes. The compiler lowers every name to an integer, emits a single NumPy call, and walks away.

```python
m = np.max(logits, axis=1, keepdims=True)
e = np.exp(logits - m)
return e / np.sum(e, axis=1, keepdims=True)
```

`class → axis=1`. `keepdims=True` — not written by the programmer, deduced by the compiler from the broadcast recorded during analysis. The generated code is identical to what you would write by hand, if you were careful.

**Door 2: Scalar.** At least one coordinate is ragged or data-dependent. The compiler emits Python loops — one per dynamic coordinate, with the bound read at runtime.

```python
for b in range(batch):
    seq_len = lengths[b]
    row = logits[b, :seq_len]
    m = np.max(row)
    e = np.exp(row - m)
    result[b, :seq_len] = e / np.sum(e)
```

`seq` is ragged. The compiler knew this — it recorded `seq` as runtime-dependent during analysis. It did not lower `seq` to an integer. It emitted a loop with `seq_len` as a dynamic bound. Same coordinate name. Different execution path.

**Door 3: Hybrid.** Three static coordinates and one ragged one, in the same operation. The compiler vectorizes over the static three and loops over the ragged one. The names tell it which is which. `batch`, `head`, and `d` get `axis=` integers. `seq` gets a loop.

The programmer did not choose the strategy. The compiler did. The programmer wrote the names. The compiler read them, classified each coordinate, and chose the door. Vectorized where possible. Scalar where necessary. The choice is per operation, not per program — a single function can mix static and dynamic coordinates, and the compiler handles each one correctly.

This is lowering. Names go in. Execution comes out.

Three doors, one compiler. The programmer never writes `axis=0` or `axis=-1`. The programmer never chooses between a vectorized loop and a scalar loop. The programmer writes coordinates. The compiler reads them, classifies them, and picks the door. Static coordinates become fused dimensions. Dynamic coordinates become guarded loops. Ragged coordinates become runtime bounds. The same name `seq` can be static in one function and ragged in another. The compiler handles each case per operation, not per program.

The three doors are not three compiler flags. They are three consequences of one fact: the coordinate `seq` has a range, and the range is known or unknown at compile time. The compiler checks that fact during range inference. The execution strategy falls out. Correctness first. Performance follows.

---

## What the Mirror Shows

The 15-line checker from Chapter 9 verified names. The passes in this chapter infer what those names imply. Range. Shape. Type. Proof that the arithmetic stays in bounds.

Together, the checker and the analyzer form a complete frontend. Source goes in. Names are checked. Ranges are inferred. Shapes are derived. Types are propagated. Index arithmetic is proved safe. And then — only then — lowering burns the names into numbers.

The mirror metaphor is not decorative. Analysis is reflection. The compiler holds the program up to itself and asks: what do these names imply that the programmer did not state? The answer is already in the names. The compiler makes it explicit.

You can do every pass by hand. You did range inference at the start of Chapter 9 when you traced `batch` and `class` through softmax and sum. You did shape analysis when you determined that `y` has shape `[batch]`. You did type propagation without thinking about it — `logits` is `f32`, so everything downstream is `f32`. The compiler does what you did, systematically, for every line.

---

## Follow the Name

Take one program. Watch every pass operate on it.

```rust
let x: [f32; batch, in] = input();
let W: [f32; out, in] = param();
let b: [f32; out] = param();
let logits[batch, out] = sum[in](x[batch, in] * W[out, in]) + b[out];
let activated[batch, out] = relu(logits[batch, out]);
```

Five lines. One linear layer. Now follow `batch` through the compiler.

**Checker.** Rule 1 — every coordinate in an index list exists on the tensor. `x` has `{batch, in}`, so `x[batch, in]` passes. `W` has `{out, in}`, so `W[out, in]` passes. `b[out]` passes. Rule 2 — the reduction coordinate `in` appears in both `x[batch, in]` and `W[out, in]`. Passes. Rule 3 — `b[out]` omits `batch`. Broadcast recorded: `b` broadcasts over `batch`. Rule 5 — `sum[in]` consumes `in`. Output `logits` has `{batch, out}`. `relu` preserves coordinates. Output `activated` has `{batch, out}`. Five checks. Zero errors. The program is consistent.

**Range inference.** `batch` appears only in index lists, never as a bound. Its range is inferred from the input — if `x` is declared with `batch in 0..32`, then `batch in [0, 32)`. `in` and `out` are inferred the same way from `W`'s declaration. No index arithmetic, so the constraint solver has nothing to prove. Trivial pass.

**Shape analysis.** `sum[in]` consumes `in`. Surviving coordinates: `{batch, out}`. Output shape: `[batch_size, out_size]`. `relu` preserves coordinates — output shape unchanged. `activated` has shape `[batch_size, out_size]`. Pass complete.

**Type propagation.** `x` is `f32`. `W` is `f32`. `b` is `f32`. `x * W` is `f32`. `sum[in](...)` is `f32`. `+ b` is `f32`. `relu(...)` is `f32`. Every tensor in this program is `f32`. Pass complete.

**Lowering.** Three named coordinates. Three integers:
- `batch` → axis=0. Static range. Vectorized.
- `in` → axis=1, consumed by sum. The sum becomes `np.sum(..., axis=1)`.
- `out` → axis=1 after reduction (was axis=1 in W, becomes axis=1 in logits). Static range. Vectorized.

The broadcast `b[out] + ...` is lowered with `keepdims` logic: `b` has shape `(out_size,)`, the sum result has shape `(batch_size, out_size)`. NumPy broadcasts automatically. The compiler recorded the broadcast during analysis; the lowering pass trusts the recording.

Generated code:

```python
logits = np.sum(x * W.T, axis=1) + b
activated = np.maximum(logits, 0)
```

No `batch`. No `out`. No `in`. The names are gone. Every one was checked before it burned.

Now add a ragged coordinate. Same program, but `batch` has dynamic extent — variable batch size per call. The compiler reclassifies `batch` as runtime-dependent. Lowering switches from vectorized to scalar:

```python
activated = np.zeros((batch_size, out_size), dtype=np.float32)
for b_idx in range(batch_size):
    logits_b = np.sum(x[b_idx] * W.T, axis=0) + b
    activated[b_idx] = np.maximum(logits_b, 0)
```

Same names. Same checks. Different execution strategy. The programmer didn't choose the loop. The coordinate `batch` being runtime-dependent chose it.

This is the compiler's contract with the programmer: *name the coordinates, declare the shapes, write the computation. The compiler checks consistency, infers ranges, derives shapes, propagates types, proves bounds, and chooses the execution strategy. Names go in. Correct execution comes out.*

---

## Lowering: The Algorithm

Lowering is the final pass. Names have been checked, ranges inferred, shapes derived, types propagated, bounds proved. What remains: translate every named operation into an operation on integers that a numerical backend can execute.

The algorithm walks the IR tree one more time. At each node, it asks: given the axis assignments for this tensor, what integer operation does this named operation become?

**Step 1: Assign positions.** Every tensor in the IR carries its layout — an ordered list of coordinate names, built during shape analysis. The lowering pass reads each layout and numbers the coordinates left to right:

```
logits layout: [batch, class]  →  batch→0, class→1
```

This mapping is local to each tensor. If a different tensor has layout `[class, batch]`, its mapping is `class→0, batch→1`. The name is the invariant. The position is per-tensor.

**Step 2: Lower index expressions.** `(index logits (batch class))` becomes: read `logits` at all positions — axis 0 and axis 1. In NumPy, the slice `[:, :]`. The compiler doesn't emit slices for full reads; it emits the tensor name. The index node confirms that every requested coordinate exists and records the access pattern for later steps.

**Step 3: Lower reductions.** `(reduction sum (class) body)` becomes `np.sum(body, axis=class_pos)`. The compiler looks up `class` in the body tensor's layout. If the body has layout `[batch, class]`, `class` is at position 1. The reduction becomes `axis=1`. If `keepdims` is needed — because a downstream operation expects `class` to still exist as a dimension of size 1 — the compiler adds it. The decision is not a flag in the source. It is a requirement of the coordinate structure: if a later operation reads `class` from the reduction's output, the reduction must preserve the dimension. If nothing downstream reads `class`, `keepdims` is omitted.

**Step 4: Lower broadcasts.** At a binary operation like `logits - max_result`, the compiler compares the operand layouts. `logits` has `[batch, class]`. `max_result` has `[batch]` — `class` was consumed by the reduction that produced it. The coordinate sets differ: `{class}` is missing. The compiler records the broadcast. In the generated code, the broadcast becomes NumPy's automatic shape broadcasting — `logits` has shape `(B, C)`, `max_result` has shape `(B,)`, NumPy broadcasts `(B,) → (B, C)` automatically. If the missing coordinate is not the last one, the compiler inserts a `reshape` or `expand_dims` to align the dimensions.

The compiler does not rely on NumPy's broadcasting rules. It relies on its own coordinate set subtraction — the same subtraction from Chapter 2. The coordinate sets tell it which axes are missing. NumPy's broadcasting is the implementation mechanism, not the specification. The specification is the coordinate sets.

**Step 5: Lower index arithmetic.** `input[b, ic, oh + kh, ow + kw]` becomes a slice with computed offsets. The compiler emits loops over `kh` and `kw` — the reduction coordinates — and computes the input indices as `oh + kh`, `ow + kw` within each loop iteration. If the ranges are static and small, the loop is unrolled. If they are dynamic, the loop is emitted as written. The index expression `oh + kh` survives lowering as an expression, not as a precomputed offset. The generated code computes it at runtime with the correct bounds.

**Step 6: Emit.** Walk the lowered tree. Emit NumPy operations for vectorized coordinates. Emit Python `for` loops for ragged coordinates. Emit `if` guards for `where` clauses. The emission is a recursive function that switches on the node type and writes the corresponding Python text.

The six steps together are:

```
lower(ir_node, layout_map):
  match ir_node:
    Index(tensor, coords)     → record access, return tensor name
    Reduction(op, c, body)    → "np.{op}({lower(body)}, axis={layout_map[c]})"
    BinaryOp(op, left, right) → record broadcast if layouts differ,
                                  return "{lower(left)} {op} {lower(right)}"
    LetDecl(name, body)       → "{name} = {lower(body)}"
```

Four cases. The actual implementation has a few more — `Where`, `If`, function calls — but the structure is the same. Walk the tree. Look up the coordinate's position. Emit the integer.

The lowering pass is the point where names die. After this pass, the IR contains no `class`, no `batch`, no `..s`. Every name has become an integer. The integers are correct because the names were checked. The fire is lit. The light is the only trace of the name that remains.

---

The compiler proved that names can be checked mechanically. The next three chapters prove that names matter in practice — placing Einlang side by side with PyTorch and NumPy on normalization, attention, and physical simulation. The question is not which notation is better. It is what each notation makes visible, and what each notation hides.
