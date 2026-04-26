---
layout: book
title: "Chapter 6: Living With Python"
---

# Chapter 6: Living With Python

Einlang is not trying to replace the entire Python ecosystem. Its purpose is to
make tensor computation, recurrence, and differentiation explicit where those
ideas matter, then cross the boundary cleanly for loading data, inspecting
values, plotting, and calling established libraries. This deliberate focus
creates power through boundaries rather than universal coverage, enabling
Einlang to excel in its domain while leveraging Python's breadth for
everything else. The result is a symbiotic relationship where each language
handles what it does best, connected through explicit interfaces that preserve
type safety and optimization opportunities.

## 6.1 `print`: Making Tensors Visible

The boundary between symbolic and concrete computation is where debugging
becomes most critical. `print` serves not just as output, but as an explicit
observation point that forces evaluation and reveals the computational
structure.

```rust
let tensor[i, j] = A[i, j] * 2.0;
print(tensor);
print(shape(tensor));
```

`print` is a built-in. It is the simplest way to make a value visible while
debugging. It also triggers evaluation of values that would otherwise remain
inside the program's expression graph.

For autodiff, bind the derivative first when you want the numeric value:

```rust
let dy_dx = @y / @x;
print(dy_dx);
```

Directly printing a raw request such as `print(@y / @x);` uses the symbolic
debugging path documented in `docs/AUTODIFF.md`.

### Print as an Observation Boundary

Until a value is observed, the implementation has freedom. It can keep an
expression symbolic, store a lazy derivative object, or delay a computation
until a later consumer needs it. `print` is one of the simplest observation
boundaries:

```rust
let z[i] = x[i] * x[i];
print(z);
```

For a learner, `print` is also a way to test the substitution model. If the
source says `z[i] = x[i] * x[i]`, a printed vector is a concrete witness that
the family of scalar expressions has been evaluated.

## 6.2 Python Interop

> **Think More.** Python interop is not only a feature; it is a boundary design.
> Einlang can stay small if it trusts Python for loading, plotting, tokenizing,
> and orchestration. Where should conversions be automatic, and where should the
> programmer be forced to state shape and type explicitly? The discussion can
> begin with ergonomics, then turn into trust: which boundary mistakes should be
> caught early, loudly, and by whom? This boundary abstraction enables modular
> system design, where each component handles its domain of expertise without
> unnecessary coupling. Implementation models range from implicit conversion to
> explicit casting, each with different safety and usability characteristics.
> Programming paradigms differ: some use seamless integration, others use
> explicit boundaries, affecting both reliability and developer experience.

The current interop syntax is module-path based:

```rust
let arr = python::numpy::array([1, 2, 3]);
let loaded = python::numpy::load("weights/W.npy") as [f32; 10, 20];
```

The standard library wraps common cases:

```rust
use std::io::load_npy;

let W = load_npy("weights/W.npy") as [f32; 10, 20];
```

Einlang does not need to own every data-loading, plotting, tokenization, or file
format story. Python is already good at those jobs. The useful split is: keep
the tensor formula in Einlang, and use Python for the surrounding ecosystem.

### Casts at the Boundary

Python values often arrive with dynamic shape. Einlang code becomes more useful
when that boundary is made explicit:

```rust
let W = python::numpy::load("weights/W.npy") as [f32; 128, 10];
```

The cast is not just documentation. It gives the compiler rank and dimension
information that later indexed declarations can use. A failed cast is an early
signal that the outside world did not provide the tensor the formula expects.

## 6.3 MNIST-Style Interop

> **Think More.** A complete workflow is rarely written in one conceptual
> language. MNIST may be small, but it already contains data loading, tensor
> formulas, differentiation, optimization, logging, and plotting. Which parts
> deserve Einlang's explicitness, and which parts are better left to Python? A
> useful debate is whether "one language for everything" is actually simpler
> than a carefully designed boundary between two languages.

The repository contains training examples that load data through Python and keep
the model step in Einlang. A representative pattern is:

```rust
let train_images = python::mnist_data::load_train_images()
    as [f32; 1437, 1, 8, 8];
let train_labels = python::mnist_data::load_train_labels()
    as [f32; 1437, 10];

let logits[n, c] = sum[d](features[n, d] * W[d, c]) + b[c];
let loss = sum[n, c]((logits[n, c] - train_labels[n, c])
                   * (logits[n, c] - train_labels[n, c]));
let dloss_dW = @loss / @W;
```

The exact examples live under `examples/mnist/`. Their lesson is architectural:
data and setup can stay in Python, while the differentiable tensor core remains
in Einlang with source-level shape checks and derivative syntax.

### The Boundary Principle

The useful Einlang/Python split is:

```text
Python: files, datasets, plotting, tokenizers, external services
Einlang: indexed tensor equations, reductions, recurrence, derivatives
```

This split keeps Einlang small without making it isolated. The language can be
specialized because it does not need to replace the entire Python ecosystem.

### A Worked Boundary

Suppose a Python helper loads a dataset:

```rust
let images = python::mnist_data::load_train_images()
    as [f32; 1437, 1, 8, 8];
let labels = python::mnist_data::load_train_labels()
    as [f32; 1437, 10];
```

From the Python side, these are arrays returned by ordinary functions. From the
Einlang side, the casts turn them into shaped tensor values. After that point,
indexed definitions can rely on the rank and dimensions:

```rust
let flat[n, d] = images[n, 0, h, w]
    where h = d / 8,
          w = d % 8;
let logits[n, c] = sum[d](flat[n, d] * W[d, c]) + b[c];
```

The boundary is explicit. Python handles loading; Einlang handles the tensor
relationship. Neither side has to pretend to be the other.

## 6.4 Interop Algorithms and Patterns

> **Think More.** Interoperability enables complex systems that combine
> specialized languages for different concerns. When boundaries are explicit,
> systems can be composed reliably while maintaining type safety and performance.
> Implementation models range from foreign function interfaces to embedded
> interpreters, each with different integration characteristics. Programming
> paradigms differ: some use tight coupling, others use loose coupling through
> data exchange, affecting both modularity and efficiency.

Explicit boundaries enable sophisticated cross-language algorithms.

### Data Pipeline Integration

Combining data processing in Python with tensor computation in Einlang:

```rust
// Python preprocessing
let raw_data = python::pandas::read_csv("data.csv");
let cleaned = python::preprocess::clean_data(raw_data);
let features = python::sklearn::extract_features(cleaned) as [f32; N, D];

// Einlang model
let weights = python::load_weights() as [f32; D, C];
let logits[i, c] = sum[d](features[i, d] * weights[d, c]);
let probs[i, c] = exp(logits[i, c]) / sum[c2](exp(logits[i, c2]));

// Python postprocessing
python::plot::confusion_matrix(probs, labels);
```

Cross-language pipelines with clear responsibility boundaries.

### Model Serving Architecture

Production model deployment with Python orchestration:

```rust
// Load model in Python
let model_params = python::torch::load_model("model.pt");

// Batch processing in Einlang
let batch_features = python::server::get_batch() as [f32; B, D];
let predictions[b, c] = sum[d](batch_features[b, d] * model_params[d, c]);

// Send results back to Python
python::server::send_predictions(predictions);
```

Scalable serving with language-appropriate optimizations.

### Implementation Models for Interop

**Foreign Function Interface**: Direct calls between languages with type
marshalling. Efficient but requires careful memory management.

**Data Serialization**: Exchange data through files or network protocols.
Flexible but adds serialization overhead.

**Embedded Interpreter**: Run one language inside the other. Convenient but
may have performance implications.

**Code Generation**: Generate code in target language from source language.
Optimizes for performance but complex.

**Protocol-Based**: Use standardized protocols for cross-language
communication. Reliable but may be slower.

Each model affects how algorithms are distributed across language boundaries.

## 6.5 Interop as Contract, Not Escape Hatch

When a small language meets a large ecosystem, there is always a danger that the
boundary will become intellectually sloppy. The user may be tempted to treat the
host language as the place where "real work" happens and the domain language as
mere ornamental notation. Or they may treat the host as an embarrassing legacy
detail that should disappear as quickly as possible. Neither attitude serves
Einlang well. The point of Python interop is not to escape the language, and it
is not to apologize for the language. It is to draw a meaningful contract
between two different kinds of strengths.

Python excels at broad ecosystem tasks: file formats, plotting, data
preparation, orchestration, notebook workflows, and access to established
libraries. Einlang excels where source-visible tensor structure matters:
indexed equations, reductions, recurrence, and derivative requests. A good
boundary is therefore not a vague "call Python when necessary." It is a
deliberate partition of responsibility.

### Contracts Need Shapes

The cast at the boundary is one of the most instructive parts of the chapter:

```rust
let W = python::numpy::load("weights/W.npy") as [f32; 128, 10];
```

This line is doing more than changing syntax. It turns a dynamically shaped
external value into a checked claim inside Einlang's world. The external source
may be flexible, messy, or user-provided. The internal tensor equation wants a
stable contract. The cast is where that transition is negotiated.

That is why the boundary should not be thought of as a casual conversion. It is
a semantic checkpoint. The user is asserting that the value crossing the line
has the rank and dimensions required by the downstream program. If that claim is
wrong, the failure belongs at the boundary, not deep inside later tensor logic.

### Why Narrow Languages Need Good Borders

A language that aims to do everything can be hazy about borders because it
pretends there are none. A small language does not have that luxury. Its
strength depends on clear scope. If the boundary to Python is vague, then the
benefit of the small core starts to dissolve. The programmer will push shape
assumptions into notebooks, scripts, and helper utilities, leaving the Einlang
program itself less informative.

By contrast, when the border is explicit, the small core becomes stronger. The
host can remain flexible without infecting the tensor language with dynamic
ambiguity. The tensor language can remain analyzable without claiming ownership
over every peripheral task in a real workflow.

## 6.6 Debugging at the Boundary: Observation, Trust, and Failure Modes

One of the real tests of an interop story is how it behaves under confusion.
Successful examples are pleasant, but publication-grade tooling has to answer a
less flattering question: when the user is wrong about what has crossed the
boundary, how quickly can they discover the problem, and how local can the
repair be?

This is where observation points such as `print` matter more than they first
appear to. Printing a tensor, printing its shape, and binding intermediate
results are not merely debugging conveniences. They are ways of restoring trust
at the place where the symbolic world of equations becomes concrete enough to
inspect.

### The Boundary Often Fails Quietly in Other Systems

In many tensor stacks, the most confusing bugs arise neither in the high-level
formula nor in the lowest-level kernel, but in the space between them. A file
loads with an unexpected layout. A label array has one-hot entries in the wrong
axis. A batch dimension is missing. Padding is retained when a mask was assumed.
These are boundary failures. The danger is that they can propagate silently if
the core language has no place to insist on shape, rank, or observation.

Einlang's small set of explicit tools helps because the repair path is direct.
The user can cast, print, slice, and name intermediate results inside the same
language that expresses the tensor logic. The debugging loop does not have to
collapse entirely into host-language inspection.

### A Worked Boundary Diagnosis

Imagine a model expects image data shaped `[f32; N, 1, 28, 28]`, but the Python
loader returns `[f32; N, 28, 28]`. A system with loose boundaries may allow the
mistake to survive until some later convolution or broadcast fails in an opaque
way. In Einlang, the cast itself is the first line of defense:

```rust
let images = python::mnist_data::load_images() as [f32; N, 1, 28, 28];
```

If the rank is wrong, the contract fails where the claim was made. If the rank
is technically acceptable but the semantics are wrong, a user can immediately
inspect:

```rust
print(shape(images));
print(images[0, 0, h, w]);
```

The point is not that printing solves all debugging. The point is that the
language provides small, compositional ways to surface the boundary facts that
matter.

### Trust Through Redundancy

Another underappreciated strength of explicit interop is that contracts can be
repeated at several levels without becoming nonsense. A Python helper may
document what it returns. The Einlang cast may restate the expected shape. A
later indexed definition may further constrain axis behavior. This is healthy
redundancy, not waste. It means that assumptions are checkable where they are
used.

That redundancy is especially valuable in collaborative settings. One person may
own the data-loading utility, another the tensor model, another the surrounding
training harness. Explicit casts and visible observation points create shared
places where misunderstandings can be discovered without requiring everyone to
know every layer equally well.

## 6.7 Practical Architectures: Where the Split Feels Natural

A reader interested in day-to-day use will eventually ask not "can Einlang call
Python?" but "what kind of system feels natural when the split is respected?"
The answer is less about a single API and more about a pattern of composition.

The most natural architecture places Python at the perimeter and Einlang at the
mathematical core. Python is responsible for datasets, configuration,
serialization, monitoring, plotting, service integration, and orchestration.
Einlang is responsible for the equations whose structure we care to expose and
optimize. Neither side pretends to be self-sufficient.

### A Research Workflow

In a research setting, a notebook or script may load experiment settings,
download data, and prepare evaluation runs. The core model step may then live in
Einlang, where axes, reductions, and recurrences remain visible. Results can
return to Python for plots, metrics, or comparison across runs. This division of
labor is especially attractive when one wants to inspect the structure of the
model itself rather than merely call prepackaged layers.

### A Systems Workflow

In a systems setting, the split may look slightly different. Python may assemble
input batches, tokenize requests, or handle service protocols. Einlang may hold
the kernel of a scoring or transformation routine whose explicit structure makes
it easier to optimize or verify. The output then returns outward. The key is
that the host remains the world of broad coordination, while Einlang remains the
world of explicit tensor semantics.

### A Pedagogical Workflow

Even for readers with no production interest, the split teaches a useful lesson:
not every language must solve every problem alone. Sometimes the most coherent
system is one in which a narrow language handles a narrow class of ideas
extremely well and delegates the rest. This is a healthier message than "all
serious tools must grow into universal platforms." It is also more realistic.

### Interop as Identity

In that sense, interop is not incidental to Einlang's identity. It is part of
the language's argument about itself. The language stays small because it can
lean on an existing ecosystem for tasks that are real but not central to the
kind of structure it wants to expose. The boundary is therefore a source of
focus, not a sign of incompleteness.

### The Chapter's Larger Claim

The larger claim of this chapter is that boundaries can be intellectually
productive. They can make contracts clearer, debugging more local, and systems
composition more honest. Python interop is not included so that Einlang can
quietly expand into a general-purpose environment by stealth. It is included so
that a narrow tensor language can remain narrow while still participating in
real work.

## 6.8 The Wider Audience for a Language With Borders

It is worth stressing that this boundary story is not only for specialists in
language design. Many readers encounter tensor programming from practical
pressure first: they have data in one place, models in another, experiments in a
third, and a great deal of notebook or service code holding everything together.
For such readers, a language that insists on clear borders can be more useful
than one that promises total ownership and delivers a blurrier reality. The
border makes responsibilities legible.

That is part of why this chapter belongs in a broader publication rather than
only in a technical appendix. It addresses a recurring experience of modern
computing: meaningful systems are often assembled from components that are good
at different things. The question is not whether a pure single-language world
would be elegant in theory. The question is whether a real system can remain
coherent while crossing tools, runtimes, and communities of practice. Einlang's
answer is that coherence comes from explicit contracts, not from pretending the
crossings do not exist.

At a glance, the boundary looks like:

```text
Python loader -> cast -> Einlang equation -> Python report
```

## 6.9 Boundary Case Studies

It helps to make the boundary argument concrete with recurring case studies.
First, consider exploratory work. A practitioner may want to load a dataset,
inspect a few examples, try several preprocessing variants, and then express the
core tensor relationship clearly. Python is an excellent medium for the
exploration, but the tensor equation benefits from Einlang's structure-visible
syntax. In that workflow, the boundary is not a nuisance. It is the line between
experimentation infrastructure and mathematical core.

Second, consider a long-lived project with multiple contributors. Data loading,
evaluation reports, and external services may evolve rapidly. The tensor logic
may need stronger stability guarantees because it is the part that most benefits
from explicit shapes, reductions, and derivative requests. Here again the
boundary is useful because it prevents every layer from inheriting the full
complexity of every other layer.

Third, consider educational or communicative work. A notebook or article may
need to explain not only what a system computes, but how its tensor structure
works. Python can handle the surrounding exposition and plotting. Einlang can
hold the exact equations worth reading closely. The split keeps the explanation
honest: auxiliary tasks remain auxiliary, while the tensor core remains visible.

A tiny example is enough to show the pattern:

```rust
let images = python::data::load_images() as [f32; N, D];
let logits[n, c] = sum[d](images[n, d] * W[d, c]);
python::report::plot_logits(logits);
```

Across all three cases, the same lesson appears. A boundary is healthiest when
it reflects a real difference in semantic responsibility. Python is the place
where the outside world is admitted. Einlang is the place where the tensor
relationships are stated with maximal clarity. The more faithfully that split is
kept, the more coherent the overall system tends to remain.

## 6.10 Why Wider Audiences Should Care About This Chapter

Readers who are not planning to adopt Einlang immediately may still find this
chapter useful because it names a general problem in contemporary technical
work. Many systems today are assembled from specialized tools rather than
written end to end in one homogeneous language. The real challenge is often not
whether those tools can call one another, but whether the boundary between them
preserves meaning instead of erasing it.

That question appears in data science, machine learning, simulation, graphics,
and even ordinary business software. One tool is good at storage, another at
analysis, another at visualization, another at serving. Systems become brittle
when these crossings are handled only by habit and implicit convention. They
become more trustworthy when contracts, casts, and observable handoff points are
part of the design.

The failure mode is usually simple:

```text
opaque helper -> hidden reshape -> later mismatch
```

Seen from that angle, the Python chapter is not merely a pragmatic appendix to a
tensor language. It is part of the book's wider argument that boundaries can be
an intellectual achievement. A system becomes clearer when it knows where one
kind of meaning ends and another begins. Einlang's interop story is one
instance of that more general principle.

## 6.11 Why the Boundary Story Matters So Much

The reason this chapter needs so much space is that boundaries are where many
technical systems become either honest or misleading. A small tensor language
can look beautiful in isolation while remaining awkward in practice if it has no
clear account of how it meets datasets, plotting tools, services, notebooks,
and surrounding libraries. Conversely, a language with very modest scope can
become surprisingly powerful if it crosses those borders well.

Einlang's answer is not magical seamlessness. It is legibility. The user is
meant to see when a value enters from Python, when a shape claim is being made,
when a tensor is being inspected, and when a result is returning outward. That
legibility helps beginners learn, helps practitioners debug, helps collaborators
divide labor, and helps implementers keep the core language focused. The chapter
therefore occupies a deeper place in the book than a mere "integration notes"
section would suggest.

There is also a broader cultural argument here. Modern technical work is almost
never done inside one pure, sealed environment. We work across data tools,
visualization layers, deployment systems, experiment harnesses, and domain
libraries. A publication about a serious language should therefore say something
intelligent about crossing worlds. By presenting interop as a structured
contract rather than an embarrassment, this chapter tries to do exactly that.

For readers outside the immediate Einlang project, that may be the most portable
lesson. Any specialized tool becomes more trustworthy when its boundaries are
clear, observable, and honest about what they promise. The Python chapter is one
instance of that wider principle, but the principle travels far beyond this
language.

## 6.12 A Publication-Worthy Lesson About Tools

A serious publication should leave the reader with something larger than a
feature checklist. This chapter's larger lesson is that tool quality is often
revealed at the edges. Many systems look impressive when viewed only at their
most elegant core examples. Far fewer remain impressive when one asks how data
enters, how assumptions are stated, how meaning survives translation, and how
responsibility is divided across layers.

Einlang's interaction with Python is meant to answer that harder question. It
does not promise that every crossing is invisible. It promises that crossings
can be explicit enough to reason about. That is a stronger and more durable kind
of sophistication than seamlessness-by-marketing. A reader deciding whether the
language is serious should care about this chapter precisely because it shows how
the project behaves when the outside world arrives.

The publication-level importance of this is easy to miss. Many books about tools
and languages dwell on internal elegance and leave boundary realities to README
files or scattered examples. But a language earns trust when its boundary story
is part of its stated design. That is what this chapter has tried to supply: not
merely proof that Python interop exists, but an argument for what kind of
clarity a good interop story should preserve.

### A Final Practical Reading

If a practitioner remembers only one operational lesson from this chapter, it
could be this: keep the outer world broad and flexible, but make the inner
tensor claims narrow and explicit. Load, inspect, orchestrate, and visualize in
the environment already good at those things. Cast, name, reduce, recur, and
differentiate in the environment whose entire reason for existing is to make
that structure visible. The cleaner that separation remains, the stronger both
sides tend to become.

## 6.13 Interop and the Shape of Real Work

Real technical work rarely arrives as a pure mathematical object waiting
obediently for one ideal language. It arrives mixed with files, services,
scripts, dashboards, experiments, and partial data. A language becomes more
useful when it can meet that reality without losing its semantic center. That is
exactly what a good interop story provides.

For Einlang, the center is explicit tensor structure. The surrounding world is
allowed to remain heterogeneous, but the boundary is where that heterogeneity is
translated into a form the tensor language can state clearly and check. This is
why the chapter belongs in the main arc of the book: it explains how a language
with narrow ambition participates in broad reality without giving up the source
of its coherence.

That lesson generalizes. Many successful tools are not universal; they are
disciplined participants in larger ecosystems. What makes them durable is not
that they eliminate boundaries, but that they make boundary crossings legible.
Einlang's relationship with Python is one example of that broader pattern.

This is also why the chapter should matter to readers who care about software
quality more broadly. Systems often fail where meanings are translated, not only
where equations are computed. A shape contract omitted at the boundary can be as
damaging as a bug in the kernel itself. By insisting that boundary claims be
spelled out, observed, and checked, Einlang is arguing for a style of technical
honesty that extends beyond tensor programming.

Seen this way, interop is part of the language's publication-level identity. It
shows how the project expects to live in the world rather than only in ideal
examples. The host language remains the place for breadth and reach. Einlang
remains the place for explicit tensor meaning. A good border lets both remain
good at what they are for.

There is also a kind of readerly relief in this arrangement. One does not need
to pretend that every surrounding concern belongs in the tensor language in
order to take the tensor language seriously. The book can say plainly: here is
where the formula should live, here is where the ecosystem should help, and here
is how the crossing between them should be made understandable. That candor is
part of what makes a narrow tool feel trustworthy.

In practice, that means the chapter is not a detour away from the core. It is a
demonstration that the core can keep its shape under real conditions. A language
with a good boundary story can remain small without becoming isolated, and it
can remain principled without becoming impractical. That is a substantial design
achievement.

This is also why the chapter speaks to a wide audience. Anyone who has watched a
beautiful internal abstraction dissolve at the point where real inputs and real
systems arrive will recognize the stakes. A language proves part of its worth by
how it survives contact with the outside world. Clear interop is one way of
surviving that contact without losing identity.

Put differently: the chapter is about practical dignity. It argues that a small
language need not become vague in order to be useful, and need not become
universal in order to participate in serious work. It can keep its center by
crossing boundaries well. That is a lesson many specialized tools could stand to
learn.

### A Final Boundary Question

Whenever a value crosses from Python into Einlang, or back again, ask one blunt
question: what meaning would be lost if this crossing were undocumented? The
answer usually tells you what contract, cast, print, or named intermediate is
still worth adding.

The chapter therefore closes not only with an interoperability recipe, but with
a general design value: boundaries are strongest when they preserve meaning
rather than merely moving bytes. That value applies to many tools beyond
Einlang, and it is one reason this discussion belongs at the center of the
book's argument about clarity.

That same value is useful whenever systems are built from specialized parts.
Clear boundaries help teams collaborate, help readers debug assumptions, and
help languages remain good at the things that justify their existence. Interop,
in other words, can be a source of rigor rather than a concession to impurity.

This chapter argues that the best boundary is neither invisible nor clumsy. It
is visible enough to inspect and disciplined enough to trust. That is the kind
of practicality a serious publication should defend.

The claim is modest in wording but ambitious in implication: software becomes
more legible when crossings between worlds are treated as semantic events worth
designing carefully. Interop, in that sense, is part of the language's theory of
clarity.

That is why the chapter is ultimately about more than Python. It is about how a
focused tool keeps its promises while living in a broader world. The answer is
not isolation, and it is not total absorption. It is a border clear enough that
meaning survives the crossing. Readers who build any kind of multi-tool system
will recognize the value of that answer.

The chapter's practical wisdom lies there: clear boundaries are a way of keeping
both power and honesty at once. That is a rare and useful combination.

In that respect, the chapter is as much about system character as about system
mechanics. It asks a language to be explicit not only at the center of the
equation, but also at the point where the equation meets the world.

That is a strong standard, and a useful one.

It is also one that scales well from notebooks to larger systems.

That scalability is part of what makes the boundary story feel mature.

It keeps practical breadth from dissolving semantic focus.

## Summary

The relationship between Einlang and Python demonstrates the power of
deliberate boundaries in language design. Rather than attempting universal
coverage, Einlang focuses intensely on tensor computation while leveraging
Python's ecosystem for everything else. This symbiotic approach creates more
power than either language could achieve alone.

The design principles that enable this collaboration are carefully chosen:

- `print` observes values and helps debug the expression graph, providing
  explicit observation points in a lazy computational model;
- Python handles the messy edge of the world, from data loading to
  visualization, freeing Einlang to focus on its core strengths;
- Casts at the interop boundary turn dynamic values into checked tensor shapes,
  maintaining type safety across language boundaries;
- Examples such as MNIST keep data loading in Python and model equations in
  Einlang, demonstrating clean separation of concerns.

This boundary-driven approach transforms what could be a weakness - limited
scope - into a strength. By being excellent at tensor computation and explicitly
handing off everything else to Python, Einlang creates a more reliable and
composable system than a language trying to do everything adequately. The
result is not compromise, but synergy: each language handles its domain with
depth and clarity.

The final chapter explores the philosophical foundations of these design
choices, examining why Einlang's boundaries are not limitations, but the
source of its focused power.
