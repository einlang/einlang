# Autodiff implementation notes

**Status:** Historical note.

This file used to describe the retired compiler-generated diff-block implementation. It no longer matches the main code path.

The current implementation instead:

- snapshots high-level IR in `AutodiffPass`
- rewrites autodiff requests to runtime intrinsics
- resolves those requests with the NumPy JVP/VJP runtime

Start here instead:

- [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md)
- [AUTODIFF_VJP_JVP_REWRITE.md](AUTODIFF_VJP_JVP_REWRITE.md)
- [AUTODIFF_PIPELINE.md](AUTODIFF_PIPELINE.md)

Use this file only as historical context when comparing the old diff-block design with the current runtime-builtins path.
