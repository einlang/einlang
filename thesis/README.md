# Einlang Thesis

This directory contains a thesis-form version of the Einlang writeup. It is
separate from the ACM-style paper in `../paper/` but reuses the same bibliography
and TikZ figure sources.

Read the generated PDF at `einlang_thesis.pdf`. For the shorter language-design
paper, see `../paper/einlang_paper.pdf`.

The thesis driver is `einlang_thesis.tex`; chapter bodies live under
`chapters/`.

Build from this directory:

```bash
../.tools/tectonic einlang_thesis.tex
```
