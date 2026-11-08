# A faster prompt enhancer on Windows and Linux

**Status:** Done (2026-11-08, unreleased; built on `feat/enhancer-fast-runtime`)

## Problem

Helpers that write with the prompt enhancer waited 10 to 15 seconds on Windows and Linux.
`generate` decoded at 19.3 and 19.8 tok/s, bound by kernel launch overhead rather than arithmetic.

## Outcome

The pinned Qwen3.5 enhancers decode one token per CUDA-graph replay (`backends/cuda/enhancer_decode.py`). That gives 83 and 63 tok/s with the same models and sampling and nothing to install, and bench quality is unchanged.

## Gate

Only `qwen3_5_text` repos with no quantization or sampling settings of their own use the graph; other models, which can break under capture, and the CPU stay on `generate`.

## Why not llama.cpp

It is faster still, but it needs a binary per platform and a second set of model files.
