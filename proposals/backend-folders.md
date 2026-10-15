# Group the backends into platform folders

**Status:** Proposed (2026-10-15)

## Problem

`zvisiongenerator/backends/` is one flat folder of 23 modules that serve three different audiences:

- **macOS / MLX:** `image_mac.py`, `image_mac_preview.py`, `video_mac.py`, `memory_mac.py`, `seedvr2_mac.py`, `prompt_enhancer_mac.py`.
- **Windows and Linux / CUDA:** `image_win.py`, `image_win_lokr.py`, `image_win_preview.py`, `image_win_quant.py`, `video_diffusers.py`, `memory_cuda.py`, `cuda_driver.py`, `lora_peft.py`, `prompt_enhancer_win.py`.
- **CUDA building blocks** added for LTX on 10 GB GPUs: `block_streaming.py`, `safetensors_stream.py`, `video_encode.py`, `ltx_cuda_weights.py`, `streaming_decode.py`, `token_chunking.py`.
- **Shared:** `__init__.py` (backend selection), `prompt_enhancer_session.py`.

Finding the module that matters for a platform means reading every name. The `_win` names are also misleading: those modules serve Linux too, which `docs/development.md` has to explain ("the filename is historical"). The CUDA LTX work made the folder grow by six modules in one change, and it will keep growing with the remaining platform-parity steps.

## Evidence

Checked 2026-10-15 on the `cuda-video-quant` branch:

| Check | Result |
|---|---|
| Modules in `backends/` | 23 plus `__init__.py` |
| `zvisiongenerator.backends.<module>` references in the package | about 60, most to `memory_cuda`, `memory_mac`, `prompt_enhancer_session`, `ltx_cuda_weights`, `block_streaming` |
| The same in `tests/` | 129 references in 25 files, many of them `monkeypatch.setattr("zvisiongenerator.backends....")` strings that must change with the path |
| `backends/` paths in `docs/development.md` | 13, plus the project structure tree |

## Proposed change

A pure move, with no behaviour change:

```
backends/
  __init__.py                 backend and accelerator selection (unchanged role; the only place that picks a platform)
  prompt_enhancer_session.py  shared
  mac/
    image.py                  was image_mac.py
    image_preview.py          was image_mac_preview.py
    video.py                  was video_mac.py
    memory.py                 was memory_mac.py
    seedvr2.py                was seedvr2_mac.py
    prompt_enhancer.py        was prompt_enhancer_mac.py
  cuda/
    image.py                  was image_win.py
    image_lokr.py             was image_win_lokr.py
    image_preview.py          was image_win_preview.py
    image_quant.py            was image_win_quant.py
    video.py                  was video_diffusers.py
    memory.py                 was memory_cuda.py
    driver.py                 was cuda_driver.py
    lora_peft.py
    prompt_enhancer.py        was prompt_enhancer_win.py
    block_streaming.py
    safetensors_stream.py
    video_encode.py
    ltx/
      weights.py              was ltx_cuda_weights.py
      streaming_decode.py
      token_chunking.py
```

- One `refactor(backends): …` commit, not a breaking change since nothing outside the package imports these paths: `git mv` every file, update imports, `backends/__init__.py`, the tests' patch strings and `docs/development.md` (structure tree and prose). The test files can follow the same split (`tests/backends/mac/…`) or stay flat; keep them flat unless the move is cheap.
- Keep the platform folders free of each other: `mac/` never imports from `cuda/` and the reverse, which the move makes visible. Anything both use moves up to `backends/`.
- `backends/__init__.py` stays the only place that picks a backend, per AGENTS.md.

## Verification

- `make check` passes with no test changes beyond paths.
- `git log --follow` works on the moved files (pure `git mv`, no edits in the same commit as the rename where possible: rename first, then fix imports in a second commit, or one commit with high similarity).
- A grep for the old module names (`image_win`, `video_diffusers`, `_mac.py`) in code, tests and docs finds nothing.

## Alternatives considered

- **Leave it flat.** Cheapest, but the folder only grows, and the `_win` naming keeps needing an explanation.
- **Split by task (`image/`, `video/`, `enhance/`) instead of platform.** Closer to the workflows, but each task folder would mix MLX and CUDA code, and the platform split is the one that decides which code runs on a machine.
- **Do it inside the CUDA LTX branch.** Rejected: hundreds of mechanical path changes would bury that branch's functional changes in review.
