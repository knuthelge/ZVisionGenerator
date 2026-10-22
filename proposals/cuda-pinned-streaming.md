# Pin streamed weights once on Windows and Linux

**Status:** Proposed (2026-10-22)

## Problem

On Windows and Linux, unquantized and q8 weights stay in system memory and stream to the GPU block by block (`backends/cuda/block_streaming.py`, `group_offload`). Every call passes `low_cpu_mem_usage=True` to diffusers' group offloading. In diffusers 0.41.0 that option keeps the weights in ordinary (pageable) memory and pins a fresh copy of each block every time it moves to the GPU. So every denoising step copies the whole transformer once in system memory before the GPU copy can start, on top of the transfer itself. diffusers' own docstring says the option "may counteract the benefits of using streams".

The option has been set since the first CUDA streaming code and was never measured. The NF4 path (`stream_blocks`) already does the opposite: it pins each parameter in place once, with `_pin_parameters`.

## Evidence

From `diffusers/hooks/group_offloading.py` (0.41.0):

- `ModuleGroup._init_cpu_param_dict` keeps `tensor.data.cpu()` for each parameter. With `low_cpu_mem_usage=True` that is the parameter's own CPU tensor, unpinned. With `False` it is a separate pinned copy (`.pin_memory()`), so the model sits in memory twice until the first onload swaps `param.data`.
- `_onload_from_memory` calls `_pinned_memory_tensors`, which runs `tensor.pin_memory()` for every tensor that `is_pinned()` reports as unpinned, on every onload. PyTorch's caching host allocator reuses the pinned buffers, so the per-step cost is a CPU memory copy of each block rather than a new allocation.
- A tensor that is already pinned is used as it is.

This affects every streamed load:

| Where | Component |
|---|---|
| `cuda/image/backend.py` `_stream_to_gpu` | Transformer and text encoder, unquantized and q8, every family except Ideogram |
| `cuda/image/ideogram4.py` `_place` | Transformer, unquantized and q8 |
| `cuda/video/backend.py` `_place_transformer` | LTX transformer at q8 in low-memory mode (`ltx-8`) |
| `cuda/video/backend.py` `_place_components`, `_encode_prompt` | Connectors (low memory) and the Gemma text encoder |

The NF4 paths (`ltx-4`, q4 Ideogram) use `stream_blocks` and are not affected.

How much time this costs is unknown. The copy may already overlap with compute on the side stream, in which case the gain is small.

## Proposed change

Measure first (below). If it pays off, pin each component's CPU parameters in place before group offloading is applied, so diffusers finds them pinned and skips its per-onload copy:

- In `group_offload`, call the existing `_pin_parameters(module)` before `apply_group_offloading`, and keep `low_cpu_mem_usage=True`. Pinning parameter by parameter in place holds one extra tensor at a time, never a second copy of the model, which `low_cpu_mem_usage=False` would.
- No new setting. `release_memory` already empties PyTorch's host cache after a job; the pinned weights are freed with the model.

## Trade-off

Pinned memory is page-locked: the kernel cannot swap it out or reclaim it. The model already lives in anonymous memory while a job runs, so the resident size does not change, but under memory pressure nothing of it can be pushed to swap. That matters most for bf16 models near the limit of a 32 GB machine (Ideogram's 18.6 GB transformer, Krea 2) and for systemd-oomd on Linux (see [linux-oomd-model-loads.md](archive/linux-oomd-model-loads.md)). Pinning the whole model also takes time at load, which the measurement must count.

Windows may cap how much memory a process can pin. This is unverified.

## Measurement plan

On the 10 GB GPU and 32 GB machine, the same seed and size per pair, current code against pinning in place, interleaved, three runs each. Each run is in its own systemd user unit.

1. `zit` unquantized and `klein9b` q8 at 1024×1024: seconds per step, load time, peak RSS, PSI memory pressure.
2. `ideo` unquantized and q8 at 1024×1024: the same, plus whether oomd or swap is triggered with the 18.6 GB transformer pinned.
3. `ltx-8` t2v at 704×448, 97 frames: seconds per step, and the text-encoder load time per new prompt (Gemma is pinned on every load).

Ship rule: ship when steps get at least 10% faster for one image family or for `ltx-8`, load time grows by less than the time saved over a typical job, and no run swaps or is killed where the current code was not. If only some components gain, pin only those (for example the transformers and not the per-prompt Gemma load).

## Testing

- `group_offload` pins a module's CPU parameters in place (same parameter objects, `is_pinned()` true) before group offloading is applied, and passes the same options as today.
- No real models; diffusers' `apply_group_offloading` is mocked.

## Open questions

- Does Windows allow pinning an 18.6 GB model, and does pinning change the WDDM behaviour of shared GPU memory?

## Alternatives considered

- **`low_cpu_mem_usage=False`.** diffusers pins a separate copy at apply time, so the model is held twice until the first step, too much for a 32 GB machine with a bf16 model.
- **More blocks per group.** Fewer, larger transfers, but more GPU memory per group on a 10 GB card, and it does not remove the copy.
- **Resident models on GPUs with room.** A separate, larger change that cannot be measured on a 10 GB card.
