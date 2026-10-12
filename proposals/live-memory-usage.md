# Show memory use while a job runs

**Status:** Proposed (feasibility study), 2026-10-11

## Question

Can we tell the user, on every platform, how much memory a job is using while it generates: during the model load, the denoising steps and the decode?

## Answer

Yes. We can do it on macOS, Windows and Linux without new dependencies, but the numbers mean slightly different things on each platform:

| | macOS (MLX, unified memory) | Linux (CUDA) | Windows (CUDA, WDDM) |
|---|---|---|---|
| Our GPU memory | Exact: `mx.get_active_memory()` + `mx.get_cache_memory()` | Exact for PyTorch: `torch.cuda.memory_reserved()` | Same as Linux |
| Whole GPU in use (all processes) | n/a (one memory pool) | `torch.cuda.mem_get_info()` | `torch.cuda.mem_get_info()` (dedicated VRAM only) |
| Our process in total | `phys_footprint` (`proc_pid_rusage`), the figure Activity Monitor shows | Anonymous RSS from `/proc/self/status` | `psutil` private bytes |
| Budget to compare against | `max_recommended_working_set_size` (already used for fit badges) | VRAM + system memory (already in `MemoryBudget`) | Same as Linux |
| Is the machine struggling? | `kern.memorystatus_vm_pressure_level` (normal / warn / critical) and swap | PSI `/proc/pressure/memory` | **Hard:** VRAM spilling into shared memory is visible only through PDH counters (see below) |
| Peak per image | `mx.get_peak_memory()` / `reset_peak_memory()` | `torch.cuda.max_memory_reserved()` / `reset_peak_memory_stats()` | Same as Linux |

The one real gap is the Windows *Sysmem Fallback* case, where we can't simply call an API. Everything else is either a counter we already have access to or one syscall.

## Evidence

### macOS: measured on an M5 with 32 GB, mlx 0.32.3

- **RSS can't be used.** After a 4 GiB MLX array was evaluated, `psutil` RSS read 0.02 GiB, because Metal buffers aren't counted in RSS. `phys_footprint` read 4.05 GiB and `mx.get_active_memory()` read 4.00 GiB.
- **The MLX counters are free.** `get_active_memory` + `get_cache_memory` + `get_peak_memory` together take about 0.14 µs per call. The `psutil` process, memory and swap queries together take about 14 µs.
- **The counters can be read from another thread.** A sampler thread read the counters at 10 Hz while the main thread ran matmuls, with no errors. The values tracked active versus cached memory as buffers moved between them.
- **No measurable cost at 1 Hz.** The test interleaved runs with no sampler, a sampler that only sleeps, MLX counters and `phys_footprint`. Run times scattered by ±25% across runs, because this Mac was swapping heavily, and showed no trend for any probe. An early 45% slowdown didn't reproduce once the runs were interleaved.
- **The pressure readings work.** `sysctl kern.memorystatus_vm_pressure_level` returned 2 (warning) while swap stood at 14 GB of 16 GB, which matches the machine's state.

### CUDA: from documentation; not yet measured on hardware

- `torch.cuda.memory_reserved()`, `memory_allocated()` and the peak counters read PyTorch's own allocator on the host. They are free and never synchronise. They miss the CUDA context, about 0.3 to 0.5 GB, and anything allocated outside PyTorch.
- `torch.cuda.mem_get_info()` calls `cudaMemGetInfo` and reports the whole device, so other apps and the desktop are included. It needs a CUDA context: calling it before the backend has started CUDA would create one early and bypass `configure_allocator()` (`backends/memory_cuda.py`). The sampler must therefore check `torch.cuda.is_initialized()` first, and never import torch itself.
- **Linux:** NVML can report each process's GPU memory. We don't need it, since the PyTorch counters cover our own use. Safetensors are memory-mapped, so plain RSS includes file pages during a load. Anonymous RSS excludes them, and it does include the pinned host memory that group offload streams from. PSI is already the signal the [systemd-oomd proposal](archive/linux-oomd-model-loads.md) measures.
- **Windows:** under WDDM, NVML's per-process GPU memory always returns `NVML_VALUE_NOT_AVAILABLE`, because Windows rather than the NVIDIA driver manages the memory. This doesn't matter, because the PyTorch counters still work.
- **Windows Sysmem Fallback:** since driver 536.40, when VRAM runs out the driver moves allocations into system memory instead of raising out-of-memory. The job then runs much more slowly, and nothing tells the user why. In that state PyTorch's counters still report the memory as allocated, and `mem_get_info` reports only dedicated VRAM, so neither shows the spill. The per-process PDH counters `\GPU Process Memory(pid_<pid>_*)\Shared Usage` and `Dedicated Usage` do show it; Task Manager reads the same data. We can read them through `pdh.dll` with `ctypes` in roughly 60 lines, without pywin32. This is the one piece that needs a Windows machine to build and verify.

## When to sample

Step callbacks alone aren't enough. The step events (`_ProgressChecker` in `image_mac.py`, `_make_step_callback` in `image_win.py` and `video_diffusers.py`) don't fire during the model load, the text encoding, the VAE decode or prompt enhancement, which are where memory peaks and failures happen.

Instead, a **sampler thread runs for the length of each job and reads the counters once a second** on the worker that already runs the job (`web/web_runner.py`). It publishes an event only when a value moves by more than about 1% of the budget, so the SSE stream stays quiet while a job is steady. When the job ends, the sampler stops and reports that job's peak.

## Proposed shape

- **Backend** (atomic, one module per platform beside the existing ones):
  - `backends/memory_mac.py` and `backends/memory_cuda.py` each gain a pure `memory_usage() -> MemoryUsage` reader.
  - `MemoryUsage` is a frozen dataclass: `process_bytes`, `accelerator_bytes`, `device_used_bytes | None`, `spilled_bytes | None`, `pressure: "normal" | "warn" | "critical" | None`.
  - `backends/__init__.py` picks the reader for the platform, as it already does for `get_accelerator_memory_budget()`.
  - A small sampler in `core/` takes the reader and a `ProgressCallback` as arguments, so it can be tested with a fake reader.
- **Event:** `memory_usage`, sent with the job's existing event context, plus `budget_gb` from the `MemoryBudget` we already compute. `generation_finished` gains `peak_gb`.
- **Web UI:** a compact readout on the job card, coloured with the fit badges' `fits` / `tight` / `too_large` thresholds so live use and the estimate read the same way.
  - macOS has one memory pool, so it gets one readout: `Memory 18.2 / 26.8 GB`.
  - CUDA gets two, `GPU 7.1 / 16 GB · RAM 21.4 / 32 GB`, each against its own budget: VRAM at 90% and system memory at 80%, as `classify_cuda_memory_fit` already uses. Unquantized and q8 weights stay in system memory and stream to the GPU, so RAM is often the readout that fills first. A warning appears on pressure `warn`/`critical` or on spill. It needs an HTML preview and sign-off before anyone builds it.
- **CLI:** the peak per image on the line that reports the finished image. A live meter there isn't worth the noise.
- **Side benefit:** recorded peaks show how far the deliberately rough estimates in `utils/model_memory.py` are off, and could be used to calibrate their working-memory margins later.

## Risks and open questions

1. **Windows spill detection** is the only part that isn't verified. If PDH proves unreliable, we fall back to a heuristic: flag when `memory_reserved()` exceeds the dedicated VRAM that `mem_get_info` reports. Or we drop spill detection and link to the Sysmem Fallback setting in the docs.
2. **What to show for RAM on CUDA.** Our process's anonymous memory or private bytes shows what we hold. The machine's available memory (`psutil.virtual_memory().available`) shows how close the machine is to swapping. The readout should use our process's figure against the budget, and raise the warning from available memory or pressure, because other apps use system memory too.
3. **The prompt enhancer loads into the same process** and is counted along with everything else. That is correct for "is this machine about to run out", but the readout shouldn't be labelled as the image model's use alone.
4. **Multiple GPUs:** we read device 0 only, which is the one the backends use.
5. **The readers need CUDA hardware to verify:** the cost of `mem_get_info` on Windows and Linux, and the PDH instance names.

## Effort

- macOS reader, sampler, event and CLI peak: small; we can build and test all of them here.
- The Linux and Windows readers without spill detection: small.
- Windows PDH spill detection: medium, and it needs a Windows machine.
- Job-card readout: small once the preview is approved.

## Sources

- [NVML `nvmlProcessInfo_t`: `usedGpuMemory` not available under WDDM](https://docs.nvidia.com/deploy/nvml-api/api/structnvmlProcessInfo__t.html)
- [NVIDIA forum: NVML per-process memory on Windows WDDM](https://forums.developer.nvidia.com/t/nvml-problems-for-windows-not-available-in-wddm-driver-model/77557)
- [hypomnesis: per-process GPU memory on Windows via DXGI and PDH](https://docs.rs/crate/hypomnesis/0.2.2)
- [`torch.cuda.mem_get_info`](https://docs.pytorch.org/docs/2.12/generated/torch.cuda.memory.mem_get_info.html)
- [NVIDIA forum: `cudaMemGetInfo` and context overhead](https://forums.developer.nvidia.com/t/why-do-cudamemgetinfo-also-occupies-a-lot-of-gpu-memory/197462)
- [PyTorch forum: slow execution from GPU shared memory (Sysmem Fallback)](https://discuss.pytorch.org/t/documented-fix-slow-execution-pytorch-using-gpu-shared-memory/218909)
- [Sysmem Fallback and the `GPU Process Memory` counters](https://runaihome.com/blog/shared-gpu-memory-slow-local-ai-sysmem-fallback-fix-2026/)
