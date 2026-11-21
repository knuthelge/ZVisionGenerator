"""The one memory offset a figure carries, applied once by the verdict engine for every CUDA job.

A CUDA figure is what the job's tensors need (PyTorch's allocated peak) plus the process's own cost of running them: the CUDA context and
the allocator's slack. That cost does not change with the job and is measured once per operating system (the allocator behaves differently under WDDM), so exactly one
constant (``CUDA_OFFSET_BYTES``, the running system's) is added to a figure. A Mac figure carries no offset: it is fitted to the process's peak footprint directly, because the footprint's excess over
MLX's peak varies by model.

``CUDA_CONTEXT_BYTES`` is not part of a figure. It is the context the loader's own process already holds when it reads the card's free memory, which it adds back
so that free memory and a figure (whose offset contains the context) are the same quantity.

Each constant carries its measurement mark. Change one only from a bench result (``docs/developer/memory-bench.md``).
"""

from __future__ import annotations

import sys

from zvisiongenerator.utils.model_memory import GIB

__all__ = ["CUDA_CONTEXT_BYTES", "CUDA_OFFSET_BYTES", "CUDA_OFFSET_LINUX_BYTES", "CUDA_OFFSET_WINDOWS_BYTES"]

# The CUDA context and the allocator's slack above PyTorch's allocated peak: what a card must give a job beyond its tensors. measured: the middle of the hold brackets of the jobs whose peak does not shrink on a tight card
# (krea2-q4-xl, ideo-l, klein9b-q4-2048, klein9b-q4-xl-ref3, klein9b-q4-m-ref4, klein9b-q4-edit-2048).
CUDA_OFFSET_LINUX_BYTES = round(0.40 * GIB)

# The same offset on Windows, where the allocator runs without expandable segments (more fragmentation). measured with expandable segments off: the middle of the hold-bracket offsets of klein9b-q4-2048,
# klein9b-q4-edit-2048, ltx4-xl-14s and ltx4-xl-10s.
CUDA_OFFSET_WINDOWS_BYTES = round(1.00 * GIB)

# The offset of the system this process runs on, the one the engine adds to a figure.
CUDA_OFFSET_BYTES = CUDA_OFFSET_WINDOWS_BYTES if sys.platform == "win32" else CUDA_OFFSET_LINUX_BYTES

# The CUDA context a process holds before it allocates a tensor, so a loader that reads the card's free memory after its own context exists can add it back: the offset above already contains it.
# measured: the card's free memory sits this far below the card's total in every case.
CUDA_CONTEXT_BYTES = round(0.22 * GIB)
