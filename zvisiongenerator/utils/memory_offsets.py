"""The one memory offset a figure carries, applied once by the verdict engine for every CUDA job.

A CUDA figure is what the job's tensors need (PyTorch's allocated peak) plus the process's own cost of running them: the CUDA context and
the allocator's slack. That cost does not change with the job and is measured once per operating system (the allocator behaves differently under WDDM), so exactly one
constant (``CUDA_OFFSET_BYTES``, the running system's) is added to a figure; nothing else is added to it. A Mac figure carries no offset: it is fitted to the process's peak footprint directly, because the footprint's excess over
MLX's peak varies by model. The status bands (:mod:`zvisiongenerator.utils.memory_grade`) are the only margin.

``CUDA_CONTEXT_BYTES`` is not part of a figure. It is the context the loader's own process already holds when it reads the card's free memory, which it adds back
so that free memory and a figure (whose offset contains the context) are the same quantity.

Each constant carries its measurement mark. Change one only from a new bench measurement (``docs/developer/memory-bench.md``).
"""

from __future__ import annotations

import sys

from zvisiongenerator.utils.model_memory import GIB

__all__ = ["CUDA_CONTEXT_BYTES", "CUDA_OFFSET_BYTES", "CUDA_OFFSET_LINUX_BYTES", "CUDA_OFFSET_WINDOWS_BYTES"]

# The CUDA context and the allocator's slack above PyTorch's allocated peak: what a card must give a job beyond its tensors. measured, bench run ``cuda-r9`` (a 10 GB
# card, jobs run under growing holds): the middle of the hold brackets of the jobs whose peak does not shrink on a tight card, which lie within +0.19..+0.64 GiB of the free-card peak
# (Krea 2 q4 at 1600x1600 +0.39..+0.64, Ideogram 4 q8 at 1408x1408 +0.36..+0.61, Klein 9B q4 at 2048x2048 +0.36..+0.61, with three references +0.36..+0.61, with
# four +0.19..+0.44, an edit at 2048x2048 +0.19..+0.44).
CUDA_OFFSET_LINUX_BYTES = round(0.40 * GIB)

# The same offset on Windows, where the allocator runs without expandable segments (more fragmentation). Measured, Windows rounds 6 and 7 with expandable segments off: middle of the span
# +0.74..+1.29 GiB of the hold-bracket offsets in budget units (Klein 9B 2048 +1.02..+1.14 in round 6 and +1.17..+1.29 in round 7, Klein 9B edit 2048 +0.92..+1.05, LTX 1600x896 14 s +0.74..+0.99,
# LTX 1600x896 10 s +0.78..+0.90). A case whose rounds disagree (Klein 9B 2048: its pooled outcomes overlap, because the other programs' baseline differed between the rounds, 0.36 and 0.21 GiB)
# is read round by round.
CUDA_OFFSET_WINDOWS_BYTES = round(1.00 * GIB)

# The offset of the system this process runs on, the one the engine adds to a figure.
CUDA_OFFSET_BYTES = CUDA_OFFSET_WINDOWS_BYTES if sys.platform == "win32" else CUDA_OFFSET_LINUX_BYTES

# The CUDA context a process holds before it allocates a tensor, so a loader that reads the card's free memory after its own context exists can add it back:
# the offset above already contains it. measured: the card's free memory 0.22 GiB below what the job's card had, in every row of bench run ``cuda-r9``.
CUDA_CONTEXT_BYTES = round(0.22 * GIB)
