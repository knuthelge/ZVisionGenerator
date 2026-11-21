---
search:
  exclude: true
---

# Memory bench

The memory verdict (`utils/memory_verdict.py`) is calibrated from measured peaks. The bench in `bench/memory/` runs the app's own command for each case in a fresh process and records the job's peak memory, and `fit` checks the verdict against it.

## Run it

```bash
uv run python -m bench.memory run --media image --family zimage --label mine
uv run python -m bench.memory fit bench/memory/runs/image/mine/results.jsonl
```

`--media` is `image`, `video` or `audio`, and the cases are in `bench/memory/cases/<media>.yaml`. Download the models first, on the Web UI's Models page (`uv run ziv-ui`): the app has no download command, and a model fetched inside a case counts against `--timeout`. `--dry-run` lists the cases without loading anything, `--force` re-runs a case that has a result, and cases marked `lora: true` need `--lora PATH`. `--budget-gib N` (CUDA) and `--total-gib N` (Mac) make the app read a smaller machine. Results go to `bench/memory/runs/<media>/<label>/`, which version control ignores. On Windows, set the NVIDIA Control Panel's **CUDA - Sysmem Fallback Policy** to **Prefer No Sysmem Fallback** for the bench's Python, so a job that does not fit stops instead of spilling into system memory. The program to add is the uv-managed base CPython interpreter, not the virtual environment's launcher; print its path with `uv run python -c "import sys; print(sys._base_executable)"`.

**Hold probe.** To find the least GPU memory a case runs on, run it under growing holds, then `fit` both files:

```bash
for hold in 1.0 1.5 2.0; do uv run python -m bench.memory run --media image --case zit-q4-m-resident --label mine-probe --hold-in-process-gib $hold; done
uv run python -m bench.memory fit bench/memory/runs/image/mine/results.jsonl bench/memory/runs/image/mine-probe/results.jsonl
```

Space the holds until one run is out of memory and the next smaller one ran; the figure is the middle of that gap.

## What a figure is

A figure is what the job itself needs, not the job card's live readout of the whole machine.

- **CUDA GPU:** the middle of the hold-probe bracket where the case was probed, otherwise PyTorch's allocated peak plus one CUDA offset per operating system (`utils/memory_offsets.py`). The reserved peak is never used: it grows to fill whatever the card has free.
- **CUDA system memory:** the highest anonymous plus shared memory the process held in one sample, plus the file-backed pages where the weights are mapped from disk. On Windows it is the process's private working set, and where the weights are mapped from disk its full working set, which holds the mapped pages. `fit` also lists the process's peak commit charge and the system's commit rise, which are not judged: a job whose commit outruns RAM plus the page file fails although its RAM fits. A Windows RAM check is left out when the job filled RAM (its peak commit charge is above the machine's physical RAM), because its figure is then a lower bound.
- **Mac:** the process's peak footprint (`phys_footprint`), with no offset.

## Calibration rule

`fit` exits 1 and names each failure when a check fails:

- A case that ran has a GPU and a RAM estimate within the larger of 10 % and 1 GiB of its measured figure.
- A case that ran never reads **Too large**, except a Mac job estimated above the machine's whole memory.
- A case that ran out of memory reads **Too large** against the card less what other programs held.
- A case that ran out of system memory (its process was killed, or PyTorch or Windows reported no room) reads **Too large**, whatever the card held.

## Change a constant

The constants of `utils/memory_verdict.py`, `utils/model_memory.py`, `utils/h3_memory.py`, `utils/memory_offsets.py` and `backends/cuda/image/placement.py` each carry a comment that marks them *measured* or *not measured*, with the case id. If `fit` fails, refit the constant it names and run `fit` over all results. Never pad: the status bands in `utils/memory_grade.py` are the only margin, so never add one to make a status come out.

To add a case, add an entry under a family in `bench/memory/cases/<media>.yaml`; a case measures only what the engine has a term for.
