# Share memory bench results

**Status:** Proposed (2026-11-22)

## Problem

The memory verdict is calibrated from the bench, and the bench has run on a few machines. A contributor with another NVIDIA card or an Apple Silicon Mac could close the gaps, but today they would have to run every case of every family, hand over a results file that carries file paths from their machine, and trust that `fit` judges it fairly: it judges rows against the machine it runs on, and the run's settings sit in a separate `meta.json` that is easy to forget. A bad command line, such as a mistyped case id, ends in a traceback.

## What a contributor sees

1. They install the app, download the models they want to measure on the Models page, and run one command:

   ```bash
   uv run python -m bench.memory run --quick
   ```

   `--model NAME` (repeatable) limits the run to the cases of those models. Each family has one `quick` case: the smallest size that still loads the model the way a real job does, at two steps.
2. The run prints each case's outcome as it goes and ends by naming the file to attach: `bench/memory/runs/<media>/<label>/results.jsonl`, with the label `<platform>-<os>` by default (`cuda-win32`, `cuda-linux`, `mac-darwin`).
3. They attach that one file to an issue. It holds nothing that identifies them.

A mistyped case or family, or a machine with no NVIDIA card and no Mac, is refused in one line that says what is accepted. The internal `one` command no longer shows in `--help`.

## What a row records

Every row carries what a maintainer needs to judge it, so `meta.json` is no longer written:

- the commit and whether the tree had uncommitted changes, and the app version;
- the OS release, the GPU name, the driver and CUDA versions, the total RAM, and on a Mac the chip.

A row never records a host name, user name, serial number or PCI bus id. The row's `os` (`sys.platform`) joins the key of a result, so rows from two systems in one folder stay apart instead of one replacing the other.

Error text and the `--dry-run` listing replace the home directory with `~`, and the repository root, the Hugging Face cache and the data folder with a named placeholder, longest path first. A LoRA file's path is already never recorded, only its size.

## How rows are judged

`fit` judges each row against the card or Mac it ran on, not against the machine `fit` runs on: the row's own GPU memory and RAM (or the Mac's total memory) replace this machine's budget, and the weights come from the shipped table when the model is not downloaded here. The report names the machine of each row, so a failure reads "12 GB card, Windows" next to the case id.

## Measured figures in one file

**Today:** the bench's measured figures are spread over several Python files: `zvisiongenerator/utils/memory_verdict.py` (image terms per family, platform and quant; RAM working terms; LoRA and mapped shares), `model_memory.py` (LTX), `h3_memory.py` (H3), `memory_offsets.py` (per-OS CUDA offset and context), and the audio `memory_gb` figures in `model_metadata.yaml`. A recalibration, or a contributor's figures, means editing several of them.

**Proposed:** a new `zvisiongenerator/utils/memory_figures.yaml` holds every measured figure. Each one is keyed by family, then platform, then term, and carries its value, whether it was measured, and its bench case ids, as fields rather than comments. The audio `memory_gb` figures move there too, and the audio preset looks its figure up by model id.

It is a separate file, not `model_metadata.yaml`, because that file is read almost everywhere (listing, downloads, help, every load), while the figures are only needed when a job is judged (the CLI warning, the Web tag, CUDA placement). Only the memory engine reads the new file, on its first verdict, and then caches it, so no other reader pays for it.

Staying in code: the formulas, the Fits/Tight/Too large bands, structural constants (the 1 MP anchor, the quant levels) and the generated weight-size table (`weight_totals_table`), which is tool-generated data of a different kind.

A typed loader validates the file on first read and fails loudly if a term a formula needs is missing or misnamed. A test checks that every figure has its measured status and, where measured, case ids.

This ties to sharing: `fit` can print the exact change to `memory_figures.yaml` that a contributor's rows suggest, so a figures PR is a data diff and never touches code.

Done when every verdict for every shipped model, platform and job type is byte-identical before and after the move, and the tests that pinned constants read them from the file.

## Documentation

`docs/developer/memory-bench.md` gets a **Share your results** section: what to have installed (the app, an NVIDIA driver or an Apple Silicon Mac, the models downloaded first, and on Windows the sysmem fallback setting), the quick command, and the file to attach. There is no separate report command: the redacted `results.jsonl` is the report.

## Alternatives considered

- **A `report` command that writes a summary.** Another format to keep in step with the rows; the redacted rows already say everything.
- **Keep `meta.json`.** It is a second file to attach and it can drift from the rows it describes.

## Open questions

- Whether a quick case whose model is not downloaded is skipped with a line or stops the run; either way no case should download inside its timed process.
