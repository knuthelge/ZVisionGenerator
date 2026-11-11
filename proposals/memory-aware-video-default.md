# Preselect a video model that fits the machine

**Status:** Proposed (2026-11-11)

## Problem

The Web UI preselects `ltx-8` as the video model on every machine (`ui.default_models.video` in `config.yaml`). `ltx-8` needs about 27 GB of system memory on Windows and Linux and a Mac with 32 GB or more. On a 16 GB Mac or a PC with 16 GB of RAM, a first-time user who presses Generate on Video starts a large download and then a job that the memory check refuses, or a machine that crawls. The docs can say "switch to `ltx-4` first", but most newcomers won't read that before pressing Generate. Both first-time-user tests of the docs overhaul hit this as their first video.

## Evidence

- The first-run dialog already picks an image model by memory: `recommended_first_run_model` in `web/first_run.py` takes `first_run.large_model` when the machine has at least `first_run.min_memory_gb` (unified memory on Apple Silicon, VRAM on CUDA), otherwise `first_run.small_model`. Video has no equivalent.
- The memory figures in the docs: `ltx-4` about 20 GB of system memory on Windows and Linux, `ltx-8` about 27 GB, and a Mac with 32 GB or more for `ltx-8`.
- `ziv-video` uses `ui.default_models.video` only when extending a clip whose model isn't available locally, so the CLI barely depends on this default.

## Proposed change

When the user has not set a video default, the Web UI preselects `ltx-8` only on machines with enough memory for it, and `ltx-4` on every other machine. A `first_run`-style pair in `config.yaml` names the two models and the threshold. Which memory counts on Windows and Linux is system memory, not VRAM, because video streams its weights.

A video default the user saved on the Config page always wins.

## Alternatives

- **Docs only.** The docs overhaul states which model is preselected and when to switch. That stays either way, but it doesn't protect the user who doesn't read first.
- **Default to `ltx-4` everywhere.** Simpler, but users with a 32 GB Mac or a PC with plenty of RAM get the less faithful model by default.
- **Pick by the memory estimate per model.** The memory check already estimates whether a model fits; preselecting the largest model that fits would extend to LTX-2.5 and H3. More general, but more to test.

## Open questions

- The threshold for `ltx-8` on Windows and Linux (27 GB measured on one machine) and on a Mac (32 GB).
- Whether `ltx25-4` should be the small pick instead of `ltx-4`.
