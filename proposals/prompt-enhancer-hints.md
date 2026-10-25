# Prompt enhancer: model hints, worked examples and a local judge

**Status:** Proposed (2026-10-22). Split out of [prompt enhancer v3](archive/prompt-enhancer-v3.md), which shipped everything else.

Three independent items, each measured on the enhancer bench (`make enhance-bench`) before it ships:

1. **[Per-model hints](#1-per-model-hints)** in text and video rewrites.
2. **[Worked examples](#2-worked-examples)** in text and video rewrites.
3. **[A local judge](#3-a-local-judge)** for the bench.

## Problem

- **No model awareness in text rewrites.** Every image model gets "a text-to-image model"; the only per-model setting is FLUX.1's word ceiling. Krea 2, Ideogram 4 and Klein have arrived since v2, each with its own prompting guidance, and [MOSS-Image](moss-image.md) will add another.
- **No example in text and video rewrites.** The caption prompt has a worked example and follows its format reliably. Text and video rewrites have rules only.
- **The bench counts, but does not rank.** Its checks cover length, format, quoted text and kept subjects. Fidelity and style adherence are judged by reading outputs and picture sheets by eye.

## Evidence

- `resolve_enhance_ceiling` in `utils/prompt_enhance.py` reads `enhance_max_words` from `model_presets` / `video_model_presets` per family (set for `flux1` in `config.yaml`). A hints key can be read the same way, so hints live in config next to the model, not in code.
- `prompt_enhance_caption.py` builds its example from `_EXAMPLE_HEAD`, `_EXAMPLE_TAIL` and a per-style `_EXAMPLE` (photo and art), so the pattern exists.
- Every system message is built by one helper, `_system_message(intro, rules, output_line)`, so a hint or an example is one more argument, not a change in three builders.
- [prompt-enhancer-v2](archive/prompt-enhancer-v2.md) rejected per-model hints "for now" (item 4); v3 reopened them and left them here.
- The bench already has `run`, `rescore`, `compare` and `sheet` (`bench/enhance/`), with 63 cases and stored baselines, so each item has a before-and-after measure.

## Proposed change

### 1. Per-model hints

- An optional `enhance_hint` string on a model preset family (image and video), resolved like `enhance_max_words`, and passed to the text and video builders as one line under the intro.
- A hint describes the prose the model reads best (sentence length, ordering, whether it wants lighting and camera terms, how it treats quoted text), in one or two sentences. First set: Klein, Z-Image, Krea 2, Ideogram 4 in text mode, LTX; MOSS-Image when it lands.
- Captions (JSON) get no hint; their format is already model-specific.
- A family without a hint gets today's prompt unchanged.

### 2. Worked examples

- One short input → output pair per mode (image text rewrite; video scene and action; video action only), chosen per style group (photo or art) as the caption example is.
- Examples use a subject unlike typical prompts, so the model does not copy it; the bench's `keep` and `forbid` checks catch leakage.

### 3. A local judge

- `uv run python -m bench.enhance judge LABEL [--model smart|REPO]` scores each stored output 1 to 5 on fidelity (kept what the user wrote) and style adherence (followed the picked options), from a fixed rubric, with the Smart model or another local chat model.
- Scores land in `results.jsonl` next to the checks and in the `compare` report. They rank runs; they never pass or fail a case.
- It runs on stored outputs, one model at a time, like `rescore`.

## Steps

1. The judge, so items 1 and 2 can be ranked as well as counted.
2. Hints, one family at a time, each compared against the baseline (checks and a picture sheet).
3. Examples, measured on their own.

## Docs and changelog

- **Hints and examples:** CHANGELOG `[Unreleased]` line ("the enhancer tailors rewrites to the selected model"); a sentence in `docs/concepts/prompt-enhancer.md`; `enhance_hint` in the config reference.
- **Judge:** `docs/developer/enhancer-bench.md`.

## Alternatives considered

- **Hints in code, keyed by family.** Config keeps them next to `enhance_max_words` and lets a user tune their own models. Rejected.
- **A hosted LLM as judge.** It sends prompts off the machine. Rejected, as in v3.
- **Several examples per mode.** Longer system prompts slow the 4B model and raise the risk of copying. One first; more only if the bench shows a gain.

## Open questions

- May a hint name the target model, or only describe the prose it wants?
