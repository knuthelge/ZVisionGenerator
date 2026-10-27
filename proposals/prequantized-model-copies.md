# Download pre-quantized copies instead of building them

**Status:** Proposed (2026-10-27)

## Problem

Most models are downloaded at full precision and then quantized on the user's machine. That costs the user in three ways:

- **A download several times larger than what runs.** LTX-2.3 on Windows/Linux downloads about 103 GB and then runs a copy of about 23 GB. The prompt enhancer downloads 9 GB (Fast) or 19 GB (Smart) in bf16 and quantizes it to 4 bits on every load.
- **A wait on first use.** The job stops at "Preparing" or "Saving q8 copy…" for 2–12 minutes. These are also the memory peaks that trip `systemd-oomd` on 30 GB Linux machines (8–9 GiB for the video copies).
- **Sources that can vanish.** Several pins are personal uploads with low download counts. A pinned revision doesn't help if the repo is deleted.

The app already writes every one of these copies itself (`utils/stored_quant.py`, `backends/cuda/video/weights.py`, `backends/cuda/video/h3/stored_copy.py`, `ideogram4_instant_builder.py`, `moss_builder.py`, `seedvr2_loader.py`). This proposal is about building them once, from the same pins and with the same code, publishing them, and downloading the copy instead of the source.

[ideogram-instant.md](archive/ideogram-instant.md) deferred this ("Publish our own converted repos … revisit if the first-run cost is a problem"). This proposal revisits it for every model. **The finding is that only some models are worth it, one can't be published at all, and on macOS the image copies already exist upstream.**

## What the user sees

- **A smaller first download.** The first-run dialog and the Models page show the smaller size: `ltx-4` about 31 GB instead of about 103 GB, Fast enhancer about 3 GB instead of 9 GB.
- **No preparation step.** The first job goes from download straight to loading. The "Preparing" or "Saving copy" phase and its memory peak don't happen.
- **Same output.** A published copy is the same bytes the app would have built (or, where a library packs nondeterministically, within the tolerance of a local rebuild; see Test).

Nothing else changes in the UI. The size labels sit in the existing first-run dialog and Models page, which already work on desktop, tablet and phone. The unquantized level of an image model still downloads the original.

## Inventory

Sizes are from the Hugging Face API at the pinned revisions (2026-10-27). Preparation costs are the measured figures from the linked proposals; "est." marks estimates from parameter counts.

### Windows and Linux

| Model | Downloaded today (source) | Prepared locally | With a published copy | Saved |
|---|---|---|---|---|
| Enhancer Fast | 9.1 GB bf16 (`coder3101/Qwen3.5-4B-heretic`, personal) | NF4 at every load, never stored | ~3 GB NF4 | ~6 GB per install, the quantize step on every load |
| Enhancer Smart | 18.8 GB bf16 (`coder3101/Qwen3.5-9B-heretic`, personal) | Same | ~6.5 GB NF4 | ~12 GB |
| `ltx-4`, `ltx-8` (LTX-2.3) | ~103 GB: transformer 46.2 (`Lightricks/LTX-2.3`, official), Gemma 3 text encoder 48.7 (fp32), connectors 6.3, VAEs and vocoder 1.9 (`diffusers/LTX-2.3-Distilled-Diffusers`, official) | `ltx-4@q4` (NF4 transformer + FP8 text encoder, ~23 GB est.), `ltx-8@q8` (FP8 transformer, same encoder, ~34 GB est.). Time and peak not measured for 2.3; LTX-2.5's identical build took 327 s at 9.2 GiB | ~31 GB (`ltx-4`), ~42 GB (`ltx-8`), ~53 GB for both | ~60–70 GB, ~5 min, the 9 GiB peak |
| `ltx25-4` (LTX-2.5) | 71 GB (`unsloth/LTX-2.5-FP8`, a byte-identical copy of the gated official repo) | `ltx25-4@q4`, 22 GB; 327 s, 9.2 GiB ([ltx-2.5.md](ltx-2.5.md)) | ~31 GB | ~40 GB, ~5.5 min |
| `h3` | 98 GB: transformer 66.3 bf16, VAEs 11 (`MiniMaxAI/MiniMax-H3`, official), NF4 Qwen3-VL shards 1–4 ~19 (`unsloth/…-bnb-4bit`), turbo LoRA 1.4 (`lightx2v`) | `h3@q4`, 12 GB (folded AdaLN tables, turbo LoRA merged); 4–11.6 min, 8.3 GiB | ~42 GB | ~56 GB, up to 12 min. Not publishable (Licences) |
| SeedVR2 | FP8 files as published, 8.5 / 3.4 GB (`numz`, `AInVFX`) | nothing | — | nothing |

### macOS

| Model | Downloaded today | Prepared locally | Already published upstream | Saved |
|---|---|---|---|---|
| Enhancer Fast / Smart | 2.4 / 16.1 GB MLX 4-bit (`McG-221`, `TheCluster`, both personal) | nothing | — | nothing (longevity only) |
| `zit` | 33 GB | mflux `save_model`, 20–30 s, 10–17 GB disk ([image-model-memory.md](archive/image-model-memory.md)) | `mflux-community/z-image-turbo-mflux-q8` 11.0 GB, `-q4` 5.9 GB | 22–27 GB |
| `klein9b` | 35 GB | same | `mflux-community/flux2-klein-9b-mflux-q8` 17.9 GB, `-q4` 9.5 GB | 17–25 GB |
| `klein4b` | 16 GB | same | `mflux-community/flux2-klein-4b-mflux-q4` 4.6 GB (its q8 lists 17.1 GB, larger than the source; check) | ~11 GB at q4 |
| `krea2` | 36 GB | same | `mflux-community/krea-2-turbo-mflux-q8` 22.2 GB, `-q4` 15.8 GB | 14–20 GB |
| `ideo` | 28 GB (Instant bf16 + FP8 components) | builds an FP8 folder (~19 GB) in seconds | only the base model, not Instant | ~9 GB |
| `moss` (after M2) | — | — | `mflux-community/moss-image-sft-mflux-q8` 14.0 GB, `-q4` 7.5 GB | for [moss-image.md](moss-image.md) M2 |
| `ltx-4`, `ltx-8`, `ltx25-4` | already MLX q4/q8 packs (`dgrauet`, `MarceloFSO`) + `mlx-community` Gemma 3 | nothing | — | nothing |
| `h3` | 107 GB: INT8 transformer 37.8 (`water1234`, personal), FL2VA components 68 incl. ~50 GB bf16 text-encoder shards | `h3@mlx-te4`, ~15 GB | `mlx-community/Qwen3-VL-32B-Instruct-4bit`, same scheme and shapes | ~35 GB; see Spike first |
| SeedVR2 | 7B fp16 16.5 GB, deleted after the copy | int8 copy, 9.3 GB | — | ~7 GB once |

`mflux-community` is the mflux project's own conversion org. Its repos are written by `mflux-save`, the same quantizer our stored-quant flow calls, so on macOS we don't need to publish image copies; we pin theirs.

## Licences

Read from each upstream licence on 2026-10-27. None of them names quantization; each defines derivatives broadly enough ("any modified version", "adapted weights") that a quantized copy is a derivative and the distribution terms apply.

| Source | Licence | Can we publish a quantized copy? |
|---|---|---|
| Qwen3.5 heretic (enhancer) | Apache-2.0 | **Yes.** Keep LICENSE and NOTICE and state the change |
| LTX-2.3, LTX-2.5 | LTX-2 Community License | **Yes, with a catch.** Licence copy, Attachment A use restrictions, modified-file notices, distribution only under this licence (§3.1–3.4). §3.5 restricts transfers to large commercial entities; see Open questions |
| Gemma 3 (the LTX-2.3 text encoder) | Gemma Terms of Use | **Yes.** Pass on the §3.2 use restrictions, a copy of the terms, modified-file notices, and a NOTICE file (§3.1) |

MiniMax H3 can't be published: its licence allows distribution only within an Applicable Territory that excludes the EU, UK, Korea and US, which a Hugging Face repo can't respect, and the unlicensed `lightx2v` turbo LoRA is merged into `h3@q4`.

## Ranked by value

1. **Prompt enhancer on Windows/Linux.** NF4 snapshots of the two pinned repos, so 3 GB and 6.5 GB instead of 9 GB and 19 GB, and no quantize step on every load. macOS already downloads MLX 4-bit. Apache-2.0, the smallest repos, and no format of our own: transformers reads `quantization_config` from the checkpoint. Every Windows/Linux user who enhances benefits.
2. **LTX-2.3 on Windows/Linux (`ltx-4`, `ltx-8`).** The largest saving: ~103 GB down to ~31 GB, and the alias fixes the level, so the copy is exactly what runs. It also takes the bulk of the download off `Lightricks/LTX-2.3`, a repo that has deleted and moved files before. The licence allows it with conditions (see Licences).
3. **LTX-2.5 on Windows/Linux (`ltx25-4`).** 71 GB down to ~31 GB, the same licence and format as 2. macOS already downloads a quantized pack, so parity holds without work there.
4. **Image models on macOS: pin `mflux-community`'s q4/q8 repos.** Saves 11–27 GB for users who run at q4 or q8, with no hosting by us. Weaker than 1–3, because the default level is unquantized: a user who never picks q4/q8 downloads the original anyway.

**Not worth it:**

- **SeedVR2.** Windows/Linux already download FP8; macOS saves 7 GB once and deletes the download afterwards.
- **`ideo` on macOS.** ~9 GB, and its conversion takes seconds.
- **The macOS LTX packs.** They are already quantized.
- **Mirrors of the macOS enhancer repos** (`McG-221`, `TheCluster`). Byte-identical copies change nothing for the user.
- **A byte-identical mirror of every other personal source** (`Hippotes`, `MarceloFSO`, `lxxxy6`, `akhaliq`, `LBH-123-AI`). Each mirror makes us the maintainer of someone else's file. Mirror a source only when we already publish that model, or when it shows signs of going away.

The macOS H3 saving comes from `mlx-community`'s text encoder instead (see Spike first).

## Proposed change

### Hosting

One Hugging Face **organization** for the project, with at least two maintainers, so no copy depends on one person's account (see Open questions). Every repo is public and ungated: the app's rule is no Hugging Face account for defaults ([first-run.md](archive/first-run.md)). Each repo carries:

- the upstream licence files and the NOTICE text each licence requires;
- a model card with `base_model`, `base_model_relation: quantized`, the source pins, what was changed, and a line saying it is not an official release.

Phases 1–2 host about 95 GB (phase 3 hosts nothing).

### Formats

| Platform | What | Format |
|---|---|---|
| Windows/Linux | Enhancer | transformers `save_pretrained` with bitsandbytes NF4 (the `BitsAndBytesConfig` the loader uses today), text model only |
| Windows/Linux | LTX transformer and text encoder | the existing prepared-copy shards (NF4 packed as `Linear4bit` saves itself, FP8 `float8_e4m3fn`) with their manifest, unchanged |
| macOS | Image models | `mflux-community`'s `mflux-save` output (MLX affine, group 64) |

No new format, and no torchao: torchao would add a dependency and a second load path to save bytes we already save.

### Pins and download plans

- **Config.** Each alias that has a published copy gets a pinned `prequantized` entry, per platform and level, in the `repo@sha` form the config already uses. The alias's source pins stay, because the unquantized level (images), the converters (LTX fine-tunes take the VAEs and connectors from the base) and the build script still read them.
- **Download plans** (`backends/cuda/download_plan.py`, `backends/mac/download_plan.py`, `downloads/plans.py`). When a published copy exists for the level being run, the plan fetches the copy and only the source files the copy doesn't replace (VAEs, connectors, vocoder, tokenizers, configs). The Models page "downloaded" check and the first-run size labels read the same plan.
- **Placement.** The copy is linked from the Hugging Face cache into `~/.ziv/models/<alias>@q<bits>`, as `moss_builder` links its sources today. The Models page then counts and deletes it like a built copy.
- **Identity.** A published copy's manifest records its repo pin instead of the source's file identities, and the loader trusts it as current while the pin matches.
- **Format tags.** Today's stored-quant format tags embed the exact library versions (`mflux-0.22.0`, `diffusers-…+transformers-…+bitsandbytes-…`), so any bump would mark a published copy stale. Published copies get a format version we own instead, with a contract test that loads a small fixture of each format under the locked libraries. When a bump breaks loading, the copy is rebuilt and the pin moves.
- **Existing local copies.** A locally built copy whose manifest uses the old tags is rebuilt or downloaded again once. That is a breaking change for disk users already have, so ask before shipping; per project rules there's no migration code.

[free-source-weights.md](free-source-weights.md) becomes unnecessary for every model with a published copy, because the original is never downloaded. It stays relevant for `h3` and for unquantized image runs.

### Reproducibility

`scripts/build_prequantized.py <alias> --platform cuda|mac --bits 4|8 --out <folder>` builds a copy from the pinned sources by calling the app's own writers (`weights.ensure_prepared`, the stored-quant writers, `save_pretrained` for the enhancer). It writes the model card, the licence files and the source pins.

Upload is a manual step by a maintainer, never CI: CI has no GPU, and a publish should be a deliberate act, like a release tag. The published pins change in a normal commit with a CHANGELOG line.

### Becoming a maintainer of weights

This is the real cost, and the reason for the short list:

- **We become the distributor.** The licence duties move to us: notices, the LTX §3.5 clause, Gemma's use restrictions, takedown and abuse reports, and upstream licence changes.
- **Rebuilds.** Every upstream pin bump, and every library change that breaks a format, means rebuilding and uploading tens of GB on the reference machines.

## Phases

Each phase ships on both platforms together.

1. **Enhancer on Windows/Linux.** macOS already downloads MLX 4-bit. Loader change: pass `BitsAndBytesConfig` only when the checkpoint has no `quantization_config`. The CPU fallback reads NF4 through bitsandbytes' CPU kernels; measure its speed first, and if it is unusable, keep the bf16 source for the CPU path only.
2. **LTX-2.3 and LTX-2.5 on Windows/Linux.** macOS needs nothing; its packs are already quantized.
3. **macOS image models from `mflux-community`**, and H3's text encoder from `mlx-community` if its spike passes. Windows/Linux keep building locally, which is the same feature by a different route.

## Spike first (one day, both platforms)

- **Enhancer.** Save the NF4 Fast model, reload it from the folder, and run `make enhance-bench`: rewrites identical to today's at the same seed. Record size and load time against quantize-at-load.
- **mflux-community.** Load `flux2-klein-9b-mflux-q4` and `z-image-turbo-mflux-q8` through our stored-quant path. The image must match a locally saved copy at the same seed, bit for bit or within mflux's own repeat tolerance. Explain the 17.1 GB Klein 4B q8.
- **LTX.** Rebuild `ltx-4@q4` twice and compare SHA-256. If NF4 packing on the GPU isn't deterministic, define the tolerance (render PSNR against a local copy) before publishing.
- **H3 text encoder (macOS).** Replace the bf16 Qwen3-VL-32B shards (FL2VA text-encoder shards 1–11 and 14) and the local conversion to `h3@mlx-te4` with `mlx-community/Qwen3-VL-32B-Instruct-4bit`, pinned. It uses the vendored `packages/minimax_h3_mlx/quantize_text_encoder.py` scheme (affine 4-bit, group 64, every 2-D tensor including `embed_tokens`) with the same shapes; only the key prefix differs (`language_model.model.*` against `model.language_model.*`). The H3 download drops from about 107 GB to about 72 GB, and the conversion and its peak go away. Pass: after renaming keys and keeping layers 0–49, every tensor equals the local `h3@mlx-te4`, or the prompt embeddings and a fixed-seed clip at `s` match it. Record the download, the time to the first step and the peak.
- **Hosting.** Confirm the org's storage quota and that ~45 GB repos upload and download at normal speed.

## Test

- **Unit:**
  - with a published pin, the plans fetch the copy plus only the components it doesn't replace, and the size labels follow;
  - the loader accepts a published copy by its pin, and rejects one whose pin changed;
  - the contract test loads a fixture of each published format.
- **Hardware:**
  - one render per published copy against a locally built copy: same seed, identical or within the spike's tolerance;
  - first-run download size and time to the first step, before and after.

## Open questions

1. Which Hugging Face org hosts the copies, and who besides the project owner maintains it? An org with one active maintainer only moves the single-maintainer risk to us.
2. Is hosting the uncensored ("heretic", abliterated) enhancer models, today personal uploads by `coder3101`, under the project's name acceptable? That is a reputational choice as much as a technical one. If not, phase 1 is dropped, or the enhancer moves to a censored base.
3. Is the LTX §3.5 clause acceptable for a public ungated repo? It forbids transferring a derivative to a Commercial Entity (≥ US$10M revenue) that has no paid licence, which an ungated repo can't check; the existing community copies (`unsloth`, `dgrauet`) take the same position. The alternative is gating with automatic approval, which brings back the Hugging Face account requirement.
4. Should MiniMax be asked for permission to publish an H3 copy? It would also need a licensed turbo LoRA.

## Alternatives considered

- **Use third-party pre-quantized repos everywhere.** Right for macOS image models (`mflux-community`) and the H3 text encoder (`mlx-community`). On Windows/Linux the NF4 copies that exist (`OzzyGT`, `Veetance`, `cookieshake`, `AcademiaSD`, …) are personal uploads in other layouts, which would trade one single-maintainer risk for another and change the output.
- **Our own Windows/Linux image-model copies (NF4/FP8).** Only if usage shows users pick q4/q8; the default level is unquantized.
- **Stream-quantize during the download.** Cuts the disk but not the download, and needs a streaming writer per loader.
