# Convert LTX-2.5 and MiniMax H3 fine-tunes

**Status:** Implemented on all three platforms; macOS gates run as a minimal set, two open findings (2026-10-25), shipping in v0.13.0. The Windows/Linux gates ran on the reference card; the macOS gates ran on a 32 GiB M5 as a minimal set (Part 3 G3 and G4, Part 4 G6). See [Implementation and results](#implementation-and-results-2026-10-25) and [macOS gate results](#macos-gate-results-2026-10-25).

Four parts, in this order:

1. MiniMax H3 on Windows/Linux
2. LTX-2.5 on Windows/Linux
3. LTX-2.5 on macOS
4. MiniMax H3 on macOS

Part 1 comes first, and its work (per-model naming and conversion) may run alongside Parts 2 and 3, since it touches different modules. Parts 3 and 4 started after the Mac test of the shipped LTX-2.5 and H3 support ([ltx-2.5-mac-test-plan.md](archive/ltx-2.5-mac-test-plan.md)) passed. Each part opens with a spike and a gate table, and is built only once its gates pass.

This follows [Convert LTX-2.3 checkpoints](archive/ltx-checkpoint-conversion.md), and it picks up an item that [LTX-2.5 and MiniMax H3](ltx-2.5.md) left out of scope: "Converting single-file fine-tunes for either family."

## Problem

People pick a video model for its fine-tunes as much as for the base. H3 has a large community of fine-tunes and merges on the Hub. LTX-2.5 has fewer, but they exist. None of them runs in the app today:

- **H3 on Windows/Linux.** H3 runs only from a diffusers folder (`model_index.json` plus `transformer/`). Most fine-tunes are single ComfyUI or MiniMax-native files, so they can't be loaded at all. A fine-tune that does come as a diffusers folder replaces the one stored copy, `h3@q4`. So stock and a fine-tune can't sit side by side, and every switch rebuilds the copy (11.6 minutes on the reference card).
- **H3 on macOS.** Only the pinned INT8 transformer runs. There is no way to make an INT8 copy of a fine-tune.
- **LTX-2.5.** The LTX-2.3 converter refuses every 2.5 checkpoint ("its feed-forward layers have no bias, as in LTX-2.5").
- **Turbo.** H3 runs at 4 or 8 steps only because the app merges lightx2v's turbo LoRA. Many H3 fine-tunes are already step-distilled, and merging the LoRA into one of those distills it twice. Nothing in a file says reliably which kind it is.

What the user wants: pick the file on the Models page (or run `ziv model model`), say what it is, and get a model under its own name. It should run like the stock model, at the same speed and memory, with LoRAs, image to video, first and last frame, Extend and upscale. It should sit next to the stock model and be deletable on its own.

## Already fixed on this branch

The discovery found three bugs in the shipped H3 code. They are fixed on this branch, outside this proposal, and the parts below build on those fixes:

- **Stale copy on Windows/Linux.** A fine-tune with the same file layout could silently reuse the stock `h3@q4`, because the copy's manifest identified source files by name and size only. Local files are now also identified by modification time and inode.
- **FP8 scale on Windows/Linux.** Scaled-FP8 transformers were stored without their `weight_scale`. They are now dequantized as they stream, per tensor or per row, and other quantized formats are refused.
- **Local folders on macOS.** A local H3 folder's own transformer was ignored, and the pinned INT8 one ran in its place. A local folder now loads its own transformer when it is in the fork's INT8 layout, and a bf16 one is refused instead of being replaced by stock.

## Decisions (2026-10-23)

The user answered the three questions this proposal first left open.

1. **LTX-2.5 sources: a verified repackage, or nothing.** The LTX-2.5 pins, the distilled LoRA and the stock test input come from an ungated repackage only when its file's SHA-256 matches the gated original's LFS id, as the Hub listing shows it. Reason: this proves the repackage is byte-identical to the original without reading gated content. Downloads stay ungated, with a link to the original. Outcome (2026-10-23): every item has a verified copy. These are the dev and distilled transformers (and so the distillation and connector pins), the distilled LoRA, and the stock test input. See [Verified sources](#verified-sources).
2. **No per-model H3 schedule for now.** Every converted H3 model runs at shift 6.0, at 4 or 8 steps. A per-model schedule is added only if Part 1's spike shows a problem (G6). Reason: no fine-tune has yet been shown to need one.
3. **Odd H3 formats: scaled INT8 in, pruned out.** Scaled-INT8 files are supported: Part 1 extends the shared dequantize helper, which now handles FP8, to INT8, with a test and a spike gate (G8). Pruned files are still refused, with a clear message. Reason: scaled INT8 is a scale multiplied out, like FP8. A pruned file's AdaLN projection has another shape, so the fold can't apply to it.

## Implementation and results (2026-10-25)

Built as planned, with these changes. The first three came from two community test fine-tunes (a pruned INT8-ConvRot H3 one and a W4A8+INT8 ConvRot LTX-2.5 one) and from a header survey of every H3 and LTX-2.5 repository on the Hub (about 470 repositories, read 2026-10-25).

- **Decision 3 reversed: pruned and rotated files are read.** The survey shows ComfyUI's own H3 templates and most community H3 files are pruned, at ranks 8, 25 and 64, and that INT8-ConvRot is the most common quantized format for both families. NVFP4 and W4A8 are common as well.
  - A pruned curve turned out to be an affine projection of stock's `silu(time embedding)` (a fit residual of 4.9e-5 on its grid, measured on the Mac gate's pruned fine-tune). The reader un-prunes it: the projection is fitted once, leaving out the embedding's noise directions (rcond 1e-5), and each curve-reading projection becomes stock-shaped. On the H3 test fine-tune, the rebuilt modulation matches the curve to 1e-5–1e-4; stock's own bf16 modulation is 1.5e-3 from the curve. One code path then serves both platforms, with LoRAs and the turbo merge working unchanged.
  - One shared decoder (`utils/quantized_weights`) reads scaled FP8, INT8 with or without ConvRot, W4A4, W4A8, W6A8, NVFP4 and MXFP8. Each format was checked against comfy-kitchen's own encoder and decoder to bf16 precision. Rotation is undone with the same block Hadamard matrix ComfyUI uses.
- **ComfyUI stores the fused QKV contiguous** (`[q; k; v]`); only MiniMax's raw sharded transformer interleaves it per head. The reader tells them apart by layout: a single file is ComfyUI, a sharded native set is MiniMax's. The MLX port reads the interleaved order, so the Mac writer interleaves again.
- **Quantized connectors** (LTX) can't be compared with stock's hashes, so they read as "can't tell" and stock's are used, rather than the file being refused.
- **No server-sent type list.** All three types run on all three platforms, so the form's list stays in the frontend.
- **Pins without full downloads.** The LTX-2.5 pins were computed with ranged reads of only the needed tensors from the verified copy (about 8 GB read, not 84 GB downloaded). The size and SHA-256 of both files were checked against the official listing first. The dev and distilled connectors are identical.
- **The macOS LTX-2.5 base** links its `quantize_config.json` unchanged for q8: the MLX loader reads the bits from the weights' shapes, so no q8 config is written.
- **The converter runs with the backend's torch settings** (TF32 for float32 matmuls). Without them the time embedding, and so the AdaLN tables, differ slightly from stock's copy.
- **Picking a Hugging Face cache file works.** The CLI and Web UI used to resolve symlinks, which turned the file into a hash-named blob and lost its shards.

Gate results on the reference machine (Windows/Linux):

| Gate | Result |
|---|---|
| H3 G1 Layouts | Pass: every layout in the table is classified; the mapped names and shapes cover stock both ways (tests on a tiny config, the H3 test fine-tune and stock's shards) |
| H3 G2 Sampled mapping | Pass: range-read tensors of `FL2VA/transformer/` (interleaved) and Comfy-Org's bf16 file (contiguous) are bit-identical to `transformer/` after mapping |
| H3 G3 Stock equivalence | Pass: stock converted from `transformer/` is byte-identical to `h3@q4` (2038 tensors, 1200 tables). The native and Comfy-Org full files were not downloaded; G2 covers their mapping |
| H3 G4 Clip | Pass: the video is identical to `h3`'s; the audio differs, as it does between two `h3` runs (the audio path is not deterministic) |
| H3 G5 Cost | Pass: 6 min 19 s and 5.7 GiB RSS for stock (Phase 2a: 11.6 min, 8.3 GiB); no swap, no kill. The pruned INT8 test fine-tune: 2 min 58 s, 6.8 GiB |
| H3 G6 Turbo flag | Not run (no full turbo file downloaded) |
| H3 G7 Fine-tune | Pass on review: the pruned INT8-ConvRot test fine-tune, turbo merged, renders sharp with its own character at 864×480, 8 steps |
| H3 G8 Scaled INT8 | Pass: decoded INT8-ConvRot Linears are within 0.9–1.0% of stock (wrong by 137% without the rotation); the fine-tune's own bf16 layers are 2.7% from stock |
| LTX G1 Mapping | Pass: every name of the LTX-2.5 test fine-tune maps onto the diffusers 2.5 transformer, keyframe embedding included, in both directions; the MLX names, dtypes and shapes equal `ltx25-4`'s transformer (7355 tensors) |
| LTX G2 Version | Pass on the files read: each reads as its own version |
| LTX G3 Pins | Pass: 8 tensors per label, 258 connectors, no projection group; the test fine-tune reads as `distilled` |
| LTX G4 Stock equivalence | Not run (needs the 42 GB distilled transformer) |
| LTX G5 Fine-tune | Pass: the W4A8+INT8 ConvRot test fine-tune converts at q4 in 8 min 15 s with 5.1 GiB RSS (this includes writing the FP8 text-encoder copy once); its quantized connectors read as "can't tell" and stock's are used. It renders sharp with its own character at `s`, 49 frames (117 s; stock `ltx25-4` 109 s at 21.6 GiB). The 1280×704 memory run was not repeated: the storage equals `ltx25-4`'s |
| LTX G6 Settings | Pass: the converted folder's load options equal `ltx25-4`'s (ancestral sampler, CRF 18, its repo_ignore and upsampler) |

macOS, checked from Linux only: the Mac H3 writer run on the H3 test fine-tune yields every one of the 1042 tensor names, dtypes and shapes of the shipped INT8 transformer (`water1234/...` at `70505b09`), and the same `config.json`. The only differences are the four layers the shipped build keeps in bf16 for its calibration, which plain rounding quantizes, as its `quant_config.json` records. The LTX-2.5 MLX writer's names, dtypes and shapes equal `ltx25-4`'s transformer. Parts 3 and 4's gates ran on the Mac afterwards; see [macOS gate results](#macos-gate-results-2026-10-25).

## macOS gate results (2026-10-25)

Run on an M5 with 32 GiB, on the fix branch `fix/mac-video` after the Mac test's fixes, as a minimal set the user chose: Part 3 G3 and G4 and Part 4 G6. Part 3 G1 and G2, Part 4 G2, G4 and G5 were not run. G6 also stands in for Part 4 G1 (strict load) and G3 (cost on a full-size file). Free disk stayed above 10 GiB, and the cached `h3` was deleted for G6 and downloaded again afterwards.

**A reader finding.** A community LTX-2.5 W4A8+INT8 ConvRot file is rejected: `model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_v.weight is stored as I8 with a scale but no quantization record`. Its quantization records live in the safetensors `__metadata__` key `_quantization_metadata` (`asym_w4a8_int8`, `group_size` 16, `convrot_groupsize` 256) with no `comfy_quant` tensors, and `utils/quantized_weights.py` reads only the `comfy_quant` form. The rejection is platform-independent, and a second community file in the same layout is rejected the same way.

**An observation to check against the design rule.** G3's fine-tune has FP8 connectors, which read as "can't tell" (96 of their Linears are FP8, so they can't match the pins), yet the Mac writer wrote its own `connector.safetensors` from the checkpoint (4.03 GB, +3.75 GiB of disk) instead of linking stock's. The Windows/Linux results above say that "can't tell" connectors use stock's. Part 3's design rule says connectors are linked only when the pins say they are stock, and otherwise written from the checkpoint; the two statements disagree. Which one is intended is open.

**Result: image to video with a converted H3 model follows the start image when the checkpoint does.** A community H3 fine-tune converted the normal way (the converter merges the turbo LoRA, then rounds to 8 bit) gives frame 0 at 27.5 dB PSNR and 0.857 SSIM against the start image, equal to stock `h3` (27.7 dB, 0.866) at the same size. Two other community H3 checkpoints ignored the start image (9.5 dB and 5.1 dB): one was converted with `--turbo` (nothing merged by the converter) and one with the converter's merge, so the behavior follows the checkpoint, not the converter. The staged path itself runs as designed on a converted folder (`--size xs --frames 33 --image <start>`: keyframes peak 14.1, vision 1.2, prompt 13.5 GiB; 162.5 s, 14.0 GiB, no swap).

A control on stock `h3` with the turbo LoRA off gave frame 0 at 12.9 dB, so the turbo is needed for start-image adherence at 8 steps. A converted model keeps it, merged before the rounding.

**Conversion cost.** Three H3 conversions peaked at 8.8, 9.1 and 9.4 GiB, with no swap (the 9.4 GiB one took 57.4 s, with a max RSS of 11.35 GB). Part 4's 8 GB bar was set too low. G3's bar is kept as written and recorded as missed.

Part 4's build decision (G0–G3 and G5 pass) is met in substance: G3's bar was set too low, and G6 covers G5.

### Follow-ups

- **Read `__metadata__._quantization_metadata`.** Extend `utils/quantized_weights.py` to take the quantization records from the safetensors metadata when no `comfy_quant` tensors are present, so the W4A8 files above convert.
- **The connectors' design rule.** Decide whether "can't tell" connectors link stock's or are written from the checkpoint (see the observation above).

## Evidence

The code was read at `9ffb6bb`, and the line numbers below are from that commit. The files the fixes above touch have moved since. Hub facts come from Hugging Face metadata and safetensors headers read on 2026-10-23. No weights were downloaded, and no gated file's content was read. The official LTX-2.5 SHA-256 values come from the gated repo's file listing (metadata only), read with an account that has accepted the LTX-2.5 licence.

### What a new model type plugs into

LTX-2.3 conversion is the template on both platforms:

- **CLI.** `converters/convert_checkpoint.py:47-50`: `LTX_MODEL_TYPE` and `MODEL_TYPES`, which the CLI choices and the Web validator share. `_cmd_model` is at `:529` and `_cmd_ltx` at `:634`. `--quantize`, `--distill` and `--force` are at `:741-744`.
- **Conversion steps.** `converters/ltx_convert.py:77-124` (`convert_ltx`): the name checks, the alias refusal, the partial folder, the promote step, and the manifest written last.
- **Checkpoint reader.** `converters/ltx_checkpoint.py`:
  - `inspect_checkpoint` (`:111`) reads the header only.
  - `_check_version` (`:162-171`) refuses 2.5 by its missing feed-forward bias.
  - `distillation` (`:190`) and `connectors_status` (`:207`) check against the pins in `ltx23_hashes.json`.
  - `transformer_tensors` (`:273`) streams the weights, dequantizes them and merges the LoRA.
- **Platform writers.**
  - Platform dispatch is `backends/__init__.py:319-346`.
  - CUDA: `backends/cuda/video/converted.py` calls `ltx_diffusers_repo(config)` with no version (`:61`) and refuses changed connectors (`:93-94`).
  - macOS: `backends/mac/ltx_converted.py` has `_BASE_FILES` (`:46-56`) and writes one tensor at a time (`_write_transformer`, `:211`).
- **Manifest.** `utils/ltx_converted.py:17-35` writes `ziv-ltx.json`, and its kind must be `ltx-2.3`. This manifest is the only record of a converted model's settings; nothing goes into `config.yaml`.
- **Web UI.**
  - `web/server.py:1719-1780`: `_convert_model_from_form` and `_start_ltx_conversion` (a background job that holds the exclusive slot).
  - `web/ltx_conversion.py:34-51`: the inspection and the recommended level.
  - `ModelsPage.svelte`: `:27` hard-codes `LTX_TYPE = 'ltx-2.3'`, the type list is at `:757-767`, and the LTX block at `:772-800`.
- **Detection, listing, level, delete.**
  - Detection: `utils/video_model_detect.py:109-111`. H3 is recognised by its `model_index.json` class, LTX by its path or its contents.
  - Listing: `converters/list_assets.py:64-76`.
  - The level shown on a row: `web/workspace_api.py:423-428`.
  - Delete: `web/model_delete.py:40-70` removes the installed folder plus its `<name>@q4`, `@q8` and `@mlx-te4` copies.
- **Load options.** `ltx_load_options(config, model_name)` (`utils/config.py:812`) picks the LTX version (sampler, CRF, upsampler) from the alias name, through `ltx_version` (`:262`). A converted folder's name is in no version block.

### MiniMax H3 today

- **Windows/Linux.**
  - There is one stored copy, `h3@q4` (`backends/cuda/video/h3/stored_copy.py:43`). Its only call site is `backend.py:543`.
  - `write_copy` (`:91`) streams the weights, merges the turbo LoRA (`:110`), folds AdaLN into per-step tables and packs NF4. `load_copy` (`:137`) applies user LoRAs at run time.
  - `model_name` is "accepted and ignored" (`backend.py:510`).
  - The source must be a diffusers folder (`sources.py:36`, `:90`).
- **Fold keys.** They come from the preset's schedules (`config.yaml:387-389`: shift 6.0, 4 and 8 forwards; `adaln.py:113`), not from the weights. The copy drops the bf16 AdaLN projection, so a missing table can't be rebuilt without the source. `adaln.py:201-209` raises "rebuild the H3 copy" on the first forward.
- **macOS.**
  - The vendored fork's `load_dit` (`packages/minimax_h3_mlx/load.py:58`) replays `quant_config.json` (`:83-101`) with a strict key match.
  - The app always applies the turbo LoRA (`h3_video.py:608-623`). The fork skips it when the LoRA path is `None` (`pipeline.py:265`).
  - The text-encoder copy is named after `model_name` (`h3_video.py:624`), so a second model name would write a second copy.
- **Upstream (not vendored).** [`Argus-AiTeam/minimax-h3-mac`](https://github.com/Argus-AiTeam/minimax-h3-mac) at `3fcc9f8c`:
  - Two INT8 producers: `scripts/build_quant.py` (round-to-nearest, RTN) and `scripts/build_calibrated_int8.py`, which needs recorded activations and a CUDA device. Both load the whole bf16 transformer into memory (`build_quant.py:87`, `build_calibrated_int8.py:203`), which doesn't fit a 32 GB Mac.
  - `reference/diffusers/convert_minimax_h3_to_diffusers.py` gives the native-to-diffusers mapping, with no transposes:
    - de-interleave the fused per-head QKV into `[q; k; v]` and split it (`:110-151`, `:256-262`)
    - swap the fc1 halves from `[gate; value]` to `[value; gate]` (`:265-270`)
    - rename `out_proj` to `to_out.0` and `fc2` to `ff.net.2`, and drop `rope.inv_freq`

### Input layouts on the Hub

MiniMax H3, from headers:

| Layout | Example | Names and precision |
|---|---|---|
| Diffusers folder | [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) `transformer/` (14 shards, 66.3 GB) | `transformer_blocks.N.attn.to_q/to_k/to_v`, `ff.net.0.proj`; BF16 |
| MiniMax native | the same repo, `FL2VA/transformer/` (13 shards) | `blocks.N.attn.qkv_proj` (fused), `mlp.fc1/fc2`, `token_refiner.*`; BF16 |
| ComfyUI single file | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) `minimax_h3_fl2va_bf16.safetensors` (66.3 GB, `config` metadata) | native names, no prefix |
| ComfyUI-saved, prefixed | community full fine-tunes, e.g. `WarmBloodAban/Minimax-h3_Singularity` | `model.diffusion_model.blocks.N.*`; BF16 |
| Scaled FP8 | `rzgar/minimax_h3_fl2va_fp8_e4m3fn`; Comfy-Org's `..._pruned_fp8_scaled` | F8 plus `weight_scale` and `comfy_quant` |
| Full turbo, diffusers names | [`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo) `..._turbo_4step_v1.1_768p_fp8.safetensors` (34 GB) | F8 with a per-row BF16 `weight_scale` |
| Pruned | Comfy-Org's `..._pruned_bf16` (40.2 GB) | `adaln_proj.linear.weight` is `[96768, 8]`, so shapes differ from stock |
| Scaled INT8 | community hybrids | I8 plus `comfy_quant` and an F32 `weight_scale` |
| MLX INT8 | [`water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8`](https://huggingface.co/water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8) (37.8 GB) | U32 plus scales and biases, group 32, `quant_config.json` |

Windows/Linux reads only diffusers names, and macOS only native or MLX names. Community fine-tunes come in every one of these layouts.

LTX-2.5, from headers:

- **Lightricks layout.** Read from the header of the dev transformer `ltx-2.5-22b-dev-transformer-bf16.safetensors` in `lxxxy6/LTX-2.5` at `3506f52c`, with HTTP range reads only. The file is byte-identical to the official one: its oid equals the official file's (see [Verified sources](#verified-sources)).
  - `model.diffusion_model.*` names, 4349 tensors (4059 BF16, 290 F32).
  - The text connectors are included: 258 tensors, 129 each for `video_embeddings_connector` and `audio_embeddings_connector`. There is no `text_embedding_projection`. It lives in the Gemma 4 file, `gemma4-12b-with-proj-ltx-2.5`.
  - All eight distillation tensors are there in BF16: the weight and the bias of `patchify_proj` (`[4096, 128]`), `audio_patchify_proj` (`[2048, 128]`), `proj_out` (`[128, 4096]`) and `audio_proj_out` (`[128, 2048]`).
  - There are no VAEs. The `config` metadata has `transformer` and `scheduler` sections, `ltx_version: 2.5.0` and `gemma_version: gemma4-12b-ltx-v1`.
  - The distilled transformer's header has the same names, shapes, dtypes, offsets and `config` as the dev one. So pins for both come from these two files, and the connector pins don't need the Diffusers repo.
- **What changed from 2.3.**
  - The video feed-forward has no bias, while the audio feed-forward keeps one (`audio_ff.net.0.proj.bias`, BF16 `[8192]`).
  - It adds `keyframes_abs_pos_embedding` (`[1, 4096]`). `_normalized_name` (`ltx_checkpoint.py:143`) drops it when it has no prefix, yet the MLX pipeline needs it for keyframes (`ltx_pipelines_mlx/_base.py:373-389`).
  - The text encoder is Gemma 4.
  - There are 258 connector tensors, against the 262 in the 2.3 pins: 2.3 also pins its 4 `text_embedding_projection` tensors, and 2.5 has none in the transformer.
- **Community fine-tunes.** Full ones are few. One is a ComfyUI-scaled FP8 file (21.5 GB, `weight_scale` of shape `[1]`). Most 2.5 repos are GGUF, FP8, NVFP4 or W4A8 quantizations, repackages, or LoRAs. unsloth's 2.5 single file uses torchao's `Float8Tensor` (`_weight_qdata`, `_weight_scale`), which no reader here knows.
- **The distilled LoRA.** The original is [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5) `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (8,899,889,568 bytes, gated). `lxxxy6/LTX-2.5` has an ungated copy with the same oid: rank 450 and alpha 450, with `diffusion_model.<module>.lora_A/B.weight` keys that `_lora_pairs` (`ltx_checkpoint.py:261`) already reads. The 2.3 distillation and connector pins don't carry over to 2.5.
- **macOS.** The shipped `ltx25-4` repo, [`MarceloFSO/ltx-2.5-mlx-q4-fast`](https://huggingface.co/MarceloFSO/ltx-2.5-mlx-q4-fast), uses the MLX names and the q4 group-64 scheme that the 2.3 writer already produces. Two things block the 2.3 writer from making a 2.5 folder:
  - `_BASE_FILES` lacks what the 2.5 pack needs (`model_files.py:98-109`, `mac/video.py:766-771`): the Gemma 4 bundle, the conv and AV VAEs, and the duration head.
  - Without `config` metadata, the writer falls back to 2.3's `embedded_config.json`, which the vendored loader then refuses (`ltx_core_mlx/utils/weights.py:114-141`).
- **No upstream converter.** `dgrauet/ltx-2-mlx` at `ae3b7ea9` only quantizes in memory and has no checkpoint-to-MLX writer.

### Turbo and distillation can't be read from a header

- **LoRA or full checkpoint:** yes. `lora_A` and `lora_B` keys, `.alpha` tensors and the metadata all show it.
- **A full checkpoint with turbo merged in:** no. lightx2v's full 4-step file and community fused-turbo checkpoints have stock names, shapes and dtypes. "Turbo" appears only in the file name. Metadata hints exist (titles, `*_video_shift` keys), and they can be wrong.
- **Hash pins** identify only exact published files, and most community turbo work is a merge.
- **A weight-space test** (candidate minus stock, compared with the LoRA's delta) would need the stock base and tensor reads on every conversion. Decided: it is not built.

### Sources and licences

- **Licences.** LTX-2.5 is under the [LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x), and every Lightricks 2.5 repo is gated (auto-approved). H3 is under the MiniMax H3 community licence, and the MiniMaxAI repo is ungated. Converted models are derivatives and keep those terms.
- **The standing rule** (2026-10-22, [ltx-2.5.md](ltx-2.5.md)): the app downloads from ungated sources and links to the original. The download dialog shows a short licence line, which is built by the download-licence work on another branch (see [Licence notice](ltx-2.5.md#licence-notice)). A download this proposal adds gets a licence entry there, and nothing more.

## Proposed change

### Shared across the four parts

**CLI.** Two new `--model-type` values in `MODEL_TYPES`:

```bash
ziv model model -i <file> --model-type ltx-2.5 --quantize 4|8 [--distill] [--name <name>] [--force]
ziv model model -i <file> --model-type h3 [--turbo] [--name <name>] [--force]
```

| Flag | `ltx-2.3` | `ltx-2.5` | `h3` |
|---|---|---|---|
| `--quantize 4\|8` | required | required | refused: the level is fixed (NF4 on Windows/Linux, INT8 on macOS) |
| `--distill` | yes | yes. The LoRA downloads from a pinned, verified ungated copy, with a link to the official file ([Verified sources](#verified-sources)) | refused |
| `--turbo` | refused | refused | the checkpoint is already step-distilled, so the turbo LoRA is not merged |
| `--name`, `--force` | yes | yes | yes |

- **Input.** `-i` is a `.safetensors` file. For a sharded set (a diffusers `transformer/` or a native `FL2VA/transformer/`), pick any shard, and the converter reads the whole set from the index beside it. The Web UI's path field and browser stay file-only.
- **Names.** The LTX-2.3 rules apply:
  - an existing name is refused without `--force`
  - alias names (`h3`, `ltx25-4` and the rest) are refused, and so are stored-quant names (`<name>@q4` and the like)
  - the model is written into a hidden partial folder and moved into place once complete, with the manifest written last
- **Platforms.** On a platform where a part hasn't shipped yet, its type is refused with a message saying where it runs. That check lives in `backends/__init__.py`, beside `convert_ltx_checkpoint`.

**`--turbo`.** The flag means "this checkpoint is already step-distilled":

- With `--turbo`, the lightx2v turbo LoRA is not merged.
- Without it, the LoRA is merged, as it is for stock.
- The app never decides this on its own. The inspection shows hints: "turbo", "lightning", "distill" or "4step"/"8step" in the file name or metadata, or a shift written in the metadata. The box still starts unchecked, and only the user ticks it.

This mirrors LTX's `--distill`, with one difference: `--distill` has a reliable detector (the pins) that preselects the box where its pins exist, and turbo has none.

**The Models page form.**

- **Types.** Two new model types, **LTX-2.5 (video)** and **MiniMax H3 (video)**. The server sends the list of types this platform can convert, in the models payload. It replaces the hard-coded list, so a type that can't run here is never offered.
- **LTX-2.5** shows the LTX-2.3 controls: the inspection, the quantize level and **Distill**.
- **MiniMax H3** shows the inspection and a **Turbo** checkbox ("Already step-distilled"). It has no quantize choice; the helper text names the level that is stored.
- **The job.** Both run as the existing background conversion job, with progress, holding the exclusive slot. `web/ltx_conversion.py` is generalised rather than copied. Then `make frontend-build`.

**Inspection.** `/api/models/convert/inspect` takes the model type. It reads only the header, works offline, and returns:

- the layout and precision found (bf16, or scaled FP8)
- for LTX, the distillation result and the recommended level
- for H3, the turbo hints
- a refusal with its reason, when there is one: a file of the other family or version names the right type ("this is an LTX-2.5 checkpoint; pick LTX-2.5"), and a LoRA is sent to the LoRAs tab

**Detection, listing and delete.**

- **Detection.** A converted folder is recognised by its manifest, through one helper shared by detection, listing and `has_complete_weights`.
- **Manifests.** LTX keeps `ziv-ltx.json`, with kind `ltx-2.3` or `ltx-2.5`. H3 gets a sibling manifest of kind `h3`, with a storage field: folded NF4, or MLX INT8. One reader serves both: the LTX one, generalised to a set of kinds.
- **Platform.** A folder converted on the other platform is refused with "convert it again on this machine".
- **The row** shows the level.
- **Delete** removes the folder and nothing else. It never touches the stock download, `h3@q4`, a shared text encoder, or files linked from a base.

**Recorded settings.**

- **Outputs.** Each output records the model by its name, as for any model, and **Reuse settings** picks the same folder. Outputs get no new keys.
- **The manifest** records the conversion:
  - the source file's name and size
  - the level and the storage format
  - the LTX version, or the H3 turbo choice (`merged`, with the LoRA's repo and revision, or `distilled`)
  - the H3 fold keys
  - LTX's element counts for the memory estimate
- **`config.yaml`** gets nothing.

### Part 1: MiniMax H3 on Windows and Linux (size M)

**What exists and can be reused:**

- `write_copy`, `load_copy` and the exact AdaLN fold
- `ensure_stored_copy`'s partial-folder rules
- `LoraMerger`, and the user LoRAs applied at run time
- the scaled-FP8 reader from the fix above, extended here to scaled INT8
- `convert_ltx`'s name and folder handling
- the conversion job and its slot
- the H3 memory check: shapes are held to stock's, so the stock estimate applies

**The change:**

1. **A header-only H3 reader** (`converters/h3_checkpoint.py`):
   - It classifies the layout:
     - diffusers names
     - native names, bare or with a `model.diffusion_model.` or `diffusion_model.` prefix, in one file or a shard set
     - a ComfyUI single file
   - It checks every name and shape against the stock transformer config.
   - It streams the tensors under diffusers names in bf16, using the upstream mapping above written as our own code, with scaled FP8 and scaled INT8 dequantized.
   - **Scaled INT8** ([Decision 3](#decisions-2026-10-23)): the shared helper from the FP8 fix (`dequantized_tensors` in `backends/cuda/quant.py`, with `dequantized` in `ltx_checkpoint.py`) is extended to INT8, so no second dequantize path exists.
     - An I8 weight is multiplied by its F32 `weight_scale`, per tensor or per output row, as FP8 is.
     - The `comfy_quant` formats it accepts are the INT8 names the spike records.
     - A format that needs more than a scale, such as a rotation, is refused and named in the message.
   - **Pruned files** are refused with a clear message: "This is a pruned MiniMax H3 checkpoint: its AdaLN projection is `[96768, 8]`, where stock has the full projection, so it can't be converted. Convert the full (unpruned) file instead."
   - It refuses other shapes or block counts, other quantized formats, GGUF, and LoRAs.
2. **The writer.** `write_copy` takes a tensor source instead of shard paths, and so does its time-embedder read for the tables. It also takes an optional turbo LoRA, which is left out with `--turbo`. The output is the `h3@q4` layout, written to `<models>/<name>/` with the manifest.
3. **Loading a converted folder.** `load_model`:
   - takes the components (VAEs, scheduler, tokenizer) and the text encoder from the preset's repo, as for stock, but without the stock `transformer/` shards (66 GB), which a converted folder doesn't need
   - takes the stored copy from the folder
   - applies no turbo LoRA, since it is merged or built in, and applies user LoRAs as today
   - never writes `h3@q4`
   - refuses at load time, with "convert it again", when the preset's fold keys differ from the manifest's (a schedule changed in `config.yaml`). Today that failure would come at the first forward. A converted folder serves only the schedules it was built with, because the copy keeps no bf16 AdaLN weights.
4. **CLI, form, inspection, detection, listing and delete** as described under Shared.

**Spike (before building).**

- **Where it runs.** The CPU steps use headers and ranged reads. The GPU steps run on the RTX 3080 10 GB / 30 GB RAM reference machine, each in its own systemd user unit and into a scratch data folder, so the user's stored copies are never rewritten.
- **Disk.** Each layout's source is about 66 GB, and they are converted one at a time.

Steps:

1. **Layouts.** Classify the layouts in the table above from their headers, and map each one onto the diffusers transformer built on the meta device.
2. **Sampled bit-exact check.** Take blocks 0, a middle block and the last, plus every tensor outside the blocks, from `FL2VA/transformer/` and from Comfy-Org's bf16 file. Map them and compare them with `transformer/` byte for byte. This also shows whether ComfyUI stores QKV in the native order.
3. **Full equivalence.** Convert stock `transformer/` without `--turbo`, then the native set and the Comfy-Org file. Compare the folders with each other and with `h3@q4`.
4. **A clip** from the converted stock folder, against `h3`: text to video, 864×480, 8 steps, fixed seed.
5. **`--turbo`** on lightx2v's full 4-step FP8 file, at 4 and 8 steps.
6. **One community full fine-tune** in a supported layout, chosen during the spike and recorded by repo and revision.
7. **Scaled INT8.** Record the `comfy_quant` format names and scale shapes of every scaled-INT8 H3 file found. Then dequantize the sampled Linears of one file (ranged reads) and compare them with stock bf16, next to that file's own unquantized bf16 layers. Convert it, and render one clip.

| Gate | Bar | Result |
|---|---|---|
| G1 Layouts | Every probed header is classified. The mapped names and shapes cover the meta model both ways | not run |
| G2 Sampled mapping | Every sampled tensor is bit-identical, for both the native and the ComfyUI layout | not run |
| G3 Stock equivalence | The converted stock folder's weights and tables are byte-identical to `h3@q4`'s, and so are the native and ComfyUI folders' | not run |
| G4 Clip | The clip from the converted stock folder is identical to `h3`'s | not run |
| G5 Cost | At most 20% more time and 1 GiB more system memory than the stock copy build as measured in Phase 2a (11.6 min, 8.3 GiB); swap at most 0.25 GiB; no oomd or kernel kill | not run |
| G6 Turbo flag | Review: the `--turbo` file renders sharp at its step count | not run |
| G7 Fine-tune | Review: sharp, shows the fine-tune's character, and image to video and Extend work as on stock | not run |
| G8 Scaled INT8 | Every recorded format is either read or refused by name. The sampled dequantized Linears are no further from stock than the file's own bf16 layers plus INT8 rounding (about 1%), and none is off by a scale factor. Review: the clip is sharp | not run |

Decision: build once G1–G5 pass, and add INT8 once G8 passes.

- If a layout fails G2 or G3, the user decides whether it is dropped or delayed.
- If G6 or G7 looks wrong at shift 6.0, the per-model schedule goes back to the user ([Decision 2](#decisions-2026-10-23)).

**Risks:**

- **Revisions.** The official `transformer/` and `FL2VA/transformer/` may not hold the same weights, for example if they come from different revisions. Then G3 compares against `FL2VA/transformer/` converted to diffusers with the upstream script at the same revision.
- **A wrong `--turbo` answer** renders broken output: distilled twice, or not at all. The fix is to convert again, and the inspection hint and the form's text say so.
- **Shift.** Fine-tunes trained for another shift (some state 12.0) still run at the app's 6.0 ([Decision 2](#decisions-2026-10-23)).
- **INT8 variants.** Community INT8 formats vary. One that needs more than a scale is refused until someone asks for it.
- **Disk.** Each converted H3 model is a full NF4 copy, as large as `h3@q4`.

### Part 2: LTX-2.5 on Windows and Linux (size M)

**What exists and can be reused:**

- the whole LTX-2.3 converter: the reader, the scaled-FP8 dequantization, the LoRA merge and the CUDA writer (`write_transformer`)
- text-encoder sharing, through the Gemma 4 path that already exists (`weights.py:71-78`, `write_text_encoder`)
- the memory estimates from the manifest
- the job and the form
- the `ltx25` version block in `config.yaml`

**The change:**

1. **`ltx-2.5` in `MODEL_TYPES`**, with the same flags as 2.3.
2. **A version-aware reader.**
   - 2.5 is recognised by its missing video feed-forward bias while the audio one is present, on top of 2.3's checks for gated attention, prompt AdaLN and 48 blocks.
   - It is not recognised by `audio_ff_bias`: the vendored docstring gets that one wrong.
   - `keyframes_abs_pos_embedding` is kept whether it is prefixed, bare or written as `unsloth_root::`.
   - torchao `Float8Tensor` files are refused as quantized repacks, not fine-tunes.
   - The diffusers layout stays refused.
3. **Per-version pins.** A 2.5 hashes file (beside `ltx23_hashes.json`) holds two sets, both computed from the verified transformers:
   - **Connector pins:** 258 tensors, with no `text_embedding_projection` group.
   - **Distillation pins:** the eight tensors per label (`dev`, `distilled`), the weight and the bias of the four modules, as `ltx23_hashes.json` does for 2.3. ("Four tensors" in the 2.3 design counted modules, not tensors.)

   `distillation` and `connectors_status` read the file of the checkpoint's version. A design detail for 2.5: they must not assume 2.3's counts.
4. **Version-aware config accessors.** `ltx_diffusers_repo`, `ltx_distilled_lora_source` and `ltx_mlx_base_repo` (`config.py:692-718`) take the version. The 2.5 distilled LoRA goes into `versions.ltx25.distilled_lora`: `lxxxy6/LTX-2.5` at `3506f52c`, with a link to the official file.
5. **Manifest kind `ltx-2.5`.** The video runner reads the version from a converted folder's manifest and passes it to `ltx_load_options`. A converted 2.5 model then runs with 2.5's ancestral sampler, CRF 18, components and upsampler. Without this it would silently run with 2.3's settings, because today the version comes only from the alias name.
6. **Mapping fixes** from the spike, if any, go into `transformer_name_map`, covered by `test_diffusers_ltx2_contract.py`.

#### Verified sources

This is how [Decision 1](#decisions-2026-10-23) is applied. It was checked on 2026-10-23 and is pinned in `config.yaml` and the 2.5 hashes file. Nothing is checked again at conversion time.

- **The method.**
  - The official `lfs.sha256` (oid) and size of each file come from the file listing of `Lightricks/LTX-2.5` at its main revision. That is metadata only: the Hub's `paths-info` or tree API, `HfApi.get_paths_info` or `list_repo_tree`.
  - The gated listing needs an account that has accepted the LTX-2.5 licence. That account belongs to the maintainer, once, for this check. Users never need one, because the app downloads only the ungated copies.
  - The ungated copy's listing is read anonymously, pinned at a revision taken from the API.
  - A copy counts as verified when its size and oid equal the official ones. The download step then hashes the file and checks it against the pinned oid before it is used (spike step 3).
  - No gated content is read.
- **The verified sources.** `lxxxy6/LTX-2.5` at `3506f52caf688795fc35edbf0d1b197bc30f3df1` holds all three:

  | Official file (`Lightricks/LTX-2.5`) | Ungated copy at `3506f52c` | Bytes | oid (SHA-256) |
  |---|---|---|---|
  | `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | 8,899,889,568 | `86370bbf79a9eb4edaa158907e2b48a5188fe4c5dc8ce30c7eb8f2f131a9bbf5` |
  | `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | `ltx-2.5-22b-dev-transformer-bf16.safetensors` | 42,018,190,584 | `792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584` |
  | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | `ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 42,018,190,584 | `31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4` |

- **Other copies with the same oids**, if this repo goes away:
  - **The LoRA:** comfyicu/LTX-2.5, yuvraj108c/LTX-2.5, tugrulv89/LTX_2.5-handler, pinecoresystems/LTX-2.5-serving, MrMofer/ltx-2.5-mlx-q8 and SceneWorks/ltx-2.5-mlx.
  - **The dev transformer:** comfyicu, yuvraj108c, tugrulv89 and unsloth/LTX-2.5-GGUF.
  - **The distilled transformer:** `MarceloFSO/ltx-2.5-distilled-nvidia` at `586b19e45ed03b554f2f61f2d349751b9b17babc` (`diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors`), and pinecoresystems.

  Each one would be re-pinned after the same check.
- **What uses them.** The LoRA is `--distill`'s download. The two transformers are the source of the pins, and the distilled one is the stock test input (Part 2 G4, Part 3 G1).
- **Not needed.** `Lightricks/LTX-2.5-Diffusers` is a separate gate, and its LoRA is a different file (9,699,368,752 bytes).
- **Fallback, only if a pinned source disappears with no verified replacement.** That item's pins are left out, and the inspection says "can't tell". The user then decides `--distill`. `--distill` is hidden if the LoRA is the item that went.

**Spike.**

1. **CPU:** map every 2.5 name, in the Lightricks layout and in ComfyUI's, through `transformer_name_map` onto the diffusers LTX2 transformer built on the meta device from the pinned unsloth config.
2. **CPU:** read the version signal from every probed 2.3 and 2.5 header.
3. **Pins, once, by a maintainer script.** The script is kept outside the package and ships nothing. It:
   - downloads the two verified transformers at their pinned revision
   - checks each file's SHA-256 against its pinned oid
   - hashes the eight distillation tensors of each file, and the 258 connector tensors, the way `distillation` and `connectors_status` read them (converted to bf16 first)
   - writes the 2.5 hashes file

   If the dev and distilled connectors differ, both sets are recorded as stock. It is about 84 GB of downloads, done once by a maintainer.
4. **GPU:** convert the verified distilled transformer at q4, and compare the result with `ltx25-4@q4`'s transformer and with a clip.
5. **GPU:** convert the ComfyUI-scaled FP8 community fine-tune at q4, then render a clip at `s` and one at 1280×704.

| Gate | Bar | Result |
|---|---|---|
| G1 Mapping | All names and shapes are covered both ways | not run |
| G2 Version | Every header reads as its own version | not run |
| G3 Pins | The script's local SHA-256 equals the pinned oid for both transformers. The hashes file has 8 tensors per distillation label and the 258 connector tensors. The stock dev and distilled files read as `dev` and `distilled`, and their connectors as `stock` | not run |
| G4 Stock equivalence | The NF4 transformer is byte-identical to `ltx25-4@q4`'s, and the clip is identical | not run |
| G5 Fine-tune | Conversion peak at most 6 GB. The clip's system memory drop is at most 1 GiB above `ltx25-4`'s (21.4 GiB at 1280×704). Swap at most 0.25 GiB, no kill. Review: sharp | not run |
| G6 Settings | The converted model's run uses the 2.5 sampler, CRF and components, as recorded in the run log | not run |

Decision: build once G1–G6 pass.

**Risks:**

- **A pinned source can disappear.** The other copies with the same oids are the replacement, after the same check. Only if none is left does the item use the fallback in [Verified sources](#verified-sources).
- **Thin real-world testing.** Full 2.5 fine-tunes are rare, so there is only one file to test with.
- **q8.** It doesn't fit with margin on 10 GB / 30 GB: `ltx25-8` was cut at 24.0–24.4 GiB at 768×448. It is offered as it is for 2.3. The form preselects the level that fits, and the memory badge warns.

### Part 3: LTX-2.5 on macOS (size S to M)

Part 3 started after the Mac test passed (2026-10-25).

**What exists and can be reused:**

- `backends/mac/ltx_converted.py`, as it is: the names (`mlx_name`, `_RENAMES`), the quantized set, the one-tensor writer and the one-to-one connector mapping
- Part 2's reader
- 2.5 detection: the vendored `is_ltx25_pack` reads `ff_bias: false` from `embedded_config.json`, so the backend's 2.5 path follows from the folder's files

**The change:**

1. **A base-file list per version.** For 2.5, it adds the Gemma 4 bundle (`text_encoder.safetensors`, `text_encoder_config.json`, the tokenizer, processor, chat template and generation config), the `vae_*_conv` and `vae_*_av` files, and `duration_head`.
2. **The 2.5 base** is the shipped `ltx25-4` repo (ungated and pinned), which anyone who uses `ltx25-4` has already downloaded. It only has q4, so for q8 the converter writes its own `quantize_config.json`.
3. **`embedded_config.json`** comes from the checkpoint's metadata, in stock's spelling. When the checkpoint has none, it comes from the 2.5 base, never from 2.3's.
4. **Connectors** are linked when the pins say they are stock. Otherwise they are written from the checkpoint, which costs about 4 GB more disk.
5. **The level.** The form preselects q4 until the Mac test has measured q8's headroom ([ltx-2.5.md](ltx-2.5.md), open question 4).

**Spike**, on the 32 GB Mac:

| Gate | Bar | Result |
|---|---|---|
| G0 Precondition | The Mac test of the shipped LTX-2.5 and H3 support has passed | Passed: the Mac test and its re-run on the fix branch (`ltx25-4` frames identical before and after the trimmed re-download) |
| G1 Bit identity | Stock 2.5 distilled, converted at q4, equals the `ltx25-4` repo's `transformer-distilled.safetensors` bit for bit. The input is the verified distilled transformer ([Verified sources](#verified-sources)) | Not run (user: minimal set; names, dtypes and shapes matched from Linux) |
| G2 Load | The converted folder is read as a 2.5 pack, and its clip is identical to `ltx25-4`'s | Not run (user: minimal set); G3's folder loads as a 2.5 pack and renders |
| G3 Fine-tune | It converts at q4 with a peak of at most 6 GB (2.3: 5.0 GB). Review | Pass: a community LTX-2.5 FP8 fine-tune converts in 25.7 s at a 2.8 GiB peak, no swap. The clip (768×448, 121 frames, 179 s, 17.9 GiB) is sharp and coherent. Disk grew 14.32 GiB: transformer 10.54 plus the connector 3.75, written from the checkpoint (see the observation below) |
| G4 q8 | It loads and renders on 32 GB, with no more swap than the 2.3 q8 run | Pass on a re-run: converts in 29.4 s at 2.9 GiB; renders at a 22.7 GiB peak (the 2.3 q8 run: 22.67 GiB, swap 0). The first render gained 0.60 GiB of swap for no identified reason; the same command again gained 0.05 GiB. Sharp and coherent, with a small green speck on the fox's shoulder in every frame |

Decision: build once G0–G3 pass. q8 is offered only if G4 passes.

**Risks:**

- **The mirror.** The base is a mirror of a gated repo. Its file hashes are recorded in [ltx-2.5.md](ltx-2.5.md).
- **Determinism.** `mx.quantize` must round the same way across MLX versions. It reproduced 2.3 bit for bit on Metal.

### Part 4: MiniMax H3 on macOS (size L)

Part 4 started after the Mac test passed (2026-10-25). It is the largest part, because no streaming INT8 writer exists upstream.

**What exists and can be reused:**

- Part 1's reader, which yields diffusers names with FP8 and INT8 dequantized, plus its inverse to native names. The dequantize helper moves out of `backends/cuda/` into a shared module so that the Mac can use it
- the vendored quantization recipe (`quantize.py:50-70`: `CORE_LINEARS`, `NEVER_QUANTIZE`, `QuantConfig`), for which tensors are quantized and how
- `ShardWriter`, for the shards and their index
- the fork's `load_dit`, as the strict load check
- the local-folder fix above, since the converter writes the INT8 layout that fix already loads
- the fork's `turbo_lora_path=None`

**The change:**

1. **A streaming RTN INT8 writer** (`backends/mac/h3_converted.py`). It holds one tensor in memory at a time, and for each one:
   - merges the turbo LoRA, unless `--turbo` was given
   - maps it to native names
   - quantizes it with `mx.quantize` to the shipped recipe: 8 bit, group 32, AdaLN at 8 bit, and the same layers kept in bf16

   It writes the shards, the index, `config.json`, and a `quant_config.json` without a calibration record.
2. **The converted folder** is a local H3 folder in the layout the fix loads: its own INT8 `transformer/`, the `FL2VA/` components linked from the stock download, and the manifest.
3. **Loading.** A converted folder gets no turbo LoRA. User LoRAs are combined without it (`combine_loras`). The text-encoder copy is shared with stock (`h3@mlx-te4`), never written again as `<name>@mlx-te4`.

**Spike**, on the 32 GB Mac:

| Gate | Bar | Result |
|---|---|---|
| G0 Precondition | The Mac test has passed | Passed: the Mac test and its re-run on the fix branch |
| G1 Four blocks | Four stream-quantized blocks strict-load with `load_dit` | Not run alone (user: minimal set); G6's full folder strict-loads and renders text to video |
| G2 Inverse mapping | The sampled tensors, mapped from diffusers to native names, are bit-identical to `FL2VA/transformer/` | Not run (user: minimal set; passed on Linux, same code) |
| G3 Cost | The full stock conversion peaks at 8 GB or less, with no swap. Time recorded | Bar missed: three H3 conversions peaked at 8.8, 9.1 and 9.4 GiB, with no swap; the 9.4 GiB one took 57.4 s (peak footprint 9.38 GiB, 10.07 GB, max RSS 11.35 GB). The bar was set too low |
| G4 Quality | A clip from RTN-converted stock against the shipped calibrated INT8 `h3`. Review: no visible loss. PSNR recorded | Not run (user: minimal set) |
| G5 Turbo | A clip with the LoRA merged before quantizing, against stock's run-time LoRA clip. Review: no visible loss | Not run as specified (user: minimal set). G6's result stands in: a fine-tune converted with the turbo merged keeps the start image (27.5 dB against stock's 27.7 dB). Stock with the turbo off gave 12.9 dB, so the turbo is needed, and the merge keeps it |
| G6 Fine-tune | A community fine-tune converts and renders. Review | Pass: a community fine-tune converted with the turbo merged renders sharp, and its image to video frame 0 is 27.5 dB / SSIM 0.857 against the start image, equal to stock `h3` (27.7 dB / 0.866). Two other community checkpoints ignored the start image (9.5 dB and 5.1 dB), one converted with `--turbo` and one with the merge, so that follows the checkpoint, not the converter. The 5.1 dB one (pruned rank 8, INT8, 20.98 GB, converted in 57.4 s) renders text to video sharp and coherent (`xs`, 73 frames, 257 s, 11.7 GiB) |

Decision: build once G0–G3 and G5 pass. If G4 fails (RTN visibly worse than calibrated), stop and ask the user. The calibrated path needs recorded activations and a CUDA device, so it would be a different proposal.

**Risks:**

- **Quality.** RTN INT8's quality on H3 is unknown.
- **Revisions.** The shipped INT8 was built from MiniMaxAI revision `939557dc`, while the Windows/Linux preset pins `42ed227e`. So G4 compares across revisions.
- **Disk.** About 38 GB per converted model.
- **Time.** Conversion time on a Mac is unknown.

### Not in scope

- **A per-model H3 schedule** (shift, step counts): only if Part 1's spike shows a need ([Decision 2](#decisions-2026-10-23)).
- **Other input formats:** GGUF, SVDQuant and torchao's `Float8Tensor`. (Pruned H3 files and the rotated and 4-bit ComfyUI formats were moved into scope, see [Implementation and results](#implementation-and-results-2026-10-25).)
- **A weight-space turbo detector:** decided against.
- **Changed LTX connectors on Windows/Linux:** refused, as for 2.3.
- **Converting H3 LoRAs** from ComfyUI's format.
- **Calibrated INT8** on the Mac.
- **H3 Ref2VA fine-tunes:** Ref2VA isn't supported ([ltx-2.5.md](ltx-2.5.md)).
- **The licence notice:** the download-licence work on another branch.

### Docs and tests

- **Docs.**
  - `docs/cli/models.md` gets LTX-2.5 and H3 sections beside LTX-2.3's: inputs, levels, `--distill` and `--turbo`, names, and what is linked. It drops the line saying LTX-2.5 checkpoints are refused.
  - `docs/concepts/models.md`'s name lookup rules mention converted folders.
  - Each part adds a `CHANGELOG.md` entry under `[Unreleased]`.
- **Tests**, on synthetic tensors and with no real weights:
  - **Mapping:** each layout's mapping; the diffusers-to-native-to-diffusers round trip as an identity; the fc1 swap and the QKV de-interleave against hand-built examples.
  - **Reading and refusals:** the 2.3 and 2.5 version signal and the cross-type refusals; `keyframes_abs_pos_embedding` kept in every naming; the refusals for pruned (with its message), torchao, LoRA and diffusers-layout LTX files.
  - **Dequantization:** scaled INT8 with a known per-tensor and per-row scale through the shared helper; an unknown `comfy_quant` format refused by name; the FP8 cases unchanged.
  - **Pins:** the 2.5 hashes file's contract (8 tensors per label, 258 connector names, no projection group); a checkpoint reading as `dev`, `distilled` or "can't tell" against synthetic pins; the LoRA source pinned in config with its link.
  - **Flags:** `--turbo` skips the merge and its absence merges; hints never change the flag; `--quantize` is refused for `h3` and `--turbo` for LTX.
  - **Loading:** a converted H3 folder loads without writing `h3@q4` and refuses changed fold keys; `ltx_load_options` receives a converted 2.5 folder's version.
  - **Detection, listing and delete:** delete leaves `h3@q4`, the cache and the shared text encoder.
  - **Payloads:** the type list in the models payload per platform, and the inspection payload per type.
  - **macOS:** the text-encoder copy's name.

## Alternatives considered

- **A weight-space turbo detector.** Rejected by the user. It needs the stock base downloaded and tensor reads on every conversion. A flag with hints is honest, and it mirrors `--distill`.
- **Diffusers-layout H3 only.** Rejected by the user: most fine-tunes are single files.
- **Reading the gated files for the LTX-2.5 pins.** Not needed: the gated listing gives the official oids, and the ungated copies with equal oids are the same bytes. The gated content is never read, and users never need an account.
- **No LTX-2.5 pins.** Rejected: verified sources exist for every item.
- **Applying the turbo LoRA at run time for converted H3 models.** Rejected: the 2b decision in [ltx-2.5.md](ltx-2.5.md) kept the turbo LoRA merged into the stored copy, and a converted model follows stock.
- **Vendoring upstream's `build_quant.py` for the Mac.** Rejected: it loads the whole bf16 transformer, which doesn't fit 32 GB, and it needs `save_sharded`, which isn't vendored.
- **Calibrated INT8 on the Mac.** Deferred: it needs activations recorded on the same source revision, plus a CUDA device.
- **Keeping the bf16 AdaLN projection in converted H3 folders**, so a schedule change could rebuild the tables without the source. Rejected for now: it would add the 13.0B AdaLN parameters (about 26 GB in bf16) to every converted model. Converting again is the remedy, as it is for LTX.
- **A folder as input** (`-i <folder>`). Rejected: the Web UI's path browser picks files, and picking any shard covers sharded sources.
- **Re-pointing the one shared `h3@q4` copy at the fine-tune.** That is today's behaviour, and it can't keep stock and a fine-tune side by side.
- **A text-encoder copy per model name on the Mac.** Rejected: it would duplicate the encoder for every fine-tune.

## Open questions

None at the moment. The three that were here are now [decisions](#decisions-2026-10-23). The spikes' decision rules say where a result goes back to the user.
