# Convert LTX-2.3 checkpoints

**Status:** Proposed (2026-10-19)

## Problem

`ziv model` converts single-file fine-tunes of Z-Image, FLUX.2 Klein and Krea 2 into a models-folder entry, but not LTX. Community LTX-2.3 fine-tunes (and merges with a LoRA baked in) ship as one ComfyUI-style `.safetensors` file, and there is no way to run one: `ltx-4` / `ltx-8` always load the stock distilled transformer.

## What exists today

- **Windows/Linux already read a Lightricks single-file checkpoint.** `backends/cuda/video/weights.py` maps the `model.diffusion_model.*` tensors to diffusers names (`transformer_name_map`: diffusers' `convert_ltx2_transformer_to_diffusers` plus the LTX-2.3 renames) and streams them into a prepared NF4/FP8 copy, `<models>/<alias>@q<bits>`, one tensor at a time. The only input it accepts is the one global `(repo, filename)` in `video_model_presets.ltx.diffusers.transformer`.
- **macOS loads an MLX folder** (`dgrauet/ltx-2.3-mlx-q4` / `-q8`) through the vendored `ltx_pipelines_mlx`, which reads `transformer-distilled*.safetensors`, `connector.safetensors`, `embedded_config.json`, the VAEs, vocoder and upsamplers from that folder. The vendored code has no converter (the repos are made with mlx-forge), and it can't be edited by hand.
- **The converter** (`converters/convert_checkpoint.py`) loads the whole state dict into memory, which is fine for 4–20B image transformers but not for LTX's 22B one (about 44 GB in bf16).

### Layouts, checked from the published file headers

The Lightricks checkpoint `ltx-2.3-22b-distilled-1.1.safetensors` holds 4444 `model.diffusion_model.*` tensors (the transformer plus `video_embeddings_connector` / `audio_embeddings_connector`), `text_embedding_projection.*`, and the VAE, audio VAE and vocoder; its `__metadata__` carries a `config` JSON with `transformer`, `vae`, `scheduler`, `audio_vae`, `vocoder`.

The MLX q4 `transformer-distilled-1.1.safetensors` uses the **same native names** with:

- `model.diffusion_model.` → `transformer.`
- `.to_out.0.` → `.to_out.`, `ff.net.0.proj` → `ff.proj_in`, `ff.net.2` → `ff.proj_out` (and the `audio_ff` equivalents), `.linear_1.` / `.linear_2.` → `.linear1.` / `.linear2.`. These are exactly the renames in the vendored `LTXV_LORA_COMFY_RENAMING_MAP`.
- Every Linear weight inside `transformer_blocks` (including `to_gate_logits`) quantized with MLX affine quantization at group size 64 (`weight` U32 plus `scales` / `biases`); biases and norms stay bf16, `*scale_shift_table*` stays f32, and everything outside the blocks stays bf16.
- The connectors live in a separate `connector.safetensors`.

So the Mac conversion is a rename plus `mx.quantize`, with no layout surgery.

## Proposed change

### 1. Input handling (shared)

- A new `--model-type ltx-2.3` in `MODEL_TYPES`, for the CLI and the server's validation. The Web UI convert form keeps its own list in `ModelsPage.svelte`, and needs more than a new entry (§5).
- Accept the Lightricks full checkpoint (`model.diffusion_model.*` plus bundled VAE/vocoder) and transformer-only exports (`diffusion_model.*` or unprefixed). Normalize the transformer names to `model.diffusion_model.*` first: `transformer_name_map` keeps only that prefix, so an unnormalized export maps to nothing and fails the coverage check. Only native transformer keys get the prefix; top-level `text_embedding_projection.*` stays as it is, as in the Lightricks layout. A single file in diffusers' layout (top-level `proj_in.` / `time_embed.` where native has `patchify_proj.` / `adaln_single.`) is refused with a clear message: prefixing it would pass conversion and fail at load. Supporting that layout is out of scope; the image converters' `DIFFUSERS_PREFIXES` branch is the pattern if it's ever needed. Bundled VAE, audio VAE and vocoder tensors are ignored; those components come from the base repo, as for the image families.
- Scaled FP8 exports: reuse `_dequantize_scaled_fp8`'s rule, but per tensor while streaming (it loads the whole dict today). A weight can come before its `weight_scale` in the file, so read all the scalar scales first (small reads located through the header), then stream the weights and multiply each by its scale. Never buffer FP8 weights while waiting for a scale. The spike's patched `_transformer_tensors` does exactly this. GGUF is out of scope. Community files mix precisions within one checkpoint: the test file had the video attention and feed-forward Linears of blocks 2–45 in `float8_e4m3fn` with a scalar f32 `weight_scale` and a `comfy_quant` tag (no `input_scale`), and everything else in bf16. Its metadata described it as bf16 anyway, so detect FP8 from the tensors, never from the description.
- Tensors stock stores in f32 (the `*scale_shift_table*`s) may arrive as bf16; cast them back to f32 for the MLX layout.
- **Dev or distilled: detect, recommend, let the user confirm.** The app runs LTX at 8 steps without guidance, so the checkpoint must be step-distilled. Many community merges are dev plus style LoRAs, and their pages say "8 steps" because they assume Lightricks' distilled LoRA is loaded beside them. The test file was one (results below). The check uses four small tensors that Lightricks' distilled LoRA changes but style LoRAs typically leave alone: `patchify_proj`, `audio_patchify_proj`, `proj_out` and `audio_proj_out` (about 1 MB each in bf16). Their SHA-256 hashes for stock dev, distilled 1.0 and distilled 1.1 are pinned in a small data file in the package, generated once from the Lightricks checkpoints. The converter hashes the same four tensors in the user's file, converted to bf16 first (stock stores them as bf16, so a file that stores them in f32 still matches; one that stores them in scaled FP8 can't, since FP8 is lossy, and comes out as "can't tell"), reading a few MB from it and nothing from the network:
  - all four match dev: **plain dev**. It says so and recommends `--distill`; the Web UI preselects a Distill checkbox. (In the test file, all four tensors' weights and biases are bit-identical to dev's and differ from distilled 1.1's. Both Lightricks distilled LoRAs, 1.0 and 1.1, target all four, so a merge that bakes one in changes them and can't be read as plain dev.)
  - all four match one distilled version: **already distilled**, convert as is.
  - anything else: **can't tell**. It says so, points to the model page, and recommends nothing.

  `--distill` merges Lightricks' distilled LoRA (`ltx-2.3-22b-distilled-lora-384-1.1`, pinned in config) into the bf16 weights before quantizing. The LoRA (7.6 GB) is downloaded once into the Hugging Face cache; offline without it, `--distill` fails with a clear message before converting anything. While merging, each target's `lora_A` / `lora_B` pair is read from the LoRA file through its header as that tensor streams by, never the whole file (as the vendored `_fuse_distilled_lora` does with `mx.load`). On a 32 GB machine, loading it whole would add about 7.6 GB to a conversion peak of about 21 GB. Merging at conversion is more accurate than a run-time LoRA on q4/q8 weights, keeps LoRA slots free, and avoids a run-time LoRA's cost (about 50% slower per clip in the spike). The merge is never applied silently: a file distilled some other way would be distilled twice, so the user confirms it.
- Refuse anything that isn't LTX-2.3, checked by the converter itself on both platforms, from the header: LTX-2.0 has no `to_gate_logits` or `prompt_adaln_single`, LTX-2.5 has no feed-forward input bias (`ff.net.0.proj.bias` in native and ComfyUI headers; `ff.proj_in.bias` exists only after the MLX rename), and the block count and hidden sizes must match stock 2.3. Don't rely on the vendored `validate_config_matches_weights`: it fires only when the config says `ff_bias=True`, and the converter writes the checkpoint's own config, so a 2.5 file would pass it; CUDA has no such check at all.
- Read the checkpoint one tensor at a time with `utils/safetensors_stream` (`read_header`, `iter_tensors`), and write one tensor at a time, so a 32 GB Mac or PC can convert it. CUDA writes through the existing `ShardWriter` (numbered shards plus an index: the prepared-copy layout). macOS can't: `ShardWriter` writes shards, while the vendored MLX loader reads exactly one `transformer-distilled.safetensors`, and saving the whole dict in one go would hold the 12–22 GB output in memory. So this needs a **new single-file streaming writer** in `utils/safetensors_stream`. Every output name, dtype and shape follows from the input header and the quantization rules (packed `weight` plus `scales` / `biases` per group of 64), so it can write the complete header first, then stream the data in header order.
- For `ltx-2.3` only, refuse a `--name` that already exists, and leave the existing folder alone; `--force` replaces it. Today `_cmd_model` writes into an existing folder (`mkdir(exist_ok=True)`), which for LTX would leave a mix of old and new files. Image conversion keeps its current behaviour: re-converting a Z-Image, Klein or Krea 2 model under the same name still works, so no existing workflow changes. Also for `ltx-2.3` only, refuse configured alias names (`ltx-4`, `ltx-8` and the rest of `model_aliases`). `resolve_model_path` picks a models-folder entry before an alias, so `--name ltx-4` would hide stock `ltx-4`, clash with the alias table's quantize level, and make `stored_quant_dirs_for` count the alias's `ltx-4@q4` as the fine-tune's, so deleting one deletes the other.
- **Never leave a half-written model.** Write the whole folder into a hidden partial folder (`stored_quant`'s `partial_dir`) and move it into place with `promote_partial` once complete. The manifest is written last. On any exception, Ctrl-C included, the partial folder is discarded (`discard_partial`), as `ensure_prepared` already does. Only a hard kill (an out-of-memory kill) can leave one behind, under a hidden name. It's swept by the existing rule (`sweep_stale_partials`: same name, nothing written for an hour) the next time that name is converted. Either way, it's never a model that is listed and fails at load. This matters on macOS especially, where the single-file writer puts a valid header in front of data that may never arrive; and on CUDA, `has_complete_weights` skips the transformer for `LTX2Pipeline` folders. With `--force`, the old folder is replaced only after the new one is complete.
- `--quantize 4|8` is **required** for `ltx-2.3`, and that level is the only one stored. This is a different meaning from the image types, where `--quantize` saves an extra `<name>@q<bits>` copy through `_store_quantized_copy`. The LTX path must skip that call (on a video folder it would crash or leave a stray copy), and the flag's help text and the Web UI's success message ("saved a q8 copy") must say which meaning applies. Both backends run LTX quantized only, so the level picked at conversion is the model. No unquantized copy is kept. The level is recorded in the folder (in the manifest), and the Web UI preselects the level that fits this machine's memory: q8 needs roughly 10 GB more than q4 for the transformer alone.
- One level per name, as with `ltx-4` / `ltx-8`. To get both levels of a fine-tune, convert twice under two names (e.g. `myft-4`, `myft-8`). There is no `<name>` / `<name>@q<bits>` pair, because a base folder would hold no weights of its own. Changing level later means converting again from the original file.

### 2. macOS output: an MLX folder

`<models>/<name>/`:

- `transformer-distilled.safetensors`: the renamed transformer, block Linears quantized at the `--quantize` level, the rest cast as listed above. It must be the folder's only `transformer*` file. The vendored loader (and `ltx_mlx_transformer_file`, which mirrors it) prefers `transformer.safetensors`, then the newest versioned `transformer-distilled-*.safetensors`. A linked stock `transformer-distilled-1.1.safetensors` would win and silently replace the fine-tune.
- `connector.safetensors`: linked from the base when the checkpoint's `video_embeddings_connector`, `audio_embeddings_connector` and `text_embedding_projection` equal stock (they did in the test file: all 262 tensors identical), or when the checkpoint has none of them (transformer-only exports). Otherwise, written from the checkpoint in the MLX connector layout, if the spike shows the names map one to one; if not, refused with a clear message until that mapping exists. "Equal to stock" is checked locally against pinned SHA-256 hashes of stock's 262 tensors, kept in a small data file in the package and generated once from the stock checkpoint. Reading stock's tensors instead would download up to 1 GB on every conversion and fail offline. A connector tensor in another dtype than stock's counts as different.
- `embedded_config.json` from the checkpoint's `config` metadata (to be confirmed against dgrauet's file in the spike), so `LTXModelConfig.from_checkpoint_dir` never falls back to defaults. Transformer-only exports usually carry no such metadata; for those, use stock's `embedded_config.json` from the dgrauet repo, after checking that the checkpoint's shapes (hidden sizes, block count, the `ff.net.0.proj.bias` tensors) match stock's. Refuse when they don't. Normalize enum-like values to stock's spelling: the test file wrote `text_encoder_norm_type` as `PER_TOKEN_RMS` where stock has `per_token_rms`. The vendored MLX code doesn't read that key, but a later version might.
- Everything else (VAEs, vocoder, upsamplers, `config.json`, `quantize_config.json`, `split_model.json`) linked from `dgrauet/ltx-2.3-mlx-q<bits>`, which is the same `download_base_components` idea with a different file list. That is an explicit allow-list, never "everything we didn't write": no `transformer*` file from the base is linked.

### 3. Windows/Linux output: one quantized folder

CUDA only runs LTX at q4 or q8, so a bf16 diffusers copy (44 GB) would never be loaded. Everything goes into the one folder `<models>/<name>/`:

- `transformer/`: the NF4 (q4) or FP8 (q8) transformer, written by the existing `write_transformer` with the user's file as the source. That is the code path `ltx-4` / `ltx-8` already use, but its tensor source needs new code first: prefix normalization (§1), per-tensor scaled-FP8 dequantization (without it, FP8 weights are stored unscaled and silently wrong; see the CUDA run below), and the `--distill` merge. The spike's patched `_transformer_tensors` is the prototype for the first two. NF4 packing needs the GPU, as today.
- `text_encoder/`: the shared FP8 text encoder, hard-linked from another prepared copy when one exists, plus a link to the diffusers repo's `text_encoder/config.json`. The prepared copy holds only shards, an index and its manifest, but `_empty_text_encoder` builds the model from `<folder>/text_encoder` with `AutoConfig.from_pretrained`, so without the config every generation fails. Build it with `ensure_prepared` exactly as today, so `text_encoder/` keeps its own prepared-copy manifest (`ziv-quant.json`, tied to the diffusers repo's text encoder, which stays in the Hugging Face cache anyway). Reuse checks that per-part manifest (`manifest_matches` on the candidate folder), so it must stay; the folder-level manifest below covers only the transformer. Then widen `_prepared_weights`' search from `*@q*/text_encoder` to converted LTX folders too, and reuse works both ways instead of writing another 12 GB.
- `model_index.json`, the transformer `config.json`, and links to the connectors, VAEs, vocoder, tokenizer and scheduler of `diffusers/LTX-2.3-Distilled-Diffusers`.
- The folder's manifest, for the transformer: the level, the layout version and the storage format. Unlike today's prepared copies, it does not tie the folder to the source checkpoint (see §4). So the q4 check must not be `_storage_format`'s `nf4-bitsandbytes-<major.minor>`: that would make every converted q4 model refuse to load after a routine bitsandbytes minor update, with no source to rebuild from. Record the serialized NF4 layout instead (the part names in `_NF4_PARTS`, block size 64, `nf4`, `uint8` storage, double quantization), plus a dequantized sample: 64 values of a fixed row of one block weight, dequantized at conversion and stored as numbers (not a hash, which can't be compared with a tolerance). Loading one weight through `Params4bit.from_prequantized` wouldn't be enough, since `_place_nf4` already raises on missing parts and a changed interpretation would load without error. So on the first load with a different bitsandbytes version, dequantize the same row and compare it with the stored values within a small tolerance, so a harmless rounding change passes. Only on a mismatch does the folder refuse to load, saying to reconvert from the original file, and it is marked that way on the Models page. The bitsandbytes version is recorded so the check runs once per version.

As on macOS, the connectors are linked when they equal stock (by the pinned hashes) or are absent. Otherwise the conversion is refused on CUDA: `convert_ltx2_transformer_to_diffusers` drops the connectors, and nothing maps the checkpoint's connector and `text_embedding_projection` names to `LTX2TextConnectors`. Writing a fine-tune's own `connectors/` is follow-up work, with its own spike.

### 4. Making the backends load a converted folder

This is the part the image families didn't need:

- **Detection.** `detect_video_model` and `list_video_models` recognise a local folder only when its path contains "ltx", a rule `docs/guides/video.md` and `ziv video`'s error text document, and that hand-placed folders (the CLI help's `models/ltx-mlx`) rely on. Keep that rule, and add detection by contents beside it, with the helpers `utils/model_files.py` already has: `is_ltx_mlx_layout` (marker `connector.safetensors`) plus `ltx_mlx_transformer_file` on macOS, and a `model_index.json` whose `_class_name` is in `_SEPARATE_TRANSFORMER_PIPELINES` (`LTX2Pipeline`, `LTX2ImageToVideoPipeline`) on CUDA. Put that in one helper used by detection, listing and `has_complete_weights`, so they can't disagree. Then a fine-tune can be named anything, and it shows up as a video model on the Models page and in `ziv model list`. Detection doesn't require the manifest: hand-placed folders have none, and writing through a partial folder already means a half-written conversion never sits under its final name.
- **Transformer source.** `ltx_load_options` always passes the global `transformer_source`, which would replace the converted transformer with the stock one. It only receives `(config, model_name)`, with no resolved path, so it can't tell a converted folder from an alias, and it shouldn't read files: it's a pure config helper. The CUDA backend's `load_model` decides instead, from the model path it already receives. For a converted folder (a local folder with the converter's manifest), it ignores `transformer_source`, loads the stored `transformer/` and `text_encoder/` from that folder, and never calls `_prepared_weights`. Otherwise `_prepared_weights` would name a copy `<name>@q<bits>` and build it from the global stock transformer, so the user would silently run stock.
- **Load what's stored (CUDA).** `ensure_prepared` checks a prepared copy against its source files by resolved name and size, so it needs the original checkpoint on every load; deleting or moving the file after converting would break the model. A converted folder loads its stored `transformer/` and `text_encoder/` directly and never looks at the source.
- **The level comes from the folder.** `ltx_diffusers_quantize` knows only the `ltx-4` / `ltx-8` names, and `load_model` refuses anything else. For a converted folder, `load_model` reads the level from the folder's manifest, ignoring the `quantize` that `ltx_load_options` passes (that one comes from the alias table and is `None` for any other name). Video has no quantize choice to reconcile: `ziv video` has no `--quantize` flag, and the Web UI's video settings report `supports_quantize: False`. So the level is simply what was converted, shown on the Models page. On macOS the level is already implied by the quantized weights.
- **Memory estimates.** On CUDA, `memory_estimate.py` and `ltx_cuda_system_memory` key off the alias and the global transformer source. For a converted folder, feed the existing `estimate_ltx_cuda_system_memory` the same inputs it gets for stock: record `block_matrices` and `transformer_rest` in the manifest at conversion, counted from the source tensors' real shapes. The connector and text-encoder inputs come from the linked components as they do for stock. A converted folder is then estimated exactly like `ltx-4` / `ltx-8`, and a stock-converted folder gets the same fit badge as the alias. Don't count elements from the stored transformer's headers: `transformer_elements` would read NF4's packed `uint8` shapes and come out at about half at q4. Don't size it from bytes on disk either: that's a second model that disagrees with the stock one (connectors are bf16 on disk but FP8 in memory, and `directory_bytes` skips the linked files). The fit badge and `_check_system_memory` use the same estimate. `_check_system_memory`'s refusal and warning hard-code their advice (`smaller = "ltx-4" if quantize == 8`), which for a converted q8 folder would point the user to the stock model. The spike hit this warning at q8 on a 30 GB machine. For a converted folder, the advice is instead to convert the fine-tune again at `--quantize 4` under a new name. On macOS, keep the existing estimator: it already works from the folder's contents (`_files_bfloat16_bytes` on `ltx_mlx_transformer_file`, the connector, decoders and Gemma); it only has to be given the converted folder.
- LoRAs, i2v, extend and the two-stage upscale need no changes: they run on whatever transformer was loaded.

### 5. Web UI, docs, tests

- **Web UI convert.** An "LTX-2.3 (video)" option in the hardcoded type list in `ModelsPage.svelte`, with a required quantize choice and a Distill checkbox. Two server changes come with it:
  - **An inspect endpoint**, called when a file and the LTX type are picked and before submit. It reads the header and hashes the four distillation tensors (offline, a few MB), and returns the layout (full or transformer-only, FP8 mix), whether Lightricks' distillation was found, and the level that fits this machine. The form preselects the quantize level and, for a plain-dev file, the Distill checkbox.
  - **Conversion as a background job with progress.** `_convert_model_from_form` runs the converter as one blocking `subprocess.run(capture_output=True)`. That's fine for image models, but an LTX conversion takes minutes. Report the stage and tensors done, the way other long Web UI jobs report progress. The job holds the exclusive slot (`_claim_quantize_slot`, through `web_runner.admit_exclusive`) for its whole run, as the blocking call does today. Otherwise the handler would release it as soon as the job started, and a generation could be admitted while the conversion holds about 21 GB and the GPU.

  Then `make frontend-build`.
- `docs/` model-conversion page and a `CHANGELOG.md` entry. Update `docs/guides/video.md`'s model-selection rules and `ziv video`'s unknown-model message (`video_cli.py`) to say that converted LTX folders are recognised by their contents, besides local paths containing `ltx`.
- Tests: key mapping on a few synthetic tensors per family of names; refusal of 2.0/2.5-shaped headers; `--quantize` required for `ltx-2.3`; content-based detection; `load_model` given a converted folder loading its stored transformer and level from the folder, never the global `transformer_source` or the alias table, and never writing a `<name>@q<bits>` copy; the level read from the manifest; an NF4 layout the installed bitsandbytes can't read refusing to load with the reconvert message, and a minor-version change alone not refusing; an interrupted conversion leaving only a partial folder, which isn't listed and is swept; `--force` replacing the old folder only after the new one is complete; configured alias names refused as `--name`; 2.0- and 2.5-shaped headers refused on both platforms even when their config says otherwise; a diffusers-layout file refused; `text_embedding_projection.*` left unprefixed; connectors linked when hashes match or when absent, written (macOS) or refused (CUDA) otherwise; `--quantize` for `ltx-2.3` never calling `_store_quantized_copy`; refusing an existing `--name` without `--force`; prefix normalization of transformer-only exports, and their config fallback and shape check; exactly one `transformer*` file in a macOS folder, and `ltx_mlx_transformer_file` returning it; the single-file writer's header matching the data it then streams; `text_encoder/config.json` present in a CUDA folder; text-encoder reuse in both directions; detection through the shared helper, including an `LTX2ImageToVideoPipeline` index; estimates from the manifest's element counts matching the stock alias's for a stock-converted folder; the memory refusal and warning for a converted q8 folder advising a q4 reconversion, not `ltx-4`; the inspect endpoint's response; the distillation check on synthetic tensors against pinned hashes (all match dev, all match a distilled version, mixed or unknown), and its recommendation or lack of one; `--distill` failing cleanly before converting when the LoRA can't be downloaded; the NF4 sample (unchanged and slightly rounded pass, a changed dequantization refuses); the conversion job holding the exclusive slot until it ends; `--distill` merging a synthetic LoRA delta before quantization, and no merge without it; scaled-FP8 tensors dequantized per tensor in a mixed file; a converted folder loading with its source file gone. No real weights.

## Spike (before building)

The spike answers three things the plan assumes: (1) is `embedded_config.json` the checkpoint's own `config` metadata, (2) do the checkpoint's connector tensors map one-to-one onto `connector.safetensors`, and do fine-tunes change them, and (3) does `mx.quantize` reproduce dgrauet's quantized weights exactly. Parts 1 and 2 read file headers and small tensors. Part 3 needs MLX and runs **on a Mac only**; it is not emulated on a CPU elsewhere.

### On Windows/Linux

With a fine-tune test file: compare its header with the stock `ltx-2.3-22b-distilled-1.1.safetensors` (prefix style, key set, dtypes, metadata, and any dev/distilled hint), and compare the connector and `text_embedding_projection` tensors with the stock ones. That answers question 2's second half and open question 1. Then convert it with the existing `write_transformer` on the GPU to check the CUDA path end to end.

**Results (2026-10-19), from a 34 GB community LTX-2.3 merge.** This was read on the CPU from headers and sample tensors; the GPU runs follow below.

- **Layout.** Lightricks full-checkpoint layout: the same 5947 keys and shapes as stock, plus 440 `weight_scale` / `comfy_quant` pairs. The VAE, audio VAE and vocoder (1499 tensors) are byte-identical to stock.
- **Precision.** Scaled FP8 as described in §1, mixed with bf16. All 290 tensors stock keeps in f32 were bf16.
- **Connectors.** The connectors and `text_embedding_projection` (262 tensors) are byte-identical to stock.
- **Config.** The `config` metadata equals stock except for the case of `text_encoder_norm_type`.
- **Base model.** Plain dev, with large changes concentrated in attention and no step distillation. Its model page says 8 steps, which (as the renders below show) assumes Lightricks' distilled LoRA is loaded beside it. Relative distance of sample tensors:

  | Tensor | to dev | to distilled 1.1 |
  |---|---|---|
  | `patchify_proj` | 0 (bit-identical) | 0.0051 |
  | `scale_shift_table` | 0.0017 | 0.0075 |

  Inside the blocks, `test − dev` is 9–31% of the attention weights and under 1% of the feed-forward ones. It doesn't align with any Lightricks distillation (cosine within ±0.006 against the distilled 1.0 and 1.1 checkpoints' deltas and both distilled LoRAs). The method itself works: stock `distilled − dev` aligns with the LoRA (cosine 0.76–0.98). The renders below confirm the reading: the file is plain dev and needs the distilled LoRA.

**CUDA run (2026-10-19), same file, q4 on a 10 GB NVIDIA GPU with 32 GB of RAM.** The existing `write_transformer` was patched for the run only, to dequantize scaled FP8 per tensor as it streams.

- **Silent bug in today's code.** `write_transformer` casts each tensor with `tensor.to(storage dtype)` and skips names it doesn't know, so a scaled-FP8 weight would be stored without its `weight_scale`. That doesn't raise an error. Measured on the stored NF4 weights, the FP8 layers are 99.4–99.8% away from the unscaled weights, and 9.4–9.6% from the dequantized ones, which is ordinary NF4 rounding (bf16 layers: 9.7–11%). The per-tensor dequantize in §1 is required, not optional.
- **Cost.** Streaming the 34 GB file into the q4 transformer took 103 s; the text encoder was hard-linked from `ltx-4@q4`. Convert plus a 49-frame `s` clip took 4:19 with a 21.3 GB peak RSS, against 2:08 and 21.7 GB for the same clip on stock `ltx-4`. So conversion adds about 2 minutes and no memory. The output is 9.8 GB, the same as stock q4.
- **Output.** Five prompts (beach with a kite, portrait, dog at a lake, rainy city at night, barista), all at 8 steps, seed 7 and size `s`, against stock `ltx-4`:
  - **As converted, q4 and q8:** broken on every prompt. The scene layout is right, but the frames are ghosted, smeared and noisy, and details never resolve. q8 is only marginally sharper than q4, so quantization isn't the cause.
  - **q4 with Lightricks' distilled LoRA 1.1 at 1.0 (run-time `--lora`):** sharp and well formed on every prompt, at stock quality or better, with the merge's own style. That style is strong: it changed the subjects in ways the prompts didn't ask for, e.g. the dog came out as an upright, human-bodied figure. That's the merge's training showing through, not the conversion.
  - **Timing and memory per pass of five clips** (wall time; peak RSS): stock q4 9:51, 22.1 GB; converted q4 8:30, 22.1 GB; converted q8 16:04 including writing the copy (156 s), 27.1 GB, as the app's 27 GB estimate warned; q4 with the run-time LoRA 13:22, 25.2 GB.
  - **q8 tensors:** FP8 layers 2.4–3.1% from the dequantized source (bf16 layers 2.8–3.8%), 99.4–99.8% from the unscaled one.

### On the Mac

Needs: any Apple Silicon Mac with 16 GB or more (the script holds one sample at a time, at most about 0.4 GB with its f32 copy), the repo on this branch, `make install`, and network access to Hugging Face. Nothing is fully downloaded: the script reads the headers and the 18 sampled Linears with ranged requests, about 1.5 GB in all (the Lightricks samples once, plus the quantized ones from both MLX repos). Expect a few minutes on a fast connection, longer on a slow one.

1. Save the script below outside the repo, e.g. `~/ltx-spike/ltx_spike.py`.
2. Run it from that folder, with the repo's environment:

   ```bash
   cd ~/ltx-spike
   uv run --project <path to the repo> python -I ltx_spike.py | tee ltx-spike-report.txt
   ```

   If Hugging Face refuses the Lightricks file, log in first with `uv run --project <path to the repo> hf auth login`.
3. Paste `ltx-spike-report.txt` back into the conversation. It holds only model file names, key names, shapes and numbers.

How to read the report:

- **Section 1:** `True` on the first line means the converter copies the checkpoint's `config` metadata as `embedded_config.json`. Otherwise, the key lists show what to build or link instead.
- **Section 2:** "matched" counts how many checkpoint connector tensors map onto `connector.safetensors` by prefix with the same shape. All matched means the converter writes the connectors from the checkpoint; anything unmatched is listed, to work out the renames.
- **Section 3:** `exact` on every line (one of the two input dtypes) means `mx.quantize(group_size=64)` is what mlx-forge did, so the full conversion can be checked tensor by tensor. Otherwise, the error numbers show whether ours is as close to the bf16 weights as dgrauet's. `quantize_config.json` at the top shows dgrauet's own settings. Unquantized tensors should all be `equal`.

```python
"""Compare the Lightricks LTX-2.3 checkpoint with dgrauet's MLX folders for the checkpoint conversion spike (macOS)."""

from __future__ import annotations

import json
import re
import struct

import mlx.core as mx
import numpy as np
from huggingface_hub import HfFileSystem, hf_hub_download, list_repo_files

LIGHTRICKS_REPO = "Lightricks/LTX-2.3"
LIGHTRICKS_FILE = "ltx-2.3-22b-distilled-1.1.safetensors"
MLX_REPOS = {4: "dgrauet/ltx-2.3-mlx-q4", 8: "dgrauet/ltx-2.3-mlx-q8"}
MLX_TRANSFORMER = "transformer-distilled-1.1.safetensors"
PREFIX = "model.diffusion_model."
CONNECTOR_PREFIXES = (f"{PREFIX}video_embeddings_connector.", f"{PREFIX}audio_embeddings_connector.", "text_embedding_projection.")
# Native → MLX renames (the vendored LTXV_LORA_COMFY_RENAMING_MAP).
RENAMES = (
    ("audio_ff.net.0.proj", "audio_ff.proj_in"),
    ("audio_ff.net.2", "audio_ff.proj_out"),
    (".ff.net.0.proj", ".ff.proj_in"),
    (".ff.net.2", ".ff.proj_out"),
    (".to_out.0", ".to_out"),
    (".linear_1.", ".linear1."),
    (".linear_2.", ".linear2."),
)
# One Linear of each kind inside a block, checked in the first and the last block.
BLOCK_LINEARS = ("attn1.to_q", "attn1.to_gate_logits", "attn2.to_k", "ff.net.0.proj", "ff.net.2", "audio_attn1.to_v", "audio_ff.net.2", "audio_to_video_attn.to_out.0", "video_to_audio_attn.to_q")
UNQUANTIZED = (
    "transformer_blocks.0.scale_shift_table",
    "transformer_blocks.0.attn1.to_q.bias",
    "transformer_blocks.0.attn1.q_norm.weight",
    "patchify_proj.weight",
    "adaln_single.emb.timestep_embedder.linear_1.weight",
    "scale_shift_table",
)
# safetensors dtype → (numpy dtype the bytes are read as, MLX dtype to view them as)
DTYPES = {
    "BF16": (np.uint16, mx.bfloat16),
    "F16": (np.float16, None),
    "F32": (np.float32, None),
    "F64": (np.float64, None),
    "U32": (np.uint32, None),
    "U8": (np.uint8, None),
    "I8": (np.int8, None),
    "I32": (np.int32, None),
}

fs = HfFileSystem()


def header(path: str) -> tuple[dict, int]:
    """Return a remote safetensors file's header and the byte offset its data starts at."""
    with fs.open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size)), 8 + size


def tensor(path: str, head: tuple[dict, int], name: str) -> mx.array:
    """Read one tensor of a remote safetensors file with a ranged request."""
    info = head[0][name]
    start, end = info["data_offsets"]
    with fs.open(path, "rb") as f:
        f.seek(head[1] + start)
        raw = f.read(end - start)
    if info["dtype"] not in DTYPES:
        # Not a KeyError: callers report KeyError as a missing tensor.
        raise ValueError(f"{name} has dtype {info['dtype']}, which this script can't read")
    np_dtype, view = DTYPES[info["dtype"]]
    array = mx.array(np.frombuffer(raw, dtype=np_dtype).reshape(info["shape"]))
    return array.view(view) if view is not None else array


def mlx_name(native: str) -> str:
    """Return the MLX transformer name of a native name (without ``model.diffusion_model.``)."""
    for old, new in RENAMES:
        native = native.replace(old, new)
    return f"transformer.{native}"


def check_config(lightricks: tuple[dict, int]) -> None:
    """Section 1: compare embedded_config.json with the checkpoint's config metadata."""
    stored = json.loads(lightricks[0]["__metadata__"]["config"])
    print("== 1. embedded_config.json")
    for bits, repo in MLX_REPOS.items():
        with open(hf_hub_download(repo, "embedded_config.json"), encoding="utf-8") as f:
            embedded = json.load(f)
        print(f"q{bits}: equal to checkpoint config: {embedded == stored}; equal to config['transformer']: {embedded == stored.get('transformer')}")
        if embedded not in (stored, stored.get("transformer")):
            print(f"  embedded keys:   {sorted(embedded)}")
            print(f"  checkpoint keys: {sorted(stored)}")
            transformer = stored.get("transformer", {})
            print(f"  differing transformer keys: {sorted(k for k in set(embedded) | set(transformer) if embedded.get(k) != transformer.get(k))}")


def check_connectors(lightricks: tuple[dict, int]) -> None:
    """Section 2: map the checkpoint's connector tensors onto connector.safetensors by name and shape."""
    print("\n== 2. connectors")
    ours = {k: v["shape"] for k, v in lightricks[0].items() if k.startswith(CONNECTOR_PREFIXES)}
    theirs_head = header(f"{MLX_REPOS[4]}/connector.safetensors")[0]
    theirs = {k: v["shape"] for k, v in theirs_head.items() if k != "__metadata__"}
    print(f"checkpoint connector tensors: {len(ours)}; connector.safetensors tensors: {len(theirs)}")
    print(f"connector.safetensors dtypes: {sorted({v['dtype'] for k, v in theirs_head.items() if k != '__metadata__'})}")
    guessed = {k: "connector." + k.removeprefix(PREFIX) for k in ours}
    matched = [k for k, name in guessed.items() if theirs.get(name) == ours[k]]
    print(f"matched by 'connector.' + name: {len(matched)}")
    used = {guessed[k] for k in matched}
    for key in sorted(set(ours) - set(matched))[:40]:
        print(f"  unmatched checkpoint: {key} {ours[key]}")
    for key in sorted(set(theirs) - used)[:40]:
        print(f"  unmatched connector.safetensors: {key} {theirs[key]}")


def check_quantize(lightricks: tuple[dict, int]) -> None:
    """Section 3: quantize sample Linears with mx.quantize and compare with dgrauet's weights."""
    print("\n== 3. quantization")
    blocks = sorted({int(m.group(1)) for k in lightricks[0] if (m := re.match(rf"{re.escape(PREFIX)}transformer_blocks\.(\d+)\.", k))})
    path_lightricks = f"{LIGHTRICKS_REPO}/{LIGHTRICKS_FILE}"
    references = {}
    for bits, repo in MLX_REPOS.items():
        if "quantize_config.json" in list_repo_files(repo):
            with open(hf_hub_download(repo, "quantize_config.json"), encoding="utf-8") as f:
                print(f"q{bits} quantize_config.json: {f.read().strip()}")
        path_mlx = f"{repo}/{MLX_TRANSFORMER}"
        references[bits] = (path_mlx, header(path_mlx))
    # One Lightricks sample at a time: read once, compare against both repos, then drop it.
    for native in [f"transformer_blocks.{block}.{linear}" for block in (blocks[0], blocks[-1]) for linear in BLOCK_LINEARS]:
        try:
            weight = tensor(path_lightricks, lightricks, f"{PREFIX}{native}.weight")
        except KeyError as exc:
            print(f"{native}: missing {exc}")
            continue
        full = weight.astype(mx.float32)
        for bits, (path_mlx, mlx_head) in references.items():
            try:
                ref = [tensor(path_mlx, mlx_head, f"{mlx_name(native)}.{part}") for part in ("weight", "scales", "biases")]
            except KeyError as exc:
                print(f"q{bits} {native}: missing {exc}")
                continue
            results = []
            from_bf16 = None
            for label, source in (("bf16", weight), ("f32", full)):
                ours = mx.quantize(source, group_size=64, bits=bits)
                from_bf16 = from_bf16 or ours
                # Scales and biases are stored in the reference's dtype, so compare them in it.
                stored = [a.astype(b.dtype) if a.dtype != b.dtype else a for a, b in zip(ours, ref, strict=True)]
                exact = all(a.shape == b.shape and mx.array_equal(a, b).item() for a, b in zip(stored, ref, strict=True))
                results.append(f"{label} {'exact' if exact else 'differs'}")
            ours_err = mx.abs(mx.dequantize(*from_bf16, group_size=64, bits=bits).astype(mx.float32) - full).max().item()
            ref_err = mx.abs(mx.dequantize(*ref, group_size=64, bits=bits).astype(mx.float32) - full).max().item()
            print(f"q{bits} {native}: {', '.join(results)}; max error ours {ours_err:.3g}, dgrauet {ref_err:.3g}")
    for native in UNQUANTIZED:
        try:
            a = tensor(path_lightricks, lightricks, f"{PREFIX}{native}")
        except KeyError as exc:
            print(f"{native}: missing {exc}")
            continue
        for bits, (path_mlx, mlx_head) in references.items():
            try:
                b = tensor(path_mlx, mlx_head, mlx_name(native))
            except KeyError as exc:
                print(f"q{bits} {native}: missing {exc}")
                continue
            equal = a.shape == b.shape and mx.array_equal(a.astype(mx.float32), b.astype(mx.float32)).item()
            print(f"q{bits} {native}: {'equal' if equal else 'differs'} ({a.dtype} → {b.dtype})")


def main() -> None:
    """Run the three spike checks and print the report."""
    lightricks = header(f"{LIGHTRICKS_REPO}/{LIGHTRICKS_FILE}")
    check_config(lightricks)
    check_connectors(lightricks)
    check_quantize(lightricks)


if __name__ == "__main__":
    main()
```

## Verification

Convert the stock `ltx-2.3-22b-distilled-1.1.safetensors`:

- macOS: every tensor of the output `transformer-distilled.safetensors` should equal dgrauet's `transformer-distilled-1.1.safetensors` at the same level (bit-exact if mlx-forge used plain `mx.quantize` at group 64; otherwise within quantization noise). Then a short t2v clip should match `ltx-8` at the same seed.
- CUDA: the converted transformer is the same function on the same input as `ltx-8@q8`, so its tensors should match. Then one real community fine-tune on each platform, as a hand test, including a load after the source file is moved away.

### macOS test plan (after the build)

Needs: a 32 GB Apple Silicon Mac (the smallest that runs `ltx-8`), the repo on this branch with `make install` (run every step from the repo folder), and free disk for each step's files:
  - steps 1–7: the 46 GB stock checkpoint, dgrauet's 22 GB q8 transformer (already in the Hugging Face cache if `ltx-8` is installed), the 22 GB q8 output, and a few GB of base files linked from the q8 repo: about 95 GB. Step 3's `--force` keeps the old 22 GB copy until the new one is complete, so allow about 22 GB more during it;
  - step 8: dgrauet's 12 GB q4 transformer, the 12 GB q4 output, the fine-tune file (34–46 GB), its output at the level that fits (22 GB at q8, which a 32 GB Mac picks; 12 GB at q4), and Lightricks' distilled LoRA (7.6 GB, downloaded once if `--distill` is used): about 93 GB more at q8;
  - step 9: the Web UI conversion of the same fine-tune under a new name, another 22 GB at q8. Or delete step 8's fine-tune output first, which frees the same.

Step 8 converts the stock checkpoint again, so keep it until step 8 is done. Deleting the q8 output after step 7 frees 22 GB; the stock checkpoint can go after step 8. Record wall time and peak memory for each conversion; `/usr/bin/time -l` prints the peak as "maximum resident set size".

1. **Fetch the inputs.** Download the stock checkpoint, plus dgrauet's q8 transformer to compare against:

   ```bash
   uv run hf download Lightricks/LTX-2.3 ltx-2.3-22b-distilled-1.1.safetensors
   uv run hf download dgrauet/ltx-2.3-mlx-q8 transformer-distilled-1.1.safetensors
   ```

   Both print their local path; use those below.
2. **Convert at q8.** Expect minutes, not hours; peak memory should stay well under the machine's 32 GB, since the converter streams.

   ```bash
   /usr/bin/time -l uv run ziv model model -i <stock checkpoint> --model-type ltx-2.3 --quantize 8 --name ltx-stock-q8
   ```

3. **Refusals.** Leaving out `--quantize` must fail with a message asking for 4 or 8. Converting again with `--name ltx-stock-q8` must refuse and leave the folder unchanged; with `--force` it replaces it.
4. **Compare tensors.** Save the script below outside the repo (e.g. `~/ltx-spike/compare_ltx.py`) and run it from that folder with the two paths. The summary should report every shared tensor equal; if the spike found mlx-forge differs, the differences should be no larger than the spike measured. It reads one tensor at a time.

   ```bash
   (cd ~/ltx-spike && uv run --project <path to the repo> python -I compare_ltx.py ~/.ziv/models/ltx-stock-q8/transformer-distilled.safetensors <dgrauet q8 transformer>)
   ```

   The subshell keeps your shell in the repo, so the plain `uv run ziv ...` commands in the other steps run this branch's code.

5. **Listing.** `uv run ziv model list` shows `ltx-stock-q8` under video models, and so does the Web UI Models page, with a memory-fit badge for q8. Then rename the folder to a name without "ltx" (e.g. `stock-q8`), check it is still listed as a video model, and rename it back to `ltx-stock-q8` for the next steps.
6. **Generate.** Run the same short clip twice and compare. With identical weights the clips should be identical; with quantization differences they should look the same.

   ```bash
   uv run ziv video -m ltx-8 --prompt "A red kite over a beach, slow pan" --frames 49 -s s --seed 7 -o ~/ltx-compare/stock
   uv run ziv video -m ltx-stock-q8 --prompt "A red kite over a beach, slow pan" --frames 49 -s s --seed 7 -o ~/ltx-compare/converted
   ```

7. **Stored level.** The Models page shows `ltx-stock-q8` at q8, with a fit badge based on its size on disk. The folder holds exactly one `transformer*` file.
8. **q4 and a fine-tune.** Repeat steps 2, 4 and 6 at `--quantize 4` against `dgrauet/ltx-2.3-mlx-q4` (and `ltx-4`). Then convert one real community fine-tune at the level that fits. Pass `--distill` when the converter recommends it (plain-dev merges render broken at 8 steps without it, as the spike showed); note what it reported. Run a t2v, an i2v, an extend and a LoRA clip with it, at the default 8 steps.
9. **Web UI convert.** Convert the fine-tune again from the Models page under a new name. After picking the file and LTX-2.3, the form preselects the level that fits (q8 on a 32 GB Mac) and, for a plain-dev file, Distill. The conversion shows progress while it runs.

`compare_ltx.py`, saved outside the repo:

```python
"""Compare two safetensors files tensor by tensor, reading the raw bytes (bfloat16 included)."""

from __future__ import annotations

import json
import struct
import sys

import numpy as np

# safetensors dtype → numpy dtype its bytes are read as; bfloat16 is widened by hand.
DTYPES = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U32": np.uint32, "U8": np.uint8}


def header(path: str) -> tuple[dict, int]:
    """Return a safetensors file's header without its metadata, and the offset its data starts at."""
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(size))
    head.pop("__metadata__", None)
    return head, 8 + size


def raw(path: str, head: tuple[dict, int], name: str) -> bytes:
    """Read one tensor's bytes."""
    start, end = head[0][name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(head[1] + start)
        return f.read(end - start)


def values(data: bytes, dtype: str) -> np.ndarray:
    """Return a tensor's values: floats as float64, packed integers as int64 so no bit is lost."""
    array = np.frombuffer(data, dtype=DTYPES[dtype])
    if dtype == "BF16":
        return (array.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return array.astype(np.int64 if np.issubdtype(array.dtype, np.integer) else np.float64)


def main(ours_path: str, theirs_path: str) -> None:
    """Print one line per tensor that is missing or differs, then a summary."""
    ours, theirs = header(ours_path), header(theirs_path)
    ours_keys, theirs_keys = set(ours[0]), set(theirs[0])
    for key in sorted(ours_keys ^ theirs_keys):
        print(f"only in {'ours' if key in ours_keys else 'theirs'}: {key}")
    differing = 0
    for key in sorted(ours_keys & theirs_keys):
        a_info, b_info = ours[0][key], theirs[0][key]
        if (a_info["dtype"], a_info["shape"]) != (b_info["dtype"], b_info["shape"]):
            print(f"differs: {key} {a_info['dtype']}{a_info['shape']} vs {b_info['dtype']}{b_info['shape']}")
            differing += 1
            continue
        a, b = raw(ours_path, ours, key), raw(theirs_path, theirs, key)
        if a != b:
            error = np.abs(values(a, a_info["dtype"]) - values(b, b_info["dtype"])).max()
            print(f"differs: {key} max {error:.3g}")
            differing += 1
    common = len(ours_keys & theirs_keys)
    print(f"{common - differing} of {common} shared tensors equal; {len(ours_keys ^ theirs_keys)} not shared")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

For quantized weights (U32), a "max" difference counts packed integers, not weight values. If any differ, compare the dequantized weights as in the spike script.

## Open questions

1. **Dev fine-tunes.** Settled by the spike: the test file is plain dev, its "8 steps" assumes the distilled LoRA, and with it the output is good. §1 detects this and recommends `--distill`. Still open: whether the detection ever flags a file that was distilled another way. That needs more community files; until then the user confirms the merge. A proper dev run mode (about 30 steps with guidance) stays out of scope.
2. **Connectors.** The one test file left them unchanged; linking when equal and writing them otherwise covers both cases. One file isn't proof that fine-tunes never train them.

## Effort

About 6–8 days. New code on both platforms: the shared tensor source (prefix normalization, layout and version checks, per-tensor FP8 dequantization, the hash-based distillation check and the `--distill` merge), writing through a partial folder, the pinned connector hashes, and on macOS the quantizing single-file writer, checked against dgrauet's file. The CUDA side is otherwise plumbing: detection through the shared helper, load options, loading a stored folder without its source, the level and storage format from the manifest, text-encoder config and reuse, and estimates from bytes on disk. The Web UI needs the inspect endpoint and conversion as a job with progress. The Mac spike (about an hour) should confirm `embedded_config.json`, the connector mapping and the bit-exact comparison before the macOS writer is built.
