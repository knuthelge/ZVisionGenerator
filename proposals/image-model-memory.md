# Keep image models resident and reuse quantized weights

**Status:** Done (2026-10-04, branch `feat/image-model-memory`; unreleased)

## Problem

Batches of images on macOS spend most of their time outside the denoising steps. In real Web UI jobs on a 32 GB Apple M5, an image whose steps and VAE decode take about 15 s took 60–140 s in total, and the time grew as the batch went on.

Two causes, both in the mflux backend (`backends/image_mac.py`):

1. **Weights are never kept in memory.** mflux loads weights lazily (`mx.load`), and `MfluxBackend.load_model` never evaluates them. Every `generate_image` call rebuilds the parameters from the safetensors files, so each image re-reads the weights from disk and, when a quantize level is selected, quantizes the full bf16 weights again. Active memory after an image stays at 3–6 GB for a model that should hold 10–17 GB, while the peak during each image is close to the bf16 size.
2. **MLX's buffer cache grows unchecked.** Freed buffers are kept for reuse up to MLX's memory limit. The cache held 16–29 GB in real runs, on top of the model, pushing macOS into swap and slowing the steps themselves.

Re-quantizing at the start of every job (6–8 s for Klein 9B q8, with a peak near the bf16 size) is a smaller, third cost.

## Evidence

Measured with a JSON-lines timing trace on the `debug/image-load-timing` branch (not shipped). Klein 9B, 944×528, 4 steps, 5 images per job, auto-enhanced prompts, no LoRA:

| Run | Per image | Job total | Active / peak memory | MLX cache | Swap | Before step 1 |
|---|---|---|---|---|---|---|
| q8, current (lazy) | 28 → 61 → 68 → 80 → 82 s | 342 s | 6 / 30 GB | 18 GB | 9 GB | 38–47 s |
| q4, current (lazy) | 56–67 s | 331 s | 3 / 27 GB | 20 GB | 7 GB | 29–30 s |
| **q8, eager + cache cap** | **27.7–29.0 s** | **178 s** | 16.8 / 24 GB | 4.1 GB | 3.8 GB, flat | 7–8 s (text encoding) |
| **q4, eager + cache cap** | **24.5–25.7 s** | **157 s** | 9.0 / 16 GB | 4.2 GB | 3.7 GB, flat | 6–7 s (text encoding) |
| bf16, eager + cache cap | 114–140 s | 694 s | 31.3 / 35 GB | 0 GB | 16–18 GB | 83–110 s |

Controlled benchmark (512×512, warm page cache) for the stored-quant part:

| Model | Evaluate weights after load: quantized at load → pre-saved | Peak while loading |
|---|---|---|
| Z-Image q8 | 3.6 s → 1.45 s | 10.8 → 10.4 GB |
| Klein 9B q8 | 7.4 s → 2.8 s | 17.8 → 16.8 GB |

Other phases are small: prompt-enhancer load 1–3 s, about 3 s per "extra"-length rewrite, gaps between images under 2 s, memory release after a job about 1 s.

bf16 Klein 9B needs about 31 GB and does not fit a 32 GB Mac with either lazy or eager loading; the Web UI already reports memory fit, so no new warning is proposed.

## Proposed change

Scope: the mflux backend on macOS. The Windows (diffusers + bitsandbytes) backend is unchanged.

### 1. Eager weight loading

At the end of `MfluxBackend.load_model`, after LoRAs and the VAE upcast, evaluate every parameter once (`mx.eval(model.parameters())`). Weights, quantization and baked LoRAs are then resident for the whole job instead of being rebuilt per image. This applies to every family and precision.

Trade-off: a model larger than memory is now paged in at load instead of per image. It is as slow as today (see the bf16 row) and already flagged by the memory-fit indicator.

### 2. MLX buffer-cache policy

- After load, call `mx.clear_cache()` to drop what loading and quantization left behind.
- Cap the free-buffer cache at a fixed 4 GB with `mx.set_cache_limit` (no new setting).
- Clear the cache after each generation (`text_to_image` / `image_to_image`).

The cap only limits freed buffers kept for reuse, never the model or live activations. Too low a cap costs extra allocations; at 4 GB no step slowdown was measured. The setting is process-wide, so LTX video jobs in the same process are affected too; they already clear the cache between stages. The limit is set from the macOS backend layer (`backends/`), not from generic code.

### 3. Store a selected quant on first use

Quantization stays opt-in through the existing Quant dropdown. The first job that loads a model at a given level saves the quantized base weights next to the source model; later jobs load that copy.

- **Location:** `~/.ziv/models/<name>@q<bits>/` (e.g. `snofs@q8`). For a Hugging Face model picked through an alias, `<name>` is the alias (e.g. `zit@q8`) and the source is the downloaded snapshot; a new revision has different files, so its copy goes stale and is saved again. A model not downloaded yet is downloaded by the first load and saved afterwards. Raw repo ids and folders outside the models directory are quantized at load. Deleting a model (or an alias's download) deletes its copies too. Stored quants are listed on the Models page with their size and can be deleted with the existing delete action. The model dropdown lists only `<name>`; picking a quant uses the stored copy transparently. `@` is safe in image model names: only enhancer references are parsed as `REPO@REVISION`.
- **Lookup:** when a job selects model `<name>` with quant `<bits>` and `<name>@q<bits>` exists and is current, load it with `quantize=None` (the weights are already packed; `model_memory` already treats packed U32 weights as pre-quantized).
- **Saving:** after the first job's model has loaded (and before generating), save with mflux's `save_model`. The job shows a "Saving q8 copy…" status. Saving takes about 20–30 s and 10–17 GB of disk. A failed save (for example, a full disk) warns and the job carries on with the in-memory model. Writes go to a temporary folder that is renamed when complete; on failure, or when the user stops the job during the save, the save is abandoned and the temporary folder removed, and the next job with that quant tries again.
- **Base weights only:** mflux's saver bakes LoRAs into the weights, so a LoRA job never saves its own weights. Whether LoRA jobs can use stored quants is decided by a check done first (see Open questions): if LoRAs apply correctly and at full speed on pre-quantized weights, LoRA jobs load the stored quant and apply LoRAs at runtime, and a missing quant is saved from a LoRA-free load of the base model; otherwise LoRA jobs keep quantizing at load.
- **Detection files:** mflux does not write `model_index.json` or `transformer/config.json`, which `detect_image_model` needs (family and Klein size). Copy them, plus `scheduler/`, from the source model.
- **Staleness:** record the source path, total weight size and modification time, the quant level and the mflux version in a small manifest inside the stored folder. If any differ, quantize again and replace the copy.

### 4. Converter: save directly as a quant

Add `--quantize {4,8}` to `ziv-model model` (and the matching Web UI converter option), producing `<name>@q<bits>` through the same save path as section 3, so a quant can be prepared ahead of time.

## Out of scope

- A warning when a model does not fit memory: the Web UI already shows memory fit.
- Keeping a loaded model between Web UI jobs: loading takes 3–10 s per job once quants are stored; reconsider later if needed.
- Changing how preflight enhances prompts: it costs 20–35 s per job before the first image and is acceptable.
- Speeding up text encoding (7–8 s per image for Klein with ~100-word prompts).

## Open questions

None remaining. **LoRAs on a stored quant** was checked before implementation (Klein 9B q8, `k2-flat` LoRA, 512×512, seed 7): the stored quant plus LoRA produced pixel-identical output to quantizing at load (mean absolute difference 0.0; the same copy without the LoRA differed by 18.6), at the same speed (11.7 s vs 11.3 s per image). LoRA jobs therefore load the stored quant and apply LoRAs at runtime, and a missing quant is saved from a LoRA-free load first.

## Decisions

- Quantization stays opt-in; no memory-fit warning is added (the Web UI already shows fit).
- Stored quants live next to the model as `<name>@q<bits>`, are saved automatically on first use, and stay out of the model dropdown.
- The MLX buffer-cache cap is a fixed 4 GB.
- Stopping a job during a save abandons the save and removes the temporary folder.
- The timing trace used for the evidence stays on the debug branch and is not shipped.

## Testing

- `load_model` evaluates parameters, clears the cache and sets the cache limit (mocked `mx`).
- Generations clear the cache in their cleanup path, including on skip and on errors.
- Stored-quant lookup: picks `<name>@q<bits>` when current; ignores it when the manifest differs; never used when no quant is selected.
- Saving: writes the manifest and detection files, renames atomically, removes the temporary folder on failure, warns without failing the job, and never saves LoRA-baked weights.
- Converter: `--quantize` produces the same layout and manifest as the first-use save.
- No real models are loaded; mflux and MLX calls are mocked.
