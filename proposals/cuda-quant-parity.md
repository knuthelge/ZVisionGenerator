# Quantization parity on CUDA: FP8 q8, stored quants and memory fit

**Status:** Done (2026-10-09, unreleased)

## Problem

macOS stores a selected quant next to the model and reuses it ([Keep image models resident and reuse quantized weights](image-model-memory.md)), and the Web UI shows whether each model fits the Mac's memory. On Windows and Linux (the shared diffusers/CUDA backend) none of this worked:

1. **No stored quants.** `DiffusersBackend.stored_quant_format()` returned `None`, so every q4/q8 job quantized the full bfloat16 weights again at load.
2. **q8 was capped by VRAM.** q8 was bitsandbytes INT8, which places the quantized weights straight on the GPU and cannot stream them from system memory. Krea 2's transformer is about 12.8 GB at INT8, so q8 could never load on a 10 GB card, although the same model ran unquantized (streamed from system memory). On top of that, the text encoder was moved off the GPU only at q4, so at q8 it stayed resident too.
3. **Unquantized Krea 2 did not run on 10 GB.** Block-level group offloading only splits a top-level layer list. Krea 2's Qwen3-VL text encoder nests its layers under `language_model` and calls its embedding directly, so all 8 GB moved to the GPU at once and the job ran out of memory while encoding the prompt.
4. **No memory-fit badges** on CUDA.

## Evidence

Measured on an RTX 3080 (10 GB) with 30 GB of system memory, Krea 2 Turbo, 1024×1024, 8 steps, warm runs unless noted.

| Krea 2 | Weights in system memory | Peak VRAM | Generation | Load |
|---|---|---|---|---|
| bf16, streamed (after fix 3) | 34.5 GB | 4.7 GB | 334–508 s (weights stream from disk) | 17 s |
| FP8 storage, streamed, from source | 17.7 GB | 4.7 GB | 25.5 s | 114 s |
| FP8 storage, stored copy (16.5 GB) | 17.7 GB | 4.7–5.3 GB | 26 s | 26–29 s |
| NF4, quantized at load | — | 6.8–7.7 GB | — | 125 s |
| NF4, stored copy (9.2 GB) | — | 6.6–7.7 GB | 28 s (cold) | 7.5–18.5 s |
| INT8 (bitsandbytes) | — | out of memory at load | — | — |

- A stored copy reproduces the image of a load from source: FP8 bit-identical; NF4 within 0.06/255 mean (GPU kernel nondeterminism).
- FP8 against bfloat16 (same seed): equal quality on inspection, but the composition shifts (PSNR 11.9 dB from a different pose, not from artifacts). mflux's q8 is not bit-exact either.
- Z-Image Turbo q8: stored copy 10.8 GB, loads in 17.5 s (150 s on first use, including the write), 18 s per 1024×1024 image, 2.9 GB peak VRAM.

## Change

### q8 is FP8 weight storage on CUDA

The transformer's and text encoder's layer weights are stored in `float8_e4m3fn` and computed in bfloat16 (diffusers layerwise casting; norms and embeddings stay in bfloat16). The components then stream to the GPU exactly like unquantized ones (block-level for the transformer; for the text encoder, block-level when its layers form a top-level list, else leaf-level), so every model that runs unquantized also runs at q8, with half the system memory and half the data per step. The FP8 storage needs no FP8 compute support; it ran on Ampere.

The FP8 transformer is never held in bfloat16. It is built empty, prepared with layerwise casting (which fixes each parameter's storage dtype), and filled one saved tensor at a time, each cast to that dtype as it is read and given memory of its own. Building it in bfloat16 first (25.6 GB for Krea 2) and casting afterwards made `systemd-oomd` kill desktop apps on a 30 GB machine. Streamed, Krea 2 q8 peaked at 13.1 GB of anonymous memory while writing its copy, under a 20 GB cap. Tensors read without a dtype change are cloned: `safe_open` tensors map the file, and mapped weights are dropped under memory pressure and read from disk again at every step.

q4 stays bitsandbytes NF4 on the GPU (model CPU offload for Krea 2, whose NF4 pair does not fit 10–12 GB together).

### Stored quants on CUDA

The loader (`image_model_loader.py`) and the stored-quant folder handling (`utils/stored_quant.py`) are unchanged in shape; the backend protocol gains two methods:

- `quantizes_from_files(bits)`: whether a level's copy is written from the source files instead of saved from a loaded model.
- `write_quantized_files(source, path, bits, cancelled)`: write it.

| Level | How the copy is made | How it loads |
|---|---|---|
| q8 (FP8) | Before the first load: the text encoder is loaded, cast and saved, then the transformer is streamed in (below) and saved; `model_index.json` and the other components it lists are hard-linked (copied across file systems). The write checks for a stop between components. | Text encoder through transformers (upcast, then cast back: lossless). Transformer streamed in like a source model; its FP8 tensors pass through unchanged. |
| q4 (NF4) | After the first load, `pipeline.save_pretrained` (diffusers' pre-quantized format), as on macOS. | `from_pretrained` without a quantization config; the saved configs carry it. |

The backend recognises a stored copy by its manifest's `bits`. The format tag depends on the level: diffusers and transformers versions for q8, plus bitsandbytes for q4, so upgrading a library re-creates only the copies that depend on it (`stored_quant_format(bits)`). Interrupted saves are swept by the newest write inside their partial folder, after an hour. `copy_detection_files` now keeps component configs a backend wrote (a stored NF4 transformer's `config.json` holds its quantization settings) and never writes through a hard link. The converter's `--quantize` goes through the same `create_stored_quant`.

An alias that is not downloaded yet loads at q8 from Hugging Face (downloading it) and the next job writes the copy, rather than loading the model twice in one job.

### Memory fit on CUDA

`get_accelerator_memory_budget()` returns a `MemoryBudget`: GPU memory, plus system memory on a discrete GPU (`nvidia-smi` and `psutil`, so the badge never starts CUDA in the server process). `estimate_cuda_image_memory` mirrors the backend: unquantized and q8 need about 4 GB of GPU memory plus the VAE, and hold their weights in system memory; q4 holds its NF4 weights on the GPU. A model is *too large* when its GPU need exceeds the card, *tight* above 90% of the card or with weights above 80% of system memory (they then stream from disk), and otherwise *fits*. The Web UI badge explains both numbers. CUDA video is not estimated.

### LoRAs on CUDA

LoRAs did not work on Windows and Linux at all before this change: `peft`, which diffusers needs to apply them, was not a dependency. It now is (Windows and Linux only). Two more fixes make LoRAs work at every level, matching macOS:

- **q8.** peft creates a LoRA's adapter layers in the base layer's dtype, FP8 here, which CUDA cannot multiply and which flushes typical LoRA values to zero. FP8-stored layers now declare `compute_dtype = bfloat16`, the attribute peft already honours for bitsandbytes layers, so adapters load and run in bfloat16 on top of the FP8 weights.
- **LyCORIS LoKr** (`lokr_w1`/`lokr_w2`, optionally decomposed) is applied the way mflux applies it: each target layer gets an extra output term `scale · w1 @ X @ w2ᵀ`, never building the Kronecker product or changing the layer's weights, so it works on bfloat16, FP8 and NF4 layers alike. The scale is `alpha / rank` for a decomposed factor, else 1, as in mflux. Layer names are mapped with diffusers' own LoRA key converters, run on stand-in LoRAs whose `lora_B` rows carry their own index, which also maps fused layers the converters split (a FLUX.2 `qkv` target gets its slice of the delta). The factors stay in pinned system memory and are copied to the GPU as each layer runs: group offloading looks up every tensor a block holds at load time, so adapters added afterwards cannot be buffers, and keeping them on the GPU pushed q4 Krea 2 out of memory.
- Like mflux, other LyCORIS tensors (`diff`, `diff_b`, LoHa) are skipped with a warning.

Measured on Krea 2 (1024×1024, 8 steps, cold first image): a standard LoRA at q4 loads from the stored copy in 16.5 s and generates in 30 s; at q8 it loads in 34 s and generates in 67–69 s, against 33 s without a LoRA. A large LoKr file (about 0.8 B adapter parameters) takes 91 s at q8 and 37 s at q4, alongside a standard LoRA. Z-Image q8 with a file mixing LoRA and `diff` tensors applies the LoRA and skips the rest.

## Decisions

- FP8 replaces INT8 for q8 on CUDA rather than adding a third level: it keeps the user-facing contract (8-bit, about half the memory) and removes the VRAM cap.
- q8 copies are written from files, so first use never holds a full bfloat16 model and its FP8 copy at the same time.
- A stop during a q8 write loads nothing: the loader returns no model, and the image and upscale runners quit before their first image without using it.
- Krea 2's VAE is tiled through the VAE itself; its pipeline has no `enable_vae_tiling` wrapper, so it decoded whole frames and could run out of GPU memory.

## Not done

- **LoRA speed at q8.** A LoRA roughly doubles the time of a q8 image (cold runs; warm runs not measured). Likely the adapters moving between system memory and the GPU every step; not investigated.
- **FLUX.2 Klein 9B LoKr** was covered by unit tests only (fused `qkv` slicing); Klein 9B was not run on hardware.
- **Very large models at q4.** An NF4 transformer above ~9 GB hits the same VRAM limit INT8 did; streaming NF4 is not supported by bitsandbytes.
- **Windows** was not tested on hardware; the backend is shared with Linux.
