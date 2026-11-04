# More accurate q8 weights on Windows and Linux

**Status:** Proposed (2026-11-04)

## Problem

On Windows and Linux, `q8` stores each layer weight as a plain `float8_e4m3fn` cast of the bfloat16 weight, with no scale (`cast_to_fp8` in `backends/cuda/quant.py`, through diffusers' layerwise casting). Two things go wrong:

- **FP8 has only 3 mantissa bits.** Every weight is rounded to about 1 part in 16, wherever it sits in the range.
- **Without a scale, small weights fall off the bottom of the range.** Below 2⁻⁶ FP8 goes subnormal and loses even those bits; below about 2⁻¹⁰ a weight becomes zero. Diffusion and text-encoder weights are small, so a few percent of every matrix is flushed to zero, and up to 17% in some text-encoder layers.

`q8` is used for:

| Component | Where |
|---|---|
| Transformer and text encoder at `q8` | Every image model: `zit`, `klein4b`, `klein9b`, `krea2`, `ideo`, `moss` |
| Transformer | `ltx-8` |
| Gemma text encoder and text connectors | `ltx-4` and `ltx-8` (every LTX-2.3 run, both levels) |
| Gemma text encoder | `ltx25-4` |

So every LTX-2.3 clip runs through a text encoder that loses part of its weights, even at `ltx-4`.

The community's answer is GGUF Q8_0, which is generally treated as almost lossless. Q8_0 is not FP8: it stores weights as 8-bit integers with one scale per 32 weights.

## Measurement

40 random linear weights per component (at least 256 wide, no embeddings or norms), read from the downloaded originals on the CPU. Error is ‖W − Ŵ‖ / ‖W‖ per matrix, median and worst over the 40.

| Storage | Bits per weight | `zit` transformer | `klein4b` transformer | `krea2` transformer | `zit` text encoder | LTX-2.3 Gemma |
|---|---|---|---|---|---|---|
| FP8, no scale (**today**) | 8 | 2.65 / 2.68% | 3.01 / 4.66% | 2.72 / 3.09% | 3.13 / 3.92% | 3.87 / 12.23% |
| FP8, one scale per row | ~8 | 2.64 / 2.64% | 2.64 / 2.65% | 2.64 / 2.65% | 2.64 / 2.65% | 2.64 / 2.64% |
| FP8, one scale per 32 | 8.5 | 2.37 / 2.40% | 2.39 / 2.41% | 2.35 / 2.40% | 2.39 / 2.40% | 2.38 / 2.64% |
| INT8, one scale per row | ~8 | | 0.96 / 1.63% | | | 1.05 / 1.34% |
| INT8, one scale per 128 | 8.125 | | 0.67 / 0.99% | | | 0.68 / 0.83% |
| **INT8, one scale per 64** | 8.25 | | 0.61 / 0.84% | | | 0.62 / 0.72% |
| INT8, one scale per 32 (= GGUF Q8_0) | 8.5 | 0.56 / 0.62% | 0.55 / 0.70% | 0.59 / 0.71% | 0.54 / 0.67% | 0.55 / 0.61% |

The plain cast flushes a median 0.5% (`zit`) to 5.2% (Gemma) of weights to zero, worst 17.5%.

**The finding:** adding scales to FP8 removes the bad tails (Gemma's worst layer goes from 12% to 2.6%) but barely moves the median, because the floor is FP8's 3-bit mantissa. INT8 with a scale per block is about 4–5× more accurate at the same size. The format to switch to is INT8 with block scales, not scaled FP8. This is what Q8_0 does; we don't need GGUF to get it.

Weight error is a proxy. Whether it shows in images and clips is what the spike measures.

## What the user sees

- `q8` images and LTX clips closer to unquantized, if the spike shows a visible difference. The text-encoder fix reaches every LTX-2.3 run, `ltx-4` included.
- The same disk size within about 3%, the same memory, the same step time (to be confirmed by the spike).
- Existing `q8` stored copies and LTX prepared copies are rebuilt once on first use, with the usual "Saving q8 copy…" step, because their recorded format changes.

There is no UI change, so desktop, tablet and phone layouts are unaffected. macOS is not affected either: mflux and MLX quantize with their own affine scheme, which already has group scales.

## Design

- **Storage.** Each quantized linear weight is saved as `int8` `[out, in]` plus `weight_scale` in bfloat16 `[out, in / 64]` (absmax / 127 per block of 64 along the input). A layer whose input width isn't a multiple of 64 gets one scale per row. The layers kept in the compute dtype today (norms, embeddings, `proj_in`/`proj_out`, `lm_head` and each model's `keep_patterns`) stay as they are.
- **Compute.** A pre-forward hook on each quantized `Linear` rebuilds the bfloat16 weight on the GPU (`int8 → bf16`, multiplied by the expanded scale) and drops it after the forward, the same lifetime layerwise casting gives the FP8 weight today. The scale is a buffer of the layer, so group offloading and block streaming move it with the weight.
- **One helper set in `backends/cuda/quant.py`** replaces `cast_to_fp8` for these uses: quantize one tensor (used while writing a copy, a tensor at a time as now), prepare an empty model's layers, and the hook. `fill_weights`, `write_fp8_copy`, the LTX `write_*` and `load_*` functions and MOSS keep their streaming shape and call the new helpers. `ideogram4_instant.fp8_rows` stays: mflux reads that layout on macOS.
- **LoRAs.** peft creates adapters in a layer's `compute_dtype` when it has one; `_declare_compute_dtype` sets it on the INT8 layers as it does on FP8 ones now.
- **Format strings.** The stored-copy and prepared-copy format becomes `int8-b64` (today `fp8-e4m3` in `video/weights.py` and `video/converted.py`), so existing copies read as stale and are rebuilt.

## Spike first

On the 10 GB GPU, through the GPU queue, before building:

1. **Images.** `zit`, `klein4b` and `krea2` at `q8`, unquantized, today's FP8 and INT8 per 64, five prompts with fixed seeds. Report PSNR and LPIPS against unquantized and a side-by-side sheet for a blind look.
2. **Video.** `ltx-4` with today's FP8 Gemma against an INT8 Gemma, three prompts with fixed seeds. This is the case with the largest weight error.
3. **Speed and memory.** Step time and peak GPU and system memory for `zit` and `klein9b` at `q8`, and an `ltx-8` clip. The extra multiply per layer should be lost in the streaming cost.
4. **LoRA.** One LoRA on `klein9b` at `q8` loads and changes the output as it does today.

**Ship if** INT8 is visibly or measurably closer to unquantized on at least the LTX text encoder or one image model, and step time is within 5% of today. **Drop it if** no difference shows anywhere: then FP8's error is below what these models show, and the rebuild isn't worth it.

## Open questions

- **Converted LTX fine-tunes at `q8`.** `ziv-model model --model-type ltx-2.3 --quantize 8` writes a folder that loads without its source file. Those folders can't be rebuilt automatically; with the format change they'd stop loading until converted again. Default is not to keep reading the old format; decide before building.
- **Block size.** 64 is the proposal; 32 matches Q8_0 for 3% more size and about 10% less error. The spike can settle it if the two differ visibly.

## Alternatives considered

- **Scaled FP8** (per row or per block). Measured above: fixes the tails, keeps the 2.4–2.6% floor.
- **GGUF Q8_0.** Same numbers as INT8 per 32, but a new dependency, a loader and an untested path through block streaming, and the community files are mostly in ComfyUI's layout. Our own safetensors layout reuses the streaming code we have.
- **Native FP8 matmul** (`torch._scaled_mm`). Faster on RTX 40 and 50 cards, but RTX 30 cards have no FP8 matmul, and it keeps the less accurate format.
- **INT8 matmul** (`torch._int_mm`). Would need the activations quantized too, which changes the numerics beyond the weights; out of scope.

## Related

[Download pre-quantized copies instead of building them](prequantized-model-copies.md) would publish `q8` copies. If both go ahead, this one goes first, so the published copies use the new format.
