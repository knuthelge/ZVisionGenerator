# LTX-2.5 and MiniMax H3

**Status:** In progress. Phase 1a is done and 1b is decided (2026-10-22): go with `ltx25-4` only. 1c is implemented and GPU-verified: a smoke test on the RTX 3080 10 GB / 30 GB RAM ran t2v and i2v at 1280×704 in 172 and 176 s, with peak RSS about 22.5 GB and no kills. An Extend of 2 s joined cleanly, with a frame-to-frame PSNR across the seam of 35.8 dB, which is inside the clip's normal motion range. Phase 2a is done and 2b is decided (2026-10-22): go at 864×480. 2c is done. 2d is built with full parity (text to video, image to video, first and last frame, Extend, 2× upscale with review through SeedVR2, user LoRAs, 4 or 8 steps) and is implemented on Windows and Linux and GPU-verified on the RTX 3080 (M1, 2026-10-23, see [Phase 2 M1 results](#phase-2-m1-results-measured)); the final GPU verification (M2, 2026-10-23) is done and H3 passed on CUDA (see [Phase 2 M2 results](#phase-2-m2-results-measured)); after M2, H3's 2× upscale was fixed to fit the card at every size and to stop redrawing the frame, but it still looks clearly worse than LTX's own upscale, and a latent upscale for H3 is being researched (see [H3 upscale after M2](#h3-upscale-after-m2)); 2e, the LTX backend split, remains. Phase 3 is implemented on macOS and verified on a 32 GiB M5 (2026-10-25, [test plan](archive/ltx-2.5-mac-test-plan.md)): LTX-2.5 (the `ltx25-4` alias, 2.5 sigmas and the ancestral loop, Gemma 4 handling) and H3 on a vendored MLX port (everything except upscale). Everything built so far ships in v0.13.0.

**Mac test plan:** [ltx-2.5-mac-test-plan.md](archive/ltx-2.5-mac-test-plan.md) is the test plan for LTX-2.5 and MiniMax H3 on a 32 GB Mac. It ran on 2026-10-25, and its results and follow-ups are recorded in it.

Three phases, each opened by a benchmark and a decision point before any implementation:

1. **LTX-2.5** on Windows/Linux: spike (done), decision (done), build.
2. **MiniMax H3** on Windows/Linux: spike, decision, a consolidation refactor of the shared weight code, build, then a split of the LTX backend. This phase adds the video-family seam.
3. **macOS:** both families, tested afterwards on a Mac.

This proposal supersedes the "LTX-2.5 not planned" decision of 2026-10-10 in [deps-cuda](archive/deps-cuda.md), [deps-macos](archive/deps-macos.md) and [platform-parity](archive/platform-parity.md). That decision rested on the bf16 DiT being no smaller than 2.3's, so there was "no memory gain". The premise here is different: the app already runs a 19B LTX DiT on a 10 GB card as a streamed NF4 copy. A 2.5 DiT of the same shape fits the same path. Since then, community work has also brought H3 within reach of 10-16 GB cards.

## Problem

The app runs only distilled LTX-2.3. Two newer open video models matter:

- **LTX-2.5** (Lightricks, open weights 2026-08-11) is the current LTX. Users report better lip-sync, audio and faces than 2.3, and most 2.3 LoRAs carry over.
- **MiniMax H3** (open weights 2026-08-03) is stronger on dynamic scenes and has a large LoRA ecosystem. It's heavier: a 33B DiT and a 32B text encoder. In the last month, third-party work has made it practical on 12-16 GB cards: sparse attention, a 50-layer text-encoder quant, AdaLN precompute and refreshed turbo LoRAs.

Hard requirement: each family runs on Windows/Linux with a 10 GB Ampere card (sm_86, no FP8/FP4 compute) and 30 GB of system RAM, with systemd-oomd active. Preferred: it runs on a 32 GB Mac. Distilled or few-step use only; base models at 20-30+ steps are out.

The video stack is LTX-only in practice. `get_video_backend(family)` exists, but these all assume LTX:

- the registry and capability flags
- the step cap of 8
- the memory estimates
- frame alignment (`8k + 1`)
- every `ltx_*` config accessor

### Not in scope

- **New LTX-2.5 features:** DFR, keyframe slots, multishot, auto duration, HDR, the temporal upscaler and the Gemma 4 prompt enhancer. Each is a follow-up.
- **The LTX-2.5 diffusion VAE decoder.** Its default attention materializes the full score matrix. The conv VAE ships beside it, unchanged from 2.3.
- **NVFP4 and SVDQuant on CUDA.** They are Blackwell-only, or emulated on Ampere.
- **H3 reference-to-video (Ref2VA) and the hosted-only H3 variants** (Max, Regenerate-2K).
- **Converting single-file fine-tunes for either family.**

## Evidence

Checked against `main` at `bd82706` on 2026-10-22, using:

- diffusers 0.41.0, transformers 5.19.0, torch 2.14.1+cu130 and bitsandbytes 0.50.2
- the vendored `ltx_*_mlx` packages at `ae3b7ea` (v0.16.3)
- Hugging Face metadata and safetensors/GGUF headers; the research itself downloaded no weights. The Phase 1a spike then ran them

*est.* marks an estimate from the app's memory constants or a FLOP model. *author* marks a figure the uploader reported and we haven't reproduced. *measured* marks a figure from the Phase 1a or 2a spike (see [Phase 1a results](#phase-1a-results-measured) and [Phase 2a results](#phase-2a-results-measured)).

### Phase 1 evidence: LTX-2.5

**What changed from 2.3.**

| Part | 2.5 compared with 2.3 | Effect on us |
|---|---|---|
| DiT | Same shapes. 2.3 has 4,186 tensors (diffusers names); 2.5 has 4,091: 96 video FF-bias tensors dropped (`ff_bias=false`; the audio FF keeps its bias) and `keyframes_abs_pos_embedding` added. All 4,091 names and shapes match the diffusers model, so the folder needs no renames (*measured*) | NF4/FP8 prepared copies and block streaming carry over. diffusers 0.41 reads both switches from config (`transformer_ltx2.py:453, 1186`) |
| Text encoder | Gemma 4 12B fine-tuned by Lightricks (`Gemma4UnifiedForConditionalGeneration`, `gemma4-12b-ltx-v1`, about 12.0B). Projection heads live in the connectors | New class and module names. Measured: the Gemma 3 renames (`weights.py:67-71, 441-448`) need only two more prefixes, both for the vision embedder, and the FP8 skip patterns (`quant.py:48`) need no change (the copy is 12.13 GiB, as estimated). Google's base Gemma 4 is not a substitute |
| Connectors, audio VAE, vocoder, conv VAE | Unchanged | Reuse |
| Spatial x2 upscaler | Same keys and shapes, new weights | Reuse `_UPSAMPLER_CONFIG`, new file |
| Distilled sampler | Euler ancestral (eta 1.0, s_noise 1.0). Noise seed offset 10000 on stage 1 and 20000 on stage 2 (`ltx_pipelines/distilled.py:482, 491`). 8 + 3 steps | diffusers has `LTXEulerAncestralRFScheduler`, but `LTX2Pipeline`'s loop needs a terminal sigma of 0 without an extra forward, and the audio scheduler is a deep copy that must share the generator. We wrap the scheduler (measured: works) |
| Image-to-video CRF | 18 (2.3: 33) | Per-version default |

diffusers 0.41.0 and transformers 5.19 already have every class. No dependency bump is needed.

**Sources and licence.** Every Lightricks 2.5 repo is gated (auto-approve). `unsloth/LTX-2.5-FP8` is ungated and byte-identical to `Lightricks/LTX-2.5-Diffusers` (same LFS ids). The LTX-2.x Community License applies: free under $10M annual revenue, and derivatives (our stored quants) keep the terms.

Decided 2026-10-22: both families download from ungated sources. When a model first downloads, the download dialog shows one short line with a link to its licence, saying that the app expects the user to have agreed to it. See [Licence notice](#licence-notice).

**Memory on 10 GB / 30 GB.** System RAM binds first, because the DiT copy is pinned in RAM and streamed block by block (`block_streaming.py:44-75, 145-150`). The app's estimate is `max(text encoder, transformer) + connectors + 6 GiB` (`utils/model_memory.py:510-512`). Phase 1a measured two of the options below; the rest are still estimates.

| Option | DiT | System peak | Verdict |
|---|---|---|---|
| NF4 DiT + FP8 Gemma 4 (today's `ltx-4` path) | 9.7 GiB | 21.1-21.4 GiB at 1280×704 (*measured*; *est.* 21.08) | Fits, the same as 2.3 (21.5). Shipped as `ltx25-4` |
| NF4 DiT + NF4 or disk-streamed Gemma 4, embeddings cached | 9.7 GiB | about 19 GiB (*est.*) | Not needed: the go bar was met. NF4 Gemma 4 stays untested |
| GGUF UD-Q3_K_XL / UD-Q4_K_XL | 12.0 / 14.0 GB | about 21 GiB (*est.*) | Would fit; needs a GGUF loader. Out for 1c |
| FP8 DiT (today's `ltx-8` path) | 17.8 GiB | 24.0-24.4 GiB at 768×448 (*measured*; *est.* 27.0) | Doesn't fit with margin. Out for 1c |
| LTX's official `--offload disk` with an INT8-ConvRot DiT | 2 blocks resident | about 5 GB plus page cache (*est.*) | Lowest RAM. Re-reads the DiT every pass (about 236 GB per clip) and needs an INT8 kernel outside diffusers |

No LTX-2.5 run on any 8-12 GB Ampere card had been published; the nearest was an RTX 5080 16 GB at 1280×704 for 5 s with sound in 125 s, on NVFP4. Phase 1a is now the reference for this card.

#### Phase 1a results (measured)

Run on 2026-10-22 on the RTX 3080 10 GB / 30 GB RAM machine, with revision `7b40724` of `unsloth/LTX-2.5-FP8`. Despite its name the repo is bf16 throughout (DiT 35.4 GiB, text encoder 22.3 GiB). Each row ran in its own systemd unit with a cold page cache and was sampled every 0.5 s. LTX-2.3 `ltx-4` ran beside it in the same session. Times are wall time for the whole process, including loading.

- **Memory metric.** The go metric is the system-wide drop of `MemAvailable` during the row. Pinned memory is charged to the unit's cgroup as shmem, inside `file`, so the cgroup's "non-file" figure misses about 10 GiB of pinned DiT. 1c's memory constants must count it.
- **1280×704×121, two stages (8 + 3, stage 1 at 640×352), 5 s with audio:**

  | | 2.5 `ltx25-4` | 2.3 `ltx-4` |
  |---|---|---|
  | t2v | 196 s, peak 21.1 GiB | 199 s, 21.5 GiB |
  | i2v | 201 s, peak 21.4 GiB | 203 s, 21.5 GiB |

  The peak matches 2.3 within noise, and the cgroup cross-check differs by about -0.1 GiB. There was no swap, user-session pressure stayed at 0, and there was no oomd or kernel kill. GPU peak was 6.5 GiB (CUDA allocated 5.05 GiB). s/step is 3.10 in stage 1 and 11.38-11.41 in stage 2, identical to 2.3. The peak is a transient at the handover from the text encoder to the DiT (transformer attach), not during denoising, which holds about 16.5 GiB.
- **768×448×121, one stage:** 145-157 s (2.3: 158 s), of which about 38 s is encoding, about 20 s pipeline load and 26-30 s DiT attach; s/step 4.33. Use 768×448 (the `s` preset), not 704×448. A warm-run figure (earlier estimate: 65-90 s) was not measured.
- **FP8 DiT at 768×448:** the system drop was 24.0-24.4 GiB and `MemAvailable` bottomed at about 3 GiB; one run had 0.6 GiB of swap and 23% user-session pressure. 1280×704 was not run because its gate failed. The app's q8 estimate of 27.0 GiB overstates by about 3 GiB, while the q4 estimate of 21.08 matches.
- **Pinning and NF4 Gemma 4:** pinned streaming met every bar, so the unpinned arm was not needed, and the NF4 Gemma 4 arm was not triggered.
- **Stored-copy build:** 327 s, about 9.2 GiB `memory.peak`. The text-encoder copy is 12.13 GiB.

**Quantization quality.** The only LTX-2.5 set is unsloth's: LPIPS/PSNR against bf16, 1280×704×121, 8 + 3 steps, 4 seeds. Compiled and eager bf16 already differ by 0.265, so that's the noise floor.

| File | Size | LPIPS / PSNR |
|---|---|---|
| Q8_0 | 22.8 GB | 0.242 / 21.6 |
| INT8-ConvRot | 25.1 GB | 0.265 / 20.2 |
| UD-Q3_K_XL | 12.0 GB | 0.310 / 19.1 |
| UD-Q4_K_XL | 14.0 GB | 0.320 / 19.4 |
| Q4_K_M | 13.6 GB | 0.352 / 18.1 |
| FP8 e4m3 | 19.1 GB | 0.366 / 17.6 |
| UD-Q2_K_XL | 9.5 GB | 0.416 / 17.1 |

NF4 (bitsandbytes) has no published number. It's what ships for 2.3. Phase 1a measured it only against FP8 (below), not against bf16.

GGUF in diffusers needs `gguf>=0.10`, `convert_ltx2_transformer_to_diffusers`, the prompt-AdaLN rename and a connector filter. Its default path dequantizes every Linear on every forward (`quantizers/gguf/utils.py:607-618`).

**Quality in Phase 1a (*measured*).** The video is deterministic: a repeat run is bit-identical, while the audio differs slightly between runs. These are luma PSNR and SSIM over 121 frames, not LPIPS, so they aren't comparable to the table above, and there is no bf16 reference.

- NF4 DiT against FP8 DiT on the same seed: 24.8 dB / SSIM 0.83 on the first prompt and 21.0 / 0.70 on a high-texture prompt. Same subject and motion, but this doesn't show which is better.
- Deterministic against ancestral sampling: 13-17 dB, i.e. different samples. The ancestral sampler is the upstream one.
- i2v frame 0 against the input image: 27.9 dB (2.3: 28.1).
- 2.5 looks softer than 2.3, at least partly because it follows "shallow depth of field" prompts.

**Speedups on Ampere.**

- **SageAttention 2** has sm_86 kernels, and diffusers has a `"sage"` backend. The 2× figure is per kernel. Measured in Phase 1a on sm_86, on self-attention only (cross-attention passes a mask that Sage rejects): stage 2 is 9.5% faster per step (10.30 s against 11.38 s), with no wall-time gain (198 s against 196 s), and quality against SDPA is 34.9 dB / SSIM 0.95 at 768×448. PyPI has no 2.x wheel. It builds from source in about 3.5 minutes with no sudo, using pip-provided CUDA 13 nvcc and headers, but torch 2.14 needs a C++17 to C++20 patch to its `setup.py`, pinned to one commit. Below the 15% bar, so out.
- **INT8 tensor cores** exist on Ampere and NF4 doesn't use them. INT8-ConvRot ran 1.3-2× faster than bf16 on a 3090 for a different model. There's no diffusers kernel.
- **Not applicable on Ampere:** Sol-Attn (measured on H100, B200 and 5090 only), `LTX2_SOL_TAU` (MLX only), and step caching (little to skip at 8 + 3).

**Recent changes:**

- LTX-2 v1.5.0 (2026-10-18): faster passes and sampler, mostly multi-GPU.
- Wan2GP LTX-2.5 VAE and kernel speedups (2026-10-11 to 2026-10-21).
- Unsloth quants (2026-10-13).
- No LTX-2.5.x or LTX-3 announced.

### Phase 2 evidence: MiniMax H3

**Model.**

| Part | Facts |
|---|---|
| DiT | 33.1B dense, 50 blocks, hidden 5376, 56×128 heads, FFN 14336. 13.0B are per-block `adaln_proj` Linears (2688 → 96768) that depend only on the timestep. Guidance-distilled: one forward per step, no CFG or negative prompt |
| Text encoder | Qwen3-VL-32B. H3 reads `hidden_states[50]`, the un-normalized output of decoder layer 49, so layers 50-63 and `lm_head` are unused. The vision tower is needed only for image or reference input |
| Video VAE | 16× spatial, 4× temporal, 24 channels; decoder 2.42B (9.0 GiB fp32 on disk) |
| Audio VAE | 151M params, stereo, 32 kHz |
| Output | 24 fps fixed. Frames snap to `17n + 5` (124-345, so 5-15 s). Canvas short edge 768 by default; smaller canvases are fine |
| diffusers 0.41 | `MiniMaxH3ModularPipeline` (t2va, fl2va, ref2va), plus a single-file loader (#14839) for the official layout. Pruned Comfy/GGUF files (`adaln_t_table`) are refused. The modular loop has no step callback or interrupt |
| Repo | 464 GiB in all. The t2va set is about 122 GiB; the download must filter out `transformer_ref/`, `FL2VA/` and `Ref2VA/` |

**Licence.** MiniMax H3 Community License:

- **Excluded territories (§I.5, §V.4):** use and outputs are unauthorized in the EU, UK, Republic of Korea and US.
- **Revenue (§IV.1):** authorization needed above $20M annual revenue.
- **Attribution (§IV.2):** "MiniMax H3" shown in the UI of commercial products.
- **LoRAs:** count as Model Derivatives.

The QA page calls the territory limit temporary. The licence text governs. H3 uses the same [licence notice](#licence-notice) as LTX-2.5, linking the licence text.

**What shrinks it.**

| Lever | Effect | Source |
|---|---|---|
| Truncate the text encoder to 50 layers, no `lm_head` (and no vision for t2va) | 33.4B → about 25B. NF4 about 13.2 GiB; FP8 about 24 GiB | Header arithmetic. Comfy-Org's bf16 file is already 50 layers |
| Community 50-layer GGUF (`pottokao/MiniMax-H3-TextEncoder-16GB-GGUF`, 2026-10-20) | 12.65 / 13.70 / 14.80 GB. Conditioning error against bf16 is 7.9 / 6.4 / 4.9% | *author*. Error is measured on conditioning, not video |
| Fold AdaLN into per-timestep tables | 13.0B dropped exactly. DiT 20.1B → NF4 about 10.8 GiB | Same technique in FastVideo, LightX2V and PipeNetwork's MLX port (25-26 GB of bf16 saved) |
| Pruned community DiTs | Comfy pruned int8 21.0 GB, w6a8 16.0 GB; GGUF Q3_K 8.8 GB, Q4_K 11.4 GB; FastVideo Trim (42 blocks) int8 18.9 GB | These change numerics (the unsloth NOTICE says so), and LoRAs trained on the full model don't fit them |
| Encode first, cache embeddings, unload | Text encoder and DiT never coexist. About 10 MB per 1k-token prompt | Standard pattern; a ComfyUI node caches conditioning |
| Unpinned or disk-streamed DiT | On an RTX 3090 with 31 GB RAM, ComfyUI hit 29.9 GB host RAM pinned (OOM-killed) and 7.5 GB with `--disable-pinned-memory` | *author*. Our `stream_blocks` pins today |

System RAM peak for t2va with our formula (*est.*):

| Combination | Peak |
|---|---|
| NF4 50-layer text encoder (13.2) + folded NF4 DiT (10.8) | about 24 GiB with the VAE decoder resident; about 19 GiB loaded lazily |
| Today's machinery unchanged (64-layer FP8 text encoder) | 43 GiB. Refused |

GPU peak with streamed blocks (*est.*):

| Clip | Peak |
|---|---|
| 384×672, 124 frames | about 5 GiB |
| 768×1344, 124 frames | about 7-8 GiB |
| 15 s at 768p | over 10 GB unless attention is query-chunked |

**What speeds it up.** Compute, not memory, is the real limit, because full attention over 10k-100k tokens dominates.

| Lever | Effect | Ampere? |
|---|---|---|
| Turbo LoRAs: LightX2V FL2V 8-step v1.1 and 4-step v1.3 768p (refreshed 2026-10-20); a 4-step T2V LoRA (larryvrh); FastVideo 4/8-step distilled models; 2/4-NFE PDMD LoRAs | 20 → 8 or 4 steps. Per-step cost is unchanged. 4 steps can smear fast motion | Yes. Merge into bf16 before quantizing (as the LTX converter does with the distilled LoRA) |
| SageAttention 2 | About 2× (Comfy docs). INT8 attention can morph clip ends or garble text. On LTX (Phase 1a) it gave 9.5% per step on self-attention only | sm_86 kernels exist; builds from source with a patch |
| Veda learned sparse attention (ComfyUI node, Triton) | RTX 5070 12 GB, 1344×768, 124 frames, 8 turbo steps: 342 s dense, 231 s with Sage, 130 s with Veda. On a 4090, 1.8-2.8× end-to-end | Node says Triton SM80+. Untested on a 3080 |
| Sol-H3 (NVlabs), Spark-H3, ComfyUI core `sol-attn`, Turbo-SLA, FastH3 VSA | 2-4.5× claimed | Mostly RTX 40/50 builds. Wan2GP says FastH3 Omni runs on RTX 30 |
| Smaller VAEs: LightVAE 12-layer, INT8-ConvRot VAE (about 2× decode), Sol-H3-VAE | Decode time and memory | INT8 VAE on Ampere plausible |
| Lower resolution | 960×544 is about 2.3× faster per step than 1344×768 (diffusers docs) | Yes |
| Step caching (TeaCache etc.) | 0 skips with the default hook in one MLX test; risky at 8 turbo steps | Unproven |

Reported runs near our class:

| Card | Clip | Steps / tricks | Time | Source |
|---|---|---|---|---|
| RTX 3060 12 GB | 1280×736, 5 s | 20 steps | 1006 s | secondary |
| RTX 3060 12 GB | 1280×736, 5 s | 4-step turbo + EasyCache | 420 s | secondary |
| RTX 3060 12 GB | 480p, 5 s | INT8 | 3-4 min | secondary |
| RTX 4070 12 GB | 576×832, 124 frames | 8-step turbo | 131 s | secondary |
| RTX 4070 12 GB | 576×832, 124 frames | 4-step turbo | 79 s | secondary |
| RTX 3090 24 GB | 1280×720, 5 s | pruned W4A8, VSA | sampling 57.5 s + decode 36.8 s | *author* |
| RTX 5070 Ti 16 GB | 1344×768, 61 frames | 4-step turbo, 50-layer GGUF text encoder | about 75 s per clip | *author* |

No 3080 timing exists. Our FLOP model, calibrated against the 3060 runs, gives these estimates at 8 turbo steps with SageAttention:

| Clip | DiT | Total |
|---|---|---|
| 864×480, 5 s | about 1.3-2.4 min | about 3-5 min |
| 1280×736, 5 s | about 5-7 min | about 6-10 min |

Sparse attention could roughly halve those, if it runs on sm_86.

**What exists in our code.** These reuse as they are:

- `stream_blocks` and group offload, since H3 has a `transformer_blocks` ModuleList
- NF4 packing (`weights.py:302-360`) and `ensure_prepared`
- the encode-then-unload pattern (`backend.py:1104-1145`)
- `cast_to_fp8`
- `release_memory`
- the taking-turns estimate
- `encode_video`

These are new:

- a 50-layer text-encoder loader with its own hidden-state tap. diffusers raises when `num_hidden_layers <= 50`, but a 50-layer tower with `tie_last_hidden_states=False` gives the same tensor.
- the AdaLN fold (a shim per block returning the precomputed table)
- a chunk-by-chunk VAE decode under `no_grad`. H3's VAE already tiles and chunks itself, but its stock whole-video float conversion runs out of GPU memory at 1280×736 (see [Phase 2a results](#phase-2a-results-measured))
- our own step loop over diffusers' `MiniMaxH3Scheduler`, for progress and cancel
- query-chunked attention for long clips
- 4-D input support in `chunk_tokens`

### Phase 3 evidence: macOS

LTX-2.5:

- **Already supported in the vendored packages:** the vendored MLX packages support 2.5 at our pin. `dgrauet/ltx-2.5-mlx-q4` (11.3 + 10.6 + 4.0 GB) should peak at about 17-19 GB (*est.*). `-q8` is about 26-27 GB and needs DiT streaming, which we turn off above 24 GiB (`backends/mac/video.py:33`). Only dgrauet's packs put the encoder at the root, the layout the pinned loader needs.
- **Gaps in our Mac code (fixed in Phase 3):**
  - `_generate_one_stage` runs 2.3's sigmas and the deterministic loop (`backends/mac/video.py:396, 447-455`).
  - The download plan always adds Gemma 3 (`backends/mac/download_plan.py:86-93`).
  - The estimator reads Gemma 3 and 2.3 decoder names (`utils/model_memory.py:100, 347, 422`).
  - `make update-ltx` to v0.17.0 is optional; it adds `--low-ram` with unfused LoRAs.

H3 candidates to test:

| Candidate | What it is | Known numbers |
|---|---|---|
| PipeNetwork/minimax-h3-mlx | Python MLX port with AdaLN precompute | 4-bit DiT 11.5 GB resident; unquantized 50 GB text encoder |
| antirez/h3.c | Metal-native engine with SSD block streaming | DiT storage 36.5 → 2.0 GiB; int8 peak 25.9 GiB |
| FastVideo FastH3 MLX | INT4 dense DiT and an Omni MLX kit | INT4 DiT 10.74 GiB |

Reported Mac speeds: about 1.2 h projected for 8 steps at 1344×768 for 5 s on an M3 Ultra (bf16), and 23 min for 768×448 for 5 s at 8 steps on an M4 Pro 24 GB (INT8 + turbo, secondary).

## Proposed change

### Licence notice

Shared by every family that needs it, built once in Phase 1:

- **Config:** a `licence` entry per video model preset (or alias), holding a display name and a URL. For LTX-2.5 it points at the LTX-2.x Community License; for H3, at the MiniMax H3 licence text.
- **Download dialog:** when a model with a `licence` entry is about to download for the first time (the first-run picker and the Models-page download), the dialog shows one short line, for example: "LTX-2.5 is under the [LTX-2.x Community License](…). Downloading it means you have agreed to it." There is no separate checkbox.
- **CLI:** the same line is printed once before a CLI download starts.
- **Contract test:** the workspace and models payloads carry the licence name and URL for those models. Copy isn't tested.

### Phase 1: LTX-2.5

**1a. Spike (done 2026-10-22; throwaway, not merged).** A bench script outside the package drives `DiffusersVideoBackend` on the RTX 3080 10 GB / 30 GB RAM machine, with spike-only patches instead of code changes. Each run gets its own systemd user unit. The weights are the Diffusers folders of `unsloth/LTX-2.5-FP8` (67 GB downloaded). The NF4 DiT and FP8 Gemma 4 copies are built with the existing `ensure_prepared`, into a scratch data folder so the user's stored copies are never rewritten. The patches add:

- the ancestral step, with a terminal sigma of 0 and noise seeds offset by 10000 (stage 1) and 20000 (stage 2)
- CRF 18 for image-to-video
- no prompt enhancer, processor or duration head at load (the repo lists them but ships no weights for them)

Steps, in order; a broken smoke run stops the spike:

1. **Preflight, no model run.** Record the revision. Compare checkpoint tensor names and shapes with the diffusers models both ways (DiT, Gemma 4, connectors, upsampler). List which Gemma 4 layers the FP8 skip patterns keep in bf16. Check that the GPU, RAM and disk are free.
2. **Build the copies, then a smoke run** (t2v, 768×448, 121 frames, one stage), repeated once as the noise floor for every quality comparison.
3. **Go/no-go runs:** t2v and i2v at 1280×704, 121 frames, two stages (8 + 3, stage 1 at 640×352). LTX-2.3 `ltx-4` runs the same prompts, seeds and sizes in the same session.
4. **Follow-ups:**
   - unpinned streaming, only if the go runs miss the memory bar
   - deterministic against ancestral sampling
   - NF4 Gemma 4, by connector-output cosine, only if memory is tight
   - the FP8 DiT, smaller size first
   - SageAttention 2 on the self-attention layers only (it refuses masks), with 30 minutes to build it

   GGUF is deferred.

Each run records:

- from outside the unit, every 0.5 s: the system's available memory (the go metric is its drop during the row), the unit's cgroup memory as a cross-check, swap and pressure, and the user session's pressure; `memory.peak` for information only. Pinned memory is charged to the cgroup inside `file`, so a non-file figure would miss it
- from inside: the CUDA peak per phase, s/step per stage, time per phase and total wall time
- oomd and kernel OOM kills, from the system journal
- automated output checks: frame count and fps, an audio stream with sound, no NaN or frozen frames, and for i2v the PSNR of frame 0 against the input image
- PSNR/SSIM between paired runs on the same seeds, next to the noise floor, and contact sheets beside 2.3 for one visual review at the end

A watchdog outside the unit kills the run before oomd would (swap, pressure, memory), and any oomd kill stops the spike. Seeds match upstream in order and meaning, not bit for bit. Results go to a JSON file and a summary.

**1b. Decision point.** Go if both 1280×704 runs (t2v and i2v) meet all of these:

- the system memory drop at most 1 GiB above 2.3's, and at most 24 GiB (`memory.peak` counts page cache, so it's only recorded)
- swap at most 0.25 GiB and no oomd kill
- the user session's memory pressure stays at or below 20%
- every output check passes

The end review can veto a go. Decide then:

- **Levels:** `ltx25-4` on a go. `ltx25-8` only if its 1280×704 run finishes without a kill or swap, within 26 GiB of system memory drop.
- **Sampler:** ancestral, unless it fails the checks where deterministic passes, or looks worse.
- **NF4 Gemma 4:** only if memory is tight, and only if the connector outputs keep a mean per-token cosine of at least 0.99 against FP8.
- **Pinning:** stop pinning only if pinned streaming misses the memory bar, unpinned meets it, and it's at most 10% slower per step.
- **SageAttention 2:** in only if all of these hold:
  - it builds within the time box
  - a wheel exists, or one documented command reproduces the build
  - stage 2 at 1280×704 runs at least 15% faster per step
  - it loses no quality beyond the noise floor
- **GGUF:** a follow-up spike only if NF4 looks clearly worse than FP8.
- **What 1c needs from the preflight:** the components to leave out at load, and any Gemma 4 renames and FP8 skip patterns.

**1b. Decision (2026-10-22, approved by the user): go, `ltx25-4` only.** NF4 DiT + FP8 Gemma 4, the upstream ancestral sampler and pinned block streaming. Both 1280×704 runs met every bar (see [Phase 1a results](#phase-1a-results-measured)): 21.1 and 21.4 GiB against 2.3's 21.5, no swap, pressure 0, no kill, all output checks passed.

Out:

- **`ltx25-8` (FP8 DiT):** 24.0-24.4 GiB at 768×448 with swap in one run; the 1280×704 gate failed, so it was never run. It doesn't fit with margin.
- **SageAttention:** 9.5% faster per step in stage 2, below the 15% bar, with no wall-time gain.
- **GGUF:** out for 1c. A separate spike only if NF4 later looks clearly worse than FP8; the NF4 and FP8 pairs above don't show that.
- **NF4 Gemma 4:** the memory bar was met, so the arm never ran. Its quality stays open.
- **Unpinned streaming:** not needed. Pinning stays.

**1c. Build (implemented and GPU-verified 2026-10-22).** `ltx25-4` only: no `ltx25-8`, Sage or GGUF. Windows and Linux; on macOS the alias was first a message that points to `ltx-4` and `ltx-8`; Phase 3 replaces it with the MLX pack.

- **Config:** a per-version layer, `video_model_presets.ltx.versions.ltx25`, with `aliases: [ltx25-4]`, merged over the family block (which stays LTX-2.3). The version sets `default_repo` (a pinned revision of `unsloth/LTX-2.5-FP8`), a `null` `transformer` (the repo's own), the upsampler from the same repo, `repo_ignore` for the root single-file checkpoints, `image_crf: 18` and `sampler: euler_ancestral`. In 1c only the `diffusers.*` keys are read from a version block. The step defaults stay family-level (the version layer for them is cut and left to Phase 2), and the 2.3 aliases are unchanged. The label is "LTX-2.5 4-bit".
- **Detection** of the 2.5 repos by the `unsloth/ltx-2.5` prefix. `LTX2DFRPipeline` is not added to the pipelines that load a separate transformer, as the earlier review required.
- **Weights:** the text-encoder class in `model_index.json` selects the rename set. Gemma 4 adds two vision-embedder prefixes ahead of the Gemma 3 set; no FP8 skip-pattern change; the DiT folder needs no renames.
- **Missing components:** `processor`, `prompt_enhancer` and `duration_head` are passed as `None` when the pipeline loads, and completeness checks ignore the components the app never loads (`LTX2_UNLOADED_COMPONENTS`, which includes `diffusion_decoder`).
- **Sampler:** `backends/cuda/video/sampler.py` wraps diffusers' `LTXEulerAncestralRFScheduler` with a terminal sigma of 0 and the last timestep dropped. Stage 1 uses seed + 10000 and stage 2 a fresh scheduler at seed + 20000, and the audio scheduler shares the one generator. **Extend uses the deterministic scheduler** whatever the version configures, because the ancestral step would re-noise the pinned tail; the Extend seam check passed in the GPU verification.
- **Image-to-video CRF:** `backends/cuda/video/image_crf.py` re-compresses the conditioning image at the version's `image_crf` (18) before the app's fit to the clip size, as upstream does. The 2.3 aliases keep diffusers' CRF 33.
- **Memory:** the q4 estimate (21.08 GiB) matches the measurement and carries over, and the estimate and the download plan read the version's sources.
- **Tests and docs:** mocked tests for detection, config resolution, renames, sampler selection, the Extend exception and the download plan, plus new diffusers contract tests. CHANGELOG and the video docs describe the alias.

Not in 1c: the [licence notice](#licence-notice) and a prompt-embedding cache across jobs.

### Phase 2: MiniMax H3

**2a. Spike (throwaway, not merged).** A bench package outside the app, run on the RTX 3080 10 GB / 30 GB RAM machine with the 1a harness: a driver in its own systemd user unit starts one unit per row behind idle-machine gates, with a scratch data folder and Hugging Face offline mode. The go memory metric is again the system's `MemAvailable` drop, sampled from outside the unit. A watchdog kills only the spike's own unit, and contact sheets are batched for one visual review at the end. Nothing is installed into the app's environment: SageAttention and the sparse kernel load from a side folder.

Weights, at pinned revisions: the official `transformer/`, `vae/` and `audio_vae/`; unsloth's NF4 Qwen3-VL-32B (shards 1-4, 50 layers used); the lightx2v 8-step v1.0 and 4-step v1.2 768p turbo LoRAs (v1.3 is reported degraded); and larryvrh's v4 LoRA. With approval (granted), the official text-encoder shards 1-11 (50.0 GiB) are fetched as a bf16 reference. Before the run, each LoRA's step count and shift are taken from its model card. Disk: about 81 GiB free once everything is down, and about 26 GiB at the end with every copy built (about 76 GiB if the reference is removed after S1).

- **S0, CPU, no weights:** on tiny random-init models, prove:
  - a 50-layer Qwen3-VL tap (final norm replaced by identity, or `tie_last_hidden_states=False`) equals the full tower's `hidden_states[50]`
  - the AdaLN fold reproduces the transformer's output bit for bit (tables computed by the same module with the same k rows)
  - the schedules: N sigmas give N − 1 forwards, with video and audio lists of equal length
  - the packed-sequence and row-timestep index sets
  - the LoRA merge for both file formats, loaded through diffusers' public LoRA loader
  - a tiny end-to-end run that uses the intended sigma lists
- **S1, text encoder:** the NF4 50-layer encoder without vision or `lm_head`, streamed through the GPU. Record load and encode time, RSS, memory drop and VRAM. The bf16 reference runs layer by layer, both prompts per layer, so the 50 GiB is read once. Its time is the cost of a bf16 fallback, and its embeddings feed one S4 clip.
- **S2, DiT:** one pass over the official transformer writes four stored copies: the lightx2v 8-step and 4-step LoRAs merged, larryvrh's merged, and a plain one. In each:
  - every converted LoRA target is merged in fp32, including the refiner, the output norm and (larryvrh) the AdaLN projection, before the fold
  - AdaLN is folded into per-step tables, for the cards' schedules and, as insurance, video shifts 3-12 at 4 and 8 steps; a GPU check compares folded and unfolded blocks
  - block linears are NF4, `adaln_proj` is excluded, and the token refiner stays in its checkpoint dtype

  Then measure the memory drop, VRAM and s/step at about 15k (864×480) and 34k (1280×736) video tokens with SDPA, SageAttention 2 (the 1a wheel, transformer blocks only) and Spark-H3 sparse attention. A repeated forward shows whether runs are deterministic. Pinning: H3's tensor sizes aren't powers of two, and PyTorch rounds each pinned allocation up to one, which would add about 70% to the pinned DiT and about double the pinned text encoder (*est.*). LTX's sizes happened to be exact. The spike pins without the rounding, measures the rounded and unpinned modes where the rules need them, and runs the text encoder unpinned if the setting doesn't work.
- **S3, decode:** diffusers' decode already tiles (256 px, 64 overlap) and chunks in time (5 latent frames plus 2 overlap, as trained). At 1280×736, compare it with the same chunks decoded one at a time and moved off the GPU as they finish, for frames, peaks and time, with the cold read of the fp32 VAE included. A temporal chunk smaller than the trained one would change the output, so it isn't tested.
- **S4, end to end:** t2va at 864×480 and 1280×736, 124 frames, through our own denoise loop over diffusers' schedulers and layout helpers. diffusers 0.41's modular blocks can't set a per-request shift. Runs:
  - lightx2v 8-step and 4-step, and larryvrh v4
  - each attention arm at 864×480
  - a second prompt with fast motion
  - a clip from the bf16 text embeddings
  - a plain copy with the LoRA applied at run time, if memory allows

  Each clip records time and memory per phase and passes automatic checks: 124 frames at 24 fps, a stereo audio stream with sound, and no NaN or frozen frames.

Order: S0, S1, the copies, S2 at 15k, S3, the 864×480 8-step go row, then the rest. A failed S0 proof, a kill or a broken go-row output stops the spike.

Spark-H3 is the sparse kernel because it is the only one with a diffusers hook for H3 and a report on an sm_86 card (RTX 3090). comfy-kitchen's fast modes pair with their own trained LoRAs, and cicalooo's flex kernel is a ComfyUI node that would need porting.

**2a. Done (2026-10-22).** Results are below; the decision is under 2b.

#### Phase 2a results (measured)

Run on 2026-10-22 on the RTX 3080 10 GB / 30 GB RAM machine, with `MiniMaxAI/MiniMax-H3` at revision `42ed227e`. The text encoder is `unsloth/Qwen3-VL-32B-Instruct-bnb-4bit`; Qwen3-VL-32B is byte-identical to H3's text encoder. The turbo LoRA is `lightx2v/Minimax-h3-Turbo` (8-step v1.0, 768p), merged at scale alpha/rank = 0.0625. M is the system `MemAvailable` drop. Every row ran with no swap and no oomd or kernel kill unless noted.

- **Go row, 864×480, 5 s, 124 frames, stereo audio, 8 turbo steps (8 DiT forwards), SDPA:** 276 s cold, 274 s on a rerun. M 16.9 GiB.
- **4-step (4 forwards):** 207 s with the same RAM and VRAM. It saves only about 25% of the time.
- **DiT:** 16.9 s per forward at 15k tokens with SDPA (14.4 s with SageAttention). GPU reserved 6.3 GiB. Video is deterministic: a repeat forward is bit-identical.
- **AdaLN fold:** exact. The table equals stock modulation bit for bit on real blocks 0 and 49. The 13.0B `adaln_proj` parameters become tables of about 0.14 GiB per schedule (the app's stored copy holds the tables of both schedules and every conditioning mode: 24 keys, about 0.7 GB). Building the folded NF4 copy took 11.6 min with M 8.3 GiB.
- **Text encoder, unsloth NF4, 50 layers:** about 1.2 s to encode a prompt, M 16.6 GiB, GPU 2.1 GiB. The mean cosine against bf16 is 0.996 and 0.993. A clip from bf16 embeddings looks equivalent. The bf16 layer-streamed fallback costs about 2 min per prompt.
- **Pinned memory:** PyTorch rounds pinned host allocations up to a power of two, so H3's DiT tensors pin about 1.74× their size (15.6 against 9.0 GiB). `pinned_max_round_threshold_mb:1` in `PYTORCH_ALLOC_CONF` removes this, as measured.
- **VAE decode:** chunk by chunk with the default 256/64 tiles is bit-identical to stock decode and fits. Peak 5.0 GiB at 864×480 (24 s) and 5.5 GiB at 1280×736 (44 s). Stock decode runs out of memory at 1280×736 on the whole-video float conversion. Smaller tiles save nothing and change the output. The decode must run under `no_grad`, with the VAE's fp32 parameters (`scale1`/`scale2`, norms) kept in fp32.
- **SageAttention:** 15.2% faster per step, but PSNR is 26.6 dB against SDPA, below the 30 dB bar. Out.
- **Spark-H3:** untested, because its PyPI install failed. (M1 built it from source: faster, but it fails the quality bar and is not shipped; see [Phase 2 M1 results](#phase-2-m1-results-measured).)
- **1280×736 (34k tokens):** the DiT runs out of GPU memory with both SDPA and Sage. Not offered yet. (M1 showed that the feed-forward up-projection, not attention, is what overflows; chunking it fixes the size.)
- **LoRA applied at run time instead of merged:** 297-301 s and 19.9 GiB, with a visibly different result. Out.
- **larryvrh v4 LoRA:** comparable, not clearly better.

**2b. Decision point.** Go if an 8-step 864×480 run (any attention arm) finishes 5 s with audio within 330 s of cold-start wall time (300-330 s is marginal), and meets the 1b bars:

- a system memory drop of at most 24 GiB
- swap at most 0.25 GiB
- no oomd or kernel kill
- user-session pressure at or below 20%
- every output check passes

If only the 4-step run meets them, the go is "4-step only", provided the review accepts both its clips (including the fast-motion one). The end review can veto a go. Decide then:

- **Variants:** t2va only (i2v through fl2va later). 8 steps and/or 4 steps, by the same bars and the review.
- **Attention:** decided per size tier. At 864×480, from the 15k runs:
  - SageAttention 2 if it's at least 15% faster per step and keeps PSNR ≥ 30 dB and SSIM ≥ 0.90 against SDPA (else the review decides)
  - Spark-H3 if it's at least 25% faster than the best dense backend, the review sees no loss, and one command installs it

  The 34k benchmark decides the 1280×736 tier by the same speed bars, with a 1280×736 Sage clip for quality if Sage didn't qualify at 864×480.
- **Turbo LoRA:** lightx2v (8-step v1.0, 4-step v1.2). larryvrh v4 only if the review rates it clearly better on both prompts.
- **Merged or plain:** a plain copy with the LoRA applied at run time, instead of merged copies, only if all of these hold:
  - it's at most 5% slower per step
  - its denoising memory drop exceeds the merged copy's by no more than the LoRA's size plus 0.25 GiB
  - it passes every bar
- **Pinning:** pinned without rounding, unless it misses the memory bar where unpinned meets it at most 10% slower.
- **Text encoder:** unsloth's NF4 unless the review sees a change against the clip made from bf16 embeddings. The fallback is the bf16 encoder streamed layer by layer, at the cost S1 measured. The embedding cosine is informational.
- **1280×736:** offered whenever it fits in memory with no swap and no kill, with no time limit; the expected time is shown so the user decides (see the 2b decision below).
- **Decode:** diffusers' decode, unless decoding chunk by chunk lowers the 1280×736 GPU peak by at least 1 GiB with identical frames.
- **2d inputs:**
  - the cost of building fold tables per schedule (the copy drops `adaln_proj`, so a schedule not built in needs the bf16 AdaLN weights again)
  - the pinning fix for `stream_blocks`: the setting is appended to one combined `PYTORCH_ALLOC_CONF` string, since setting it can mask `PYTORCH_CUDA_ALLOC_CONF`

**2b. Decision (2026-10-22, approved by the user): go at 864×480.** The user's review the same day widened the build to full parity with LTX (see 2d).

- **Shipped variant:** the 8-step lightx2v LoRA merged into one folded NF4 copy, the unsloth NF4 50-layer text encoder, SDPA, pinned streaming without rounding, and chunk-by-chunk decode. The go row met every bar (see [Phase 2a results](#phase-2a-results-measured)).
- **4-step:** first left out (about 25% faster at the same memory, with rougher output), then shipped after the review: the steps control offers 4 and 8 from the one stored copy. M1 confirmed 224 s against 293 s for a 5 s clip at 864×480, with the same scene and slightly softer.
- **Sizes:** the LTX size table with its notes, `xl` included, instead of hiding what does not fit. The memory note and the expected time in the UI and CLI let the user decide; nothing is blocked. 1280×736 fits a 10 GB card once the feed-forward layers are chunked (M1).
- **Out:** SageAttention, Spark-H3, larryvrh v4 and a run-time turbo LoRA. The turbo LoRA stays merged into the stored copy; user LoRAs are applied at run time on top of it.

**2c. Consolidate the shared weight code (refactor, no behaviour change).** This lands after the decision and before any H3 code, as a short series of commits on the LTX-2.5 branch. Its aim is narrow: the weight code H3 would otherwise copy gets one home. Re-checking the code trimmed the first draft of this table:

| Drift | Decision |
|---|---|
| The video NF4 pre-pack is hard-wired to `transformer_blocks.` (`backends/cuda/video/weights.py`) | **Do.** Move it into `backends/cuda/quant.py` as `nf4_linear_names(model, *, prefix, exclude=())`, `pack_nf4`, `swap_in_linear4bit`, `place_nf4` and `nf4_storage_format()`. H3 packs the same way but must leave `adaln_proj` out |
| Images quantize NF4 at load through diffusers' `BitsAndBytesConfig` | **Drop.** It is a different mechanism with nothing to share |
| Two stored-copy entry points, `image_model_loader.store_copy` and `video/weights.ensure_prepared` | **Partial.** `utils/stored_quant.py` gets `ensure_stored_copy(target, manifest, write, *, reuse=(), flush=False)` and `file_identities(paths)`; `ensure_prepared` becomes a wrapper and H3 calls the core with its own manifest (fold schedule, LoRA). The image path stays: it warns and falls back, checks cancel and flushes, so merging it would only add flags |
| Rename tables with their own loops (FLUX.2, Krea 2, LTX-2.3, Gemma, macOS) | **Drop.** There are three rename semantics, so one helper doesn't fit; each loop is a few lines next to its table, and H3 adds no table |
| `cuda/video/backend.py` at 1,488 lines | **Defer** to [2e](#phase-2-minimax-h3). H3 needs nothing LTX-specific from it, and the 2d registry points at the backend class wherever its code lives. In 2d, only leaf helpers move: when H3 first needs a model-neutral helper that calls no other patched name, 2d moves it to a shared module and the backend imports it back under the same name. A caller that has to move takes its tests, repatched on the shared module, in the same commit |
| `_to_uint8`, `chunk_tokens` and the taking-turns estimate | **Drop.** They are already in shared modules. Extending `chunk_tokens` to 4-D inputs is a 2d decision |
| Duplicate `_numel` and element-count loops | **Partial.** `math.prod` replaces the three loops. The two block/rest counters stay, because their name predicates differ |

Order: golden tests first (the prepared-copy manifest bytes, the FP8 and NF4 copy layouts with a fake bitsandbytes on the CPU, and the element counts), together with a test guard that fails any test reaching the Hugging Face Hub download functions unmocked; then the element counts, the NF4 move, the stored-copy core and the developer docs. Each commit passes `make check` on Linux, and the branch tip passes macOS CI. Every test uses mocks or tiny synthetic tensors, with no real weights, no Hugging Face cache and no downloads. CI has no CUDA, so the CUDA NF4 test is run by hand on a CUDA machine after the NF4 move.

LTX-2.3 and 2.5 behaviour, outputs, stored-copy layouts and manifests stay byte-for-byte the same, so existing stored copies aren't rebuilt. Image copies, SeedVR2 and Ideogram are unchanged.

**2d. Build.** Scope (user decision, 2026-10-22): full parity with LTX (text to video, image to video, first and last frame, Extend, upscale with review, user LoRAs, the LTX durations and sizes) on Windows and Linux, plus macOS in the same work, with only the Mac test left for afterwards.

1. **A video-family seam,** a refactor with no behaviour change:
   - a frozen `VideoFamily` and `VideoCapabilities` read from `video_model_presets.<family>`
   - a `(platform, family) → factory` registry, with per-family `supports_video_*`
   - `align_frames(frames, FrameRule)`, which also covers `17n + 5`; the rule is sent to the frontend, so `videoDuration.ts` stops recomputing it
   - one `settle_video_steps` with a per-family cap
   - per-family memory estimators

   The LTX tests are the guard.
2. **`backends/cuda/video/h3/`,** in atomic modules:
   - `frames.py`
   - `schedule.py` (schedules, row sets, row timesteps)
   - `text_encoder.py` (50-layer NF4 from unsloth's repo, plus the vision tower for keyframes; no embedding cache, since encoding a prompt takes about a second)
   - `adaln.py` (fold)
   - `lora.py` (turbo merge, user LoRA routing)
   - `conditioning.py` (keyframes, Extend tail)
   - `stored_copy.py` (the folded 4-bit copy)
   - `sampler.py` (own loop over the diffusers scheduler, cancel, no preview)
   - `decode.py` (chunk by chunk, under `no_grad`)
   - `audio.py`
   - `sources.py`
   - `backend.py`

   There is no `attention.py`: M1 found that the feed-forward up-projection overflowed, not attention, so the feed-forward layers run through `chunk_tokens`.

   H3 builds on 2c's NF4 pre-pack and stored-copy routine. `chunk_tokens` needed no 4-D extension. Decisions carried in from 2b:
   - **Sizes.** The LTX size table with its notes, `xl` included. The UI and CLI show the memory note and the expected time.
   - **One stored copy.** The 8-step LoRA is merged into the folded NF4 copy. The fold bakes in the distinct timesteps of a forward, so the copy holds the tables of both schedules (4 and 8 steps) and every conditioning mode (24 keys); a changed shift or schedule needs a rebuild from the bf16 AdaLN weights.
   - **Pinned allocator setting.** `configure_allocator()` in `backends/cuda/memory.py` adds `pinned_max_round_threshold_mb:1` for all CUDA backends, merged into one string written to both `PYTORCH_CUDA_ALLOC_CONF` and `PYTORCH_ALLOC_CONF`, since torch 2.14 reads the first. A `PYTORCH_CUDA_ALLOC_CONF` the user set is left exactly as written.
   - **Decode.** H3's VAE tiles and chunks itself, so decode needs no tiling attributes. The fixes are `no_grad`, the VAE's fp32 parameters kept in fp32, and decoding chunk by chunk with the default 256/64 tiles.
   - **Text encoder.** The NF4 weights come from `unsloth/Qwen3-VL-32B-Instruct-bnb-4bit`. The official `MiniMaxAI/MiniMax-H3` repo is ungated, so no access step is needed.
3. **A download plan** with allow and ignore patterns for the transformer, VAEs, tokenizer and processor, the first four shards of the NF4 text encoder and the turbo LoRA; the macOS plan lists the INT8 transformer, the `FL2VA` components and the turbo LoRA file by file.
4. **Config, docs, CHANGELOG, frontend build and mocked tests.**

**2e. Split the LTX backend (refactor, no behaviour change).** Its own series of commits after 2d and before the model-naming cleanup, kept apart from any user-facing rename. `backends/cuda/video/backend.py` becomes LTX modules under `backends/cuda/video/ltx/` (state, load, place, sample, extend, decode) beside `h3/`, with `DiffusersVideoBackend` as a thin class. Conditions:

- `_check_system_memory` keeps `stacklevel=4`, so its warning still names the caller of the public load.
- The `importlib.reload` test of the backend module stays: importing the backend sets no allocator environment.
- Every test patch target moves in the same commit as the function it patches, and callers move with their helpers.
- The Hub guard from 2c is the backstop for a patch that silently stops applying.
- The golden tests from 2c and the LTX tests pass unchanged apart from patch targets.

#### Phase 2 M1 results (measured)

GPU diagnosis and parity measurements on 2026-10-23 on the RTX 3080 10 GB / 30 GB RAM machine, through the real CLI on the shipped code. No oomd or kernel kill in any run.

- **Memory finding (R23):** the 1280×736 and 345-frame overflows were the feed-forward up-projection (`ff.net.0`), not attention. SDPA is flash on sm_86 and grows linearly. Chunking the feed-forward layers (`chunk_tokens` at 4096 rows, no time cost) brings 34,554 rows to 6.42 GiB allocated / 7.25 GiB reserved (8.02 / 8.70 without), and 42,560 rows to 7.39 / 7.93 where the unchunked forward ran out of memory. Chunking the q/k norms or the attention queries as well added nothing. The next limit is the rotary temporaries, +1.72 GiB at 42,560 rows.
- **End to end with the fix:** 864×480×345 (14.4 s) completes in 790 s with 8.9 GiB of GPU in the denoise and 6.2 GiB in the decode; 1280×736×124 completes in 597 s with 7.84 GiB (8.98 without) and no allocator retries. The decode does not grow with the frames.
- **Parity rows (864×480, 8 steps, seed 42):** text to video 279 s against an estimate of 277 s; image to video 355 s (the first vision load swapped 2.2 GiB); first and last frame 311 s, reaching the last image at 30.5 dB and dissolving between two different scenes; Extend by 68 frames 243 s; 4 steps 224 s; durations from 73 to 345 frames completed. First use: the 4-bit copy took 236 s and 12 GB.
- **Seam of a plain Extend:** a soft pass. The jump ratio is 2.03, ΔE76 across the seam 1.21, and the seam PSNR 26.6 dB against the clip's own median of 29.0 and p10 of 27.3. The source prefix is bit-identical and the audio joins at the video's length, with a dip of about 15 dB for about 80 ms. The tail-guide-row fallback is not built.
- **A user LoRA breaks Extend:** with the HF-hosted anti-burn LoRA at 0.5 the new part cuts to a different shot at the seam (7.4 dB). The same LoRA is coherent in text to video. Only one LoRA is tested, so it is not known whether this holds for user LoRAs in general.
- **Upscale ×2:** 346 s for a 5 s clip at 768×448 (half-size render 149 s, SeedVR2 190 s: 4.4 s per megapixel-frame, against a provisional 8.0). Review was not run, because it needs a terminal.
- **AdaLN LoRA delta:** the arithmetic is exact, but the bf16 tables keep only 25 to 39% of a tiny delta, which is the same limit as a stock bf16 model.
- **Spark-H3 (N15):** built from source at tag `comfyui-v0.1.5`. It is 20% faster per forward at 15,499 rows and 37% at 34,554, but the clip differs from SDPA at 22.5 dB PSNR and 0.735 SSIM against a bar of 30 dB and 0.90. Not shipped, and nothing was left installed.
- **Estimator refits from the same runs:** the GPU slope is 151 KiB per row with the feed-forward chunked (200 before), the desktop headroom 1 GiB (2.5 before), the user-LoRA GPU term 2.7 times the file, SeedVR2's system phase about 12 GiB for a video (21 estimated) and the seconds per megapixel-frame 4.4. Time constants held within ±5%.

#### Phase 2 M2 results (measured)

The final GPU verification on 2026-10-23 on the same RTX 3080 10 GB / 30 GB RAM machine, through the real CLI and the Web UI on the tip of the branch, against baselines of the commits before it. No oomd or kernel kill in any run.

- **Success criteria:** SC-1 passes. SC-2 fails strictly (see the next point) and passes by the noise floor. SC-3, SC-4, SC-5 (a)-(g), SC-6 to SC-9 and SC-10 to SC-13 (the Linux part) pass; SC-5 (h) and SC-14 are recorded, not judged.
- **Extend seam accepted:** the median jump ratio over three Extends of different clips is 1.12 (X1 1.12, X2 1.91, X3 1.11), against a bar of 2.1. Every seam PSNR sits at or near its own clip's 10th percentile, and the source prefix is bit-identical. M1's 2.03 was a single clip. The tail-guide-row fallback stays unbuilt.
- **LTX is not bit-reproducible on CUDA:** no LTX video frame hash matches its baseline, but two baseline runs of the same command differ from each other by as much (39.9 dB luma PSNR) as the baseline differs from the tip (39.9 to 40.6 dB). The LTX video differences fall within LTX's own run-to-run nondeterminism, so a frame-hash bar cannot be met by any code on this machine. The image paths (Z-Image Turbo, SeedVR2) are bit-identical. LTX generation time on the tip is equal or faster on every row.
- **Reproducibility:** text-to-video H3 runs are bit-reproducible (seven seed-42 runs identical). Image-conditioned H3 runs (image to video, first and last frame, an Extend of an image-to-video clip) are not, between two identical runs; the clips show the same scene. The likely cause is `cudnn.benchmark`; not investigated.
- **A user LoRA and Extend:** the anti-burn LoRA breaks the Extend seam on M1's source clip (jump ratio 27, seam PSNR 7.2 dB, reproduced on the tip) but not on the other clip tried (1.97), and the realism LoRA did not break it (1.94). The cut depends on the clip, so some user LoRAs can break an Extend on some clips.
- **Extend audio dip:** 5 to 10 dB on every Extend, against 24.7 dB on M1's clip; 0 to 4 of 15 windows of 20 ms are 6 dB or more down.
- **Spark-H3 dropped:** across four prompt pairs it is 3 to 5.5% faster per clip, uses 2.7 GiB more GPU memory, and its output differs from SDPA (19.5 to 24.8 dB PSNR). Not shipped.

#### H3 upscale after M2

Hand testing after M2 found two problems with H3's `--upscale 2` on Windows and Linux.

- **Out of GPU memory at `l` and `xl` (fixed):** SeedVR2's 1024-pixel VAE tiles ran out of memory on the 10 GB card above 768×448. A video above that size now encodes and decodes in 512-pixel tiles; the tile only affects the VAE, not the upscaler's transformer. Measured on the card: `s` 331 s at 7.1 GB of GPU, `l` 486 s at 6.4 GB, `xl` 711 s at 6.6 GB (the `xl` run overlapped another job, so it is unverified). The time estimate was refit to `frames × (0.97 + 1.48 × megapixels)` seconds.
- **Painted look (fixed):** the shared `seedvr2.softness` of 0.75 shrank each half-size frame to about a sixth of the target before SeedVR2 saw it, so SeedVR2 invented most of the detail: rock strata became rounded stones that weren't in the render. A separate `seedvr2.video_softness`, 0 by default, now feeds the frame as rendered, and the stage-2 pass is about 2× faster. The image upscale keeps 0.75.
- **Still clearly worse than LTX (resolved by the latent route):** with softness 0 the clip kept the render's shapes, but the texture was busy and not anchored to the source, and it jumped at every 5-frame batch boundary (a 50 to 70% larger frame-to-frame change there). That came from SeedVR2 itself, a one-step generative upscale in batches, and no setting removed it. H3 now upscales in latent space like LTX, with no SeedVR2: the half-size latents are enlarged 2× by LBH-123-AI's 3D latent upscaler (Apache-2.0 weights, MIT network code, 0.69 GB), then refined at full size with the last 2 forwards of the 8-step schedule, in overlapping 22-latent-frame windows (the user LoRAs applied, the start and end images held as full-size keyframe rows, the stage-1 sound kept), and decoded once. A spike on the 10 GB card measured the refine at 94 s and 4.9 GiB for `l` and 181 s and 6.6 GiB for `xl`, with no seam between windows; the estimator puts a 5 s `l` clip at about 6½ min. The SeedVR2 video engine itself is untouched, and H3 no longer reads `seedvr2.video_softness`. GPU-checked on the card through the CLI: `l` 5 s in 369 s at 6.0 GiB, `xl` 5 s in 515 s and `xl` 10 s in 1,037 s at 6.9 GiB, first and last image across 5 windows at `l` 10 s (both ends held, no seam), a user LoRA and 4 steps, all within 10% of the estimate; a 20 s `xl` clip is refused by the estimate. Still to do by hand in the Web UI: cancel during the refine, the review pause, the first-use download row, and the Models page count and delete.

**Follow-ups the user chose not to do on this branch:**

- Try LTX's 0.25 s lead-in blend on Extend to remove the 5 to 10 dB audio dip. This would reverse the earlier decision not to blend.
- Make `ziv-video` exit non-zero after a failed batch. It exits 0 today, so a script or CI wrapper cannot tell a failed batch from a good one.

**Built on this branch now:** a Last frame control in the Web UI for H3, for first and last frame parity with the CLI.

**Kept gap:** H3 has no upscale on macOS. The Mac video upscale plan is [video-upscale-existing.md](video-upscale-existing.md).

### Phase 3: macOS (tested afterwards on a Mac)

- **LTX-2.5 alias (implemented, verified on a Mac 2026-10-25):** `ltx25-4` points at `MarceloFSO/ltx-2.5-mlx-q4-fast` at a pinned revision, an ungated mirror of `dgrauet/ltx-2.5-mlx-q4` with the distilled files only (dgrauet's pack is gated). No `ltx25-8`. The mirror's hashes are below.
- **2.5 sigmas and the ancestral loop** in `_generate_one_stage` (implemented): text to video and image to video use the ancestral loop, and Extend stays deterministic.
- **Gemma 4 handling (implemented):**
  - skip Gemma 3 when the pack has a root `text_encoder.safetensors`
  - refuse a 2.5 folder whose encoder isn't at the root
  - read the 2.5 encoder and decoder files in the estimator
- **Streaming for `ltx25-8`** on 32 GB Macs (not built: there is no `ltx25-8`).
- **Optional:** `make update-ltx` to v0.17.0.
- **H3 on a Mac (implemented, verified on a Mac 2026-10-25):** a vendored copy of the Argus MLX port (`packages/minimax_h3_mlx`) behind the video-family seam. It does text to video, image to video, first and last frame, Extend (as image to video from the last frame, with a weaker seam), user LoRAs (combined with the turbo LoRA) and stereo audio; upscale is not offered, and text to video reports no per-step progress (the image modes do). The weights are the INT8 transformer, the `FL2VA` components and the turbo LoRA (about 107 GB), plus a one-time 4-bit text-encoder copy (about 15 GB). Memory and time started as estimates calibrated on an M4 Pro (about 15 GiB and 23 minutes for a 5 s clip at 768×448); on an M5 with 32 GiB text to video peaked at 12.5 GiB and took 15.1 minutes, and the staged image modes peak at about 14 GiB. The candidates in [Phase 3 evidence](#phase-3-evidence-macos) stay the fallbacks.

Files of `MarceloFSO/ltx-2.5-mlx-q4-fast` at `281bf05bd23207d6feef708fc2e270f84ba992b5` (size in bytes and LFS sha256; all `.safetensors` except `tokenizer.json`). The small JSON configs are not LFS and are not listed. The distilled files are sha256-identical to `dgrauet/ltx-2.5-mlx-q4`.

| File | Size (bytes) | sha256 |
|---|---|---|
| `audio_vae` | 106510725 | `b91400bd1cd52ba3f4df54dc6214526b1b01de1bf5bf21544d29738a926dcc7a` |
| `connector` | 4032375032 | `ad25a36abf03315a7adf1d1f7340c538efe85bb35fa7ee3130661d56020ba461` |
| `duration_head` | 3808462 | `e3ec9ff07d9aadd6e3073af63a7d66612c029b6e46ecb21f2d5ee5f43cd53c29` |
| `spatial_upscaler_x2_v1_0` | 995745333 | `df7de2dc9e32815113d8ba876dcd1e8a4a91c0ea1530b840430bf3220de90294` |
| `temporal_upscaler_x2_v1_0` | 261945827 | `31c660bba724416b3416b63802ccb5a86e835511cdade35a3b4d73c9cfdef935` |
| `text_encoder` | 10563078719 | `687a14a73d93814da5942dd55d1cf095b549c8cec31553bbaf280797492049d1` |
| `tokenizer.json` | 32169626 | `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` |
| `transformer-distilled` | 11320034990 | `413479747a7ecd39c71460c3b0e4c872399c64fdaff85f4ef2a28c435a45f21d` |
| `vae_decoder_av` | 834306031 | `6852b02be0e05fad78d87f7d74d58ec578846e505d108c1dd6b5455883d1f840` |
| `vae_decoder_conv` | 814351176 | `fd250ca3bdeb4ce67b5343d62a45883924faf5314aa60a870eb7c575f5519c80` |
| `vae_encoder_av` | 637886990 | `6092784ff3de66d8f536c7975222119d5abf81c4651fc231c0d1c3f4f38bdcd9` |
| `vae_encoder_conv` | 637887026 | `34625a28efc2c447071221b9855849d9d4a90e19d5363c7347c00896f8409502` |
| `vocoder` | 258315820 | `86fd7336e7fe23e04c0eacd3d1949de1f38bb7c8ff8241042e8166769c2cd28b` |

## Alternatives considered

- **Both families at once.** It would double the spike work before either decision, and H3 needs the family seam while LTX-2.5 doesn't.
- **GGUF first for LTX-2.5.** GGUF has the best published quality per GB, but needs a new dependency, a loader and an untested streaming path. NF4 reuses everything we have, so the spike measured NF4 and left GGUF out.
- **Pruned community H3 DiTs instead of folding AdaLN.** They're about the same size, but change numerics, are refused by diffusers and break full-model LoRAs. Folding is exact.
- **Running H3 through ComfyUI, Wan2GP or stable-diffusion.cpp as an external backend.** The fastest current kernels live there, but this would add a second runtime and lose our memory estimates, progress and cancel. Keep it as a fallback if the spike can't reach usable speed in-process.
- **Replacing `ltx-4` / `ltx-8` with 2.5.** It would change output for every existing user in one step. Keep both, and reconsider the default after measuring.

## Open questions

1. Does NF4 hold embedding quality for Gemma 4 and for the 50-layer Qwen3-VL, compared with FP8? Still open for both: the Gemma 4 arm never ran in Phase 1a.
2. Which sparse-attention kernels run on sm_86, and can one be called from a diffusers attention processor? Answered: SageAttention and Spark-H3 (built from source in M1) both run on sm_86 and are faster, and both fail the quality bar on H3 (26.6 dB and 22.5 dB against 30 dB).
3. Is 30 GB enough under oomd for 10 s clips with pinned streaming, or does streaming need an unpinned mode? Settled for 5 s clips: pinned streaming peaks at 21.1-21.4 GiB with no swap. 10 s clips are measured for H3 (243 frames at 864×480 completes in 520 s) and for 14 s with the chunked feed-forward; for LTX they are not.
4. On a 32 GB Mac, what's the usable GPU working set for `ltx25-8`, and how do the vendored H3 port's real memory and time compare with the estimates (about 15 GiB and 23 minutes for a 5 s clip at 768×448)? The Mac test answers both.
