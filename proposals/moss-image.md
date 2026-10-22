# MOSS-Image on Windows/Linux, with a 4-step Turbo

**Status:** Proposed (2026-10-20). CUDA only. macOS waits for official MOSS-Image support in mflux.

## Problem

OpenMOSS released **MOSS-Image** on 2026-10-16: a 5B text-to-image diffusion transformer under Apache 2.0, with strong Chinese and English text rendering and structured JSON layout prompts. It is smaller than Krea 2 or Klein 9B and has a permissive license, so it would make a good addition to the shipped families.

The official checkpoints are slow, though. They need about 50 steps with CFG 3, so two transformer passes per step. Within two days the community published distilled LoRAs that bring this down to 4–8 steps at CFG 1. This proposal adds MOSS-Image to the diffusers backend with a Turbo default and the full SFT model as a second option.

## Evidence

### The official checkpoints (checked 2026-10-20)

`OpenMOSS-Team/MOSS-Image-SFT` @ `2c79544…` is ungated and Apache 2.0. Its `model_index.json` declares `MOSSImagePipeline`:

| Component | Class | Size on disk |
|---|---|---|
| `transformer/` | `MOSSImageTransformer` (custom; 42 layers, dim 3072, 24 heads, text_dim 4096) | 10.2 GB BF16 (~5.1B params) |
| `text_encoder/` | transformers `Qwen3_5Model` (32 layers, hidden 4096, hybrid linear/full attention) | 16.8 GB BF16 (~8.4B params) |
| `vae/` | diffusers `AutoencoderKLFlux2` with the small FLUX.2 decoder (`decoder_block_out_channels` 96…384) | 0.25 GB |
| `scheduler/` | `MOSSImageScheduler` (custom, a `FlowMatchEulerDiscreteScheduler` subclass, shift 5.0) | — |
| `tokenizer/` | `Qwen2Tokenizer` | — |

- **Custom code.** The transformer, scheduler and pipeline live in one self-contained file, `pipeline_moss_image.py` (440 lines). It imports only torch, diffusers and transformers. The official quick start loads it with `custom_pipeline=…, trust_remote_code=True`.
- **Recommended settings.** 50 steps, CFG 3, shift 5, Euler. Block 22 is skipped in the negative branch (`uncond_skip_layers=(22,)`), and `max_sequence_length` goes up to 4096. Dimensions must be multiples of 16. The model card targets 1–2 MP and says it extrapolates from 0.25 MP to 4 MP.
- **Precision.** The transformer and text encoder run in BF16. Noise, RoPE, CFG, the Euler updates and the VAE run in FP32.
- **Base vs SFT.** `MOSS-Image-Base` is the raw pretrained model, meant for LoRA training. SFT is the supervised fine-tune and is the one to ship. Base is not offered.

### Our stack

- **diffusers 0.41.0** has `AutoencoderKLFlux2`. **transformers 5.19.0** has `Qwen3_5Model`; both import cleanly. Without the `fla` and `causal_conv1d` kernels, Qwen3.5's linear attention falls back to torch. That only affects text encoding, once per prompt.
- `detect_image_model` maps `_class_name` to a family (`_CLASS_NAME_MAP`). `MOSSImagePipeline` currently maps to `unknown`.
- `backends/cuda/image/ideogram4.py` is the existing pattern for a family whose loading and generation do not fit the shared diffusers path: its own `load_pipeline`, `generate` and `save_quantized`, dispatched by family in `backend.py`.
- The stored-quant flow (`quant.py`: NF4 via bitsandbytes, FP8 storage with BF16 compute, `empty_diffusers_model` + `fill_weights`) works on any `ModelMixin`. Both `MOSSImageTransformer` and the Qwen3.5 encoder are standard enough to go through it.
- **LoRAs.** diffusers' `load_lora_weights` needs a pipeline with a LoRA mixin, and the MOSS pipeline has none. `lokr.py` already shows how we apply adapters to modules ourselves.

### Few-step distillations (checked 2026-10-20)

All of these are community releases under Apache 2.0, and none is official. One blog post (2026-10-17) says an official distilled model is in production.

| Repo | What | Steps / CFG | Fit |
|---|---|---|---|
| `akhaliq/moss-image-turbo` @ `2ded95c…`, file `moss_image_sft_4step_v2.0.safetensors` | Rank-128 LoRA on SFT (data-space DMD), 1.19 GB, on `q/k/v/o/ffn_up_gate/ffn_down` of every block | 4 at 1024², 6 at ~1536 px, 8 at 2048²; CFG 1, no negative branch. A 3-step draft file also exists | **Chosen.** The LoRA has to stay unmerged and run in FP32 (merging it into BF16 loses fine text). About 5.6 s per 1024² image on an A100 |
| `darask0/MOSS-Image_4-step_adapter_lora` @ `c88f33b…` | 4-step LoRA (DMD + DMAD) | 4 / 1 | Alternative if Turbo disappoints in the A/B test (step 5) |
| `lodestones/MOSS-Image-SFT-8-steps-dmd` | Full distilled transformer | 8 / 1 | No: has to run in FP32 (a 20 GB transformer). Casting it to BF16 discards the distillation |
| `lodestones/Kromoss` | 8-step LoRA on **Base** | 8 / 1 | No: built on Base, not SFT |

The Turbo LoRA uses kohya-style keys: `diffusion_model.layers.{i}.proj.{name}.lora_down.weight`, `.lora_up.weight` and `.alpha`. Its helper (`apply_lora.py`) wraps each target linear as `y = W x + (alpha/rank) · up(down(x))`, with the LoRA in FP32 and its output cast back to BF16.

### Reported consumer GPU timings

- **RTX 3060 12 GB:** about 24 s at 1024² (8-step DMD on an INT8 transformer, ComfyUI).
- **RTX 3050 8 GB with 32 GB RAM:** under 40 s at 2 MP (Turbo, 16 steps).

None of these were measured on our stack. Step 5 measures them.

### Memory on a 10 GB GPU

| Part | BF16 | FP8 (q8) | NF4 (q4) |
|---|---|---|---|
| Transformer | 10.2 GB | ~5.1 GB | ~2.9 GB |
| Text encoder | 16.8 GB | ~8.4 GB | ~4.8 GB |
| Turbo LoRA (FP32, unmerged) | 1.19 GB | 1.19 GB | 1.19 GB |

At q4 or q8 the transformer and the LoRA fit on the GPU. The text encoder is bigger than the transformer, so it runs once per prompt and is then offloaded, the same way the NF4 Ideogram 4 encoder (4.7 GiB) is today. The full BF16 download is 27 GB. Quantizing it on a 32 GB machine is the kind of load that triggered systemd-oomd before, so the measurements use q4 and q8 and treat BF16 as optional.

## Change

### 1. Vendor the pipeline file

Copy `pipeline_moss_image.py` at the pinned SFT revision into `packages/moss_image/`, with the repo's `LICENSE`. Add a `make update-moss` target, the same way `make update-ltx` and `make update-seedvr2` work, and add the rule to AGENTS.md: never edit by hand.

The backend imports `MOSSImageTransformer`, `MOSSImageScheduler` and `MOSSImagePipeline` from there. Nothing is loaded with `trust_remote_code`, and no code from the Hub runs at load time.

### 2. A `moss` family on CUDA

- **Detection.** Add `"MOSSImagePipeline": "moss"` to `_CLASS_NAME_MAP`.
- **Module.** A new `backends/cuda/image/moss.py`, modeled on `ideogram4.py`:
  - `load_pipeline(model_path, dtype, bits, *, prequantized)`: builds the transformer through `empty_diffusers_model(..., model_class=MOSSImageTransformer)` + `fill_weights`, at BF16, FP8 or NF4 through the stored-quant flow. Loads the Qwen3.5 encoder at NF4 (q4) or FP8 (q8) and moves it to the CPU after encoding. Keeps the VAE in FP32.
  - `generate(...)`: calls the pipeline with the variant's steps, guidance, `uncond_skip_layers` and the size. Hooks the existing step callback for progress, skip and live preview.
  - `save_quantized(model, path)`: saves `moss@q4` / `moss@q8`. The Turbo LoRA is never baked into these copies.
- **Dispatch.** Family branches in `backend.py`, the same way `ideogram4` is dispatched.
- **Alias.** `model_aliases.moss: { win32: "OpenMOSS-Team/MOSS-Image-SFT", linux: "OpenMOSS-Team/MOSS-Image-SFT" }`, pinned to `2c79544…`, and `model_alias_families.moss: moss`. There is no `darwin` entry (section 5).

### 3. Turbo as the default variant

Add a `moss` preset with two variants, the same way `flux2_klein` has `distilled` and `base`:

| | `turbo` (default) | `sft` |
|---|---|---|
| Steps | Picked from the output size: 4 up to ~1.2 MP, 6 up to ~2.4 MP, 8 above | 50 |
| Guidance | 1.0, locked (warns and ignores other values, like `ideo`) | 3.0 |
| Negative prompt | no | yes |
| Negative-branch skip | — | block 22 |

- **The Turbo LoRA.** Pinned in config (`moss_turbo: {repo, revision, filename}`) and downloaded through the normal Hugging Face cache. A new `backends/cuda/image/moss_lora.py` applies it at load time as unmerged FP32 adapters, wrapping each target `nn.Linear`, or `Linear4bit` at q4. That matches `apply_lora.py`, written as our own code rather than imported.
- **User LoRAs.** The same module applies kohya-format user LoRAs (`diffusion_model.layers.*.lora_down/lora_up/alpha`) on top, with the strength the user sets. Files that match no target warn through `warn_unmatched_lora`. LoKr waits until someone asks for it.
- **Steps.** An explicit `--steps` still wins. On Turbo, a step count other than 3, 4, 6 or 8 warns that the distillation was trained for those counts. The variant is chosen with `--variant turbo|sft` and in the Web UI's model settings.

### 4. Scope of the first version

- **In:** text-to-image only, at 0.25–4 MP, multiples of 16, with q4/q8 stored quants and live preview.
  - The preview uses the FLUX.2 latent-to-RGB factors if the 128-channel latents match Klein's. Otherwise previews stay off for `moss`.
- **Out, and why:**
  - **img2img and upscale refine** (`supports_img2img: false`, `supports_upscale: false`): the pipeline has no image input. Adding one means encoding with the FLUX.2 VAE encoder and starting part-way down the sigmas, like the Klein path. That is a follow-up once the base works.
  - **JSON layout prompts** (`supports_json_prompt: false`): MOSS's layout schema (bounding boxes, text elements) is not Ideogram's. Mapping the enhancer's JSON format and the Web UI's JSON toggle onto it is its own proposal.
  - **Prompt-enhancer hints:** the per-model hints in [prompt-enhancer-hints.md](prompt-enhancer-hints.md) should get a MOSS entry stressing quoted text and long prompts.

### 5. macOS: wait for mflux

There is no MOSS-Image support in mflux 0.22. Issue mflux-community/mflux#888 was closed by the person who opened it, who pointed to discussion #884. That thread only gauges interest: a maintainer liked the samples, but there is no PR or timeline. A community fork (`vanch007`, branch `feature/moss-image-sft`, MLX 4- and 8-bit builds) runs at about 17 s per step at 1024² on an M3 Max.

We do not depend on the fork or add a diffusers-on-MPS path. Until mflux ships it:

- **Alias.** `moss` has no `darwin` alias, so the macOS Models page does not list it.
- **Explicit model on macOS.** Passing `-m OpenMOSS-Team/MOSS-Image-SFT` detects the `moss` family and fails in the mflux backend's existing "not supported by the mflux backend" error. The message should add that MOSS-Image runs on Windows/Linux for now.
- **Re-check.** When an mflux release adds MOSS-Image, the macOS half becomes a follow-up proposal: load through mflux with `quantize`, and check whether the Turbo LoRA applies through mflux's LoRA path.

### 6. Measure before shipping

On a 10 GB NVIDIA GPU with 32 GB of system memory:

1. **Load paths.** First-use download plus q4 and q8 stored-quant creation, from a pre-downloaded cache: peak RSS, peak VRAM, wall time. BF16 only if the q8 run shows the headroom.
2. **Speed.** Seconds per step and per image at 1024² and 2 MP, for Turbo (4 and 6 steps) and SFT (50 steps), each at q4 and q8.
3. **Quality.** Turbo at q4 and q8 against SFT at q8, on a fixed prompt and seed set heavy on text rendering, including Chinese. Record whether NF4 hurts text more than FP8 does.
4. **Turbo vs darask0.** An A/B on the same set. Ship whichever reads text better at 4 steps.
5. **Vendored file.** It imports and runs on diffusers 0.41 / transformers 5.19 (it was written for 0.38 / 5.3).
6. **User LoRA.** One user LoRA from the Hub applies on top of Turbo and visibly changes the image.

Tests (no real models, as usual): family detection from a `MOSSImagePipeline` index, variant defaults and the Turbo step picker, the guidance lock warning, the LoRA key mapping and FP32 wrapper on a toy module, the stored quant leaving the LoRA out, and the macOS error.

### 7. Docs

- **Docs:** add a CHANGELOG entry and a model guide section for `moss`: Apache 2.0, CUDA only for now, Turbo vs SFT, and q4/q8 recommended on 10–12 GB cards.

## Alternatives

- **Load with `trust_remote_code`.** Rejected: it runs whatever the Hub repo holds at load time. Vendoring a pinned file costs one make target.
- **Merge the Turbo LoRA into a stored transformer.** Rejected: the LoRA author reports that merging into BF16 degrades fine text, and the merged FP32 file is 20 GB.
- **lodestones' 8-step full transformer.** Rejected: FP32-only, 20 GB.
- **SFT only, no Turbo.** Rejected as the default: 50 steps with CFG is several times slower per image than the other shipped families. SFT stays available as a variant.
- **Use the vanch007 mflux fork on macOS.** Rejected: it means depending on an unofficial fork branch, and it is slow (about 17 s per step on an M3 Max). We wait for upstream.
- **Diffusers on MPS for macOS.** Rejected: it would be the only image family on macOS outside mflux.
