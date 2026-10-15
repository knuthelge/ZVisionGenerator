# CLI Reference

## Entry Points

| Command | Description |
|---|---|
| `ziv` | Unified parent command with subcommands: `image`, `video`, `model`, `ui` |
| `ziv-ui` | Standalone Web UI launcher |
| `ziv-image` | Image generation (standalone entry point) |
| `ziv-video` | Video generation (standalone entry point) |
| `ziv-model` | Model/LoRA management (standalone entry point) |

## `ziv` — Unified Command

`ziv` is the parent command that dispatches to subcommands:

```bash
ziv image [options]    # Same as ziv-image
ziv video [options]    # Same as ziv-video
ziv model [options]    # Same as ziv-model
ziv ui [options]       # Same as ziv-ui
```

Running `ziv` with no subcommand prints command-discovery help and exits without starting a server. Use `ziv ui` or `ziv-ui` when you want to launch the Web UI.

Run `ziv --help` for available subcommands, or `ziv <command> --help` for command-specific options.

## `ziv ui` / `ziv-ui` — Web UI Launcher

The base package install includes the Web UI runtime, so `ziv ui` and `ziv-ui` work after a normal `uv tool install z-vision-generator` or `uv sync`. If a local environment is missing required Web UI packages, the launcher exits with repair guidance for the base install instead of pointing users at a second install step.

| Argument | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Host interface to bind the local Web UI server to |
| `--port` | `8080` | Preferred local port; the app chooses the next available port if this one is busy |
| `--no-browser` | off | Start the server without opening a browser tab |

The server only answers requests addressed to an IP address, `localhost`, the `--host` value, or (when bound to `0.0.0.0`, `::` or a LAN IP) this machine's hostname in both bare and `.local` form, and rejects state-changing requests from other sites. This protects against DNS-rebinding and cross-site attacks from pages open in the same browser. To reach the UI through another hostname, list it in `ZIV_UI_ALLOWED_HOSTS` (comma-separated), for example `ZIV_UI_ALLOWED_HOSTS=studio.lan ziv-ui --host 0.0.0.0`.

## `ziv-image` — Image Generation

| Argument | Default | Description |
|---|---|---|
| `-m`, `--model` | *(required)* | Model name, local path, alias, or supported HuggingFace repo ID |
| `-q`, `--quantize` | `None` | Quantization level: `4` or `8` |
| `--scheduler` | model-aware | Scheduler name (e.g., `beta`) |
| `--lora` | `None` | LoRA specifier: `name:weight,name2:weight2` (weight optional, default 1.0) |
| `--prompt` | `None` | Inline prompt; overrides `--prompts-file` (mutually exclusive with `--json-prompt`) |
| `--json-prompt` | `None` | Inline structured JSON caption for a one-off generation (mutually exclusive with `--prompt`). Takes a JSON value, must be a valid JSON object, skips random-choice `{a|b|c}` expansion, is passed verbatim, and overrides `--prompts-file` |
| `-p`, `--prompts-file` | `prompts.yaml` | YAML prompt file path |
| `-r`, `--runs` | `1` | Number of generation runs |
| `-s`, `--size` | `m` | Size preset: `xs`, `s`, `m`, `l`, `xl` |
| `--ratio` | `2:3` | Aspect ratio: `1:1`, `16:9`, `9:16`, `3:2`, `2:3` |
| `-W`, `--width` | from preset | Override image width |
| `-H`, `--height` | from preset | Override image height |
| `--steps` | model-aware | Number of diffusion steps |
| `--guidance` | model-aware | Guidance scale |
| `--first-sigma` | `1.004` | Ideogram 4 only: override the first-step sigma (valid range `(0.0, 2.0]`, default `1.004`). Higher values (e.g. `1.005`-`1.006`) more reliably avoid the grey "blocked by safety filter" frame (best-effort mitigation, not a guaranteed bypass) |
| `--seed` | `None` (random) | Seed for reproducible image generation |
| `--upscale` | disabled | Upscale factor: `2` or `4` |
| `--upscale-denoise` | `0.3` (2×) / `0.4` (4×) | Denoising strength for upscale pass |
| `--upscale-steps` | `steps / 2` | Refinement steps for upscale |
| `--upscale-guidance` | same as `--guidance` | Override guidance scale for the upscale refine pass only |
| `--upscale-sharpen` | `True` | CAS sharpening step before upscale refinement (`--no-upscale-sharpen` to disable) |
| `--upscale-save-pre` | `False` | Save pre-upscale image alongside final |
| `--review` | `False` | Pause after each small render to approve (`y`) or discard (`n`) the upscale. Needs `--upscale` and a terminal |
| `--sharpen [AMOUNT]` / `--no-sharpen` | enabled | Apply CAS sharpening. Optional amount overrides config (0.0–1.5) |
| `--contrast [AMOUNT]` / `--no-contrast` | disabled | Apply contrast adjustment. Optional amount (1.0 = no change) |
| `--saturation [AMOUNT]` / `--no-saturation` | disabled | Apply saturation adjustment. Optional amount (1.0 = no change) |
| `--image` | disabled | Path to reference image for img2img steering |
| `--image-strength` | `0.5` | Denoising strength for reference (0.0–1.0) |
| `-o`, `--output` | `.` | Output directory for generated images |
| `--enhance [SPEC]` | disabled | Rewrite each prompt with a local LLM before the model loads. `SPEC` is optional, e.g. `style=photo,details=lighting+camera,length=longer`. Overrides prompt-file `enhance:` entries. Ignored with `--json-prompt` |
| `--enhance-note TEXT` | — | Free-text direction for the enhancer, e.g. `"at night, in the rain"` (up to 300 characters). Implies `--enhance` |
| `--no-enhance` | — | Disable enhancement, including prompt-file `enhance:` entries |
| `--enhance-model` | config | Enhancer LLM as `REPO[@REVISION]` or a local folder |

Run `ziv-image --help` for the full list.

### `--enhance` spec

Comma-separated `key=value` pairs; join several values with `+`. Omitted keys use their defaults.

| Key | Values | Default |
|---|---|---|
| `style` | `keep`, `photo`, `candid`, `street`, `film`, `bw`, `portrait`, `product`, `cinematic`, `illustration`, `anime`, `comic`, `3d`, `painterly` | `keep` |
| `mood` | `keep`, `serene`, `joyful`, `romantic`, `melancholic`, `mysterious`, `eerie`, `dramatic`, `epic`, `whimsical`, `nostalgic` | `keep` |
| `details` | `lighting`, `composition`, `camera`, `materials`, `color`, `environment`, `subject` | `lighting+composition` |
| `length` | `shorter`, `same`, `longer`, `extra` | `same` |
| `motion` (`ziv-video` only) | `action`, `camera-move`, `pacing` | `action` |
| `sound` (`ziv-video` only, used with audio on) | `ambient`, `effects`, `dialogue`, `music` | none |
| `parts` (`ziv-video` only) | `both`, `action` | `action` with `--extend` when the source has a scene, otherwise `both` |

`parts=action` keeps the scene exactly as written and rewrites only the action, so `style`, `mood` and `details` have no effect. It needs a scene (from `--scene`, the entry's `scene:` or the extended clip); with none, the run uses `parts=both` and warns.

The free-text direction is not part of the spec; pass it with `--enhance-note`.

See [Enhancing Prompts](../guides/prompts.md#enhancing-prompts).

The `ideo` alias selects Ideogram 4 (`ideogram-ai/ideogram-4-fp8`), a macOS/MLX-only image model. Its width and height must be in the 256–2048 range and multiples of 16. Size presets that exceed this range — `--size xl` with `--ratio 16:9` or `--ratio 9:16` (2112 px) — are rejected before the model loads, as are `--image` and `--upscale` (both require img2img, which Ideogram 4 does not support).

## `ziv-video` — Video Generation

Platform-aware aliases:

- `ltx-4` and `ltx-8` (4-bit and 8-bit LTX-2.3) on every platform. On Windows and Linux they need about 20 GB and 27 GB of system memory.

| Argument | Default | Description |
|---|---|---|
| `-m`, `--model` | *(required)* | Model alias, local path, or supported/configured LTX HuggingFace repo ID. With `--extend`, defaults to the source video's model |
| `--prompt` | `None` | Inline prompt, the action: what happens; overrides `--prompts-file`. May be empty when `--scene` or `--extend` is given |
| `--scene` | `None` | Scene: who and where, look, style, camera. Sent before the prompt. Alone it makes a scene-only run; with a prompt file it fills entries that have no `scene:` of their own. With `--extend` it defaults to the source video's recorded scene |
| `-p`, `--prompts-file` | `prompts.yaml` | YAML prompt file path |
| `--extend SRC` | `None` | Continue the video `SRC`: generate what happens next and append it, writing a new file (the source is not changed). macOS only. See [Extending a video](#extending-a-video) |
| `--extend-seconds S` | `2` | Length to add with `--extend`, in seconds at the source's frame rate, rounded to a multiple of 8 frames. Exclusive with `--extend-frames` |
| `--extend-frames N` | from `--extend-seconds` | Frames to add with `--extend`, rounded to a multiple of 8 (minimum 8). Exclusive with `--extend-seconds` |
| `--image` | `None` | Input image for image-to-video |
| `-W`, `--width` | model-aware | Video width (LTX: auto-aligned to 32px, 64px with `--upscale`) |
| `-H`, `--height` | model-aware | Video height (LTX: auto-aligned to 32px, 64px with `--upscale`) |
| `--frames` | model-aware | Number of frames (LTX: auto-aligned to 8k+1) |
| `--ratio` | `16:9` | Aspect ratio preset (16:9, 9:16, 1:1) |
| `-s`, `--size` | `m` | Size preset (`s`, `m`, `l`, `xl`) |
| `--steps` | model-aware | Inference steps (LTX: up to 8; on Windows/Linux, fewer than 8 keeps the end of the distilled schedule) |
| `--seed` | `None` (random) | Seed for reproducible video generation |
| `--low-memory` / `--no-low-memory` | enabled | Low-memory mode for LTX. On macOS with 24 GB of memory or less it also streams the transformer from disk, which cuts the memory held while denoising; steps are slightly slower, and about twice as slow with a LoRA. `--no-low-memory` turns that off too |
| `--upscale` | disabled | Upscale factor: `2` only. Renders at half size, upsamples the latents 2×, then refines at full size, so the output keeps the requested size. On Windows/Linux the first use downloads the 2× upsampler (about 1 GB) |
| `--review` | `False` | Pause after each small render to approve (`y`) or discard (`n`) the upscale. Needs `--upscale` and a terminal. macOS only for now |
| `--audio` / `--no-audio` | enabled | Include or strip audio in video output |
| `--lora` | `None` | Comma-separated LoRAs with optional weights: `name:0.8,name2:0.5`. Windows/Linux require diffusers-compatible LTX LoRA adapters. |
| `-r`, `--runs` | `1` | Number of batch runs |
| `-o`, `--output` | `.` | Output directory |
| `--format` | `mp4` | Output format |
| `--enhance [SPEC]` | disabled | Rewrite each prompt with a local LLM before the model loads (see the `--enhance` spec above; `motion` and `sound` are allowed) |
| `--enhance-note TEXT` | — | Free-text direction for the enhancer (up to 300 characters). Implies `--enhance` |
| `--no-enhance` | — | Disable enhancement, including prompt-file `enhance:` entries |
| `--enhance-model` | config | Enhancer LLM as `REPO[@REVISION]` or a local folder |

Operational notes:

- Video generation requires `ffmpeg` on every platform.
- Windows and Linux video generation are CUDA-only.
- On Windows and Linux a model that cannot fit in system memory stops before loading, and the first run of each model takes a while to download and prepare it.

### Extending a video

```bash
ziv-video --extend clip.mp4 --prompt "The camera pans to the tower."
ziv-video --extend clip.mp4 --extend-seconds 4 --scene "A knight in a grey courtyard."
ziv-video --extend clip.mp4 --extend-frames 96 -p prompts.yaml
```

`--extend` checks the source before the model loads and refuses a video it cannot continue, naming every problem (see [What can be extended](../guides/video.md#what-can-be-extended)). The output is a new file with the source's size and frame rate, and the original part is kept untouched when it can be stream-copied.

Defaults taken from the source when you do not pass them:

- **Scene:** its recorded scene. `--scene` replaces it.
- **Model:** the model it was made with, when that model is available locally; otherwise the configured default video model, with a warning.
- **LoRAs:** the LoRAs it was made with that still exist; missing ones are skipped with a warning. `--lora` replaces them.
- **Prompt:** none, so the extension continues naturally. A video recorded with a single prompt and no scene (made before scenes existed) keeps that prompt when you pass none, with quoted lines dropped while it is unchanged; if `--prompt` still equals it, a warning says to change it to what happens next.

Options refused with `--extend`, since the extension keeps the source's size and takes its own length: `--image`, `--upscale`, `--ratio`, `--size`, `--width`, `--height` and `--frames`. `--extend-seconds` and `--extend-frames` need `--extend`. In a prompt file, every entry needs a `prompt`, which is the action that follows the source. On Windows and Linux, `--extend` is refused. `--no-audio` leaves the result without sound, and a silent source gives a silent result. When the extension may not fit in memory, a warning is printed and the run continues.

## `ziv-model model` — Checkpoint Conversion

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | *(required)* | Path to  `.safetensors` checkpoint |
| `--name` | input filename | Custom model folder name |
| `--model-type` | `zimage` | Model type: `zimage`, `flux2-klein-4b`, `flux2-klein-9b`, `krea2-turbo` |
| `--base-model` | `Tongyi-MAI/Z-Image-Turbo` | Base HF repo (only for zimage type) |
| `--copy` | off | Copy files instead of symlinking |
| `--quantize` | off | Also save a quantized copy as `<name>@q4` or `<name>@q8` (`4` or `8`) |

## `ziv-model lora` — LoRA Import

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | — | Path to local `.safetensors` file (mutually exclusive with `--hf`) |
| `--hf` | — | HuggingFace repo ID (mutually exclusive with `-i`) |
| `--file` | auto-detect | Specific `.safetensors` file in the HF repo |
| `--name` | filename stem | Custom LoRA name |

## `ziv-model list` — Asset Listing

| Flag | Default | Description |
|---|---|---|
| `--models` | off | Show only models |
| `--loras` | off | Show only LoRAs |

## Image Size Presets

Default ratio is `2:3`. Dimensions vary by `--ratio`.

| Preset | 1:1 | 16:9 | 9:16 | 3:2 | 2:3 |
|--------|-----|------|------|-----|-----|
| `xs` | 512×512 | 672×384 | 384×672 | 608×400 | 400×608 |
| `s` | 704×704 | 944×528 | 528×944 | 864×576 | 576×864 |
| `m` | 1024×1024 | 1344×768 | 768×1344 | 1216×832 | 832×1216 |
| `l` | 1440×1440 | 1888×1056 | 1056×1888 | 1728×1152 | 1152×1728 |
| `xl` | 1600×1600 | 2112×1184 | 1184×2112 | 1936×1296 | 1296×1936 |

## Video Size Presets

Default ratio is `16:9`. Dimensions vary by `--ratio`.

| Preset | 16:9 | 9:16 | 1:1 |
|--------|------|------|-----|
| `s` | 512×256 (49f) | 256×512 (49f) | 384×384 (49f) |
| `m` | 704×448 (49f) | 448×704 (49f) | 512×512 (33f) |
| `l` | 960×512 (33f) | 512×960 (33f) | 768×768 (33f) |
| `xl` | 1408×896 (25f) | 896×1408 (25f) | 1024×1024 (25f) |
