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
| `--host` | `127.0.0.1` | Host interface to bind the local Web UI server to. `local` is short for `0.0.0.0`, which makes the UI reachable from other devices on your network; the launcher then also prints this machine's network address |
| `--port` | `8080` | Preferred local port; the app chooses the next available port if this one is busy |
| `--no-browser` | off | Start the server without opening a browser tab |
| `--background` | off | Start the server in the background and return to the prompt once it is reachable. It keeps running after the terminal closes; its output goes only to the error log |
| `--background-stop` | off | Stop every running Web UI server, whether started with `--background` or running in another terminal. Any job they are running stops too. Cannot be combined with `--background` |

Every running server records itself in `~/.ziv/run/` (or `$ZIV_DATA_DIR/run/`), which is how `--background-stop` finds them. A server is asked to shut down first and is force-stopped if it has not exited after 10 seconds, together with any process it started. On Windows a server is stopped immediately. Servers started by versions before this flag existed are not recorded and must be stopped by hand.

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
| `--enhance-model` | `fast` | Enhancer LLM: `fast`, `smart` (larger, needs more memory; falls back to `fast` with a warning where it does not fit), or `REPO[@REVISION]` or a local folder. Use `./smart` for a folder named `smart` |
| `--enhance-format` | `text` | `json` writes an Ideogram 4 JSON caption instead of a rewritten prompt, Needs a model that takes JSON captions (`ideo`). Implies `--enhance` for `--prompt`; with `--prompts-file` it applies to entries with their own `enhance:` unless `--enhance` is given |

Run `ziv-image --help` for the full list.

### `--enhance` spec

Comma-separated `key=value` pairs; join several values with `+`. Omitted keys use their defaults. Nothing is picked by default: `--enhance` with no spec, or every key at its default, gives a general rewrite that adds detail where the prompt is thin.

| Key | Values | Default |
|---|---|---|
| `style` | `keep`; photo: `photo`, `candid`, `street`, `film`, `bw`, `portrait`, `product`, `cinematic`; art: `illustration`, `anime`, `comic`, `3d`, `painterly`, `watercolor`, `gouache`, `sketch`, `pixel`, `concept`, `graphic`, `isometric`, `claymation`, `ukiyo-e`, `papercraft` | `keep` |
| `mood` | `keep`, `serene`, `cozy`, `joyful`, `romantic`, `whimsical`, `nostalgic`, `melancholic`, `mysterious`, `eerie`, `dramatic`, `epic` | `keep` |
| `theme` | `keep`, `zen`, `spiritual`, `rustic`, `sensual`, `glamorous`, `gothic`, `noir`, `horror`, `dystopian`, `fairytale`, `mythic`, `surreal`, `retro`, `futuristic`, `cyberpunk`, `steampunk` | `keep` |
| `details` (Focus on) | `lighting`, `composition`, `camera`, `materials`, `color`, `environment`, `subject`, `expression` | none |
| `length` | `shorter`, `same`, `longer`, `extra` | `same` |
| `camera` (`ziv-video` only) | `keep`, `auto`, `static`, `push-in`, `pull-back`, `pan`, `tilt`, `tracking`, `orbit`, `handheld`, `aerial` | `keep` |
| `pace` (`ziv-video` only) | `keep`, `slow-motion`, `slow`, `brisk`, `fast` | `keep` |
| `motion` (`ziv-video` only) | `action`, `secondary` | none |
| `sound` (`ziv-video` only, used with audio on) | `ambient`, `effects`, `dialogue`, `music` | none |
| `parts` (`ziv-video` only) | `both`, `action` | `action` with `--extend` when the source has a scene, otherwise `both` |

`parts=action` keeps the scene exactly as written and rewrites only the action, so `style`, `mood`, `theme` and `details` have no effect. It needs a scene (from `--scene`, the entry's `scene:` or the extended clip); with none, the run uses `parts=both` and warns.

The free-text direction is not part of the spec; pass it with `--enhance-note`.

See [Enhancing Prompts](../guides/prompts.md#enhancing-prompts).

The `ideo` alias selects Ideogram 4 (`ideogram-ai/ideogram-4-fp8`), an image model on macOS (MLX), Windows and Linux (CUDA, using `unsloth/ideogram-4-nf4-diffusers`). Its width and height must be in the 256–2048 range and multiples of 16. Size presets that exceed this range — `--size xl` with `--ratio 16:9` or `--ratio 9:16` (2176 px) — are rejected before the model loads, as are `--image` and `--upscale` (both require img2img, which Ideogram 4 does not support).

## `ziv-video` — Video Generation

Platform-aware aliases:

- `ltx-4` and `ltx-8` (4-bit and 8-bit LTX-2.3) on every platform. On Windows and Linux they need about 20 GB and 27 GB of system memory.
- `ltx25-4` (4-bit LTX-2.5) on every platform. On Windows and Linux it needs about 21 GB of system memory; on macOS it downloads about 30.5 GB with its own text encoder. Extend uses the deterministic sampler (see the [Video Guide](../guides/video.md#ltx-25)).
- `h3` (MiniMax H3) on every platform, with the exceptions in the [Video Guide](../guides/video.md#minimax-h3): no `--upscale` on macOS. It downloads about 98 GB on Windows and Linux and about 107 GB on macOS.

| Argument | Default | Description |
|---|---|---|
| `-m`, `--model` | *(required)* | Model alias, local path, or supported/configured LTX or MiniMax H3 HuggingFace repo ID. With `--extend`, defaults to the source video's model |
| `--prompt` | `None` | Inline prompt, the action: what happens; overrides `--prompts-file`. May be empty when `--scene` or `--extend` is given |
| `--scene` | `None` | Scene: who and where, look, style, camera. Sent before the prompt. Alone it makes a scene-only run; with a prompt file it fills entries that have no `scene:` of their own. With `--extend` it defaults to the source video's recorded scene |
| `-p`, `--prompts-file` | `prompts.yaml` | YAML prompt file path |
| `--extend SRC` | `None` | Continue the video `SRC`: generate what happens next and append it, writing a new file (the source is not changed). See [Extending a video](#extending-a-video) |
| `--extend-seconds S` | `2` | Length to add with `--extend`, in seconds at the source's frame rate, rounded to a multiple of the model's extension step: 8 frames for LTX, 17 for MiniMax H3. Exclusive with `--extend-frames` |
| `--extend-frames N` | from `--extend-seconds` | Frames to add with `--extend`, rounded to a multiple of the extension step (8 for LTX, 17 for MiniMax H3; minimum one step). Exclusive with `--extend-seconds` |
| `--image` | `None` | Input image for image-to-video: the first frame |
| `--end-image` | `None` | Last-frame image: the clip runs from `--image` to it. Needs `--image` and a model that supports it (MiniMax H3) |
| `-W`, `--width` | model-aware | Video width (LTX: auto-aligned to 32px, 64px with `--upscale`) |
| `-H`, `--height` | model-aware | Video height (LTX: auto-aligned to 32px, 64px with `--upscale`) |
| `--frames` | model-aware | Number of frames (LTX: auto-aligned to 8k+1, default 121; MiniMax H3: 17k+5, default 124, another count moves up to the next) |
| `--ratio` | `16:9` | Aspect ratio preset (16:9, 9:16, 1:1) |
| `-s`, `--size` | `s` | Size preset (`xs`, `s`, `m`, `l`, `xl`) |
| `--steps` | model-aware | Inference steps (LTX: up to 8; on Windows/Linux, fewer than 8 keeps the end of the distilled schedule. MiniMax H3: `4` or `8`, default 8; another count moves to the nearest, a tie to 8) |
| `--seed` | `None` (random) | Seed for reproducible video generation |
| `--low-memory` / `--no-low-memory` | enabled | Low-memory mode for LTX (MiniMax H3 always streams its blocks). On macOS with 24 GB of memory or less it also streams the transformer from disk, which cuts the memory held while denoising; steps are slightly slower, and about twice as slow with a LoRA. `--no-low-memory` turns that off too |
| `--upscale` | disabled | Upscale factor: `2` only. Renders at half size, upsamples the latents 2×, then refines at full size, so the output keeps the requested size. On Windows/Linux the first use downloads the 2× upsampler (about 1 GB). MiniMax H3 does the same with its own 2× latent upscaler (Windows and Linux only; the first use downloads 0.69 GB; no upscale on macOS) |
| `--review` | `False` | Pause after each small render to approve (`y`) or discard (`n`) the upscale. Needs `--upscale` and a terminal |
| `--audio` / `--no-audio` | enabled | Include or strip audio in video output |
| `--lora` | `None` | Comma-separated LoRAs with optional weights: `name:0.8,name2:0.5`. Windows/Linux require diffusers-compatible LTX LoRA adapters; `h3` takes MiniMax H3 LoRAs |
| `-r`, `--runs` | `1` | Number of batch runs |
| `-o`, `--output` | `.` | Output directory |
| `--format` | `mp4` | Output format |
| `--enhance [SPEC]` | disabled | Rewrite each prompt with a local LLM before the model loads (see the `--enhance` spec above; `motion` and `sound` are allowed) |
| `--enhance-note TEXT` | — | Free-text direction for the enhancer (up to 300 characters). Implies `--enhance` |
| `--no-enhance` | — | Disable enhancement, including prompt-file `enhance:` entries |
| `--enhance-model` | `fast` | Enhancer LLM: `fast`, `smart`, or `REPO[@REVISION]` or a local folder (see `ziv-image`) |

Operational notes:

- Video generation requires `ffmpeg` on every platform. If it is missing, `ziv-video` offers to install it with Homebrew or MacPorts (macOS), `winget` (Windows), or `apt`, `dnf` or `pacman` (Linux), and asks before running anything. On Windows, open a new terminal afterwards so `PATH` picks it up. On Fedora, `dnf install ffmpeg` needs RPM Fusion enabled.
- Windows and Linux video generation are CUDA-only.
- MiniMax H3 prints its estimated time when a job starts. The first run on Windows and Linux builds a 4-bit copy of the transformer (about 4 min); on macOS it converts the text encoder once. The estimate leaves this out.
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

MiniMax H3 extends 24 fps sources whose width and height are multiples of 32, and carries the source's last 22 frames into the new part (on macOS it continues from the last frame instead).

Options refused with `--extend`, since the extension keeps the source's size and takes its own length: `--image`, `--end-image`, `--upscale`, `--ratio`, `--size`, `--width`, `--height` and `--frames`. `--extend-seconds` and `--extend-frames` need `--extend`. In a prompt file, every entry needs a `prompt`, which is the action that follows the source. `--no-audio` leaves the result without sound, and a silent source gives a silent result. When the extension may not fit in memory, a warning is printed and the run continues.

## `ziv-model model` — Checkpoint Conversion

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | *(required)* | Path to  `.safetensors` checkpoint |
| `--name` | input filename | Custom model folder name |
| `--model-type` | `zimage` | Model type: `zimage`, `flux2-klein-4b`, `flux2-klein-9b`, `krea2-turbo`, `ltx-2.3`, `ltx-2.5`, `h3` |
| `--base-model` | `Tongyi-MAI/Z-Image-Turbo` | Base HF repo (only for zimage type) |
| `--copy` | off | Copy files instead of symlinking |
| `--quantize` | off | Also save a quantized copy as `<name>@q4` or `<name>@q8` (`4` or `8`). Required for `ltx-2.3` and `ltx-2.5`, where it is the only level stored; `h3` takes none |
| `--distill` | off | `ltx-2.3` and `ltx-2.5` only: merge Lightricks' distilled LoRA into a plain-dev checkpoint |
| `--turbo` | off | `h3` only: the checkpoint is already step-distilled, so the turbo LoRA is not merged |
| `--force` | off | Video types only: replace an existing model of that name once the new one is complete |

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
| `xs` | 512×512 | 672×384 | 384×672 | 576×384 | 384×576 |
| `s` | 704×704 | 960×544 | 544×960 | 864×576 | 576×864 |
| `m` | 1024×1024 | 1344×768 | 768×1344 | 1216×832 | 832×1216 |
| `l` | 1408×1408 | 1920×1088 | 1088×1920 | 1728×1152 | 1152×1728 |
| `xl` | 1600×1600 | 2176×1216 | 1216×2176 | 1920×1280 | 1280×1920 |

## Video Size Presets

Default ratio is `16:9` and default size is `s`. Dimensions vary by `--ratio`.

| Preset | 16:9 | 9:16 | 1:1 |
|--------|------|------|-----|
| `xs` | 576×320 (121f) | 320×576 (121f) | 384×384 (121f) |
| `s` | 768×448 (121f) | 448×768 (121f) | 576×576 (121f) |
| `m` | 1024×576 (121f) | 576×1024 (121f) | 768×768 (121f) |
| `l` | 1280×704 (121f) | 704×1280 (121f) | 960×960 (121f) |
| `xl` | 1600×896 (121f) | 896×1600 (121f) | 1216×1216 (121f) |
