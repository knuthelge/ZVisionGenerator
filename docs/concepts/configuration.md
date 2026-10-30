# Configuration

The app's defaults ship in a `config.yaml` inside the package. To change one, put just that setting in your own `config.yaml` in the [data folder](data-folder.md), `~/.ziv/config.yaml`:

```yaml title="~/.ziv/config.yaml"
generation:
  default_size: l

sharpening:
  normal: 0.8
```

Your file is merged over the defaults, key by key, so you only list what you change. The [Config page](../web-ui/config.md) writes the most common settings to the same file. Most changes apply from the next job; restart the Web UI if one doesn't.

## Which value wins

For a setting that exists at several levels, the most specific wins:

1. A CLI flag or a Workspace setting.
2. The model's preset for its variant, such as FLUX.2 Klein distilled.
3. The model family's preset.
4. The global default, such as `generation.default_steps`.

## `generation`

Defaults for images.

| Key | Default | Description |
|---|---|---|
| `default_ratio` | `"2:3"` | Aspect ratio |
| `default_size` | `m` | Size preset. The Config page's **Base resolution** |
| `default_steps` | `10` | Steps, for models whose preset sets none |
| `default_guidance` | `3.5` | Guidance, for models whose preset sets none |
| `seed_min`, `seed_max` | `4`, `4294967295` | Range random seeds are drawn from |

## `sizes`

The pixel size of each image preset, per ratio: `sizes.<ratio>.<preset>` with `width` and `height`. See [Images](images.md#sizes) for the defaults.

## `video_generation` and `video_sizes`

| Key | Default | Description |
|---|---|---|
| `video_generation.default_ratio` | `"16:9"` | Aspect ratio of videos |
| `video_generation.default_size` | `s` | Size preset of videos |

`video_sizes.<ratio>.<preset>` sets `width`, `height` and `frames` of each video preset. Keep widths and heights multiples of 64 and frames `8k+1`. See [Video](video.md#sizes).

## `sharpening`

The CAS sharpening amounts used when no amount is given.

| Key | Default | Used for |
|---|---|---|
| `normal` | `1.0` | The final pass of a plain image |
| `upscaled` | `1.2` | The final pass of an upscaled image |
| `pre_upscale` | `0.8` | Before the refine pass of an upscale |
| `existing_upscaled` | `1.2` | The final pass of a creative upscale from the viewer |
| `existing_pre_upscale` | `0` | Before refining an existing image; `0` is off, since the image is already sharpened |

## `contrast` and `saturation`

| Key | Default | Description |
|---|---|---|
| `contrast.default_amount` | `1.0` | Amount used when contrast is turned on without one |
| `saturation.default_amount` | `1.0` | Amount used when saturation is turned on without one |

## `upscale`

| Key | Default | Description |
|---|---|---|
| `default_denoise_2x` | `0.4` | Refine strength of a 2× upscale while generating |
| `default_denoise_4x` | `0.4` | Refine strength of a 4× upscale while generating |
| `existing_denoise_small` | `0.4` | Refine strength of a creative upscale from the viewer, for outputs up to `existing_large_megapixels` |
| `existing_denoise_large` | `0.2` | The same, for larger outputs. High values at large sizes invent detail and seams |
| `existing_large_megapixels` | `2` | Where small turns into large |
| `max_megapixels` | `20` | Largest output a creative upscale from the viewer may make |

## `seedvr2`

The plain upscale of existing images. See [Plain upscale](upscaling.md#plain-upscale).

| Key | Default | Description |
|---|---|---|
| `model` | `auto` | `auto`, `seedvr2-3b` or `seedvr2-7b` |
| `softness` | `0.75` | 0–1: how much an image is softened before it's enlarged; `0` is off |
| `video_softness` | `0.0` | The same for each frame of a video upscale (Windows and Linux) |
| `reference_megapixels` | `16` | Output size the automatic choice must fit on a Mac (a typical 4×) |
| `tiers` | | macOS: the models in quality order, with the memory figures that price them |
| `discrete_tiers` | | Windows and Linux: the same, with the system memory each needs (`system_gb`) |

With `model: auto`, the first tier that fits wins, and the last one runs when none does. On a Mac a tier fits when its estimate at `reference_megapixels` is within the GPU memory budget; on Windows and Linux when its `system_gb` is within the system memory. A tier's estimate, in GPU memory, is `weights_gb` plus the larger of `activation_floor_gb` and `activation_base_gb + activation_gb_per_mp × megapixels`. The same estimate marks sizes as **high memory**.

## `model_aliases`

Short names for models. A value is a Hugging Face repo, or a map from platform (`darwin`, `win32`, `linux`) to a repo:

```yaml
model_aliases:
  mymodel: "my-org/my-model"
```

A folder `~/.ziv/models/<alias>/` wins over an alias of the same name. See [Models](models.md#how-a-name-is-found).

## `model_labels`

How the Web UI names an alias, and its download size in the first-run dialog:

```yaml
model_labels:
  mymodel:
    name: "My Model"
```

An alias without a label shows its alias.

## `first_run`

The model the Web UI's first-run dialog preselects.

| Key | Default | Description |
|---|---|---|
| `large_model` | `klein9b` | For machines with at least `min_memory_gb` |
| `small_model` | `klein4b` | For every other machine |
| `min_memory_gb` | `32` | Unified memory on a Mac, GPU memory on a graphics card |

## `ui`

The Web UI.

| Key | Default | Description |
|---|---|---|
| `default_models.image` | `klein9b` | Default image model. The Config page's **Default image model** |
| `default_models.video` | `ltx-8` | Default video model. The Config page's **Default video model** |
| `output_dir` | unset | Where outputs are saved. Unset means `outputs` in the data folder, so `ZIV_DATA_DIR` moves it too. The Config page's **Output folder** |
| `startup_view` | `workspace` | The page that opens first: `workspace`, `gallery` or `config` |
| `gallery_page_size` | `12` | How many items the Gallery loads at a time |
| `first_run_skipped` | `false` | Set when the first-run dialog is skipped |
| `discover_huggingface` | `false` | Set by **Show Hugging Face results** in Discover LoRAs; **Turn off** sets it back to `false` |
| `discover_content_filter` | `on` | The Discover LoRAs [content filter](loras.md#content-filter). Set `off` if it hides a LoRA you want to use; repos tagged as adult stay hidden |

## `prompt_enhancer`

| Key | Default | Description |
|---|---|---|
| `user_model` | none | A Hugging Face `owner/name[@revision]` or a local folder, used instead of Fast and Smart. The Config page's **Prompt enhancer model** |
| `model`, `revision` | per platform | The Fast model, pinned |
| `download_size_label` | per platform | The Fast model's first-use download, shown in the Web UI |
| `smart.model`, `smart.revision` | per platform | The Smart model, pinned |
| `smart.download_size_label` | per platform | The Smart model's first-use download, shown in the Web UI |
| `smart.min_memory_gb` | 20 (macOS), 9 (Windows, Linux) | Accelerator memory Smart needs, in GiB |
| `temperature` | `0.7` | How varied the rewrites are |
| `max_new_tokens` | `700` | Longest rewrite the model may write |
| `idle_release_seconds` | `120` | How long the Web UI keeps the enhancer loaded after a manual Enhance |
| `length.percent` | 50 / 100 / 200 / 300 | Target length of Shorter, Same, Longer and Extra long, in percent |
| `length.floor_words` | 12 / 25 / 40 / 80 | The fewest words each length aims for |
| `length.max_words` | `300` | The most words any rewrite aims for |

## `model_presets` and `video_model_presets`

Per-family defaults and capabilities, such as `default_steps`, `default_guidance`, `default_scheduler`, `supports_negative_prompt` and `max_reference_images`. `video_model_presets.ltx` and `video_model_presets.h3` also hold each video family's frame rule, steps, capabilities and the pinned repositories and files it loads on each platform; `video_model_presets.ltx.versions` holds LTX-2.5's differences from LTX-2.3.

A family can split into variants (`variants.<name>`), such as FLUX.2 Klein distilled and base, or MOSS-Image Turbo and SFT. A variant sets its own steps, guidance and upscale steps, and may also set capability flags such as `supports_negative_prompt`; a flag it sets wins over the family's value. A variant that turns off the negative prompt may also set `negative_prompt_hint`, a sentence added to the warning shown when a negative prompt is ignored. For MOSS-Image, Turbo turns off the negative prompt and points to `moss-sft`; SFT keeps it.

`max_reference_images` is how many [reference images](images.md#references) a family takes; `0`, the default, means none. FLUX.2 Klein sets `4`.

You rarely need to change these. They are documented in the packaged `config.yaml`.

## `ideogram4_instant`

The pinned sources Ideogram 4 Instant is built from: the transformer, the base components per platform, and the transformer format. Changing a pin rebuilds the model folder. See [Ideogram 4 Instant](models.md#ideogram-4-instant).

## `moss_image`

The pinned sources MOSS-Image is built from, on Windows and Linux only: `sft`, the SFT repository as `owner/repo@revision`, and `turbo_lora`, the repository, revision and file name of the 4-step LoRA that makes the `moss` (Turbo) model. The `moss` and `moss-sft` model folders are built from them on first use. Changing a pin rebuilds the affected folder and, for a new LoRA, leaves the SFT folder alone. The packaged SFT revision is the one the bundled pipeline code was written for, so change it only together with that code.

## Other sections

`schedulers`, `platforms` and `model_alias_families` are internal: they map scheduler names to classes, platform ids to names, and aliases to model families.
