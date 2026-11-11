# Configuration

Every setting of the app's `config.yaml`, with its default. Settings that the app reads from the shell, such as `ZIV_DATA_DIR`, are [environment variables](../cli/reference.md#environment-variables). Error? See [Troubleshooting](troubleshooting.md#configuration-file).

The app's defaults ship in a `config.yaml` inside the package. To change one, put just that setting in your own `config.yaml` in the [data folder](data-folder.md) (`~/.ziv` by default):

```yaml title="config.yaml"
generation:
  default_size: l

sharpening:
  normal: 0.8
```

Your file is merged over the default settings, key by key, so you only list what you change. It holds settings only: which models the app offers, where they download from and the terms they come with are part of the app, and a key for them in your file has no effect. The [Config page](../web-ui/config.md) writes the most common settings to the same file. Most changes apply from the next job; restart the Web UI if one doesn't.

## Which value wins { #which-value-wins }

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

The pixel size of each image preset, per ratio: `sizes.<ratio>.<preset>` with `width` and `height`. See [Images](../guides/images.md#pick-a-size) for the defaults.

## `video_generation` and `video_sizes`

| Key | Default | Description |
|---|---|---|
| `video_generation.default_ratio` | `"16:9"` | Aspect ratio of videos |
| `video_generation.default_size` | `s` | Size preset of videos |

`video_sizes.<ratio>.<preset>` sets `width`, `height` and `frames` of each video preset. Keep widths and heights multiples of 64 and frames `8k+1`. See [Video](../guides/video.md#sizes).

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

The plain upscale of existing images. See [Plain upscale](../concepts/upscaling.md#plain-upscale).

| Key | Default | Description |
|---|---|---|
| `model` | `auto` | `auto`, `seedvr2-3b` or `seedvr2-7b` |
| `softness` | `0.75` | 0–1: how much an image is softened before it's enlarged; `0` is off |
| `video_softness` | `0.0` | The same for each frame of a video upscale (Windows and Linux) |
| `reference_megapixels` | `16` | Output size the automatic choice must fit on a Mac (a typical 4×) |

With `model: auto`, the 7B model runs when it fits and the 3B model runs when it does not. On a Mac a model fits when its estimated GPU memory at `reference_megapixels` is within the GPU memory budget; on Windows and Linux when the system memory it needs is within the system memory. The same estimate marks sizes as **high memory**.

## `model_aliases`

Your own short names for models. A value is a Hugging Face repo, or a map from platform (`darwin`, `win32`, `linux`) to a repo:

```yaml
model_aliases:
  mymodel: "my-org/my-model"
```

A name the app already uses, such as `zit` or `ltx-4`, is taken: the built-in model keeps it. A folder `models/<alias>/` in the data folder wins over an alias of the same name. An alias is shown by its name in the Web UI. See [Models](models.md#how-a-name-is-found).

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
| `discover_content_filter` | `on` | The Discover LoRAs [content filter](../concepts/loras.md#content-filter). Set `off` if it hides a LoRA you want to use; repos tagged as adult stay hidden |

## `prompt_enhancer`

| Key | Default | Description |
|---|---|---|
| `user_model` | none | A Hugging Face `owner/name[@revision]` or a local folder, used instead of Fast and Smart. The Config page's **Prompt enhancer model** |
| `temperature` | `0.7` | How varied the rewrites are |
| `max_new_tokens` | `700` | Longest rewrite the model may write |
| `idle_release_seconds` | `120` | How long the Web UI keeps the enhancer loaded after a manual Enhance |
| `length.percent` | 50 / 100 / 200 / 300 | Target length of Shorter, Same, Longer and Extra long, in percent |
| `length.floor_words` | 12 / 25 / 40 / 80 | The fewest words each length aims for |
| `length.max_words` | `300` | The most words any rewrite aims for |

Which Length is picked when you pick none is **Same**; no setting changes that. The `length` keys change the numbers each choice aims for, and a key you leave out keeps its default. To make rewrites shorter overall, lower the floor of Same, so a short prompt doesn't grow to 25 words, the target of Shorter, and the cap:

```yaml title="config.yaml"
prompt_enhancer:
  length:
    percent:
      shorter: 30     # Shorter aims for 30% of your prompt's length
    floor_words:
      same: 12        # a short prompt grows to about 12 words
    max_words: 120    # no rewrite aims for more than 120 words
```

To pick a Length for one run, use [`--enhance length=shorter`](../cli/reference.md#the-spec) or the Web UI's **Length**. See [Length](../concepts/prompt-enhancer.md#length).

## `model_presets` and `video_model_presets`

Per-family defaults for the steps, guidance and scheduler of a model. What a family can do (the sizes it accepts, whether it takes a negative prompt, a start image or reference images, and the rules for a video's frames) is part of the app.

`model_presets.<family>` sets `default_steps`, `default_guidance`, `default_scheduler` (a scheduler per backend, `mflux` and `diffusers`, where `null` is the model's own), `upscale.default_upscale_steps` and `upscale.default_sharpen`. A family can split into variants (`variants.<name>`), such as FLUX.2 Klein distilled and base, or MOSS-Image Turbo and SFT; a variant sets its own steps, guidance and upscale steps, and they win over the family's.

| Family | Steps and guidance | Also |
|---|---|---|
| `zimage` | `10`, `0.0` | Scheduler `beta`; upscale steps `3` |
| `flux2_klein` | `distilled`: `4`, `1.0`; `base`: `50`, `1.5` | The model's own scheduler |
| `flux2` | `20`, `3.5` | The model's own scheduler |
| `flux1` | `20`, `3.5` | The model's own scheduler |
| `ideogram4` | `8`, `1.0` | Guidance is built in, so other values are ignored |
| `krea2` | `8`, `1.0` | Scheduler `beta`; upscale steps `5`; upscale sharpen off |
| `moss` | `turbo`: `4`, `1.0`; `sft`: `50`, `3.0` | Upscale steps `2` and `12` |
| `unknown` | `10`, `3.5` | For a model whose family isn't recognised |

### `model_presets.<family>.upscale.default_sharpen`

Whether sharpening before the refine pass of an upscale is on when you give neither `--upscale-sharpen` nor `--no-upscale-sharpen`. Unset means on; `krea2` sets `false`, because pre-sharpening can leave coloured specks in fabric.

`video_model_presets` holds these settings:

| Key | Default | Description |
|---|---|---|
| `ltx.default_steps` | `8` | Steps of an LTX video |
| `ltx.upscale.default_upscale_steps` | `8` | Steps when a video is upscaled and no steps are given; without it, the family's `default_steps` |
| `ltx.upscale.existing_video_steps` | `2` | Refine steps, from `1` to `3`, of [upscaling an existing video](../concepts/upscaling.md#macos-ltx) on macOS |
| `h3.default_steps` | `8` | Steps of a MiniMax H3 video: `4` or `8` |
| `h3.default_frames` | `124` | Frames of a MiniMax H3 video, 5 s at 24 fps; it replaces the frame count of the size presets |
| `h3.diffusers.ffn_chunk_tokens` | `4096` | Windows and Linux: rows per feed-forward call. Smaller values lower the peak memory of long and large clips |

You rarely need to change these.

## `audio_model_presets`

The settings of the audio models, keyed by the model id you pass with `-m`: `yue2`, `qwen3_tts` and `sa3`. The models themselves, their downloads and limits are part of the app, and the Audio tab and `ziv audio` offer them on every machine with a CUDA card or Apple silicon. You rarely need to change a setting. The shipped values:

```yaml
audio_model_presets:
  yue2:
    settings:
      instrumental_sections: "[Intro] [Instrumental] [Solo] [Instrumental] [Solo] [Bridge] [Instrumental] [Outro]"
      abc_max_tokens: 4096
      cfg: 1.0
      ode_steps: 32
      memory_budget_gib: 24
      vae_core_frames: 512
      fade_seconds: 2
  qwen3_tts:
    settings: { language: auto, join_pause_ms: 300, join_fade_ms: 10, part_level_match_db: 3, edge_silence_max_ms: 150 }
  sa3:
    settings: { steps: 8, cfg_scale: 1.0, sample_size: 16777216 }
```

| Setting | Model | Description |
|---|---|---|
| `instrumental_sections` | `yue2` | The section list an empty Lyrics box stands for, which makes an instrumental. YuE2 needs it |
| `abc_max_tokens` | `yue2` | The most tokens the model may use to plan a song's score |
| `cfg` | `yue2` | Guidance of that planning step |
| `ode_steps` | `yue2` | Steps of the synthesis stage, which turns the plan into sound |
| `memory_budget_gib` | `yue2` | The accelerator memory in GiB the model plans its work within |
| `vae_core_frames` | `yue2` | How many frames the decoder handles at a time; fewer frames use less memory |
| `fade_seconds` | `yue2` | Length of the fade-out when a song is cut off at the model's length limit |
| `language` | `qwen3_tts` | `auto` lets the model detect the language of the text, or name one it supports to fix it |
| `join_pause_ms`, `join_fade_ms`, `part_level_match_db`, `edge_silence_max_ms` | `qwen3_tts` | How the parts of a long text are joined |
| `steps` | `sa3` | Steps of a take |
| `cfg_scale` | `sa3` | How closely a take follows the description |
| `sample_size` | `sa3` | The audio window the model works in, in samples; the shipped value is its longest take, 380 s |

An entry can also hold `platforms.cuda.settings` and `platforms.mps.settings`, which win over `settings` on that accelerator.
