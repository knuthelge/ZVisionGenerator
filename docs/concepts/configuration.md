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

## `licences`

The terms the built-in models come with. Each entry is a licence record that covers one or more models, so a model can have two (LTX-2.3 has its own licence and Gemma's):

```yaml
licences:
  flux-nc:
    name: "FLUX Non-Commercial License v2.1"
    supplier: "Black Forest Labs"
    summary: "Non-commercial"
    url: "https://huggingface.co/unsloth/FLUX.2-klein-9B/blob/455f1273c2f08f1fc93d725c2e6f1d7fc681cd94/LICENSE.md"
    notice_url: "https://huggingface.co/unsloth/FLUX.2-klein-9B/blob/455f1273c2f08f1fc93d725c2e6f1d7fc681cd94/NOTICE"
    models: [klein9b]
```

| Key | Required | Description |
|---|---|---|
| `name` | yes | The licence's name |
| `supplier` | yes | Who licenses the model |
| `summary` | yes | The terms in a short, plain phrase |
| `url` | yes | The licence text, pinned to a revision |
| `models` | yes | The models it covers, at least one: an alias from `model_aliases`, the id of an `audio_model_presets` entry, or `matte` for the [background removal](images.md#transparent-background) models |
| `notice_url` | no | The licence's NOTICE file, pinned to a revision |
| `url_label` | no | What the link to `url` is called. Default `Licence` |

A URL is pinned when its path holds a 40-character commit, as in `.../blob/<commit>/LICENSE`, or `owner/repo@<commit>` on jsDelivr. The app refuses to start with a URL that has none, a record without a required key, an empty `models` list or a model that doesn't exist. Models under MIT or Apache-2.0 have no record.

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

A video family's `capabilities` mapping lists what its models can do. Every capability is required and is `true` or `false`: `i2v`, `end_frame`, `extend`, `upscale`, `upscale_review`, `lora`, `audio`, `low_memory`, `audio_track` and `audio_track_upscale`. `audio_track` says the model can follow an audio file you give it, and `audio_track_upscale` says it can also upscale a clip made that way; a family that lacks the first refuses a track, and one that lacks the second refuses Upscale together with a track. The packaged config sets both to `true` for LTX, and `audio_track` to `true` and `audio_track_upscale` to `false` for MiniMax H3. `platform_capabilities` overrides capabilities for one platform (`darwin`, `win32` or `linux`).

A family can split into variants (`variants.<name>`), such as FLUX.2 Klein distilled and base, or MOSS-Image Turbo and SFT. A variant sets its own steps, guidance and upscale steps, and may also set capability flags such as `supports_negative_prompt`; a flag it sets wins over the family's value. A variant that turns off the negative prompt may also set `negative_prompt_hint`, a sentence added to the warning shown when a negative prompt is ignored. For MOSS-Image, Turbo turns off the negative prompt and points to `moss-sft`; SFT keeps it.

`max_reference_images` is how many [reference images](images.md#references) a family takes; `0`, the default, means none. FLUX.2 Klein sets `4`.

`video_model_presets.ltx.upscale.existing_video_steps` (default `2`, from `1` to `3`) sets the refine steps of [upscaling an existing video](upscaling.md#macos-ltx) on macOS.

You rarely need to change these. They are documented in the packaged `config.yaml`.

## `audio_model_presets`

The audio models: one entry per model, keyed by the model id you pass with `-m`. A model is music, speech or sound, and each entry says what it can do, how long it can run, which files it downloads and how it runs on each platform. The packaged config ships these three models with their pinned sources, and the Audio tab and `ziv audio` offer them on every platform that has an entry:

```yaml
audio_model_presets:
  yue2:
    label: "YuE2 3B"
    kind: music
    capabilities: { lyrics: true, duration: false, voice_clone: false }
    limits: { max_seconds: 360 }
    settings:
      cot_song: full
      cot_instrumental: melody
      instrumental_sections: "[Intro] [Instrumental] [Solo] [Instrumental] [Solo] [Bridge] [Instrumental] [Outro]"
      abc_max_tokens: 4096
      cfg: 1.0
      ode_steps: 32
      memory_budget_gib: 24
      vae_core_frames: 512
      fade_seconds: 2
    memory: { cuda_gb: 6.7, mps_gb: null }
    platforms:
      cuda:
        family: yue2
        files:
          - repo: "m-a-p/YuE2-3B@8f24312f187763b9854adeec87e437e2b03bbff1"
            files: [config.json, generation_config.json, yue2_generation_config.json, model.safetensors, modeling_yue2.py, qwen.tiktoken, weights_manifest.json]
          - repo: "m-a-p/YuE2-Vae@152733a19ad43aa67e367f9b5503ef8075bb5126"
            files: [config.json, model.safetensors, modeling_vae.py, weights_manifest.json]
        settings: { weight_swap: true }
      mps:
        family: yue2
        files:
          - repo: "m-a-p/YuE2-3B@8f24312f187763b9854adeec87e437e2b03bbff1"
            files: [config.json, generation_config.json, yue2_generation_config.json, model.safetensors, modeling_yue2.py, qwen.tiktoken, weights_manifest.json]
          - repo: "m-a-p/YuE2-Vae@152733a19ad43aa67e367f9b5503ef8075bb5126"
            files: [config.json, model.safetensors, modeling_vae.py, weights_manifest.json]
        settings: { weight_swap: false }
  higgs3:
    label: "Higgs TTS 3 4B"
    kind: speech
    capabilities: { lyrics: false, duration: false, voice_clone: true }
    limits: { max_new_tokens: 1500, part_chars: 500, max_text_chars: 10000, min_voice_seconds: 3, max_voice_seconds: 30 }
    settings: { temperature: 0.7, top_p: 0.95, join_pause_ms: 300, join_fade_ms: 10, part_level_match_db: 3, edge_silence_max_ms: 150 }
    memory: { cuda_gb: 8.2, mps_gb: null }
    platforms:
      cuda:
        family: higgs3
        files:
          - repo: "bosonai/higgs-tts-3-4b@239f63fb7b02b1aa085f98d9efae5e35cc5523e8"
            files: [config.json, model.safetensors, tokenizer.json, tokenizer_config.json]
          - repo: "bosonai/higgs-audio-v2-tokenizer@815a03f1b57dcb5ae918db415f22cd0c8a6f0f54"
            files: [config.json, model.safetensors, preprocessor_config.json]
        settings: { codec_device: cpu }
      # mps: the same entry with settings: { codec_device: mps }
  sa3:
    label: "Stable Audio 3 Medium"
    kind: sound
    capabilities: { lyrics: false, duration: true, voice_clone: false }
    limits: { max_seconds: 380 }
    settings: { steps: 8, cfg_scale: 1.0, sample_size: 16777216 }
    memory: { cuda_gb: 6.1, mps_gb: null }
    platforms:
      cuda:
        family: stable_audio_3
        files:
          - repo: "Comfy-Org/stable-audio-3@75aad8836271cb5a5123fb58304fae5d3bb4b0f9"
            files: [checkpoints/stable_audio_3_medium.safetensors]
          - repo: "cocktailpeanut/stable-audio-3-medium@a30034d70bd58f6e6f967ef28ecaa6c52c185d6c"
            files: [model_config.json]
          - repo: "mlx-community/stable-audio-3-small-sfx@ca6c38718d27cd2b446a4661c8851f1f3d0a3870"
            files: ["t5gemma-b-b-ul2/*"]
      # mps: the same entry
```

| Key | Required | Description |
|---|---|---|
| `label` | yes | The name the Web UI shows |
| `kind` | yes | `music`, `speech` or `sound`. The Web UI lists a kind's models in the order they are written here |
| `capabilities.lyrics` | yes | `true` if the model sings lyrics |
| `capabilities.duration` | yes | `true` if you choose the length in seconds |
| `capabilities.voice_clone` | yes | `true` if the model can follow a voice recording |
| `limits` | yes | The model's own ceilings. Music and sound models give `max_seconds`, the longest take. Speech models give `max_new_tokens` (per part), `part_chars`, `max_text_chars`, `min_voice_seconds` and `max_voice_seconds` |
| `settings` | no | The model's own knobs, shared by every platform. Any keys the model reads, such as `temperature` or `ode_steps` |
| `memory.cuda_gb` | yes | The GPU memory the model peaks at on a CUDA card, in GiB |
| `memory.mps_gb` | yes | The same on Apple silicon; `null` while it is unmeasured |
| `platforms.cuda`, `platforms.mps` | yes, at least one | Where the model runs. A model without an entry for your platform is not offered |
| `platforms.<platform>.family` | yes | Which engine runs the model: `yue2`, `higgs3` or `stable_audio_3` |
| `platforms.<platform>.files` | yes | The downloads, in order. Each entry has `repo`, as `owner/name@commit` with the full 40-character commit, and `files`, the file names or patterns to fetch from it |
| `platforms.<platform>.settings` | no | Settings for this platform only. They win over the entry's `settings` |

Each limit and capability is required, and the app refuses to start when one is missing, when `kind` is unknown, or when a platform has no `family` or `files`.

## `ideogram4_instant`

The pinned sources Ideogram 4 Instant is built from: the transformer, the base components per platform, and the transformer format. Changing a pin rebuilds the model folder. See [Ideogram 4 Instant](models.md#ideogram-4-instant).

## `moss_image`

The pinned sources MOSS-Image is built from: `sft`, the SFT repository as `owner/repo@revision`, and `turbo_lora`, the repository, revision and file name of the 4-step LoRA that makes the `moss` (Turbo) model. The `moss` and `moss-sft` model folders are built from them on first use. Changing a pin rebuilds the affected folder and, for a new LoRA, leaves the SFT folder alone. The packaged SFT revision is the one the bundled pipeline code was written for, so change it only together with that code.

## Other sections

`schedulers`, `platforms` and `model_alias_families` are internal: they map scheduler names to classes, platform ids to names, and aliases to model families.
