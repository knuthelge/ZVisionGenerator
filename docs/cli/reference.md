# Command reference

Every command and option. For worked examples, see the how-to pages: [images](images.md), [videos](video.md), [audio](audio.md), [batches](batches.md), [enhancing](enhance.md) and [models](models.md).

## Commands

| Command | Same as | Does |
|---|---|---|
| `ziv` | | Prints help; runs the subcommands below |
| `ziv-ui` | `ziv ui` | Starts the Web UI |
| `ziv-image` | `ziv image` | Generates images |
| `ziv-video` | `ziv video` | Generates and extends videos |
| `ziv-audio` | `ziv audio` | Generates music, speech and sound |
| `ziv-model` | `ziv model` | Converts checkpoints, imports LoRAs, lists assets |

`ziv` with no subcommand prints help and starts nothing. `ziv --version` prints the version. `ziv <command> --help` lists a command's options.

## `ziv ui` / `ziv-ui`

| Option | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Interface to listen on. `local` is short for `0.0.0.0`, which makes the Web UI reachable from other devices on your network, and prints this machine's network address |
| `--port` | `8080` | Preferred port; the next free one is used if it's busy |
| `--no-browser` | off | Don't open a browser tab |
| `--background` | off | Run in the background and return to the prompt once it's reachable. Keeps running after the terminal closes; its output goes only to the error log |
| `--background-stop` | off | Stop every running Web UI, started with `--background` or running in another terminal, and any job it is running. Can't be combined with `--background` |

Every running server records itself in `run/` in the data folder, which is how `--background-stop` finds it. A server is asked to shut down and is force-stopped, with any process it started, if it hasn't exited after 10 seconds. On Windows it is stopped at once.

The server only answers requests addressed to an IP address, `localhost`, the `--host` value, or (when listening on `0.0.0.0`, `::` or a network address) this machine's name, bare and with `.local`. It refuses state-changing requests from other sites. Add other names to `ZIV_UI_ALLOWED_HOSTS`.

## `ziv-image`

| Option | Default | Description |
|---|---|---|
| `-m`, `--model` | *(required)* | Model alias, name in `~/.ziv/models/`, local path, or Hugging Face repo |
| `-q`, `--quantize` | none | Quantization: `4` or `8` |
| `--prompt` | none | Inline prompt; overrides `--prompts-file`. Can't be combined with `--json-prompt` |
| `--json-prompt` | none | A structured JSON caption (Ideogram 4), sent as is: must be a JSON object, skips `{a\|b}` choices, overrides `--prompts-file`. Can't be combined with `--prompt` |
| `-p`, `--prompts-file` | `prompts.yaml` | YAML prompt file |
| `-r`, `--runs` | `1` | How many times to run every prompt |
| `-s`, `--size` | `m` | Size preset: `xs`, `s`, `m`, `l`, `xl` |
| `--ratio` | `2:3` | Aspect ratio: `1:1`, `16:9`, `9:16`, `3:2`, `2:3` |
| `-W`, `--width` | from preset | Exact width |
| `-H`, `--height` | from preset | Exact height |
| `--steps` | per model | Denoising steps |
| `--guidance` | per model | Guidance scale |
| `--scheduler` | per model | `beta` or `simple` |
| `--seed` | random | Seed, for repeatable images |
| `--first-sigma` | `1.004` | Ideogram 4 only: first-step sigma, in `(0.0, 2.0]`. `1.005`–`1.006` more reliably avoid a grey "blocked by safety filter" frame for harmless prompts |
| `--lora` | none | LoRAs with optional weights: `name:0.8,name2:0.5` (default weight 1.0) |
| `--image` | none | Start image for image-to-image |
| `--image-strength` | `0.5` | How far the result may move from the start image, 0–1 |
| `--reference` | none | Comma-separated reference images whose content the result keeps, named "image 1", "image 2" in the prompt; FLUX.2 Klein takes up to 4. Combines with `--image` |
| `--upscale` | off | Upscale factor: `2` or `4` |
| `--upscale-denoise` | from config | Denoise strength of the upscale pass |
| `--upscale-steps` | per model | Steps of the upscale pass |
| `--upscale-guidance` | `--guidance` | Guidance of the upscale pass only |
| `--upscale-sharpen` / `--no-upscale-sharpen` | per model: on, off for Krea 2 | Sharpen before the upscale pass |
| `--upscale-save-pre` / `--no-upscale-save-pre` | off | Also save the image before upscaling, with a `_preupscale` suffix |
| `--review` | off | Pause after each small render to approve (++y++) or discard (++n++) the upscale. Needs `--upscale` and a terminal |
| `--sharpen [AMOUNT]` / `--no-sharpen` | on | Sharpen the result. An amount (0–1.5) overrides the config |
| `--contrast [AMOUNT]` / `--no-contrast` | off | Adjust contrast (1.0 = no change) |
| `--saturation [AMOUNT]` / `--no-saturation` | off | Adjust saturation (1.0 = no change) |
| `--transparent` | off | Cut the subject out and save a PNG with a transparent background. Non-commercial use only. See [Transparent background](../concepts/images.md#transparent-background) |
| `-o`, `--output` | `.` | Output folder |
| `--enhance [SPEC]` | off | Rewrite each prompt before the model loads. See [the spec](enhance.md#the-spec). Overrides prompt-file `enhance:` entries. Ignored with `--json-prompt` |
| `--enhance-note TEXT` | none | A direction for the enhancer, up to 300 characters. Implies `--enhance` |
| `--enhance-model` | `fast` | `fast`, `smart`, `REPO[@REVISION]` or a local folder |
| `--enhance-format` | `text` | `json` writes an Ideogram 4 JSON caption. Needs a model that takes them (`ideo`) |
| `--no-enhance` | | Turn enhancement off, including prompt-file `enhance:` entries |

## `ziv-video`

| Option | Default | Description |
|---|---|---|
| `-m`, `--model` | *(required)* | `ltx-4`, `ltx-8`, `ltx25-4`, `h3`, a converted LTX model, a local path, or a supported LTX or MiniMax H3 repo. With `--extend`, defaults to the source's model |
| `--prompt` | none | The action: what happens. Overrides `--prompts-file`. May be empty with `--scene` or `--extend` |
| `--scene` | none | Who and where, the look and the camera. Sent before the prompt. Fills prompt-file entries without a `scene:`. With `--extend`, defaults to the source's scene |
| `-p`, `--prompts-file` | `prompts.yaml` | YAML prompt file |
| `-r`, `--runs` | `1` | How many times to run every prompt |
| `--image` | none | First frame, for image-to-video |
| `--end-image` | none | Last frame: the clip runs from `--image` to it. Needs `--image` |
| `-s`, `--size` | `s` | Size preset: `xs`, `s`, `m`, `l`, `xl` |
| `--ratio` | `16:9` | Aspect ratio: `16:9`, `9:16`, `1:1` |
| `-W`, `--width` | from preset | Width; aligned to 32 px (64 with `--upscale`) |
| `-H`, `--height` | from preset | Height; aligned to 32 px (64 with `--upscale`) |
| `--frames` | 121 (`h3`: 124) | Length in frames: `8k+1` for LTX (another count moves to the nearest), `17k+5` for MiniMax H3 (another count moves up to the next) |
| `--steps` | `8` | Steps, at most 8. MiniMax H3 takes `4` or `8`; another count moves to the nearest, a tie to 8 |
| `--seed` | random | Seed, for repeatable clips |
| `--lora` | none | LoRAs with optional weights: `name:0.8,name2:0.5`. LTX on Windows and Linux needs diffusers-compatible LTX LoRAs; `h3` takes MiniMax H3 LoRAs |
| `--upscale` | off | `2` only: render at half size, upsample 2×, refine at full size. The first use on Windows and Linux downloads the upsampler (about 1 GB; 0.69 GB for `h3`). `h3` has no upscale on macOS |
| `--review` | off | Pause after each small render to approve (++y++) or discard (++n++) the upscale. Needs `--upscale` and a terminal |
| `--audio` / `--no-audio` | on | Keep or remove the sound |
| `--audio-file PATH` | none | An audio file the clip follows; its sound is exactly this track. LTX and MiniMax H3, with `--extend` too. See [Audio track](../concepts/video.md#audio-track) |
| `--audio-start SECONDS` | `0` | Where in the track the clip's sound starts. Needs `--audio-file`. With `--extend` and the same file the source followed, the default is where the source's sound ended |
| `--low-memory` / `--no-low-memory` | on | LTX: free each part of the model once it has run. On Macs with 24 GB or less it also streams the transformer from disk. MiniMax H3 always streams its blocks |
| `--extend SRC` | none | Continue the video `SRC` and write a new file. See [Extending a video](../concepts/video.md#extending-a-video) |
| `--extend-seconds S` | `2` | Length to add with `--extend`, at the source's frame rate, rounded to a multiple of the model's step: 8 frames for LTX, 17 for MiniMax H3 |
| `--extend-frames N` | | Frames to add with `--extend`, rounded to a multiple of the step (8 for LTX, 17 for MiniMax H3; at least one step). Can't be combined with `--extend-seconds` |
| `-o`, `--output` | `.` | Output folder |
| `--format` | `mp4` | Output format |
| `--enhance [SPEC]` | off | Rewrite each prompt before the model loads. See [the spec](enhance.md#the-spec) |
| `--enhance-note TEXT` | none | A direction for the enhancer. Implies `--enhance` |
| `--enhance-model` | `fast` | `fast`, `smart`, `REPO[@REVISION]` or a local folder |
| `--no-enhance` | | Turn enhancement off, including prompt-file `enhance:` entries |

With `--extend`, these are refused because the extension keeps the source's size and sets its own length: `--image`, `--end-image`, `--upscale`, `--ratio`, `--size`, `--width`, `--height` and `--frames`.

MiniMax H3 extends only 24 fps sources whose width and height are multiples of 32. It prints its estimated time when a job starts; the estimate leaves out the one-time preparation of the first run.

`ziv-video` needs [ffmpeg](https://ffmpeg.org/). If it's missing, `ziv-video` offers to install it with Homebrew or MacPorts (macOS), `winget` (Windows), or `apt`, `dnf` or `pacman` (Linux), and asks before running anything. On Windows, open a new terminal afterwards. On Fedora, `dnf install ffmpeg` needs RPM Fusion.

## `ziv-audio`

| Option | Default | Description |
|---|---|---|
| `-m`, `--model` | *(required)* | The audio model: `yue2` (music), `qwen3_tts` (speech) or `sa3` (sound), whose settings are in [`audio_model_presets`](../concepts/configuration.md#audio_model_presets). The model decides which of the options below it takes |
| `--kind` | the model's first kind | What to make: `music`, `speech` or `sound`. A model that makes one kind needs none. A kind the model doesn't make is refused, naming the kinds it does. The kind decides which of the options below apply |
| `--prompt` | *(required)* | The style of a song, the text to read (up to 10,000 characters, read in parts, with [Delivery marks](../concepts/audio.md#marks) in square brackets), or the description of a sound |
| `--lyrics` | none | The lyrics of a song, with `[verse]` and `[chorus]` tags. Without lyrics, or with only `[section]` tags, the take is an instrumental, and the tags set its length. Music models only |
| `--lyrics-file PATH` | none | Read the lyrics from a text file. Can't be combined with `--lyrics` |
| `--seconds` | Auto | Length in seconds, from 1 up to the model's limit. Left out, a sound is 12 s long (ambience 12, effect 5, one-shot 2, by `--sound-type`) and music from a model without lyrics is 30 s. Models that set their own length (YuE2 from the lyrics, speech from the text) refuse it |
| `--tempo BPM` | Auto | The tempo of a song, a whole number from 40 to 220. Music only |
| `--key ID` | Auto | The key of a song: `C`, `Db`, `D`, `Eb`, `E`, `F`, `F#`, `G`, `Ab`, `A`, `Bb`, `B` for the major keys, and `Cm`, `C#m`, `Dm`, `Ebm`, `Em`, `Fm`, `F#m`, `Gm`, `G#m`, `Am`, `Bbm`, `Bm` for the minor ones. Music only |
| `--sound-type TYPE` | `auto` | The type of a sound: `auto`, `ambience`, `effect` or `one-shot`. It sets the Auto length of `--seconds`. Sound only |
| `--emotion EMOTION` | `neutral` | How speech in a built-in voice feels: `neutral`, `elated`, `amused`, `enthusiastic`, `proud`, `content`, `affectionate`, `relieved`, `longing`, `thoughtful`, `determined`, `awed`, `surprised`, `confused`, `sad`, `bitter`, `ashamed`, `helpless`, `afraid`, `angry` or `disgusted`. Speech in a built-in voice only; refused together with `--voice` |
| `--manner MANNER` | `natural` | `natural`, `whisper` or `shout`. Same rule as `--emotion` |
| `--pace PACE` | `normal` | `very-slow`, `slow`, `normal`, `fast` or `very-fast`. Same rule as `--emotion` |
| `--pitch PITCH` | `normal` | `low`, `normal` or `high`. Same rule as `--emotion` |
| `--expressiveness LEVEL` | `normal` | `low`, `normal` or `high`. Same rule as `--emotion` |
| `--vibe ID` | none | Start the delivery of speech in a built-in voice from a named style: `calm-narrator`, `storyteller`, `excited-host`, `announcer`, `sad-confession` or `whispered-secret`. `--emotion`, `--manner`, `--pace`, `--pitch` and `--expressiveness` then change one. Same rule as `--emotion` |
| `--speaker ID` | the model's first voice | The built-in voice a speech model reads in, such as `ryan` or `vivian`. Refused together with `--voice` |
| `--voice PATH` | none | A recording of the voice to follow, 3 to 30 seconds long. Speech models that clone a voice only. Needs `--voice-text`. Without it the model reads in a built-in voice (`--speaker`) |
| `--voice-text` | none | What the `--voice` recording says |
| `--voice-consent` | off | Confirms the speaker of the `--voice` recording allows cloning their voice. Needed with `--voice`, except for a speech take made in a built-in voice, which anyone may follow |
| `--enhance [SPEC]` | off | Rewrite a sound's description or a song's style before the model loads. A sound takes `space`, `distance`, `over_time` and `movement`; music takes `genre`, `music_mood`, `instruments`, `without`, `singer` and `production`. Put `+` between several values, as in `instruments=guitar+violin`. See [Enhance the prompt](audio.md#enhance-the-prompt). Refused for speech |
| `--enhance-note TEXT` | none | A direction for the enhancer, up to 300 characters. Implies `--enhance` |
| `--no-enhance` | | Turn enhancement off |
| `--enhance-model` | `fast` | `fast`, `smart`, `REPO[@REVISION]` or a local folder. Needs `--enhance` |
| `--enhance-format` | `text` | Only `text` |
| `-r`, `--runs` | `1` | How many takes to make. They share one model load |
| `--seed` | random | Seed, for repeatable takes. Every take of a run uses it; omit it for a new random seed per take |
| `-o`, `--output` | `.` | Output folder. Takes are saved as FLAC files named `<kind>_<date>_<time>.flac`, with their settings inside |

An option the chosen model doesn't take as the chosen kind is refused with the reason, such as `Qwen3-TTS 1.7B has no tempo; drop --tempo`. `ziv-audio` needs [ffmpeg](https://ffmpeg.org/), which it offers to install like `ziv-video` does.

## `ziv-model model`

| Option | Default | Description |
|---|---|---|
| `-i`, `--input` | *(required)* | The `.safetensors` checkpoint |
| `--name` | file name | Name of the model folder |
| `--model-type` | `zimage` | `zimage`, `flux2-klein-4b`, `flux2-klein-9b`, `krea2-turbo`, `ltx-2.3`, `ltx-2.5`, `h3` |
| `--base-model` | `Tongyi-MAI/Z-Image-Turbo` | Base repo, `zimage` only |
| `--copy` | off | Copy the base model's files instead of linking them |
| `--quantize` | off | Also save a `<name>@q4` or `<name>@q8` copy (`4` or `8`). Required for `ltx-2.3` and `ltx-2.5`, where it is the only level stored; `h3` takes none |
| `--distill` | off | `ltx-2.3` and `ltx-2.5` only: merge Lightricks' distilled LoRA into a plain-dev checkpoint |
| `--turbo` | off | `h3` only: the checkpoint is already step-distilled, so the turbo LoRA is not merged |
| `--force` | off | Video types only: replace an existing model of that name once the new one is complete |

## `ziv-model lora`

| Option | Default | Description |
|---|---|---|
| `-i`, `--input` | | A local `.safetensors` file. Can't be combined with `--hf` |
| `--hf` | | A Hugging Face repo. Can't be combined with `-i` |
| `--file` | detected | The file to take from a repo with several |
| `--name` | file name | Name of the LoRA |

## `ziv-model list`

| Option | Default | Description |
|---|---|---|
| `--models` | off | Show only models |
| `--loras` | off | Show only LoRAs |

## Environment variables

| Variable | Does |
|---|---|
| `ZIV_DATA_DIR` | Use another [data folder](../concepts/data-folder.md) instead of `~/.ziv` |
| `ZIV_LOG` | `off` turns the [error log](../concepts/data-folder.md#error-log) off |
| `ZIV_UI_ALLOWED_HOSTS` | Extra host names the Web UI answers to, comma-separated |
| `HF_TOKEN` | Hugging Face token for gated models. See [Hugging Face token](../concepts/data-folder.md#hugging-face-token) |
| `HF_XET_HIGH_PERFORMANCE` | `1` lets downloads use all the bandwidth and CPU cores |
| `HF_HOME` | Where Hugging Face keeps its cache |
