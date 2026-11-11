# Command reference

Every command and option. Error? See [Troubleshooting](../reference/troubleshooting.md). For worked commands, see the recipes: [images](images.md), [video](video.md), [audio](audio.md), [batches](batches.md) and [models](models.md); the [CLI overview](index.md) has the first commands, where files go and how to script them.

## Commands

| Command | Same as | Does |
|---|---|---|
| `ziv` | | Prints help; runs the subcommands below |
| `ziv-ui` | `ziv ui` | Starts the Web UI |
| `ziv-image` | `ziv image` | Generates images |
| `ziv-video` | `ziv video` | Generates and extends videos |
| `ziv-audio` | `ziv audio` | Generates music, speech and sound |
| `ziv-model` | `ziv model` | Converts checkpoints, imports LoRAs, lists assets |

`ziv` with no subcommand prints help and starts nothing, and an unknown subcommand is an error. `ziv --version` prints `ziv <version>`; it is read only as the first argument. `ziv <command> --help` lists a command's options.

An option with a value is written `--option VALUE` or `--option=VALUE`. A value with spaces needs quotes: `--prompt "A red fox"`.

## `ziv ui` / `ziv-ui`

| Option | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Interface to listen on. `local` is short for `0.0.0.0`, which makes the Web UI reachable from other devices on your network, and prints this machine's network address |
| `--port` | `8080` | Preferred port; the next free one is used if it's busy |
| `--no-browser` | off | Don't open a browser tab |
| `--background` | off | Run in the background and return to the prompt once it's reachable. Keeps running after the terminal closes; its output goes only to the error log |
| `--background-stop` | off | Stop every running Web UI, started with `--background` or running in another terminal, and any job it is running. Can't be combined with `--background` |

Every running server records itself in `run/` in the data folder, which is how `--background-stop` finds it. A server is asked to shut down and is force-stopped, with any process it started, if it hasn't exited after 10 seconds. On Windows it is stopped at once.

If the preferred port and the ones after it are all taken, the command stops with `No available port found starting at <port>`.

The server only answers requests addressed to an IP address, `localhost`, the `--host` value, or (when listening on `0.0.0.0`, `::` or a network address) this machine's name, bare and with `.local`. It refuses state-changing requests from other sites. Add other names to `ZIV_UI_ALLOWED_HOSTS`.

## `ziv-image`

| Option | Default | Description |
|---|---|---|
| `-m`, `--model` | *(required)* | Model alias, name in `models/` in the data folder, local path, or Hugging Face repo. The repo id of an alias runs as that alias |
| `-q`, `--quantize` | none | Quantization: `4` or `8`. A family that has none refuses it |
| `--prompt` | none | Inline prompt; overrides `--prompts-file`. Can't be empty or combined with `--json-prompt` |
| `--json-prompt` | none | A structured JSON caption (Ideogram 4), sent as is: must be a JSON object, skips `{a\|b}` choices, overrides `--prompts-file`. Can't be combined with `--prompt`. Other families warn and take it as a plain prompt |
| `-p`, `--prompts-file` | `prompts.yaml` | YAML prompt file |
| `-r`, `--runs` | `1` | How many times to run every prompt; at least 1. Each run picks `{a\|b}` choices again and, without `--seed`, uses a new seed |
| `-s`, `--size` | `m` | Size preset: `xs`, `s`, `m`, `l`, `xl`. The default is [`generation.default_size`](../reference/configuration.md#generation) |
| `--ratio` | `2:3` | Aspect ratio: `1:1`, `16:9`, `9:16`, `3:2`, `2:3`. The default is [`generation.default_ratio`](../reference/configuration.md#generation) |
| `-W`, `--width` | from preset | Exact width: positive and a multiple of 16, for every model |
| `-H`, `--height` | from preset | Exact height: positive and a multiple of 16, for every model |
| `--steps` | per model | Denoising steps; at least 1 |
| `--guidance` | per model | Guidance scale; 0 or more |
| `--scheduler` | per model | `beta` or `simple`. A family with its own sampler, such as Ideogram 4, refuses it |
| `--seed` | random, new for every run | Seed, for repeatable images. A seed you give is used for every run, so `-r 4 --seed 42` repeats one image four times |
| `--first-sigma` | `1.004` | Ideogram 4 only, ignored with a warning on other models: first-step sigma, above 0 and up to 2. `1.005`–`1.006` more reliably avoid a grey "blocked by safety filter" frame for harmless prompts |
| `--lora` | none | LoRAs with optional weights: `name:0.8,name2:0.5` (default weight 1.0). A bare name is a file in `loras/` in the data folder. A Hugging Face repo is refused: import it with [`ziv-model lora --hf`](#ziv-model-lora) |
| `--image` | none | Start image for image-to-image; the file must exist |
| `--image-strength` | `0.5` | How far the result may move from the start image, 0–1 |
| `--reference` | none | Comma-separated reference images (PNG, JPEG or WebP) whose content the result keeps, named "image 1", "image 2" in the prompt; FLUX.2 Klein takes up to 4. Combines with `--image` |
| `--edit` | none | Edit this picture instead of generating one; the file must exist. Needs `--prompt` and a FLUX.2 Klein model (`-m`). Refuses the options that generate a new picture: `--image`, `--reference`, `--upscale`, `--review`, `--enhance`, `--enhance-note`, `--enhance-format`, `--transparent`, `--json-prompt`, `-W`, `-H`, `--ratio`, `-s`, `--guidance`, `--scheduler`, `--contrast`, `--saturation` and `--sharpen AMOUNT`. See [Edit a picture](images.md#edit-a-picture) |
| `--mask` | none | The area of the `--edit` picture to redraw, as a PNG of the picture's shape (any size of it). Where the PNG has transparency, any pixel that isn't fully transparent is painted; otherwise any pixel that isn't black. Needs `--edit`; can't be combined with `--extend` |
| `--extend` | none | Grow the `--edit` picture on one side: `top`, `right`, `bottom` or `left`, then `=` and 128 to 512 pixels in steps of 16, such as `top=256`. Needs `--edit`; can't be combined with `--mask`. [`ziv-video --extend`](#ziv-video) continues a video instead |
| `--upscale` | off | Upscale factor: `2` or `4`. An explicit `-W` or `-H` must survive it: its value divided by the factor must already be a multiple of 16 |
| `--upscale-denoise` | from config | Denoise strength of the upscale pass, 0–1. The default is [`upscale.default_denoise_2x` or `default_denoise_4x`](../reference/configuration.md#upscale) |
| `--upscale-steps` | per model | Steps of the upscale pass; at least 1 |
| `--upscale-guidance` | `--guidance` | Guidance of the upscale pass only; 0 or more |
| `--upscale-sharpen` / `--no-upscale-sharpen` | per model: on, off for Krea 2 | Sharpen before the upscale pass |
| `--upscale-save-pre` / `--no-upscale-save-pre` | off | Also save the image before upscaling, with a `_preupscale` suffix |
| `--review` | off | Pause after each small render to approve (++y++) or discard (++n++) the upscale, or quit (++q++). Needs `--upscale` and a terminal. See [Controls during a run](../guides/batches.md#keys-during-a-run) |
| `--sharpen [AMOUNT]` / `--no-sharpen` | on | Sharpen the result. An amount (0–1.5) overrides the config |
| `--contrast [AMOUNT]` / `--no-contrast` | off | Adjust contrast (1.0 = no change); 0 or more |
| `--saturation [AMOUNT]` / `--no-saturation` | off | Adjust saturation (1.0 = no change); 0 or more |
| `--transparent` | off | Cut the subject out and save a PNG with a transparent background; downloads the cut-out models first. Non-commercial use only. See [Transparent background](../guides/images.md#transparent-background) |
| `-o`, `--output` | `.` | Output folder. With `--edit`, the edited picture's folder when left out |
| `--enhance [SPEC]` | off | Rewrite each prompt before the model loads. See [the spec](#the-spec). Overrides prompt-file `enhance:` entries. Ignored with `--json-prompt` |
| `--enhance-note TEXT` | none | A direction for the enhancer, up to 300 characters. Implies `--enhance` |
| `--enhance-model` | `fast` | `fast`, `smart`, `REPO[@REVISION]` or a local folder |
| `--enhance-format` | `text` | `json` writes an Ideogram 4 JSON caption. Needs a model that takes them (`ideo`) |
| `--no-enhance` | | Turn enhancement off, including prompt-file `enhance:` entries |

With `--upscale`, a preset size that doesn't divide evenly by the factor drifts slightly (`xs` and `s` at 4×). Give `-W` and `-H` to fix the exact size.

## `ziv-video`

| Option | Default | Description |
|---|---|---|
| `-m`, `--model` | *(required)* | `ltx-4`, `ltx-8`, `ltx25-4`, `h3`, a converted LTX model, a local path, or a supported LTX or MiniMax H3 repo. With `--extend`, defaults to the source's model |
| `--prompt` | none | The action: what happens. Overrides `--prompts-file`. May be empty with `--scene` or `--extend` |
| `--scene` | none | Who and where, the look and the camera. Sent before the prompt. Fills prompt-file entries without a `scene:`. With `--extend`, defaults to the source's scene |
| `-p`, `--prompts-file` | `prompts.yaml` | YAML prompt file |
| `-r`, `--runs` | `1` | How many times to run every prompt; at least 1 |
| `--image` | none | First frame, for image-to-video; the file must exist |
| `--end-image` | none | Last frame: the clip runs from `--image` to it. Every video model takes one. Needs `--image` |
| `-s`, `--size` | `s` | Size preset: `xs`, `s`, `m`, `l`, `xl`. The default is [`video_generation.default_size`](../reference/configuration.md#video_generation-and-video_sizes) |
| `--ratio` | `16:9` | Aspect ratio: `16:9`, `9:16`, `1:1`, or any ratio of your [`video_sizes`](../reference/configuration.md#video_generation-and-video_sizes) |
| `-W`, `--width` | from preset | Width; aligned to 32 px (64 with `--upscale`). At least 64×64 after alignment |
| `-H`, `--height` | from preset | Height; aligned to 32 px (64 with `--upscale`). At least 64×64 after alignment |
| `--frames` | 121 (`h3`: 124) | Length in frames: `8k+1` for LTX (another count moves to the nearest), `17k+5` for MiniMax H3 (another count moves up to the next) |
| `--steps` | `8` | Steps, from 1 to 8. MiniMax H3 takes `4` or `8`; another count moves to the nearest, a tie to 8 |
| `--seed` | random | Seed, for repeatable clips. A seed you give is used for every run |
| `--lora` | none | LoRAs with optional weights: `name:0.8,name2:0.5`. LTX on Windows and Linux needs diffusers-compatible LTX LoRAs; `h3` takes MiniMax H3 LoRAs. With `--extend`, defaults to the source's LoRAs that are still installed |
| `--upscale` | off | `2` only: render at half size, upsample 2×, refine at full size. The first use on Windows and Linux downloads the upsampler (about 1 GB; 0.69 GB for `h3`). `h3` has no upscale on macOS |
| `--review` | off | Pause after each small render to approve (++y++) or discard (++n++) the upscale, or quit (++q++). Needs `--upscale`, a terminal and a video backend that supports it |
| `--audio` / `--no-audio` | on | Keep or remove the sound |
| `--audio-file PATH` | none | An audio file the clip follows; its sound is exactly this track, silence after it ends, and `--no-audio` is overridden. LTX and MiniMax H3, with `--extend` too. See [Audio track](../guides/video.md#audio-track) |
| `--audio-start SECONDS` | `0` | Where in the track the clip's sound starts. Needs `--audio-file`. With `--extend` and the same file the source followed, the default is where the source's sound ended |
| `--low-memory` / `--no-low-memory` | on | LTX: free each part of the model once it has run. On Macs with 24 GB or less it also streams the transformer from disk. MiniMax H3 always streams its blocks |
| `--extend SRC` | none | Continue the video `SRC` and write a new file. See [Extending a video](../guides/video.md#extend-a-clip) |
| `--extend-from SECONDS` | end of the video | Continue `--extend` from this point in the video instead of its end. The frame at that time is the last one kept and what follows it is dropped; the source file is not changed, and the new file holds the kept part plus the new part. Needs at least 25 frames before the point; a point at or past the last frame keeps the whole video. With a track from `--audio-file`, the default start moves back by the dropped frames. Needs `--extend` |
| `--extend-seconds S` | `2` | Length to add with `--extend`, at the source's frame rate, rounded to a multiple of the model's step: 8 frames for LTX, 17 for MiniMax H3 |
| `--extend-frames N` | | Frames to add with `--extend`, rounded to a multiple of the step (8 for LTX, 17 for MiniMax H3; at least one step). Can't be combined with `--extend-seconds` |
| `-o`, `--output` | `.` | Output folder |
| `--format` | `mp4` | Output format; only `mp4` |
| `--enhance [SPEC]` | off | Rewrite each prompt before the model loads. See [the spec](#the-spec) |
| `--enhance-note TEXT` | none | A direction for the enhancer. Implies `--enhance` |
| `--enhance-model` | `fast` | `fast`, `smart`, `REPO[@REVISION]` or a local folder |
| `--no-enhance` | | Turn enhancement off, including prompt-file `enhance:` entries |

With `--extend`, these are refused because the extension keeps the source's size and sets its own length: `--image`, `--end-image`, `--upscale`, `--ratio`, `--size`, `--width`, `--height` and `--frames`.

Without `--model`, `--extend` uses the model the source was made with. If that model isn't downloaded, a warning says so and the default video model (`ui.default_models.video`) runs instead. A LoRA of the source that is no longer installed is skipped with a warning. A source made before scenes were recorded has only a prompt: leaving `--prompt` out keeps it, and giving it again unchanged warns `Still the original prompt. Change it to what happens next.`

A prompt file works with `--extend` (`-p prompts.yaml`): every prompt in it continues the source and makes its own new file, and `-r` repeats each one. Without `-p` or `--prompt`, a single extension runs.

MiniMax H3 extends only 24 fps sources whose width and height are multiples of 32. It prints its estimated time when a job starts; the estimate leaves out the one-time preparation of the first run.

`ziv-video` needs [ffmpeg](#ffmpeg).

## `ziv-audio`

| Option | Default | Description |
|---|---|---|
| `-m`, `--model` | *(required)* | The audio model: `yue2` (music), `qwen3_tts` (speech) or `sa3` (sound), whose settings are in [`audio_model_presets`](../reference/configuration.md#audio_model_presets). The model decides which of the options below it takes |
| `--kind` | the model's first kind | What to make: `music`, `speech` or `sound`. A model that makes one kind needs none. A kind the model doesn't make is refused, naming the kinds it does. The kind decides which of the options below apply |
| `--prompt` | *(required)* | The style of a song, the text to read (up to 10,000 characters, read in parts, with [Delivery marks](../guides/speech-and-sound.md#marks) in square brackets), or the description of a sound |
| `--lyrics` | none | The lyrics of a song, with `[verse]` and `[chorus]` tags. Without lyrics, or with only `[section]` tags, the take is an instrumental, and the tags set its length. `yue2` only: `sa3` and speech refuse it with `… takes no lyrics; drop --lyrics` |
| `--lyrics-file PATH` | none | Read the lyrics from a text file. Can't be combined with `--lyrics`. `yue2` only, like `--lyrics` |
| `--seconds` | Auto | Length in seconds, from 1 up to the model's limit. Left out, the length depends on what you make: a sound by its `--sound-type` (`auto` 12 s, `ambience` 12 s, `effect` 5 s, `one-shot` 2 s) and music from `sa3` 30 s. `yue2` (the length comes from the lyrics) and speech (from the text) refuse it |
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
| `--voice PATH\|NAME` | none | A recording of the voice to follow, 3 to 30 seconds long, or the name of a saved voice (saved in the Web UI; the name in any case, or its file slug). A file by that name is used as a recording first. A recording needs `--voice-text`. Speech models that clone a voice only. Without it the model reads in a default voice (`--speaker`) |
| `--voice-text` | none | What the `--voice` recording says. Refused with a saved voice, which has its own transcript |
| `--voice-consent` | off | Confirms the speaker of the `--voice` recording allows cloning their voice. Needed with a recording, except for a speech take made in a default voice, which anyone may follow. Refused with a saved voice |
| `--enhance [SPEC]` | off | Rewrite a sound's description or a song's style before the model loads. A sound takes `space`, `distance`, `over_time` and `movement`; music takes `genre`, `music_mood`, `instruments`, `without`, `singer` and `production`. Put `+` between several values, as in `instruments=guitar+violin`. The values are in [the audio spec](#the-audio-spec). See [Enhance the prompt](../guides/music.md#enhance-the-prompt). Refused for speech |
| `--enhance-note TEXT` | none | A direction for the enhancer, up to 300 characters. Implies `--enhance` |
| `--no-enhance` | | Turn enhancement off |
| `--enhance-model` | `fast` | `fast`, `smart`, `REPO[@REVISION]` or a local folder. Needs `--enhance` |
| `--enhance-format` | `text` | Only `text` |
| `-r`, `--runs` | `1` | How many takes to make, at least 1. They share one model load |
| `--seed` | random | Seed, for repeatable takes, from 0 to 9223372036854775807. Every take of a run uses it; omit it for a new random seed per take |
| `-o`, `--output` | `.` | Output folder. Takes are saved as FLAC files named `<kind>_<date>_<time>.flac`, with their settings inside |

An option the chosen model doesn't take as the chosen kind is refused with the reason, such as `Qwen3-TTS 1.7B has no tempo; drop --tempo`. `ziv-audio` needs [ffmpeg](#ffmpeg).

## ffmpeg { #ffmpeg }

`ziv-video` and `ziv-audio` need [ffmpeg](https://ffmpeg.org/). If it's missing, they offer to install it with Homebrew or MacPorts (macOS), `winget` (Windows), or `apt`, `dnf` or `pacman` (Linux), and ask before running anything. On Windows, open a new terminal afterwards. On Fedora, `dnf install ffmpeg` needs RPM Fusion. If you decline, or the install fails, the command stops with status `1`; a script can't answer the question, so install ffmpeg first.

## `ziv-model model`

| Option | Default | Description |
|---|---|---|
| `-i`, `--input` | *(required)* | The `.safetensors` checkpoint |
| `--name` | file name | Name of the model folder. A model alias, or a name ending in `@q4`, `@q8` or `@mlx-te4`, is refused |
| `--model-type` | `zimage` | `zimage`, `flux2-klein-4b`, `flux2-klein-9b`, `krea2-turbo`, `ltx-2.3`, `ltx-2.5`, `h3` |
| `--base-model` | `Tongyi-MAI/Z-Image-Turbo` | Base repo, `zimage` only |
| `--copy` | off | Copy the base model's files instead of linking them |
| `--quantize` | off | Also save a `<name>@q4` or `<name>@q8` copy (`4` or `8`), which runs only when the model is used with the same level (`ziv-image -m <name> -q 8`); without `-q` the full model runs. Required for `ltx-2.3` and `ltx-2.5`, where it is the only level stored; `h3` takes none |
| `--distill` | off | `ltx-2.3` and `ltx-2.5` only: merge Lightricks' distilled LoRA into a plain-dev checkpoint |
| `--turbo` | off | `h3` only: the checkpoint is already step-distilled, so the turbo LoRA is not merged |
| `--force` | off | Video types only: replace an existing model of that name once the new one is complete |

## `ziv-model lora`

| Option | Default | Description |
|---|---|---|
| `-i`, `--input` | | A local `.safetensors` file. Can't be combined with `--hf`; one of the two is needed |
| `--hf` | | A Hugging Face repo, such as `owner/repo`. Can't be combined with `-i` |
| `--file` | detected | The file to take from a repo with several `.safetensors` files |
| `--name` | file name | Name of the LoRA, without `.safetensors`. It can't hold `/`, `\` or `..` |

## `ziv-model list`

Prints the image models in `models/` in the data folder (converted models and stored quantized copies), the video models there, the LoRAs in `loras/`, and the built-in model aliases with what they stand for on each platform. The aliases are listed whether or not they are downloaded, so the list does not show what is on disk.

| Option | Default | Description |
|---|---|---|
| `--models` | off | Show only models (and the aliases) |
| `--loras` | off | Show only LoRAs (and the aliases) |

## Enhance spec

### The spec

`--enhance` takes comma-separated `key=value` pairs; join several values with `+`. Leave a key out to keep its default. With no spec at all you get a general rewrite.

| Key | Values |
|---|---|
| `style` | `keep`; photo: `photo`, `candid`, `street`, `film`, `bw`, `portrait`, `product`, `cinematic`; art: `illustration`, `anime`, `comic`, `3d`, `painterly`, `watercolor`, `gouache`, `sketch`, `pixel`, `concept`, `graphic`, `isometric`, `claymation`, `ukiyo-e`, `papercraft` |
| `mood` | `keep`, `serene`, `cozy`, `joyful`, `romantic`, `whimsical`, `nostalgic`, `melancholic`, `mysterious`, `eerie`, `dramatic`, `epic` |
| `theme` | `keep`, `zen`, `spiritual`, `rustic`, `sensual`, `glamorous`, `gothic`, `noir`, `horror`, `dystopian`, `fairytale`, `mythic`, `surreal`, `retro`, `futuristic`, `cyberpunk`, `steampunk` |
| `shot` | `keep`, `extreme-close-up`, `close-up`, `medium`, `full`, `wide`, `extreme-wide` |
| `angle` | `keep`, `eye-level`, `high`, `low`, `overhead`, `worms-eye`, `dutch`, `over-shoulder`, `behind` |
| `palette` | `keep`, `teal-orange`, `pastel`, `earth`, `film`, `neutrals`, `forest`, `cool`, `warm`, `neon`, `jewel`, `primary`, `sunset`, `retro`, `custom` |
| `colors` | Up to five hex codes joined with `+`, such as `1b3a4b+e07a5f`; the `#` is optional. Needs `palette=custom`; with any other palette, or none, it is an error |
| `details` | Any of `lighting`, `composition`, `camera`, `materials`, `color`, `environment`, `subject`, `expression` |
| `length` | `shorter`, `same` (default), `longer`, `extra` |
| `camera` (video) | `keep`, `auto`, `static`, `push-in`, `pull-back`, `pan`, `tilt`, `tracking`, `orbit`, `handheld`, `aerial` |
| `pace` (video) | `keep`, `slow-motion`, `slow`, `brisk`, `fast` |
| `motion` (video) | Any of `action`, `secondary` |
| `sound` (video with audio) | Any of `ambient`, `effects`, `dialogue`, `music` |
| `parts` (video) | `both` or `action` |

What each option asks for, including how Custom colours are used, is in [Prompt enhancer options](../concepts/prompt-enhancer.md#options).

### The audio spec { #the-audio-spec }

`ziv-audio --enhance` takes the same `key=value` pairs, with `+` between several values, and leaves a key at `keep` (or at none, for `instruments` and `without`) when you don't give it. A music take takes the music keys and a sound the sound keys; a key of the other kind is refused, and speech takes none. A direction goes in `--enhance-note`, not in the spec as `note=`, because it may contain commas.

Music keys, for a song's style:

| Key | Values |
|---|---|
| `genre` | `keep`; pop and rock: `pop`, `synth-pop`, `indie-rock`, `rock`; electronic: `house`, `techno`, `drum-and-bass`, `synthwave`; hip-hop and R&B: `hip-hop`, `trap`, `r-and-b`, `soul`; jazz and blues: `jazz`, `smooth-jazz`, `swing`, `blues`; folk and country: `folk`, `country`, `bluegrass`, `singer-songwriter`; classical and cinematic: `classical`, `orchestral`, `film-score`, `neoclassical`; world: `latin`, `reggae`, `afrobeat`, `flamenco`; lo-fi and ambient: `lo-fi`, `ambient`, `chillout`, `downtempo` |
| `music_mood` | `keep`, `serene`, `joyful`, `romantic`, `nostalgic`, `melancholic`, `mysterious`, `dark`, `dramatic`, `epic`, `hopeful`, `energetic`, `dreamy` |
| `instruments` | Any of `guitar`, `piano`, `drums`, `bass`, `strings`, `synth`, `brass`, `saxophone`, `violin`, `flute`, `organ`, `percussion` |
| `without` | Any of the same instruments, to leave out. An instrument can't be in both `instruments` and `without` |
| `singer` | `keep`, `female`, `male`, `duet`, `choir`. Has no effect on an instrumental |
| `production` | `keep`, `polished`, `lo-fi`, `live`, `vintage`, `cinematic` |

Sound keys, for the description of a sound:

| Key | Values |
|---|---|
| `space` | `keep`, `small-room`, `large-hall`, `street`, `open-country`, `forest`, `underwater` |
| `distance` | `keep`, `close`, `medium`, `far` |
| `over_time` | `keep`, `steady`, `builds`, `hit-then-tail` |
| `movement` | `keep`, `still`, `passing`, `approaching`, `receding` |

For example, `--enhance genre=jazz,music_mood=dreamy,instruments=piano+saxophone,without=drums,singer=female,production=live` for a song and `--enhance space=large-hall,distance=far,over_time=builds,movement=passing` for a sound. What each option asks for is in [the music guide](../guides/music.md#enhance-the-prompt) and [the speech and sound guide](../guides/speech-and-sound.md#polish-the-prompt-with-the-enhancer).

### The other flags

| Flag | Does |
|---|---|
| `--enhance-note TEXT` | A direction in your own words, up to 300 characters. Turns on `--enhance` by itself. |
| `--enhance-model` | `fast` (default), `smart`, a Hugging Face `REPO[@REVISION]`, or a local folder (`./smart` for a folder named `smart`). Where Smart doesn't fit, a warning says so and Fast runs. |
| `--enhance-format json` | Write an Ideogram 4 JSON caption instead of a prompt. With `--prompt` it turns on enhancement by itself. |
| `--no-enhance` | Turn enhancement off, including `enhance:` entries in a prompt file. |

`--no-enhance` can't be combined with `--enhance`, `--enhance-note` or `--enhance-format`. A pair that isn't `key=value` is refused with `Invalid enhancement setting '<pair>'. Use key=value`, and a direction written inside the spec with `Set the direction with --enhance-note, not in the --enhance spec.` `colors` needs `palette=custom` and takes at most five codes.

A key or value the table doesn't have is refused before the model loads, and the message lists what is valid: `Unknown style 'cartoon'. Valid values: …` or `Unknown enhancement setting(s): …. Valid keys: …`.

The video keys (`camera`, `pace`, `motion`, `sound`, `parts`) don't apply to an image. In an `--enhance` spec of `ziv-image` they are refused with `'camera' only applies to video prompts.` In a prompt file's `enhance:` entry they are ignored with a warning instead, so one file can serve both: `'camera' only applies to video prompts and is ignored for images.`, and when an entry sets only video keys, `… prompt-file 'enhance:' entries set only video options …; not enhanced in this image run.`

## Environment variables { #environment-variables }

Set one in your shell before you start a command, for example `export ZIV_DATA_DIR=/mnt/fast-ssd/ziv` on macOS and Linux, or `$env:ZIV_DATA_DIR = "D:\ziv"` in PowerShell on Windows.

### `ZIV_DATA_DIR`

Use another [data folder](../reference/data-folder.md#use-another-folder) instead of `~/.ziv`. It moves everything the app keeps there: the config file, converted models, LoRAs, outputs, presets, logs and the app database. Downloaded models stay in the Hugging Face cache, which [`HF_HOME`](#hf_home) moves; see [Models and the Hugging Face cache](../reference/data-folder.md#models-and-the-hugging-face-cache).

### `ZIV_LOG`

`off` turns the [error log](../reference/data-folder.md#error-logs) off. `0`, `false` and `no` do the same.

### `ZIV_UI_ALLOWED_HOSTS`

Extra host names the Web UI answers to, comma-separated, such as `studio.local,gpu-box`. Needed when you open the Web UI from another device by a name other than this machine's own; an IP address always works.

### `HF_TOKEN`

Hugging Face token for gated models. `HUGGING_FACE_HUB_TOKEN` works the same, and so does a token saved with `hf auth login`. See [Hugging Face token](../reference/data-folder.md#hugging-face-token).

### `HF_HOME`

Where Hugging Face keeps its cache and token. Downloaded models live in its `hub` folder, which is `~/.cache/huggingface/hub` by default. Point it at another drive when the system drive is short of space. See [Models and the Hugging Face cache](../reference/data-folder.md#models-and-the-hugging-face-cache).

### `HF_XET_HIGH_PERFORMANCE`

`1` lets downloads use all the bandwidth and CPU cores. See [Download speed](../reference/data-folder.md#download-speed).

??? note "Advanced environment variables"

    The app sets these itself and keeps a value you set. Change one only to fix a memory problem.

    ### `PYTORCH_CUDA_ALLOC_CONF`

    Windows and Linux with an NVIDIA card. When you set neither this nor `PYTORCH_ALLOC_CONF`, the app runs PyTorch's memory allocator with `expandable_segments:True,garbage_collection_threshold:0.8`, plus `pinned_max_round_threshold_mb:1`, which stops PyTorch from rounding pinned memory up to a power of two. When you set it, it is left as written and the pinned-memory setting is added to `PYTORCH_ALLOC_CONF` instead, unless you set that key yourself.

    ### `PYTORCH_ALLOC_CONF`

    The newer name of `PYTORCH_CUDA_ALLOC_CONF`; the rule above applies to it the same way. The app merges both into one value, and your settings win on a repeated key.

    ### `HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE`

    How much memory a download may use to put a file back together. The app sets it to `512mb` for its downloads, so a download doesn't compete with a running job for memory; a value you set is kept.
