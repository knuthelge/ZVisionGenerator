# Data folder

Where the app keeps your settings, LoRAs, converted models and outputs, and how to move them. The models the app downloads are kept in the Hugging Face cache instead, described [below](#models-and-the-hugging-face-cache). Error? See [Troubleshooting](troubleshooting.md).

The data folder is `~/.ziv` by default. It survives an update, and backing it up keeps your LoRAs, settings, converted models and Web UI outputs.

```text
~/.ziv/
├── models/          Converted models, stored quants and prepared copies (such as `h3@q4`), Ideogram 4 Instant and SeedVR2
├── loras/           LoRA .safetensors files
├── voices/          Saved voices: a .wav recording and a .yaml record each
├── voice-design/    The candidates of New voice, until you clear them
├── cache/           Temporary files, such as a Mac job's renamed LoRA copies, and thumbnails
├── outputs/         Images, videos and audio made in the Web UI (the default output folder)
├── presets/         Web UI presets: an image and a video folder, one .yaml file each
├── logs/            The error log of each command
├── run/             Running Web UI servers
├── config.yaml      Your settings
└── ziv.sqlite3      The Web UI's app database: the Gallery index, the job log, recent prompts and LoRA sources
```

`voices/`, `voice-design/` and `config.yaml` appear once you use them.

The thumbnails the file browser and the Gallery show are kept in `cache/browse`, at most 256 MB. They stay after the files they show are deleted, and a backup of the data folder includes them; delete `cache/browse` at any time.

Presets are saved on the machine that runs the Web UI, so every browser and device sees the same ones. The command line doesn't use them.

## Models and the Hugging Face cache { #models-and-the-hugging-face-cache }

Models the app downloads from Hugging Face, such as the built-in aliases, are kept in the Hugging Face cache, not in the data folder: the `hub` folder of `~/.cache/huggingface` by default, which other tools share. Set `HF_HOME` before you start the app to put the cache somewhere else, such as a larger drive or another disk; the cache then lives in `hub` inside that folder. See [`HF_HOME`](../cli/reference.md#hf_home).

The data folder's `models/` holds only what the app builds or you add: converted models and prepared copies. Moving the data folder doesn't move the cache.

## Use another folder { #use-another-folder }

Set `ZIV_DATA_DIR` to keep the data folder somewhere else, for example on a faster or larger drive, or another disk when the system drive is short of space. Every path above then sits inside it, and the app reads and writes there instead of `~/.ziv`:

=== "macOS"

    ```bash
    export ZIV_DATA_DIR=/Volumes/Fast/ziv
    ziv ui
    ```

=== "Windows"

    ```powershell
    $env:ZIV_DATA_DIR = "D:\ziv"
    ziv ui
    ```

=== "Linux"

    ```bash
    export ZIV_DATA_DIR=/mnt/fast-ssd/ziv
    ziv ui
    ```

Set it for every command you run, so they all use the same folder.

## Output files

This is where your pictures are saved, and your videos and audio. The Web UI saves to the **Output folder** on the [Config page](../web-ui/config.md), which sets [`ui.output_dir`](configuration.md#ui); unset, it is `outputs` in the data folder. The CLI saves to the current folder, or to `-o`.

Files can sit in a [project](../concepts/projects.md): a folder directly under the output folder. The Web UI saves into the project you have picked, and files at the top of the output folder are in no project.

Files are named after the prompt set and the time: `portrait_2026-10-04_14-03-22.png`. When a name is taken, `_2`, `_3`… is added, so nothing is overwritten. An inline prompt's set is `prompt`. Videos follow the same pattern.

Audio takes are lossless FLAC files named after their kind, such as `music_2026-10-04_14-03-22.flac`, `speech_…` and `sound_…`. A take keeps its settings inside the file for **Reuse settings**, but not the speaker's consent. A take made from a recording records the recording and its transcript, and one made in a saved voice shows the voice by name while that voice exists.

Every file records the settings it was made with: prompt and negative prompt, scene, model, seed, steps, guidance, scheduler, size, LoRAs, start image and strength, references, quant, generation time, upscale method and its settings, the sharpen, contrast and saturation amounts, and whether the background was removed. An enhanced prompt also keeps your own prompt and the enhance options. The Web UI viewer shows them under **Details**, and **Reuse settings** loads them.

Paths inside your home folder, such as LoRAs, models, start images and references, are recorded as `~/…`, so a shared file doesn't show your user name. The prompt is stored as written.

## Hugging Face token

The built-in models download without an account. A **gated** model, one whose page on Hugging Face asks you to accept a license, needs a token from an account that accepted it:

1. Accept the license on the model's page, signed in to Hugging Face.
2. Save a token with `hf auth login`, or set [`HF_TOKEN`](../cli/reference.md#hf_token) (`HUGGING_FACE_HUB_TOKEN` works too) before starting the app.

The Config page and Models page show whether a token is found and where it comes from. When Hugging Face refuses a download, the app says which model, links to its page, and says whether to accept the license or run `hf auth login`.

## Download speed

Downloads use Hugging Face's Xet transfer. Set [`HF_XET_HIGH_PERFORMANCE`](../cli/reference.md#hf_xet_high_performance)`=1` before starting the app to let downloads use all the bandwidth and CPU cores:

```bash
export HF_XET_HIGH_PERFORMANCE=1
ziv ui
```

## Error log { #error-logs }

Each command records warnings and errors, with full tracebacks, in `logs/<command>.log`: `ui.log`, `image.log`, `video.log`, `audio.log` or `model.log`.

- The file is cleared every time the command starts, so it covers only the latest run, and stays empty when nothing goes wrong.
- If the logs folder can't be written, the log goes to `ziv-logs-<user name>` in the system's temp folder; `ziv ui` prints where when it starts, and a command that fails prints it with the error.
- [`ZIV_LOG=off`](../cli/reference.md#ziv_log) turns it off.

Attach the log when you [report a bug](troubleshooting.md#report-a-problem).

## Job log

The Web UI keeps a record of finished jobs, with the reason each failed, in the [app database](#app-database). Unlike the error log it is kept across restarts, and entries are kept until you remove them with **Delete** or **Clear log** in the [Job log](../web-ui/job-log.md).

## App database

`ziv.sqlite3` holds the Gallery index, the [job log](#job-log), your recent prompts, your [stars](../web-ui/gallery.md#stars), and where each LoRA you downloaded in Discover came from. The Gallery index is what was read from each file in the output folder, so the Gallery opens fast after a restart. While the server runs, `ziv.sqlite3-wal` and `ziv.sqlite3-shm` sit beside it.

- The Gallery index rebuilds itself, so deleting the database while the server is stopped is safe for the Gallery; the next Gallery load reads every file once. The job log is not rebuilt, so deleting the database also clears it and your stars.
- Files you delete, in the app or in the folder, are never shown.
- Files the CLI writes to the output folder are picked up on the next load.
- Several Web UIs, even different versions, can share it.
- If it's ever found damaged, it is moved aside as `ziv.sqlite3.corrupt-<date>-<time>` and a new one is started with an empty job log; the old jobs stay in that copy. Delete those copies when you don't need them.
