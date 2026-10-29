# Data folder

Everything the app keeps lives in one folder, `~/.ziv/`. It survives an update, and backing it up keeps your LoRAs, settings, converted models and Web UI outputs.

```text
~/.ziv/
├── models/          Converted models, stored quants and prepared copies (such as `h3@q4`), Ideogram 4 Instant and SeedVR2
├── loras/           LoRA .safetensors files
├── cache/           Temporary files, such as a Mac job's renamed LoRA copies; empty between jobs
├── outputs/         Images and videos made in the Web UI (the default output folder)
├── presets/         Web UI presets: an image and a video folder, one .yaml file each
├── logs/            The error log of each command
├── run/             Running Web UI servers
├── config.yaml      Your settings
└── ziv.sqlite3      The Web UI's app database: the Gallery index, the job log, recent prompts and LoRA sources
```

Models downloaded from Hugging Face, such as the built-in aliases, are kept in the Hugging Face cache (`~/.cache/huggingface/hub` by default), which other tools share.

## Use another folder

Set `ZIV_DATA_DIR` to keep everything somewhere else, for example on a faster or larger drive:

```bash
export ZIV_DATA_DIR=/mnt/fast-ssd/ziv
ziv ui
```

## Output files

The Web UI saves to the **Output folder** on the [Config page](../web-ui/config.md), `outputs` in the data folder by default. The CLI saves to the current folder, or to `-o`.

Files are named after the prompt set and the time: `portrait_2026-10-04_14-03-22.png`. When a name is taken, `_2`, `_3`… is added, so nothing is overwritten. An inline prompt's set is `prompt`. Videos follow the same pattern.

Every file records the settings it was made with: prompt and negative prompt, scene, model, seed, steps, guidance, scheduler, size, LoRAs, start image and strength, references, quant, generation time, upscale method and its settings, and the sharpen, contrast and saturation amounts. An enhanced prompt also keeps your own prompt and the enhance options. The Web UI viewer shows them under **Details**, and **Reuse settings** loads them.

Paths inside your home folder, such as LoRAs, models, start images and references, are recorded as `~/…`, so a shared file doesn't show your user name. The prompt is stored as written.

## Hugging Face token

The built-in models download without an account. A **gated** model, one whose page on Hugging Face asks you to accept a license, needs a token from an account that accepted it:

1. Accept the license on the model's page, signed in to Hugging Face.
2. Save a token with `hf auth login`, or set `HF_TOKEN` before starting the app.

The Config page and Models page show whether a token is found and where it comes from. When Hugging Face refuses a download, the app says which model, links to its page, and says whether to accept the license or run `hf auth login`.

## Download speed

Downloads use Hugging Face's Xet transfer. Set `HF_XET_HIGH_PERFORMANCE=1` before starting the app to let downloads use all the bandwidth and CPU cores:

```bash
export HF_XET_HIGH_PERFORMANCE=1
ziv ui
```

## Error log

Each command records warnings and errors, with full tracebacks, in `logs/<command>.log`: `ui.log`, `image.log`, `video.log` or `model.log`.

- The file is cleared every time the command starts, so it covers only the latest run, and stays empty when nothing goes wrong.
- If the logs folder can't be written, the log goes to the system's temp folder; `ziv ui` prints where.
- `ZIV_LOG=off` turns it off.

Attach the log when you [report a bug](troubleshooting.md#report-a-problem).

## Job log

The Web UI keeps a record of finished jobs, with the reason each failed, in the [app database](#app-database). Unlike the error log it is kept across restarts, and entries are kept until you remove them with **Delete** or **Clear log** in the [Job log](../web-ui/job-log.md).

## App database

`ziv.sqlite3` holds the Gallery index, the [job log](#job-log), your recent prompts, and where each LoRA you downloaded in Discover came from. The Gallery index is what was read from each file in the output folder, so the Gallery opens fast after a restart. While the server runs, `ziv.sqlite3-wal` and `ziv.sqlite3-shm` sit beside it.

- The Gallery index rebuilds itself, so deleting the database while the server is stopped is safe for the Gallery; the next Gallery load reads every file once. The job log is not rebuilt, so deleting the database also clears it.
- Files you delete, in the app or in the folder, are never shown.
- Files the CLI writes to the output folder are picked up on the next load.
- Several Web UIs, even different versions, can share it.
- If it's ever found damaged, it is moved aside as `ziv.sqlite3.corrupt-<date>-<time>` and a new one is started with an empty job log; the old jobs stay in that copy. Delete those copies when you don't need them.
