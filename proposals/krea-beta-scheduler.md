# Krea 2: beta scheduler and upscale defaults, and beta on FLUX.2 Klein

**Status:** In progress (2026-10-31)

**Shipped (unreleased, on `main`):**
- Krea 2 defaults to the `beta` scheduler on both platforms and shows **Scheduler**.
- A new `simple` scheduler gives any model's own evenly spaced steps.
- Krea 2 upscales refine for 5 steps instead of 3, and an upscale while generating no longer sharpens first.
- `beta` on FLUX.2 Klein matches ComfyUI on Windows/Linux. On macOS it has a fix for its error that hasn't been run on a Mac yet.
- Ideogram 4 hides the Scheduler control it ignored.

Windows/Linux is verified on a 10 GB card.

**Left:** the macOS check. Nothing on the macOS side has run on a Mac: the Krea 2 beta adapter, Klein's beta scheduler and its timesteps, and Klein's refine passes. Run the [Mac test plan](#mac-test-plan) and archive this proposal once it passes.

## Problem

People online report that Krea 2 gets better skin texture with "two passes". The question was what that means and whether the app should offer it.

In the community it means one of two things:

1. **A hires fix.** Generate at about 1 MP, upscale 1.5–2.6×, then refine with Krea 2 Turbo as image-to-image, usually at 8 steps and denoise 0.25–0.5.
2. **A Raw → Turbo handoff.** The undistilled Raw model runs the first 12–17% of the noise range with real CFG, then Turbo finishes at the same size.

A separate guide traces Krea 2's plastic skin to the evenly spaced ("simple") schedule, and recommends ComfyUI's `beta` schedule instead.

## Evidence

Everything below was run on Windows/Linux at q4 with fixed seeds, two prompts (a close-up portrait and a man in a cable-knit sweater), and 1:1 crops.

| Tried | Result |
|---|---|
| Same-size "detail pass" (image-to-image on a finished image), 8 steps, strength 0.2–0.4 | More texture, not more realism: spotty freckles, orange-peel pores, coloured specks in fabric. Strength barely matters. Same on Z-Image Turbo (grimy, cracked skin), Klein 9B (an over-processed HDR look) and MOSS Turbo (dark freckle dots) |
| The same at q8, and with a different refine seed | No change, so neither quantization nor the reused seed causes it |
| Half the steps, then a detail pass | About the same as full steps plus a detail pass. Broken on MOSS Turbo (2 of 4 steps is blurry) |
| Krea 2 upscale refine steps × pre-sharpen | Pre-sharpening causes the coloured specks. 5 steps give clearly more skin texture than 3 without overcooking. 8 overcook |
| Krea 2, `beta` vs the evenly spaced schedule | `beta` gives clearly more real skin texture (pores, freckles, wrinkles) on both seeds, with no overcooking, at the same speed |
| MOSS Turbo: 4, 6, 8 steps and `beta` | Small differences. The skin stays smooth in every variant, so MOSS was left as is |

Why `beta` helps Krea: Krea's evenly spaced 8-step schedule ends with one jump from sigma 0.311 to 0, so the finest detail is drawn in a single coarse step. `beta`'s last step starts at 0.161.

The schedules were checked against ComfyUI's `beta_scheduler`: the beta(0.6, 0.6) inverse CDF in timestep space, then the model's own shift. Krea 2 and Z-Image matched. FLUX.2 Klein on Windows/Linux did not. It used diffusers' `use_beta_sigmas`, which spaces the steps after the shift: 1.00/0.73/0.28/0.01 instead of 1.00/0.98/0.91/0.68 at 832×1248 and 4 steps. On macOS, Klein with `beta` failed outright, because mflux's Klein loop reads `scheduler.timesteps` and the beta scheduler had none.

## What shipped

- **Beta on Windows/Linux:** every diffusers model gets ComfyUI's beta grid as custom `sigmas`, and the pipeline applies its own shift afterwards. A fixed-shift scheduler (Z-Image's) is switched to dynamic shifting so it takes the pipeline's `mu`. Z-Image output is pixel-identical to before.
- **Beta on macOS, Krea 2:** mflux steps Krea 2 with its own Euler stepper and takes the sigmas from the scheduler registered under the sampler's name. The backend swaps that registry entry for the run and restores it afterwards.
- **Beta on macOS, Klein:**
  - The beta scheduler now provides `timesteps`.
  - A `KleinBetaScheduler` shifts the grid with Klein's step-dependent `mu`, the same one mflux and diffusers compute.
  - The backend routes Klein's `beta` to it.
- **Refine passes stay evenly spaced on both platforms.** Krea 2's and Klein's image-to-image and upscale refine passes ignore the scheduler, and the upscale settings were tuned on that.
- **Per-model upscale defaults:** a new `upscale.default_sharpen` key (Krea 2: `false`). The CLI flag, the Web UI's Sharpen checkbox and "Copy as CLI command" all follow the model's default, and an explicit choice still wins.
- **Config and docs:** `simple` in `schedulers` (no class on either backend). Ideogram 4 gets `supports_scheduler: false`. Docs and CHANGELOG are updated.

## Alternatives considered

- **An "extra detail phase" option for every model** (a same-size refine pass): rejected. On all four models it overcooks rather than adding realistic detail.
- **The community's hires settings (8 steps, denoise 0.25):** rejected here. They rely on a learned latent upscaler (SesquiLSR); after a plain resize, low denoise keeps and amplifies artifacts.
- **Raw → Turbo handoff:** not possible. Only Krea 2 Turbo ships, and Raw would be another large download.
- **Detail Daemon** (lie to the sampler about the noise level for more detail in one pass): not built. It is fragile on Krea 2 Turbo (it breaks into blocks above about 0.10). Possible later.
- **Stronger beta (0.5, 0.7):** a little more texture with a little more speckle. Standard beta (0.6, 0.6) was kept.
- **Hiding Scheduler for Klein:** rejected in favour of making Klein's `beta` correct on both platforms.

Known and accepted:
- Krea 2 and Klein image-to-image show Scheduler but run evenly spaced steps, and the file still records the chosen scheduler.
- Saved Web UI drafts from before keep Krea's Sharpen on until Reset.
- FLUX.1-schnell (not shipped) with `beta` now gets FluxPipeline's resolution shift on Windows/Linux.

Separate finding, not fixed here: a Krea 2 2× upscale to 1280×1920 runs out of memory on a 10 GB card when other processes hold about 600 MB. Image-to-image at the same size fits.

## Mac test plan

**Summary**

- **What's new on a Mac:**
  - Krea 2 gets the `beta` schedule through a swap of mflux's scheduler registry.
  - FLUX.2 Klein's `beta` gets its own scheduler with `timesteps`, on both the plain and the references path.
  - Klein's refine passes ignore the scheduler.
  - The per-model Sharpen default, `simple`, and Ideogram 4 without a Scheduler.
- **Machine:** an Apple Silicon Mac with 32 GB.
- **When:** run from `main` once the branch is merged.
- **Time:** about 1 hour 10 minutes, mostly unattended generation (table below). Downloads and the first q8 copy of each model are not included; S2 does those first.
- **Disk:**
  - Downloads of whichever models are missing: `krea2` 36 GB, `klein9b` 35 GB, `zit` 33 GB.
  - Plus one stored q8 copy per model. The Models page lists their sizes.
  - Outputs under 1 GB.
  - If none of the models are on the Mac yet, keep at least 150 GB free.
- **Order:** S1–S3, T0, K1–K5, C1–C4, Z1, W1, then V (all the looking) at the end.
- **Failures outside this work:** T0 can show Mac-only test failures unrelated to this branch. Record them, compare them with a run on the commit before the merge, and don't let them block the rest.

| Step | What | Min |
|---|---|---|
| S1–S3 | Update, helpers, first q8 run of each model (copies not counted) | 5 |
| T0 | `make check` and the Mac beta tests | 10 |
| K1–K5 | Krea 2: `beta`, `simple`, upscale, Sharpen back on, image to image | 20 |
| C1–C4 | Klein 9B: `beta` vs default, upscale, image to image, references | 15 |
| Z1 | Z-Image with `simple` | 3 |
| W1 | Web UI: Scheduler, Sharpen, Copy as CLI command, Ideogram 4 | 10 |
| V | One visual pass | 5 |
| Total | | about 70 |

**Setup**

- **S1.** `git switch main && git pull && make install`.
- **S2.** Download `krea2`, `klein9b` and `zit` if they are missing (Models page, or let the first run fetch them). Then run one throwaway `-q 8` image per model, for example `ziv image -m krea2 -q 8 --prompt test -W 512 -H 512 -o /tmp/ziv-warmup`. The first `-q 8` run of each model saves a `<name>@q8` copy and takes several minutes; later runs load it.
- **S3.** In the terminal you'll use for every step:

```bash
export OUT=~/ziv-mac-krea-test
export PORTRAIT="Close-up portrait photo of a woman in her thirties with freckles and natural skin texture, looking at the camera, soft window light from the side, 85mm lens, shallow depth of field, loose strands of hair across her forehead, plain grey wall behind her."
export SWEATER="Half-body photo of a bearded man in his fifties wearing a chunky cable-knit wool sweater, standing on a windy harbour pier on an overcast day, fishing boats and ropes behind him, weathered face with wrinkles, natural light, shot on a 50mm lens."
latest() { ls -t "$1"/*.png | grep -v _preupscale | head -1; }
# Pixels only: the files always differ, because each PNG records its scheduler and time.
same() { uv run python -I -c "import sys; from PIL import Image, ImageChops; a,b=(Image.open(p).convert('RGB') for p in sys.argv[1:3]); print('identical' if ImageChops.difference(a,b).getbbox() is None else 'different')" "$1" "$2"; }
meta() { uv run python -I -c "import json,sys; from zvisiongenerator.utils.provenance import read_png_config; c=read_png_config(sys.argv[1]); print(json.dumps({'scheduler': c.get('scheduler'), 'upscale': c['generation'].get('upscale')}))" "$1"; }
```

If a `same` check that should be `identical` prints `different`, run both commands again into new folders before calling it a failure, to rule out run-to-run noise.

**Steps**

| Step | Run | Pass if |
|---|---|---|
| T0 | `make check`, then `uv run pytest tests/backends/mac/test_mac_beta_schedule.py -v` | The second command shows `9 passed` and no skips. For `make check`, see Failures outside this work above |
| K1 | `ziv image -m krea2 -q 8 --prompt "$PORTRAIT" --seed 4242 -W 832 -H 1248 -o $OUT/k1` | `meta "$(latest $OUT/k1)"` shows `"scheduler": "beta"` |
| K2 | `ziv image -m krea2 -q 8 --prompt "$PORTRAIT" --seed 4242 -W 832 -H 1248 --scheduler simple -o $OUT/k2` | `meta` shows `"scheduler": "simple"`, and `same "$(latest $OUT/k1)" "$(latest $OUT/k2)"` prints `different`. If it prints `identical`, the beta swap did nothing |
| K3 | `ziv image -m krea2 -q 8 --prompt "$SWEATER" --seed 4242 -W 1024 -H 1536 --upscale 2 -o $OUT/k3` | The second mflux progress bar ends at `5/5`, and `meta` shows `"steps": 5` and no `pre_sharpen` under `upscale` |
| K4 | `ziv image -m krea2 -q 8 --prompt "$SWEATER" --seed 4242 -W 1024 -H 1536 --upscale 2 --upscale-sharpen -o $OUT/k4` | `meta` shows `"pre_sharpen": 0.8`, and `same` K3 K4 prints `different` |
| K5 | `ziv image -m krea2 -q 8 --prompt "$PORTRAIT" --seed 4242 --image "$(latest $OUT/k1)" --image-strength 0.3 --steps 8 --scheduler beta -o $OUT/k5b`, then the same with `--scheduler simple -o $OUT/k5s` | `same` K5b K5s prints `identical` (Krea's image to image stays evenly spaced) |
| C1 | `ziv image -m klein9b -q 8 --prompt "$PORTRAIT" --seed 4242 -W 832 -H 1248 --scheduler beta -o $OUT/c1`, then the same without `--scheduler` and with `-o $OUT/c1d` | Both finish with no error, `same` C1 C1d prints `different`, and C1 looks like a normal Klein image (V) |
| C2 | `ziv image -m klein9b -q 8 --prompt "$PORTRAIT" --seed 4242 -W 832 -H 1248 --scheduler beta --upscale 2 -o $OUT/c2` | It finishes, the second progress bar ends at `2/2`, and `meta` shows `"pre_sharpen": 0.8` (Klein keeps Sharpen on) |
| C3 | `ziv image -m klein9b -q 8 --prompt "$PORTRAIT" --seed 4242 --image "$(latest $OUT/c1d)" --image-strength 0.3 --scheduler beta -o $OUT/c3b`, then the same without `--scheduler` and with `-o $OUT/c3d` | `same` C3b C3d prints `identical` (Klein's image to image keeps its own schedule) |
| C4 | `ziv image -m klein9b -q 8 --prompt "$PORTRAIT, as image 1" --seed 4242 -W 832 -H 1248 --scheduler beta --reference "$(latest $OUT/c1d)" -o $OUT/c4` | It finishes with no error and looks like a normal Klein image (V). References run through a separate mflux loop that also reads the beta timesteps |
| Z1 | `ziv image -m zit -q 8 --prompt "$PORTRAIT" --seed 4242 -W 832 -H 1248 --scheduler simple -o $OUT/z1s`, then the same without `--scheduler` and with `-o $OUT/z1` | `meta` shows `"scheduler": "simple"` and `"beta"`, and `same` Z1s Z1 prints `different` |
| W1 | `ziv ui --host local`, Image tab, in this order | (1) **Reset**. Pick Klein 9B and set **Upscale** to 2×: **Sharpen** is on. (2) Switch to Krea 2: **Sharpen** is off, and **Scheduler** offers "Auto (model default)", `beta` and `simple`. (3) **Copy as CLI command**: contains neither `--upscale-sharpen` nor `--no-upscale-sharpen`. (4) Turn **Sharpen** on and copy again: contains `--upscale-sharpen`. (5) **Reset**, pick Klein 9B, Upscale 2×, turn **Sharpen** off, copy: contains `--no-upscale-sharpen`. (6) Pick Ideogram 4: there is no **Scheduler** row |
| V | Open K1 and K2, K3 and K4, C1 and C1d, and C4 at 100% | K1 shows more pore and freckle detail than K2. K3's knit has no coloured specks. C1 and C4 are not washed out or noisy next to C1d |

A Mac image of the same prompt and seed won't match a Windows/Linux one: the platforms draw different starting noise. Compare the character of the skin texture only, if at all.

Record wall time, pass or fail and the `meta` output per step under a Results heading here, fix any failure on the branch, and archive this proposal once every step passes.
