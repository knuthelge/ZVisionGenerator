# Krea 2: beta scheduler and upscale defaults, and beta on FLUX.2 Klein

**Status:** Done (2026-10-31, unreleased; commits e17f666, 7f0526b)

**Shipped (unreleased, on `main`):**
- Krea 2 defaults to the `beta` scheduler on both platforms and shows **Scheduler**.
- A new `simple` scheduler gives any model's own evenly spaced steps.
- Krea 2 upscales refine for 5 steps instead of 3, and an upscale while generating no longer sharpens first.
- `beta` on FLUX.2 Klein matches ComfyUI on Windows/Linux, and runs on macOS.
- Ideogram 4 hides the Scheduler control it ignored.

Verified on Windows/Linux on a 10 GB card and on a 32 GB Mac (see [Results](#results)).

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
  - Krea 2's upscale refines for 5 steps without sharpening first.
- **Already covered elsewhere, so not run here:** `make check` runs on macOS in CI. The real-mflux tests in `tests/backends/mac` pin the beta sigmas, the registry restore and that image to image keeps its own schedule on both models. The Sharpen default, the `--upscale-sharpen` flags, "Copy as CLI command" and `simple` resolving to the model's own schedule are platform-independent and unit-tested.
- **Also covered here:** a smoke test of the merged Workspace Image and Video tabs and of Blob on the Mac's real backends and in Safari.
- **Machine:** an Apple Silicon Mac with 32 GB.
- **Models:** `krea2`, `klein9b` and one video model already on disk. Use `-q N` (and the same **Quantize** in the Web UI) matching a stored `<name>@qN` copy, so nothing new is built and the full-precision model isn't loaded.
- **Time:** about 35 minutes.

| Step | What | Min |
|---|---|---|
| P0 | Mac backend tests | 1 |
| K1–K3 | Krea 2: `beta`, `simple`, upscale | 12 |
| C1 | Klein 9B: `beta` vs default | 6 |
| E–H | Web UI: Klein `beta` with a reference, image to image, persistence, video | 13 |
| I | Blob | 3 |
| Total | | about 35 |

**Setup**

Run everything from the checkout under test after `make install`, with `uv run ziv`. In the terminal you'll use for every step:

```bash
export Q=8   # the bits of the stored copies
export OUT=~/ziv-mac-krea-test
export PORTRAIT="Close-up portrait photo of a woman in her thirties with freckles and natural skin texture, looking at the camera, soft window light from the side, 85mm lens, shallow depth of field, loose strands of hair across her forehead, plain grey wall behind her."
export SWEATER="Half-body photo of a bearded man in his fifties wearing a chunky cable-knit wool sweater, standing on a windy harbour pier on an overcast day, fishing boats and ropes behind him, weathered face with wrinkles, natural light, shot on a 50mm lens."
latest() { ls -t "$1"/*.png | head -1; }
# Pixels only: the files always differ, because each PNG records its scheduler and time.
same() { uv run python -I -c "import sys; from PIL import Image, ImageChops; a,b=(Image.open(p).convert('RGB') for p in sys.argv[1:3]); print('identical' if ImageChops.difference(a,b).getbbox() is None else 'different')" "$1" "$2"; }
meta() { uv run python -I -c "import json,sys; from zvisiongenerator.utils.provenance import read_png_config; c=read_png_config(sys.argv[1]); print(json.dumps({'scheduler': c.get('scheduler'), 'upscale': c['generation'].get('upscale')}))" "$1"; }
```

**Steps**

| Step | Run | Pass if |
|---|---|---|
| P0 | `uv run pytest tests/backends/mac -rs -q` | Everything passes with no skips |
| K1 | `uv run ziv image -m krea2 -q $Q --prompt "$PORTRAIT" --seed 4242 -W 832 -H 1248 -o $OUT/k1` | `meta "$(latest $OUT/k1)"` shows `"scheduler": "beta"` |
| K2 | The same with `--scheduler simple -o $OUT/k2` | `meta` shows `"scheduler": "simple"`, `same` K1 K2 prints `different` (`identical` means the beta swap did nothing), and at 100% K1 shows more pore and freckle detail than K2 |
| K3 | `uv run ziv image -m krea2 -q $Q --prompt "$SWEATER" --seed 4242 -W 1024 -H 1536 --upscale 2 -o $OUT/k3` | The second progress bar ends at `5/5`, `meta` shows `"steps": 5` and no `pre_sharpen` under `upscale`, and the knit has no coloured specks |
| C1 | `uv run ziv image -m klein9b -q $Q --prompt "$PORTRAIT" --seed 4242 -W 832 -H 1248 --scheduler beta -o $OUT/c1`, then the same without `--scheduler` and with `-o $OUT/c1d` | Both finish with no error, `same` C1 C1d prints `different`, and C1 is not washed out or noisy next to C1d |
| E | `uv run ziv ui --host local` in Safari (not a Private window), Image tab: Klein 9B, **Scheduler** `beta`, upload C1d from Finder as a **Reference**, prompt "$PORTRAIT, as image 1" | It finishes, looks like a normal Klein image, and `meta` on its output shows `"scheduler": "beta"`. This is the only real-weights run of Klein's references loop with `beta` |
| F | Image tab: Krea 2, upload K1 as the start image, **Strength** 0.3 | The job is labelled Image to Image and finishes |
| G | Reload the page | The prompt and the uploaded images are still there. **Reset** clears only the open tab's images. **Use as → Reference** in the viewer adds a reference. At Upscale 2×, **Sharpen** is on for Klein 9B and off for Krea 2. Ideogram 4 has no **Scheduler** row |
| H | Video tab, **Low memory** on, the smallest size and fewest frames | With a first frame, the job is labelled Image to Video and finishes. Without one, it is labelled Text to Video and starts stepping (cancel it). Extend Video opens |
| I | Watch Blob while E–H run | He appears and moves between his spots, the small canvas beside him denoises during a job, and the tab title reads Z-Vision Generator. With Reduce Motion on (System Settings → Accessibility → Display), he appears at once with still frames |

A Mac image of the same prompt and seed won't match a Windows/Linux one: the platforms draw different starting noise. Compare the character of the skin texture only, if at all.


## Results

Run on 2026-10-31 at `c0c90bc`, Apple M5 with 32 GB, `-q 8`. Web UI steps in Playwright's WebKit with an isolated data folder. Every step passes.

| Step | Wall time | Result | Notes |
|---|---|---|---|
| P0 | 28 s | Pass | 394 passed, no skips |
| K1 | 3 min 33 s (incl. saving `krea2@q8`) | Pass | `{"scheduler": "beta", "upscale": null}` |
| K2 | 3 min 3 s | Pass | `{"scheduler": "simple", "upscale": null}`; `same` K1 K2: `different`; K1 shows clearly more freckle, pore and wrinkle detail |
| K3 | 5 min 20 s | Pass | Second bar `5/5`; `{"scheduler": "beta", "upscale": {"denoise": 0.4, "factor": 2, "method": "creative", "steps": 5}}`; 1024×1536, clean knit |
| C1 | 1 min 21 s (incl. saving `klein9b@q8`) + 1 min 6 s | Pass | C1 `{"scheduler": "beta", "upscale": null}`, C1d `{"scheduler": null, "upscale": null}`; `different`; C1 is clean |
| E | 3 min 30 s | Pass | `{"scheduler": "beta", "upscale": null}`; a normal portrait. Recorded as Text to Image, since a reference alone doesn't make a run image to image |
| F | 4 min 20 s | Pass | Recorded as Image to Image (`img2img`). The output is over-textured (cracked pores), the same-size detail pass described in [Evidence](#evidence); not changed by this work |
| G | — | Pass | Prompts, uploads and the Video tab's first frame survive a reload and a browser restart; Reset clears only the Image tab's images; Use as → Reference works; Sharpen on for Klein 9B and off for Krea 2 at 2×; no Scheduler row on Ideogram 4 |
| H | 31 s | Pass | `ltx-4`, Low memory, 256×256, 9 frames: Image to Video finishes; Text to Video starts stepping and cancels; Extend Video opens |
| I | — | Pass | Title "Z-Vision Generator"; Blob animates and the denoise canvas resolves during a job; with reduced motion he appears at once with still frames |

Not checked: video thumbnails in the Gallery and preview stay blank in headless WebKit while the files play; check them in Safari.
