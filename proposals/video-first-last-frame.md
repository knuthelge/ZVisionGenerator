# First and last frame for image-to-video

**Status:** Implemented (2026-10-26, unreleased): LTX first and last frame is implemented on macOS, Windows and Linux, and MiniMax H3's part shipped in v0.13.0. Left: a run of the macOS LTX path on a Mac before the release, then archive this proposal.

## Outcome

The plan below was written before LTX-2.5 and before H3's last frame shipped, so most of steps 1 and 3–6 already existed when LTX's turn came: the request field, `--end-image`, the Web UI Last frame slot, recorded settings, Reuse and the enhancer's two-frame rule. What LTX needed was step 2, a Windows/Linux backend, the capability flag and docs. The design changed in four places.

- **The last frame sits on the true last frame on both platforms.** On macOS `-1` resolves to pixel `num_frames - 1`, as planned. On Windows and Linux, instead of generating 8 extra frames and trimming, the condition pipeline's keyframe coordinates are moved from pixel `num_frames - 8` to `num_frames - 1` for the call (`_keyframe_on_last_frame`), the way the reference positions a keyframe by pixel index.
- **Windows and Linux load the condition pipeline instead of `from_pipe`.** Every run of a batch has a last frame or none, so `load_model(end_image=True)` builds `LTX2ConditionPipeline` in place of the image-to-video pipeline. `from_pipe` would have shared the transformer with the image-to-video pipeline, which breaks the transformer swap around the text encoder.
- **Passes with a last frame keep the stock scheduler.** The Euler-ancestral step that LTX-2.5 uses would re-noise the appended keyframe tokens, as it would Extend's pinned frames. The two-stage refine re-applies both conditions.
- **No separate `supports_video_end_image()`.** `capabilities.end_frame` and `platform_capabilities` already gate the CLI, the server and the Web UI per family.

### Spike results (Windows/Linux, LTX-2.5 `ltx25-4`, a 10 GB card, 768×448)

Three image pairs (a camera push-in, a loop with the same image at both ends, a pose change), at 49 and 121 frames, with three placements of the end image. The similarity is SSIM against the end image after the pipeline's cover-and-crop.

| Placement | Where the best match lands | Last frame against the end image |
|---|---|---|
| diffusers `index=-1` | 8 frames before the end, then the clip drifts away (a loop fell to 0.66 at its last frame) | 0.66–0.96 |
| Generate 8 more and trim | the last frame, but one pose run missed the end image entirely (0.32) | 0.32–0.97 |
| Keyframe moved to the last frame (shipped) | the last frame in every run | 0.75–0.97 |

- **Motion.** At 121 frames every pair moves smoothly into the end image. At 49 frames a pose change cuts to it near the end, and two images of different places cut at either length; the camera and loop pairs never cut. The docs tell users to use the same scene and give a big change a few seconds.
- **Cost.** One latent frame of extra tokens, about 6% at 121 frames. Clips took 23–51 s with a 6.75 GiB peak on the card; no run without a last frame was timed for comparison.
- **First frame.** Stays at about 0.985 SSIM, as without a last frame.

## Mac test (left before the release)

The macOS LTX path has only run in mocked tests. This run checks that a last frame works with `ltx-4` and `ltx25-4` on a Mac, in one stage and with `--upscale 2`, and that image to video without a last frame did not change. About 1 hour, mostly unattended. Both models must be downloaded (`ltx-4` 28.6 GB, `ltx25-4` 28.8 GB).

**What is new on macOS, and what can break.** `_image_conditionings` (one stage) builds a second `ImageConditioningInput` at frame `-1`, resolves it with `resolve_frame_indices` and materializes its `keyframe_latent`. `_generate_two_stage` now always passes `images=` to `generate_and_save`, also without a last frame. LTX-2.5's one-stage clip runs its ancestral loop with the appended keyframe tokens. None of these has run against real MLX.

**Setup.**

1. Check out the branch with this change and run `make install`.
2. Run the MLX tests that skip on Linux: `uv run pytest tests/test_ltx_mlx_contract.py tests/test_video_upscale.py tests/test_video_review.py`. All must pass.
3. Make the two test images in a scratch folder outside the repo: a start image, and the same scene with a changed pose made from it by image to image.

   ```bash
   ziv-image -m zit -q 8 --ratio 16:9 --seed 4242 -o start --prompt "Full body photo of a woman in a yellow raincoat standing in a city park, arms at her sides, looking at the camera, autumn trees behind her"
   ziv-image -m zit -q 8 --ratio 16:9 --seed 4242 -o end --image start/<file>.png --image-strength 0.6 --prompt "Full body photo of a woman in a yellow raincoat jumping high in the air on the path in a city park, both arms raised high, knees bent, joyful, autumn trees behind her"
   ```

   Check that the end image shows the same path and trees with the arms up. If the scene changed, try another seed: two different places make LTX cut, which is not what this run tests.

**Runs.** Each at 768×448, 121 frames, seed 7, with this prompt: "A woman in a yellow raincoat standing on a path in a city park smiles and slowly raises both arms high above her head in joy. Autumn trees, fallen leaves, natural light, static camera."

| Step | Command (add `--image start.png --prompt …`) |
|---|---|
| M1 | `ziv-video -m ltx-4 --end-image end.png -W 768 -H 448 --frames 121 --seed 7` |
| M2 | M1 with `--upscale 2` |
| M3 | M1 with `-m ltx25-4` |
| M4 | M3 with `--upscale 2` |
| M5 | Web UI, `ltx-4`, Image to Video: the **Last frame** slot shows; one job with both images; **Reuse settings** on the result loads both; the CLI preview has `--end-image` |
| R1 | M1 and M2 **without** `--end-image`, once on this branch and once on the commit before it; `ltx-4` is deterministic, so the frames must match: `ffmpeg -i clip.mp4 -f framemd5 -` gives the same lines |

**Pass when**, for M1 to M5:

- The job finishes with no error, and peak memory (Activity Monitor) stays within 1 GiB of the same run without `--end-image`.
- The last frame matches the end image. Measure it on frame 120 and frame 112 (8 earlier):

  ```bash
  ffmpeg -v error -i clip.mp4 -i end.png -filter_complex "[0:v]select=eq(n\,120),scale=256:-2,format=gray[a];[1:v]scale=768:448:force_original_aspect_ratio=increase,crop=768:448,scale=256:-2,format=gray[b];[a][b]ssim=stats_file=-" -f null - | grep -o "All:[0-9.]*" | tail -1
  ```

  Frame 120 must be at least 0.80 and at least frame 112's value. The same command gave 0.88 to 0.93 on Windows/Linux for these four runs.
- A look at the clip shows the arms rising and no hard cut or dissolve into the end image.

R1 passes when the `framemd5` lines are identical.

**If a step fails.** Record it here. If the fix is not small, turn the last frame off for LTX on macOS before the release, with `platform_capabilities: { darwin: { end_frame: false } }` in the `ltx` family block, and say so in the docs and the changelog entry. Windows and Linux are not affected.

## Original proposal

## Problem

Image-to-video takes one image, and it always becomes the first frame. There's no way to say where the clip should end. Users who want a transition between two stills (a pose change, a before/after, a camera move between two framings, a loop that ends where it began) can only describe the end state in the prompt and hope.

LTX supports this already. The model conditions on images at any frame. The vendored MLX pipelines and diffusers 0.41.0 both have APIs for adding a keyframe after the first frame. The gap is in our app layer: the CLI, runner, request types, backend interface, Web UI, recorded settings and prompt enhancer all carry exactly one image path.

### Not in scope

- **Keyframes in the middle of the clip.** The same APIs take any frame index, so the design below shouldn't block it. But there's no UI for placing images on a timeline, and no request for one.
- **An end image without a start image.** LTX can do it, but it needs `img2vid` to stop requiring a reference image, which touches the workflow contract and the gallery reuse checks. Leave it out until someone asks.
- **Per-image strength or CRF controls.** Both images use strength 1.0 and the checkpoint's default CRF, as the single image does today.
- **The dedicated keyframe interpolation pipeline** on macOS. See [Alternatives](#alternatives-considered).

## Evidence

Checked against `main` at v0.13.0b19, the vendored `ltx_pipelines_mlx` and `ltx_core_mlx`, and the installed diffusers 0.41.0 (2026-10-11).

### macOS (vendored MLX)

| Check | Result |
|---|---|
| `ltx_pipelines_mlx.utils.args.ImageConditioningInput` | `(path, frame_idx, strength, crf=None)`. `frame_idx` is a **pixel** frame. `resolve_frame_indices(images, num_frames)` turns a negative index into `num_frames + idx`, so `-1` is the true last frame |
| `utils._orchestration.combined_image_conditionings` | An image at `frame_idx == 0` becomes `VideoConditionByLatentIndex` (replaces latent frame 0). Every other image becomes `VideoConditionByKeyframeIndex` (extra tokens appended at that frame's position, with their own positions) |
| `VideoConditionByKeyframeIndex` (`ltx_core_mlx/conditioning/types/keyframe_cond.py`) | Holds `keyframe_latent`, **not** `clean_latent`. `VideoConditionByLatentIndex` holds `clean_latent` |
| One keyframe's attention mask | `update_attention_mask` returns `None` for a single keyframe (`ltx_core_mlx/conditioning/mask_utils.py`), so no dense mask is added |
| `DistilledPipeline.generate_and_save` (two-stage upscale, inherited from `TI2VidTwoStagesPipeline`) | Takes `images: Sequence[ImageConditioningInput]` next to the legacy `image=`. Resolves indices against the frame count (`distilled.resolve_stage1_frames`), re-encodes the images at full resolution for stage 2, and drops the appended tokens in both stages |
| Our one-stage path (`video_mac._generate_one_stage`) | Calls `combined_image_conditionings` with one image at `frame_idx=0`. Slices `output.video_latent[:, : F * H * W, :]`, which drops appended keyframe tokens. The audio state is unaffected. Both images are encoded inside one `_encode`, so low-memory encoder freeing still works. **But** it materializes the conditionings with `mx.eval(*(cond.clean_latent for cond in conds))`, which raises `AttributeError` on a keyframe conditioning |

On macOS, a last frame at `-1` lands on the true last frame in both one-stage and two-stage runs.

### Windows and Linux (diffusers 0.41.0)

| Check | Result |
|---|---|
| `LTX2ConditionPipeline.__call__` | Takes `conditions: LTX2VideoCondition \| list[LTX2VideoCondition]`. Its docstring example uses first frame `index=0` and last frame `index=-1` |
| `LTX2VideoCondition` | `(frames, index=0, strength=1.0, crf=None)`. `index` is a **latent** frame index. A negative index is taken modulo the latent frame count |
| Where a non-zero latent index lands | `pixel_frame_idx = (latent_idx - 1) * 8 + 1`. For `-1` that's `(L - 2) * 8 + 1 = num_frames - 8`. At 121 frames, the "last frame" is pixel frame 113. At the 9-frame minimum it's frame 1. A latent index can't place a keyframe on the true last pixel frame |
| Loading | `audio_scheduler` is optional, so the pipeline loads from the same components as `LTX2ImageToVideoPipeline` |
| Guidance defaults | The same as `LTX2ImageToVideoPipeline` (3.0 / 1.0 / 3.0 / 0.7, audio 7.0). `_build_generation_kwargs` passes no guidance, so each pipeline's own defaults apply |
| Image framing | `LTX2ImageToVideoPipeline` **stretches** the image to the video size (`video_processor.preprocess`, default `resize_mode`). `LTX2ConditionPipeline` **scales to cover and center-crops** |
| Noise default | `noise_scale` defaults differ between the two pipelines (0.0 against `None` → the first sigma) |
| `_filter_supported_kwargs` (`video_diffusers.py`) | Silently drops keys the target pipeline doesn't accept. A misrouted `image=` or `conditions=` would vanish without an error |
| `diffusers` pin | `==0.41.0` in `pyproject.toml`, which exports `LTX2ConditionPipeline` |
| Existing `from_pipe` pattern | `image_win.py` builds `AutoPipelineForImage2Image.from_pipe(model, torch_dtype=model.vae.dtype)` per call, with no extra weights |

So switching every CUDA image-to-video run to the condition pipeline would change start-only output (framing and noise). And on CUDA, `-1` doesn't reach the last frame.

Generating 8 extra frames fixes most of that. With `N' = num_frames + 8` (still `8k + 1`), `-1` lands at pixel frame `(L' - 2) * 8 + 1 = num_frames`. That's one frame after the last frame kept by trimming to `num_frames`, about 40 ms at 24 fps. No `8k + 1` count puts it exactly on the last kept frame.

### Platform capability flags

`backends/__init__.supports_video_audio()` is the existing pattern for a video feature one backend lacks:
- `_definition_for_platform` in `web/workspace_contract.py` drops the `audio` control from the contract.
- `video_cli.py` and `_submit_video_job` normalize the flag before anything loads.
- `workspace_api.py` reports it to the frontend.

### Frame counts

The CLI and the Web server snap frame counts to `8k + 1` before they reach the backend (`video_cli.py`, `server.py`). The minimum is 9 (`utils/alignment.py`).

### Where the single image is wired through our code

| Layer | Where | What carries the image |
|---|---|---|
| CLI | `video_cli.py` | `--image` → `args.image_path`. The file must exist and `model_info.supports_i2v` must be true. `mode = "i2v" if args.image_path` |
| Runner | `video_runner.run_video_batch` | Rebuilds `VideoGenerationRequest` for every prompt and run from `args` (`image_path=getattr(args, "image_path", None)`) |
| Workflow | `workflows/__init__.py` | Picks `image_to_video_stage` when `args.image_path` is set, otherwise `text_to_video_stage` |
| Request | `core/video_types.VideoGenerationRequest` | `image_path: str \| None` |
| Backend protocol | `core/video_backend.py` | `image_to_video(model, image_path: str, ...)` |
| Stage | `workflows/video_stages.image_to_video_stage` | Passes `request.image_path`, and raises if it's `None` |
| macOS backend | `backends/mac/video.py` | `_generate` → `_generate_one_stage` (one `ImageConditioningInput` at 0) or `_generate_two_stage` (`image=image_path`) |
| CUDA backend | `backends/cuda/video/backend.py` | `load_model(mode="i2v")` loads `LTX2ImageToVideoPipeline`. `image_to_video` opens the image and passes `image=` |
| Web server | `web/server.py` | `_submit_video_job` builds an `argparse.Namespace` for `run_video_batch`. `_resolve_reference_image` reads `image_file` (upload) or `image_path`. `_submitted_settings` and `_public_job_context` only know `image_path`. `_optional_path` rejects a path that no longer exists |
| Web runner | `web/web_runner.py` | `mode="i2v" if request.image_path` |
| Workflow contract | `web/workspace_contract.py` | `visible_controls` holds control ids (`reference_image`, `reference_image_path`, `reference_image_clear` for `img2vid`). `clear_fields` holds form fields, which the frontend maps to draft state |
| Recorded settings | `utils/provenance.py` | `build_video_config_payload` records `workflow: "img2vid"` and `image_path`. `RecordedSettings` reads it back through `_local_path` |
| Gallery | `web/gallery.py` | `GalleryAsset.reference_image_path`, `_asset_metadata`, `gallery_asset_to_json`, and `reuse_params["image_path"]` |
| Prompt enhancer | `utils/prompt_enhance.py`, `preflight.py`, `web/server.py` | The `enhance_context()` factory, `preflight._job_context` (CLI and Web generation), and the server's enhance-preview parser build `EnhanceContext.reference_image`. `_context_rules` picks the "first frame is supplied" rule in video mode |
| Frontend | `frontend/src/` | `SettingsPane.svelte` (inline picker), `WorkspacePage.svelte` (upload state sent as `image_file`, the CLI warning for uploads, clearing the draft image when its asset is deleted), `GalleryPage.svelte` (the same clearing), `draft.svelte.ts` (control map, `_CLEAR_FIELD_STATE_UPDATES`, route prefill), `jobSettings.ts` (`PREFILL_KEYS`), `capabilities.ts`, `assetDetails.ts`, `assetActions.ts`, `cliCommand.ts`, `promptEnhance.ts` (decides from the workflow, not from which images are set), and the `GalleryAsset` / `JobSettings` types |

## Proposed change

Keep the `img2vid` workflow and its required start image, and add an optional **end image**. Name it `end_image_path` throughout, so `image_path` keeps its meaning and older recorded settings still read correctly.

macOS comes first and makes the feature usable end to end (steps 1–7). CUDA follows as its own step (step 8), using generate-and-trim. Until step 8 lands, Windows and Linux don't offer an end image: a capability flag drops the control and refuses the CLI flag before anything loads, following the existing audio pattern.

Build it in one worktree, with a commit per step. Each step leaves the app working.

### 0. Spike, with recorded results

The spike confirms the design rather than choosing between designs. Plain `index=-1` on CUDA is ruled out by the Evidence: its keyframe sits 8 frames before the end, and at 9 frames it sits on frame 1.

Run first+last generation directly against the libraries, without the app, using four image pairs:

- a pose change
- a camera move
- a loop: the same image at both ends
- a pair whose aspect ratios differ from each other and from the video

Use 17, 49 and 121 frames for each pair, on these three paths:

- macOS one-stage (`BasePipeline`, distilled sigmas), with a second `ImageConditioningInput(frame_idx=-1)` and the `keyframe_latent` fix from step 2 applied locally.
- macOS two-stage (`DistilledPipeline.generate_and_save(images=[...])`).
- CUDA generate-and-trim: `LTX2ConditionPipeline.from_pipe(...)` with `[index=0, index=-1]`, generating `num_frames + 8` frames and keeping the first `num_frames`. Build it from a loaded `LTX2ImageToVideoPipeline` with a LoRA and with `--low-memory` (CPU offload).

The reference for every similarity measure is the end image put through the same cover-and-crop transform the pipeline applies (`resize_and_center_crop` on MLX, the condition pipeline's cover-and-crop on CUDA), not a plain resize.

| Gate | Pass when |
|---|---|
| End-frame placement | Searching only the second half of the clip (so the loop pair's identical first frame doesn't win), the frame with the highest SSIM against the reference is the last frame or within 1 frame of it. On CUDA the expected best match is the last frame, since the keyframe sits one frame past it |
| Final-frame match | The last output frame's SSIM against the reference is at least that of the frame 8 frames earlier. A visual check agrees that the clip reads as ending on the image |
| Motion | A visual check finds no hard cut or crossfade into the end image |
| `from_pipe` (CUDA) | LoRA adapters still apply (output differs from a no-LoRA run with the same seed). CPU offload still works. Record the peak VRAM of start+end against start-only at the same frame count |

Every pair must pass every gate at every frame count, on each path. One failure blocks that path. Then:

- **A macOS path fails placement or final-frame match:** that path doesn't ship an end image. If only one-stage fails, steps 1–7 go ahead with the end image accepted only together with `--upscale` (two-stage), and the request refuses it otherwise. If both fail, stop and revisit the design.
- **A macOS path fails motion:** revisit [keyframe interpolation](#alternatives-considered) before any UI work.
- **CUDA fails any gate:** step 8 doesn't start, and Windows and Linux stay without an end image. Record why.

Write the numbers into this proposal's Evidence section before step 1.

### 1. Request, protocol, stage and runner

- `VideoGenerationRequest` gets `end_image_path: str | None = None`. Its `__post_init__` raises `ValueError` when `end_image_path` is set without `image_path`. That's the one check that runs on every path, because the workflow picks `text_to_video_stage` whenever `image_path` is empty, so a check in `image_to_video_stage` would never run.
- `VideoBackend.image_to_video` gets a keyword-only `end_image_path: str | None = None`. `image_to_video_stage` passes it through. `text_to_video` is unchanged. Update the protocol and backend docstrings.
- `run_video_batch` sets `end_image_path=getattr(args, "end_image_path", None)` when it builds each request.
- Add `supports_video_end_image()` to `backends/__init__.py`, next to `supports_video_audio()`. It returns `True` for the MLX video backend and `False` for diffusers until step 8. Platform branching stays in `backends/__init__.py`, as AGENTS.md requires.
- Tests:
  - The request rejects an end image without a start image.
  - The stage forwards `end_image_path` to a mocked backend.
  - `run_video_batch` passes `args.end_image_path` into the request the stage receives, for every prompt and run.
  - `supports_video_end_image()` follows the selected video backend.

### 2. macOS backend

- Add a pure helper that returns `(path, frame_idx)` pairs: `[(image_path, 0)]`, plus `(end_image_path, -1)` when set. It imports nothing from mlx. The call site turns the pairs into `ImageConditioningInput` inside the existing lazy imports.
- Add a helper that returns each conditioning's latent to materialize: `clean_latent` for `VideoConditionByLatentIndex`, `keyframe_latent` for `VideoConditionByKeyframeIndex`. `_encode` evaluates those instead of reading `clean_latent` directly.
- `_generate_one_stage`: run the inputs through `resolve_frame_indices(..., num_frames)` and then `image_conditioner.resolve_crf`, and pass them to `combined_image_conditionings`.
- `_generate_two_stage`: pass `images=` instead of `image=`. The pipeline resolves indices and re-applies both images in stage 2.
- Tests:
  - The pair helper, for start-only and start+end.
  - The latent helper, with fakes restricted to the real attribute names (`MagicMock(spec=["clean_latent"])` and `MagicMock(spec=["keyframe_latent", ...])`), so reading the wrong attribute fails the way it does at runtime.
  - A mocked `DistilledPipeline` receives `images=` with frame indices `0` and `-1`.
  - In one-stage mode, the indices reaching the mocked `combined_image_conditionings` are resolved against the frame count.
- Contract tests in `tests/test_ltx_mlx_contract.py`:
  - `resolve_frame_indices` takes `(images, num_frames)`.
  - `generate_and_save` accepts `images`.
  - `VideoConditionByKeyframeIndex` instances have `keyframe_latent`.

### 3. CLI, recorded settings and gallery

- `ziv-video --end-image PATH`. Validated like `--image`: `~` expanded, the file must exist, the model must support i2v. `--end-image` without `--image` is a parser error, and so is `--end-image` when `supports_video_end_image()` is false. Both are checked during argument validation, before the enhancer or the model loads.
- `build_video_config_payload` records `end_image_path` (through `_portable_path`, so home paths become `~`) when it's set. The workflow stays `img2vid`. `RecordedSettings` gets `end_image_path`, read back through `_local_path`.
- Gallery:
  - `GalleryAsset` gets `end_image_path`.
  - `_asset_metadata` fills it, and `gallery_asset_to_json` returns it.
  - Reuse adds `reuse_params["end_image_path"]` only when the resolved workflow is `img2vid` and the file still exists. When a missing start image makes reuse fall back to `txt2vid`, the end image isn't carried along. If it doesn't, Reuse drops the field and reports a `missing_end_image` fallback reason, instead of letting `_optional_path` reject the whole submission.
- Tests:
  - CLI validation: a missing file, no `--image`, a T2V-only model, and a backend without end-image support.
  - `end_image_path` round-trips through the embedded payload and `RecordedSettings`. An old payload without the key reads as `None`.
  - Gallery reuse with and without the recorded end image file present, and when the start image is missing (falls back to `txt2vid` without the end image).

### 4. Web server and contract

- `_submit_video_job` sets `args.end_image_path` on the namespace it hands to `run_video_batch`.
- The form accepts `end_image_file` (upload) or `end_image_path`. Give `_resolve_reference_image` the field names as parameters and call it twice, rather than copying it.
- `_submitted_settings` and `_public_job_context` take a `{field: saved_path}` mapping instead of a single `image_path`, so an uploaded end image is recorded on the job card and available to Retry.
- Contract:
  - Add the control ids `end_image`, `end_image_path` and `end_image_clear` to `img2vid`'s `visible_controls`.
  - Add the form field `end_image_path` to `clear_fields` for `txt2img`, `img2img` and `txt2vid`.
  - `requires_reference_image` is unchanged.
  - Give `build_workflow_contract` and `_definition_for_platform` a `video_end_image` argument next to `video_audio`, and drop the three end-image controls when it's false, the way `audio` is dropped today. Report the flag from `workspace_api.py` with `video_audio`.
- `_submit_video_job` refuses an end image when `supports_video_end_image()` is false, at submit time, before the job is queued.
- An end image without a start image is refused with a field error.
- The job type label stays "Image to Video".
- Tests:
  - The contract lists the new controls for `img2vid` only, and the new clear field for the other three workflows.
  - With `video_end_image=False`, the contract drops the end-image controls, and a submission with an end image is refused.
  - A submission with an uploaded end image reaches the runner's args and appears in the job's settings.
  - An end image without a start image is refused.

### 5. Web UI

- Extract the inline picker in `SettingsPane.svelte` into a reusable image-picker component. Use it for **First frame** (the existing picker) and **Last frame (optional)** directly under it. Both stay visible together, with no toggle or tab.
- Add a swap button between the two pickers. It swaps both the paths and any pending uploaded files.
- Upload state: `WorkspacePage.svelte` keeps a second pending file and sends it as `end_image_file`. The CLI warning for uploaded files covers it. Deleting an asset clears it from the draft wherever it's used as an end image, in `WorkspacePage.svelte` and `GalleryPage.svelte`.
- Draft and settings:
  - `draft.svelte.ts`: add `end_image_path` to the control map, to `_CLEAR_FIELD_STATE_UPDATES` and to route prefill.
  - `jobSettings.ts`: add `end_image_path` to `PREFILL_KEYS`.
  - `capabilities.ts`, and the `GalleryAsset` / `JobSettings` types.
- `assetDetails.ts`: show both images for a video that recorded an end image.
- `cliCommand.ts`: emit `--end-image`, and name uploads the same way it does for `--image`.
- Asset action **Use as last frame**:
  - If the draft is already in `img2vid`, it sets `end_image_path` and keeps the first frame.
  - From any other workflow, it opens `img2vid` with the first-frame slot empty and marked required, so an img2img reference never turns into a first frame by accident.
- Run `make frontend-build` and commit the rebuilt `web/static/app/`.
- Tests (Vitest):
  - Draft state holds and clears `end_image_path` across workflow switches.
  - Swap exchanges paths and pending files.
  - The CLI preview includes `--end-image`.
  - Both cases of "Use as last frame".
  - Prefill restores `end_image_path`.

### 6. Prompt enhancer

- `EnhanceContext` gets `end_image: bool`. In video mode with both images set, `_context_rules` picks a two-frame rule instead of the first-frame rule. The two-frame rule describes how the scene moves from the first image to the last, in time order, without re-describing what the images show.
- Set the flag in the four places that build the context:
  - the `enhance_context()` factory
  - `preflight._job_context`
  - the server's enhance-preview parser
  - `promptEnhance.ts`, which must send whether an end image is set, not infer it from the workflow
- Tests:
  - `_context_rules` returns a different rule set for start+end than for start-only in video mode, and the same set as today for start-only. Check which rule is chosen, not its wording.
  - `preflight._job_context` sets `end_image` from `args`.

### 7. Docs and changelog (macOS release)

- `docs/concepts/video.md` (and the picker in `docs/web-ui/workspace.md`, the flag in `docs/cli/video.md`):
  - A section on first and last frames: the Web UI pickers and the CLI flag.
  - The last frame is a soft condition, so the clip ends close to the image but not pixel-identical, unlike the first frame.
  - Use images with the video's aspect ratio. Each image is scaled to cover the frame and center-cropped.
  - The enhancer behaviour, next to the existing enhancer text.
  - Until step 8 lands, an end image is macOS-only. Windows and Linux don't show the Last frame picker.
- `docs/cli/reference.md`: add `--end-image`.
- `CHANGELOG.md` `[Unreleased]`: one entry under Added.

### 8. CUDA backend

Starts only if the spike's CUDA path passed. Until then, `supports_video_end_image()` is false on diffusers, so the CLI and Web server refuse an end image before anything loads, and the Web UI doesn't show the picker. As a backstop, `DiffusersVideoBackend.image_to_video` also raises `RuntimeError` when it's handed an end image while unsupported.

- **Start-only stays on `LTX2ImageToVideoPipeline`.** Its output doesn't change. `load_model` and its callers don't change.
- **With an end image**, build `LTX2ConditionPipeline.from_pipe(model.image_to_video, torch_dtype=model.image_to_video.vae.dtype)` per call, the same way `image_win.py` builds its img2img pipeline. That adds no weights, and no reference outlives the job.
  - Pass `conditions=[LTX2VideoCondition(frames=start, index=0), LTX2VideoCondition(frames=end, index=-1)]`.
  - Both images are scaled to cover and center-cropped. That differs from start-only, which stretches. Document it.
- **Generate and trim.** Generate `num_frames + 8` frames (still `8k + 1`), then keep the first `num_frames`.
  - The keyframe lands one frame after the last kept frame, about 40 ms at 24 fps (see [Evidence](#windows-and-linux-diffusers-0410)).
  - Trim the stage 1 frames *before* `_upscale_video`. The latent upsampler takes the frame count from the frames it's given, so trimming first avoids upscaling frames that get thrown away. It doesn't touch the upsampler's height/width meaning (platform parity step 3).
  - Duration and the recorded `frame_count` stay right: export uses `default_fps` on the trimmed frames, and provenance records `request.num_frames`.
  - CUDA video has no audio yet ([platform parity](archive/platform-parity.md) item 2). When that lands, its export must trim the audio to the same length.
  - Put the trimming in a small pure helper, so it's tested without a pipeline.
  - **Cost:** +8 frames is +7% frames at 121, +16% at 49 and +89% at 9. Attention cost grows faster than the frame count, and peak VRAM rises too, so a job that just fits start-only can run out of memory with an end image. If platform parity's video memory badges (item 7) land, they must estimate an end-image job at `num_frames + 8`.
- **Guard against silent kwarg drops.** After `_filter_supported_kwargs`, raise `RuntimeError` when the image input the call depends on (`image` for start-only, `conditions` for start+end) was filtered out.
- **Upscale.** Today's CUDA upscale doesn't refine (platform parity step 3), so the end image only affects stage 1. If platform parity step 3 lands first, its stage 2 refine must re-apply both conditions, as the MLX pipeline does.
- **Pipeline class.** Add `condition: type[Any] | None` to `_PipelineClasses`. If it's `None` and an end image is set, raise `RuntimeError`. diffusers is pinned to 0.41.0, so there's no separate LTX 0.9 path to handle.
- **Turn it on.** `supports_video_end_image()` returns `True` for diffusers, so the contract, CLI and server accept an end image.
- **Tests:**
  - A mocked condition pipeline receives `conditions` with indices `0` and `-1`, and `num_frames + 8` frames.
  - Start-only still calls the image-to-video pipeline with `num_frames`.
  - The trim helper keeps the first `num_frames`, and the upscaler receives the trimmed frames.
  - The kwarg guard raises when `image` or `conditions` is dropped.
  - A contract test that `LTX2ConditionPipeline.__call__` takes `conditions`, and `LTX2VideoCondition` has `frames`, `index`, `strength` and `crf`.
- **Docs and changelog:** remove the macOS-only note from `docs/concepts/video.md`, describe the Windows and Linux behaviour (end image one frame past the end, extra time and memory), and add a changelog entry. Set this proposal to Done and update the index.

## Alternatives considered

- **A separate `flf2vid` workflow.** Cleaner for the contract (a workflow that requires both images), but it adds a third video workflow to the switcher for an optional addition to `img2vid`, and it duplicates every `img2vid` control. Rejected.
- **A general keyframe list (`images: list[(path, frame_idx)]`) in the request and protocol now.** Future-proof for mid-clip keyframes, but there's no UI for it, and the two libraries index frames differently (pixel against latent), so anything beyond the first and last frame needs per-backend translation. A single `end_image_path` widens into a list later. Rejected for now.
- **Moving all CUDA image-to-video to `LTX2ConditionPipeline`.** One code path, but it changes start-only output: framing goes from stretch to crop and the noise default changes. Loading it for i2v would also mean `load_model` has to know about the end image. Rejected in favour of `from_pipe` on demand.
- **`KeyframeInterpolationPipeline` on macOS.** Upstream's dedicated interpolation pipeline: half resolution with optional CFG, using the dev (non-distilled) transformer if given, then a 2× upscale and a distilled refine, with the keyframes re-encoded at each stage. Likely the best quality, but it's a third pipeline to load and maintain. It wants dev weights we don't ship, and it always runs two stages, so it doesn't fit the one-stage path. Revisit only if the spike's motion gate fails on the distilled paths.
- **Passing the end image through the prompt only.** That's what users do today, and it's the problem.

## Open questions

- **End image without a start image.** See [Not in scope](#not-in-scope). Decide after step 5, based on whether anyone asks for it.
- **Different aspect ratios.** Both platforms scale each image to cover and center-crop it, except CUDA start-only, which stretches. If the spike shows a jarring result when the two images differ, warn at submit time when either image's aspect ratio differs from the video's by more than a small tolerance.
