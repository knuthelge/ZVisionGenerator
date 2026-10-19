# Extend a video

**Status:** Done (unreleased, 2026-10-12), on macOS. Windows and Linux moved to [platform parity](platform-parity.md) step 11. The prompt design builds on [A scene for video prompts](video-scene.md).

## Problem

A video job makes one clip, and its length is limited by memory: the `xl` preset already defaults to 25 frames (about 1 s), and even `m` stops at a few seconds on a 24 GB Mac. There is no way to make a longer video from a clip you like. References are images only (`canUseAsReference` in `frontend/src/lib/state/assetActions.ts`, `_resolve_reference_image` in `web/server.py`), and `img2vid` uses one image as frame 0. Starting an `img2vid` run from a clip's last frame leaves a visible seam: motion restarts and the sound stops.

## Evidence

- LTX continues video by pinning the end of the previous clip at the start of a new window. Upstream does it in its chunk system, and the vendored `packages/ltx_pipelines_mlx/extend_distilled.py` (`ExtendDistilledPipeline.extend_from_video`) does it for a source file: the last 25 pixel frames (4 latent frames) and their audio tokens are pinned at index 0 with `VideoConditionByLatentIndex` at strength 1.0, and the rest of the window is generated.
- That pipeline is not the right base for us as it stands:
  - It is two-stage (half resolution, upsample, refine). Our normal macOS path is the one-stage `BasePipeline` loop at full resolution (`backends/video_mac.py`, `snap_output_dimensions(two_stage=False)`); `DistilledPipeline` loads only for `--upscale`. Using it would mean a different sampler, the 64-pixel grid (custom sizes only need 32), the upsampler weights, and a model reload between generate and extend jobs.
  - It encodes the whole source at two resolutions and returns latents for the whole clip, so the original part goes through the VAE again and cost grows with the source. (The latents are bfloat16; only a transient numpy intermediate is float32, so the cost is smaller than first estimated, but it still scales with the source.)
  - It reads only the first `1 + 8k` frames, so for other lengths the carried "tail" is not the real end.
- Pinning only the last 25 frames gives exactly the window: 4 latent frames carried, `4 + n` generated, `25 + 8n` pixel frames decoded, and dropping the first 25 leaves exactly `8n` new frames. With a 25-frame carry, latent 0 really is one pixel frame, which the whole-source path gets slightly wrong.
- On Windows and Linux, diffusers 0.41.0 (pinned) ships `LTX2ConditionPipeline`. `LTX2VideoCondition(frames=…, index=0, strength=1.0)` overwrites the first latent frames, which matches the MLX pin. Our backend wires only `LTX2Pipeline` and `LTX2ImageToVideoPipeline` today, and there is no audio on those platforms.

## Shipped design

Extend generates only the new part and splices it onto the original, which stays untouched.

1. **Read the tail.** `utils/video_tail.py` decodes exactly the source's last 25 frames into memory (decode the last few seconds, keep a ring buffer of 25) at the **displayed** size, with ffmpeg's auto-rotation on, and the matching audio is read at its position in the file. No temporary clip: cutting one by time is off by a frame at keyframes or when `nb_frames` is missing, and re-encoding it adds a generation of loss.
2. **Generate the window** in the existing one-stage loop, with a carry: the VAE-encoded tail pinned at latent indices 0–3 with `VideoConditionByLatentIndex`, and its audio tokens pinned in the audio state. The window is `25 + 8n` frames at the source's size and fps, with the job's own steps and guidance. A **silent source** gets a noise-initialised audio state with no pin (a zero audio latent is not silence) and no audio track in the result. Nothing in `packages/ltx_*` is edited; every newly used internal is in `tests/test_ltx_mlx_contract.py` and "Internals we rely on" in `docs/development.md`.
3. **Decode the whole window and drop the carried frames.** Decoding with the carry in place gives the causal VAE context at the join. Keep the last `8n` frames. Cut the decoded audio by samples, not tokens: skip `round(25 / fps × 48000)` samples and keep exactly `round(8n / fps × 48000)` (audio runs at 25 tokens/s, so a token cut misses the frame boundary by up to ±20 ms). The model's audio comes out about 32 ms shorter than its video; the cut zero-pads with a short fade.
4. **Splice** in `utils/video_splice.py`, called by the `join_clips` stage. It streams through ffmpeg and never loads the original into memory, since chains grow without limit:
   - **Stream-copy branch** (a source made by the app, not rotated, same codec, profile, pixel format, size, time base, frame rate and colour tags as the segment): the source's video is first remuxed on its own (video only), then joined to the segment with the concat demuxer and `-c copy`. The original part is bit-identical. The remux is needed because concatenating the source with its audio shifts the video by the AAC priming delay (about 21 ms per extend), which builds up down a chain.
   - **Re-encode branch** (any other source, including rotated ones): the original is re-encoded once at crf 12, upright, with the colour matrix and tags set inside the filter graph and copied as probed. The spike measured crf 18 at a join error of 1.36 ΔE, crf 12 at 0.87.
   - **Audio:** the source's audio is decoded to PCM (which honours AAC priming) and fitted to exactly `N / fps`. The end of the source's sound is filled from the model's own version of that moment when it is short or ends in digital silence (a previous extension's silent tail), with 8 ms blends instead of cuts, so there is no hole at the join. The new PCM is appended and AAC is encoded once. AAC streams are never joined directly.
   - The source's metadata is dropped (`-map_metadata -1`) before `embed_mp4_config` writes ours.
5. **Seam check (phase 0).** The jump at the join is about 2 to 2.5 ΔE in a single frame, whatever the prompt, carry length or seed. A plain VAE encode and decode of the 25 carried frames gives the same step, so the seam is the LTX VAE round trip of the carry, and no pin or prompt removes it. The contact sheets look seamless. **The seam is accepted**: stream copy stays and there is no crossfade (a crossfade would halve the step but force a full re-encode of the original on every extend, losing bit-identity). The carry stays at 25 frames: 49 gave no improvement at 1.25× the time.
6. **Check the source at the entry points, not in `run_preflight`.** `utils/video_extend.py` returns a list of problems with stable codes (no video stream, unreadable, shorter than 25 frames, off the 32-pixel grid, under 64 pixels, variable frame rate, non-square pixels, more than 8 bits, HDR, interlaced, upscaled, unreadable audio). The CLI (`load_extend_source`), the Web UI's inspect route (`POST /api/extend/source`), submit and the enhance route each call it, so a clip is refused with every problem named before any model loads. **Rotation is supported**: phone clips stored sideways are read upright, the displayed size is checked against the grid, and they take the re-encode branch. A memory check in `utils/model_memory.py` warns when the window likely won't fit but lets the job run, as the uncapped Duration field does. No padding or cropping.
7. **Length.** The extension is `8n` frames (n ≥ 1), converted with the **source's** fps, not the model's: 2 s by default, `--extend-seconds` or `--extend-frames` on the CLI. The Web UI shows **Extend by (s)**, "+N frames" and the new total. The Duration field's `8k + 1` snapping does not apply.
8. **Contracts.**
   - One new workflow value, `extend`: provenance `workflow`, `source_video` (absolute path, recorded `~/…` inside the home folder), `extend_frames` and the total `frame_count`; the Web workflow contract, gallery filters and aliases; `VideoGenerationRequest.extend`.
   - A capability, not a model flag: the `VideoExtender` Protocol (`extend_video`) is separate from `VideoBackend`, and `supports_video_extend()` in `backends/__init__.py` gates the Web UI contract, the routes and the CLI. A backend without it never offers extend.
9. **Pre-fill and Reuse.** Extend starts from the source's model and LoRAs (only ones still available), editable. Reuse of an extended video reopens the Extend Video tab with the source, scene, prompt and length, and says clearly when the source file has moved.
10. **Seed.** Extend follows the normal seed lock and random rules; it does not reuse the source's seed. Noise is drawn per tensor shape, and the window never has the source's shape, so a shared seed gives unrelated noise; it would also make every run in a batch identical.
11. **Audio.** A source with audio continues it across the join. A silent source stays silent. `no_audio` strips the whole output.
12. **Batches.** Several runs or prompts make sibling extensions of the same source; chaining is done by extending a result again.
13. **Surfaces.**
    - Web UI: an **Extend Video** tab (shown only when the contract lists `extend`), with a Source video section (Browse or a typed path), **Extend by (s)** and a memory note; **Extend** on video assets in the viewer (`E`) and the ⋯ menus, which opens the tab with the source set; Reuse; and a "Joining clips…" phase on the job card.
    - CLI: `ziv-video --extend SRC.mp4 --extend-seconds 4 --prompt "…"` (or `--extend-frames`).
    - Docs: `docs/guides/video.md`, `docs/guides/prompts.md`, `docs/reference/cli.md`, and a CHANGELOG entry.

### Prompts

An LTX prompt describes the whole clip in time order: scene, action, and quoted dialogue, which drives speech. The window covers the carried second plus the new frames. Reusing the source prompt would re-speak its lines and restart its actions; dropping it lets characters, setting and style drift once the pinned second is over. The design that shipped splits the prompt with the [Scene field](video-scene.md), replacing the first design (a read-only "Continues from" block with a **Copy into prompt** action, and a `scene_in_prompt` flag):

- **Scene is kept, Prompt is what happens next.** The Scene box is filled from the source's recorded scene and is editable; the Prompt box starts empty with "What happens next…". The model receives `"{scene} {prompt}"`. An empty prompt is allowed and means "continue naturally".
- **The enhancer writes only the action.** The scene is context and is never rewritten per link, which avoids drift down a chain. The previous clip's prompt goes into the system message as "what just happened" (so the user message stays the continuation), with new rules: start from where the source ends, let a moved setting win, count people from the scene as already present with their voices, and fit the action to the new frames. A repeat guard retries a rewrite that contains a quoted line from the scene or previous prompt, or is too close to it.
- **Clips made before scenes existed** (one recorded prompt, no scene): the Prompt box is pre-filled with the old prompt and Scene is empty. While Prompt still equals the old prompt, a non-blocking warning says to change it. With the enhancer on, it extracts a scene from the old prompt and writes the action from the user's input. Without it, quoted sentences are stripped from the old prompt only while it is unchanged, so old dialogue is not spoken again.
- **Without a rewrite**, what is sent is the scene followed by the prompt.
- **The server reads the scene from the source file** for the inspect, enhance and submit routes; the client sends only its own text. In extend mode the client always sends the `scene` field, even empty: absent means the source's recorded scene, blank means deliberately no scene.
- **Prompt files.** Each entry is a continuation. CLI `--extend` with `--prompt` works the same way, and `--scene` replaces the source's scene.
- **Provenance.** `scene` and `prompt` as sent, plus the user's `scene_input` and `prompt_input` when enhanced; the scene is copied unchanged down a chain, and an extend of an extend reads the previous action from the source's `prompt`.
- **Voice.** Only the carried audio, the prompt and the speaker descriptions hold a voice. A 25-frame and a 49-frame carry (about 1 s against 2 s of clean audio) gave no meaningful difference. `AudioConditionByReferenceLatent` exists in the vendored code but is used only with a lip-dub IC-LoRA; a later spike at most.

### Windows and Linux (phase 2)

Moved to [platform parity](platform-parity.md) step 11.

## Phases

0. **Spike (macOS), done:** tail reader, carry in the one-stage loop, window decode and splice on app-made clips (a talking clip with audio, fast motion, a silent calm clip), a clip from another tool and a larger size. Outcome: assertions on frame and token arithmetic held; the seam is accepted; the carry stays at 25 frames; the re-encode branch uses crf 12; a prompt that reuses the source prompt repeats its speech, which confirms the legacy rules.
1. **macOS, done:** workflow, source checks, backend, splice stage, enhancer extend mode, provenance, CLI, Web UI (the Extend Video tab), docs. A later fix moved the video remux in front of the concat (A/V sync) and added the join audio fill.
2. **Windows and Linux:** condition pipeline, capability flag, seam spike.

## Tests

Mocked, with no real models:

- the `25 + 8n → 8n` frame arithmetic and the audio sample arithmetic;
- the ffmpeg commands of each splice branch, and the join audio fill;
- each source problem code from probe fixtures, including a rotated clip that passes;
- the vendored internals in the contract test;
- provenance keys, the workflow value, its aliases and visible controls;
- extend enhancer messages (scene and previous prompt only in the system message), the repeat guard, and the length plan;
- Reuse and the legacy-prompt warning in the frontend.

Tests that run the real `ffmpeg` and `ffprobe` (tail reading, rotation, stream copy bit-identity, join audio and A/V sync) skip when the tools are missing and are run locally, not in CI.

## Alternatives considered

- **Use `ExtendDistilledPipeline` as is.** Rejected as the default: it is two-stage and needs the 64 grid, the upsampler and a model reload, it re-encodes and re-decodes the whole source, and cost grows with the source. Upscaled sources are rejected in v1 instead.
- **Use upstream's `RetakePipeline.extend_from_video`.** Rejected: it needs the dev weights, runs with CFG 3.0 and about 30 steps, and encodes the whole source, so it is slower, loads different weights from the distilled model the app already has, and its cost grows with the source.
- **Pass a trimmed temporary clip to the vendored pipeline.** Rejected: cutting by time is off by a frame, and the extra encode adds loss. Reading the tail into memory is exact.
- **`img2vid` from the last frame.** Rejected: motion restarts and audio does not continue.
- **Decode only the new latent frames.** Rejected: without the carried frames the causal decoder has no context at the join.
- **Crossfade the join.** Rejected after the spike: it halves the single-frame step, still fails on static clips, and forces a full re-encode of the original on every extend.
- **Reuse the source's seed to keep the voice.** Rejected: noise is drawn per shape, so a window of another length gets unrelated noise; voice comes from the conditioning, not the noise. It would also make batch runs identical.
- **Pre-fill the prompt with the source's prompt.** Rejected for clips with a recorded scene: its quoted lines would be spoken again and its actions restarted. The scene is kept and the prompt starts empty. Clips without a recorded scene are the one exception, with a warning.
- **A read-only "Continues from" block with Copy into prompt.** Replaced by the Scene field, which holds the same context in an editable box that the enhancer leaves alone.
- **Validate inside `run_preflight`.** Rejected: preflight plans prompts and rewrites, and the checks need the source file, so each entry point checks before it builds a plan.

## Decisions (2026-10-11)

- **Encoding:** stream-copy the original's video and append a matching new segment; one full re-encode (crf 12) only for sources that can't be copied, including rotated ones.
- **Seam:** accepted, no crossfade.
- **Sources:** any clip that passes the checks, not only clips made by the app. Rotated phone clips are supported.
- **Upscaled sources:** rejected in v1.
- **Model and LoRAs:** pre-filled from the source and editable.
- **Length:** no hard maximum; the memory check warns and the job may still run.
- **Batches:** sibling extensions only; no chain option.
- **Silent source:** stays silent.
- **Prompt:** Scene is kept and Prompt is what happens next (see [Prompts](#prompts)).
- **Seed:** normal seed rules; the source's seed is not reused.
- **Platforms:** macOS first; Windows and Linux (video only) are [platform parity](platform-parity.md) step 11, with Extend hidden by the capability until then.
