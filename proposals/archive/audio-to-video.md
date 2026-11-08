# Make a video follow your own sound

**Status:** Done (2026-11-01, unreleased; branch docs/higgsfield-ideas; verified on a 32 GB Mac the same day)

## Problem

Every video's sound is generated, so a clip can't speak your words or move to your song. "Make this photo say what I recorded" or "a shot for this track" needs another tool, and those tools are either cloud services (Higgsfield's Speak and UGC Builder, Kling Avatars) or separate lip-sync models that edit an existing video.

LTX can do it with the weights we already ship: its audio-to-video mode holds an input track fixed while the picture is generated, so the picture follows the sound. MiniMax H3 already holds a clip's last sound fixed when it extends one, and does the same for a whole track.

## What the user sees

The **Video** and **Extend Video** tabs get an optional **Audio track**, with or without a first frame: an audio file. Setting one is the whole decision: the clip is made to follow it, and its sound is exactly that track, from **Start at**. Every video model that passes the spike on both platforms offers it: LTX-2.3 (`ltx-4`, and `ltx-8`, the same weights at 8-bit), LTX-2.5 (`ltx25-4`) and MiniMax H3 (`h3`, with Upscale off) have passed on Windows/Linux and on macOS. A model that doesn't pass doesn't show it.

- **Choosing a track** works like the start and last frame: a file from the device, a path, or Browse on the server. No recording in the app, and no video files as a source.
- **Start at** is a seconds field under the track (default 0) with the timestamp beside it ("90 → 1:30"). A Start at past the end of the track is allowed and gives a silent clip.
- **Length.** Duration (and Extend's added length) works as it does today; the clip always keeps it. Where the track runs out (or from a Start at past its end), the sound is silence. The user dropped the shortening on 2026-11-01 after review: the clip no longer ends with the track.
- **Sound.** While a track is set there is nothing else to decide: the Audio switch and the enhancer's Sound options aren't shown.
- **Batches, Repeat and prompt files.** One track per job, used by every run, like a start image.
- **Extend** shows the Audio track row for any clip. For a clip made from a track, it is filled in with the same file and Start at where that clip's sound ended, so Extend carries on with the track; clear it for generated sound, or pick another file. This is how a track longer than one clip is covered: make the first part, then Extend.

## Evidence

### Upstream

- Lightricks' official audio-to-video for LTX-2.5 runs on the distilled transformer, both stages at CFG 1, a 2× upscale with a short refine, frames on the 1+8k grid, length from the trimmed track and an optional first frame. The track is "frozen through both stages" and the trimmed original is attached unchanged ([guide](https://docs.ltx.io/open-source-model/usage-guides/audio-to-video)). The matching workflow is `LTX-2.5_A2V_Two_Stage_Distilled.json` in [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) `example_workflows/2.5`; there is none for 2.3.
- Lip-sync quality is unmeasured anywhere we could find. Reports favour 2.5 over 2.3, a March 2026 post calls 2.3's audio-to-video a mood conditioner rather than lip sync, and an HF discussion reports loose lip sync on musical avatars. Upstream's own docstring says to pass isolated vocals for music or loud ambience (`a2vid_distilled.py:119`).
- A frozen stream conditions the model as clean in three places: its tokens, a per-token timestep of 0, and a modality sigma of 0, which drives its prompt modulation and the other modality's cross-attention gate (`packages/ltx_pipelines_mlx/utils/samplers.py:83-103`). The sigma is the part that tells the video the sound is final.
- The alternatives that take a photo plus audio are separate model families (InfiniteTalk on Wan); LatentSync and the LTX-2.3 Dub-It IC-LoRA re-dub an existing video instead. None is worth a new model before LTX's own mode is measured.
- MiniMax H3: our `h3` (FL2VA) has no audio parameter (`packages/minimax_h3_mlx/pipeline.py:444-459`), and Ref2VA takes 2–15 s audio clips as references, not a timed track (dropped in [references](references.md)). But H3's Extend holds the clip's last sound fixed as clean audio rows while it generates the rest: `encode_tail_audio` and the sampler's `tail_audio` on Windows/Linux (`backends/cuda/video/h3/conditioning.py:160-228`, `h3/sampler.py:57-69`, `h3/backend.py:393-394`), and the MLX loop supports clean condition audio rows (`pipeline.py:600-671`), though nothing produces them today: macOS Extend on H3 carries no sound (`backends/mac/h3_video.py:855, 873`; `packing.py:344`). Holding every audio row is a generalisation of the Windows/Linux tail, and a new conditioning path on macOS (step 1). H3 then syncs (spike results).

### macOS

- The vendored `A2VidDistilledPipeline` (`packages/ltx_pipelines_mlx/a2vid_distilled.py:28-194`) subclasses the `DistilledPipeline` the app already loads for `--upscale 2` (`backends/mac/video.py:824-831`). It freezes the track in both stages (43-94), takes the first frame through `images=` (110-112), clears its tokens in a `finally` so Next and Stop are safe (165-166), and needs no dev transformer. It always runs two stages (153-164) and raises "Audio is too short" when the clip needs more audio than the file has (`a2vid_two_stage.py:95-101`).
- The app's one-stage path (`backends/mac/video.py:459-597`) has no frozen-audio state today. Extend's carry is a `VideoConditionByLatentIndex` (385-389), which zeroes the mask but doesn't freeze the stream, so it isn't the mechanism. A one-stage track needs a frozen state instead, as the vendored pipeline builds it (`create_noised_state(initial_latent=tokens, frozen=True)`, `a2vid_distilled.py:86-94`), with tokens encoded from the window waveform the way `_encode_carry_audio` encodes Extend's carry (step 1), and the ancestral loop LTX-2.5 already uses, which handles frozen states (`samplers.py:436`).

### Windows and Linux

- diffusers' `LTX2Pipeline` takes `audio_latents` only as starting latents and denoises them (`pipeline_ltx2.py:~1596`), so stock diffusers has no audio-to-video.
- Our `pin_audio_prefix` (`backends/cuda/video/audio_carry.py:100-125`) pins Extend's carried sound by replacing the first audio tokens and setting their per-token `audio_timestep` to 0. It isn't a whole-track pin as it stands:
  - It does nothing when the call has no `audio_timestep` (112-115), and the text-to-video `LTX2Pipeline` never passes one (`pipeline_ltx2.py:1414-1435`); only the image-to-video and condition pipelines do.
  - It leaves `audio_sigma` at the step's sigma (106-107), and with `use_cross_timestep` the video's audio cross-attention gate runs on it (`transformer_ltx2.py:1477-1478, 1560-1570`), so the video would see the track as noisy through the path that drives lip sync.
- Two-stage runs refuse a carry today (`backend.py:1159-1160`), and stage 2 re-noises stage 1's audio (1061). A hook that replaces every audio token on every forward pass covers both stages by wrapping both calls.

### Cost

- Every LTX run already generates audio tokens (about 25 per second), so a track adds no tokens, only one audio encode. Reference points from this repo: `ltx25-4` image to video at 1280×704 on the 10 GB card took 176 s at about 22.5 GB peak RAM ([ltx-2.5](ltx-2.5.md)); `ltx-4 --upscale 2` at 768×448 on a 32 GB Mac took 135–147 s at 17.7 GB ([macos-speed](macos-speed.md)).

## Proposed change

### Spike first: does it sync?

Lip sync depends on the weights, not the platform, so it is answered with the least code before any app work. On macOS the vendored `A2VidDistilledPipeline` runs as it is; on Windows/Linux a standalone script runs the image-to-video pipeline under the whole-track hook from step 1. Windows/Linux ran here. macOS is checked on a Mac after the merge; a family that fails there gets `platform_capabilities.darwin.audio_track: false`.

Three photos (a face close-up, a half-body shot, a cartoon character) and a clear voice note, on `ltx25-4`, `ltx-4` and `h3`, upscale on and off. Clips are watched with sound.

| Question | Pass |
|---|---|
| Does the mouth follow the words? | The owner judges at least 2 of 3 photos in sync at normal speed, per model. A model that fails doesn't offer the track (see step 1) |
| Does a one-stage run (upscale off) sync as well as two stages? | Same judgement on both platforms |
| Does an extension with a track carry on in sync across the seam? | One clip of `ltx25-4` extended twice with Extend's filled-in track; no jump in the words or the mouth at the seams |
| Memory and time against the same settings without a track | Within 5% on the 10 GB card and the 32 GB Mac |
| Informational: two speakers, and a song with vocals | Recorded for the docs; no gate |

If no model syncs on a clear voice, stop and mark the proposal rejected. Write the results here before step 1.

#### Results on Windows/Linux (2026-10-31, RTX 3080 10 GB, 32 GB RAM)

A script outside the app wrapped the CUDA backend's one- and two-stage passes in the whole-track hook from step 1 (tokens, a per-token timestep of 0 and an audio sigma of 0 on every forward pass) and ran `ziv-video` image to video at 768×448 (`-s s`) and, with upscale, 1280×704 (`-s m --upscale 2`), 137 or 177 frames at 24 fps, then remuxed the source track. Photos made with `zit`; tracks: a synthetic voice (Piper TTS), a real female voice (Pixabay licence) started at 1.6 s, and a 7.5 s CC0 pop song with vocals.

| Question | Result |
|---|---|
| The track is held | The model's own decoded sound follows the source's loudness (envelope correlation 0.82) |
| `ltx25-4` sync | Pass. All 6 clips in sync with the synthetic voice; face and cartoon in sync with the real voice |
| `ltx-4` sync | Pass with a real voice (face and cartoon). With the synthetic voice only the cartoon synced, so a robotic voice is a poor test, not a model limit |
| A man with a woman's voice | Out of sync on both models; the same man sings along to the song. A voice that doesn't fit the face is a likely limit to note in the docs |
| Songs | Both models sing along, including the man |
| Upscale off against on | Same sync; no rule needed, so a track works with Upscale off |
| Start at | A track from 1.6 s synced as well as from 0 |
| Memory | Peak RAM 19.6–21.7 GB with a track, the same as without; 4.5 GiB VRAM (5.0 with upscale). One 177-frame `ltx25-4` clip was killed by the kernel OOM killer at 24.9 GB with other jobs holding RAM; a longer clip, not the track |
| Time | Within noise. A back-to-back `ltx25-4` pair with per-phase times, the track encoded before the transformer was attached: whole runs 178 s and 184 s with a track against 171 s and 201 s without, denoising 63.5–67.7 s against 64.5–78.8 s; encoding the track takes 0.4–1.2 s. The earlier 13–31% came from the spike encoding the track after the transformer was attached and from queue load. One encode took 375 s while the machine's swap was full |
| Extend with the filled-in track | A 5.7 s clip of the real voice extended twice by 4 s: the model's own sound follows the track in every part (envelope correlation 0.56, 0.74, 0.69), so the parts line up. The owner saw it "kinda work", with the last 3 s slowed, where the recording itself pauses (13.2–14.6 s in) |
| Music video, 10 s + 10 s | `ltx25-4`, a song with vocals ("Calling" by Avanti, freetouse.com) from 0 s, a 241-frame clip extended by 240 frames: in sync, the join fine. Peak RAM 21.3 GB and 21.9 GB; a first attempt was killed by the kernel OOM killer while other processes held the machine's memory |
| `h3` sync | Pass. Face, half-body and cartoon with the real voice and the song, 141 frames at 768×448, Upscale off, `h3@q4`, with every audio row held (step 1's H3 bullet, done as monkeypatches): 273–282 s per clip, 5.2 GiB VRAM and 17.9 GB peak RAM, the same as a clip without a track. The face drifts from the photo's identity, as it does without a track |

The sync result for upscale off against on on `ltx-4` rests on the synthetic voice. The merge gates for the Windows/Linux commits were LTX text to video with a track (`LTX2Pipeline`, the one class that passes no `audio_timestep`), H3 text to video (the no-keyframe row set), H3 Extend with a track, and H3 with a user LoRA (`_apply_user_loras` recomputes its table delta per key, so it must work for assembled keys). They ran with the build.

#### Results of the build's gate runs on Windows/Linux (2026-11-01, same machine)

| Gate | Result |
|---|---|
| LTX text to video | Held the track |
| LTX image to video with upscale | Held the track |
| LTX Extend with the filled-in track | Held the track |
| H3 text to video, image to video, Extend and a user LoRA | Held the track, at the same time and memory as the same clip without one |
| CLI smoke test | Passed |

#### Results on macOS (2026-11-01, Apple silicon, 32 GB)

`make check` passed on the Mac, including the MLX contract pins. The runs used a real male voice (a public-domain LibriVox reading) under a photo of a man made with `zit`, at 768×448 and 121 frames (73 for `h3`), each paired with a run of the same settings without the track. The sound was compared with the track sample by sample. Sync was measured as the correlation of the mouth's opening (MediaPipe face landmarks) with the track's loudness, best within ±5 frames, against 300 circular shifts of the track; every clip made without the track came out unsynced against it (p 0.43–0.78).

| Path | Sound | Sync | Time and peak memory against no track |
|---|---|---|---|
| `ltx25-4` one stage | The track from Start at | r 0.49, p < 0.001 | +2.4 %, the same |
| `ltx25-4` two stages (`A2VidDistilledPipeline`) | The track | r 0.34, p < 0.001 | −2 % over two alternating pairs, the same |
| `ltx-4` one stage | The track | r 0.45, p < 0.001 | +0.8 %, +0.8 % |
| `h3`, Upscale off | The track | r 0.58, p < 0.001 | Not slower (692 s against 803 s), 15.2 GB both |
| A 3 s track under a 5 s clip, two stages | The track, then 2.3 s of silence; 121 frames kept, no "Audio is too short" | r 0.66, p < 0.001 | – |
| `ltx25-4` Extend with the same file, Start at filled in | The new 2 s are the track's 5.0–7.0 s | Whole clip r 0.39, p < 0.001 | −1 %, 25.6 GB both |
| Extend of a clip made with `--no-audio` | The source's 5 s silent, then the track from 0 s | – | 25.6 GB |
| `h3` Extend with the same file | The new part is the track's 3.0–5.2 s | Whole clip r 0.42, p < 0.001; the 2 s new part alone r 0.21, just above chance | 15.3 GB |
| Web UI, two stages with Review | The review clip and the result both carry the track | r 0.42, p < 0.001 | – |

Every family passes, so `platform_capabilities.darwin.audio_track` stays unset. `ltx-8` wasn't run: it is `ltx-4`'s weights at 8 bits. On `h3` the track is encoded in its own phase at a 1.7 GiB MLX peak, below the keyframe (14.1 GiB) and prompt (13.7 GiB) phases.

### 1. Backends

- **Request and capability.** `VideoGenerationRequest` gets `audio_track: AudioTrack | None` (`core/video_types.py`). An `audio_track` capability in `VideoCapabilities` (`core/video_family.py:42-52`) is set in `config.yaml` on each family that passes the spike on both platforms; others are refused in `utils/video_support.py`.
- **Length, at submit.** The track never changes the clip's frame count: the clip keeps the requested Duration on the model's frame rule. The CLI and the Web UI check once, before the model loads, that the file can be read (`utils/video_probe.ffprobe_json`, `audio_decodes`); an unreadable file fails with "Can't read the audio track". A track that is shorter than the clip, or a Start at at or past its end, is allowed: `read_track` pads the stretch with silence, so a 3.6 s voice note under a 5 s clip leaves 1.4 s of silence, and a Start at past the end gives a fully silent track. (The first design shortened the clip to the track after Start at, rounded down on the frame rule, and refused "The track has ended" and "Start at is past the end of the track"; the user dropped that on 2026-11-01 after review.)
- **Backend contract.** A frozen `AudioTrack(path: str, start: float)` value on `VideoGenerationRequest`; an `audio_track=` keyword on `text_to_video`, `image_to_video` and `extend_video` (`core/video_backend.py:111-124`), passed by the stages (`video_stages.py:128-139, 207-217`); and `audio_track: bool` on `load_model`, added to the load keys of all four backends in the first commit, since the H3 backends reject unknown ones (`h3_video.py:66, 669-671`; `h3/backend.py:78, 717-719`) and the web runner passes the same keys to all (`web_runner.py:897-903`).
- **One reader.** `utils/audio_track.py` holds `read_track(path, rate, start, seconds)` (ffmpeg, stereo float, padded with silence past the end) and the window waveform, used by every backend and stage instead of four readers; it is listed in `docs/developer/project-structure.md`.
- **macOS LTX.** `load_model` takes the track flag and loads `A2VidDistilledPipeline` for two stages, called as `generate_and_save(..., audio_path=track.path, audio_start_time=track.start, images=..., stage1_steps=...)` (`a2vid_distilled.py:96-112`; its stage overrides survive the app's review shim, which calls the original `_stage1`). One stage and Extend share one helper modelled on `_encode_carry_audio` (`backends/mac/video.py:354-378`): `pipe.audio_conditioner(encode_audio…, free_after=low_memory)` on the window waveform, `fit_audio_latent(latent, compute_audio_token_count(frames, fps))`, `audio_patchifier.patchify`, then `create_noised_state(base_shape=tokens.shape, conditionings=[], positions=audio_positions, spatial_dims=(audio_tokens, 1, 1), seed=seed+1, initial_latent=tokens, frozen=True)` (`spatial_dims` has no default and isn't used for audio; the carry branch passes the same, `video.py:541`), in place of the `pinned_audio` carry branch (537-547). Both the plain and the ancestral loop handle frozen states (`samplers.py:265, 436`).
- **Windows/Linux.** A whole-track hook beside `pin_audio_prefix` that always sets the clean tokens, a per-token `audio_timestep` of 0 and `audio_sigma` of 0, adding them when the pipeline didn't pass them. Both passes run under it, and the two-stage refusal is lifted for a track. The track is decoded with ffmpeg at the audio VAE's rate and encoded before the transformer is attached, so in low-memory mode the audio VAE isn't on the GPU beside it.
- **H3, if it passes on both platforms.** H3 with a track runs with Upscale off: its refine tables cover plain and keyframe clips only (`h3/backend.py:491-499`).
  - *Windows/Linux (done in the spike, about 60 lines over the stock backend):* the held length becomes a parameter of `encode_tail_audio` (`h3/conditioning.py:160`), `Conditioning` (`h3/sampler.py:66`), `build_layout` and `_ClipSpec` instead of the 37-latent constant; `tail_audio_rows` already holds every row. `_step_generated` skips an empty mask (a 2-line guard). Two row sets for key enumeration, a track with and without a keyframe, give the keys `(t, 1.0)` and `(t, 0.999, 1.0)`, which no stored copy has (1 of the 8 needed is stored). `read_tables` assembles each missing key per block from the rows of stored keys that contain the same timestep, since each row depends only on its own timestep (`adaln.py:148-167`): 74 of 76 compared rows were bit-identical and 2 differed by bf16 rounding. No stored copy changes and the manifest check is untouched; adding to `ROW_SETS` (`h3/schedule.py:22`) would have forced every H3 user to rebuild.
  - *macOS:* the tables are built per run (`h3_staged.py:279-283`); the cost is the conditioning path:
    1. The track window is read at 32 kHz, loaded with `load_audio_vae(fl2va/"audio_vae")` and encoded with `AudioVAE.encode(wave[:, None])` (`packages/minimax_h3_mlx/audio_vae.py:481-497`, stereo channels as the batch), whose mean is normalised `(mean - latents_mean) / latents_std`, transposed and packed channel-major `(2*T, 32)`, then padded or trimmed to the canvas's audio latents, as CUDA does (`h3/conditioning.py:160-193`). The normalise and pack step is a pure numpy helper, tested on Linux.
    2. Those rows replace the audio noise draw (`h3_staged.py:273`), and the layout becomes `dataclasses.replace(layout, num_condition_audio_rows=2*audio_latents)` (`PackedSequence` is a plain dataclass, `packing.py:78-88`; condition audio rows run at 1.0, `pipeline.py:350-361`).
    3. The empty generated-audio step is skipped, and `_decode_audio` is skipped: the clip's sound is silence of its length, which the mux stage replaces with the track (`h3_staged.py:148, 306-314, 327-334`).
    4. Text to video with a track takes the staged path (`_generate` takes it when `images or track`, `h3_video.py:541-578`). The staged path gets a text-only branch: with no images it skips the keyframe and vision phases, encodes the prompt with `MiniMaxH3TextEncoder(...).encode(prompt)` (`text_encoder.py:424`), and uses `condition_rows` of shape `(0, C)` and `anchors=()` (today `generate_conditioned` raises without images, `h3_staged.py:104-105`, and `_prompt_rows` reads `vision_inputs[1]`).
  - *Row sets, both platforms:* a clip whose audio rows are all held uses the `track_keyframe` keys when it has a keyframe or an Extend tail, and the `track` keys when it has neither; never `extend`, whose layout assumes generated audio rows (`h3/backend.py:129-131`, `adaln.py:61-65`).
  - *Upscale:* H3 with a track can't upscale (its refine tables cover plain and keyframe, `h3/backend.py:491-499`). A capability `audio_track_upscale: false` on `h3` makes `unsupported_video_request` refuse it ("MiniMax H3 can't upscale a clip made from your track. Turn Upscale off."), It stays out of `_CONTROLS_WITHOUT_CAPABILITY`, which would hide Upscale on H3 for every job; instead the workspace payload gets a `supports_audio_track_upscale` flag beside `supports_extend` (`web/workspace_api.py:777-778`), and the Web UI hides the Upscale controls while a track is in effect on a model without it. Every capability is required in config (`utils/config.py:412-421`), so `audio_track` and `audio_track_upscale` are set on every video family and documented in `docs/concepts/configuration.md`.
- **Extend with a track.** One mechanism on every backend: one pure helper builds the window waveform, the source's sound under its last 25 frames (silence for a silent source) followed by the track from Start at, at the rate each backend needs. The backends encode that waveform and hold the whole window, as for text and image to video; macOS H3, which carries a single frame, gets that frame's sound followed by the track through the path above. The extension keeps the new frames the user asked for; the sound after the track ends is silence. Extend's filled-in Start at is where the source's sound ended in the track, so the two meet without a gap.
- **Text and image to video: the sound, once.** A video stage in place of `strip_audio_stage` (`workflows/__init__.py:100-102`) remuxes the track, cut to the clip's length, onto the generated MP4 with `mux_audio` (`utils/video_splice.py:273-332`), which copies the video stream and encodes the track from its own sample rate. The backends only hold the track; on macOS two-stage `A2VidDistilledPipeline` muxes its own copy of the source (`a2vid_distilled.py:178-186`), which the stage then replaces.
- **Extend: the sound.** Extend's stages are `extend_video_stage, join_clips_stage` (`workflows/__init__.py:98`), with no strip stage. With a track, backends never write `segment.wav`; a stage between the two writes it from the same window waveform as `cut_window_audio(window_48k, skip, keep, lead)` (`video_extend.py:224`), so the held sound and the written sound match. The join includes sound when a track is set, or when the source has sound and Audio is on (today `video_splice.py:369` and `video_extend.py:90` drop sound for a silent source). A silent source with a track gets a join WAV of silence for the source's length followed by the new part, and an audio graph that maps only the WAV (`build_audio_graph` reads `[source:a:0]` today, `video_splice.py:138-152`).
- **One sound rule.** A track is *effective* when it is set and the model has `audio_track`. "Generates sound" means Audio on and no effective track; an effective track always means sound. It feeds the strip stage, the join and the enhancer.

Done when: contract pins exist for every vendored name the macOS code uses (checked by `make check` on the Mac; `tests/contracts/test_ltx_mlx_contract.py`, `test_h3_mlx_contract.py`); mocked Linux tests show `A2VidDistilledPipeline` chosen with the track flag and called with `audio_path`/`audio_start_time`, `create_noised_state` called with `frozen=True` and shape `(1, compute_audio_token_count(...), 128)`, no `segment.wav` written with a track, and H3 routed to the staged path with a track; a pure test covers the H3 row packing; an assembled H3 key equals the stored rows on a tiny tables file; a unit test checks the hook's kwargs for all three pipeline classes (text to video, image to video, condition); a track shorter than the clip keeps the full frame count and is padded with silence, and a Start at past the end gives a full-length silent waveform without an error; the mux stage's output has an audio stream as long as the clip, taken from Start at; extending a silent clip with a track gives an audio stream whose new part is the track span; and the track is encoded before the transformer is attached.

### 2. CLI

- `--audio-file PATH` and `--audio-start SECONDS` on `ziv-video`, for text and image to video and `--extend` (`--audio` is already the on/off flag). `--extend` doesn't fill the track in on the CLI; scripts pass `--audio-start` themselves. Refused for a model without the capability. `--audio-file` overrides `--no-audio`: the track is the sound.

Done when: the refusal and the override are tested.

### 3. Web UI

Setting a track is the only new action; nothing else asks the user to confirm it.

- **The section.** Its own **Audio track** section in Settings right after **Images**, in the Video and Extend Video tabs (Extend Video has no Images section, so it takes the same place in the order): one row with the existing file-slot component (Choose, Browse, path) and a waveform icon in place of the thumbnail, and a **Start at** row, a seconds field with the computed timestamp beside it, as Duration shows its frame count. The collapsed summary is "song.mp3 from 1:30" (just the name at 0).
- **Extend.** The section is shown for any source clip. The extend-source inspection (`ExtendSourceInfo` and `extend_source_response`, `web/extend_api.py:136`) returns the source's recorded track and end, and `extendPrefillFor` fills the section from them when a source is chosen, as it fills scene and prompt; a source without a track clears the section, so a track filled in from one clip doesn't stay when the user switches to another. The user can clear it or choose another file.
- **Audio switch and enhancer.** While a track is effective (step 1), the Audio switch and the Sound options aren't shown, `hasAudio` (`frontend/src/lib/state/promptEnhance.ts:158`) is false, the phone Audio chip (`settingsChips.ts:82`) isn't shown, and the form doesn't send `audio`, so an Audio switch left off can't silence a track or put `--no-audio` in Copy CLI.
- **Models without it.** The `audio_track` capability joins `_CONTROLS_WITHOUT_CAPABILITY` (`web/workspace_contract.py:296`), so the section is hidden. As with an image a model can't use, the track is then not shown, not sent and not counted, but stays in the draft and comes back on a model that can use it; the Audio switch and Sound options come back meanwhile.
- **Draft fields.** One set per tab: `audioTrackPath`, `audioTrackUpload` and `audioTrackStart` for Video, and `extendAudioTrack…` for Extend Video. They join `TAB_IMAGE_KEYS.video` and `.extend` (`draft.svelte.ts:195`) so Reset clears the open tab's track, and `heldUploads.svelte.ts` and `uploadStore` get a track field per tab, so a voice memo picked on a phone survives a visit to the Gallery as an image does. The control ids join `visible_controls` for txt2vid, img2vid and extend (`workspace_contract.py:134-292`). Presets leave the track alone, as they leave images.
- **Upload and Browse.** Uploads accept audio files (wav, mp3, m4a, flac, ogg) in `_save_upload`, which takes only images and checks them with PIL today (`web/server.py:1753-1775`); `BROWSE_PURPOSES` gets an audio entry (`web/path_browser.py:85-91`).
- **Errors.** No check when the file is chosen; a track that can't be read fails the job with the message from step 1.
- **Layouts.** Desktop: the section in the inspector. Tablet (iPad, tall): the same with 44 px touch targets. Phone: a **Track** chip right after the **Images** chip (after **Source** in Extend Video, which has no Images chip), with the value "1:30" when Start at isn't 0; its popover has Choose from this device, Browse the server, the path field and Start at, laid out like a row of the Images popover.

**Left out on purpose:** a video's sound as a source and a Gallery **Use as → Audio track** (they need a has-sound check on every video), a locked "From your track" Audio row, a hint line under Duration, a waveform and a play button.

Done when: an `.m4a` upload is accepted and a `.txt` refused; a model without the capability has the track in `hidden_controls`, shows the Audio switch and doesn't send the kept track; Extend of a track clip is prefilled with its track and end time, and the Extend form sends the track fields.

### 4. Enhancer

- No new rules: with no sound generated, the existing path already drops "sound and quoted dialogue" from the Action part (`prompt_enhance_video.py:173`) and the Sound rules (`prompt_enhance.py:952`). The risk is the other way: with sound off, quoted text is treated as on-screen lettering (`prompt_enhance.py:874-876`) and the visuals-only rule says "no sounds" (871). Check on the enhancer bench that a prompt with a quoted spoken line gives no on-screen text, and that "speaks" or "sings" survives the rewrite; adjust the rule for track runs only if it doesn't.

### 5. Provenance, Reuse, Retry, Extend

- The output's settings (`utils/provenance.py`) and gallery facts record the track's path, its Start at and where the clip's sound ends in the track (Start at plus the clip's, or the extension's, new frames over fps). There is no existence check; a missing file fails the job with the message from step 1.
- **Gallery Reuse** goes through `reuse_params` (`web/gallery.py:~285-305`), `_URL_PARAM_CONTROL_IDS` and `loadFromUrl` (`draft.svelte.ts:~98, 576-700`); **Load settings** from the queue and Job log through `PREFILL_KEYS` (`frontend/src/lib/state/jobSettings.ts:6`). Both put the track in the tab the prefill opens, as `image_path` lands (Video for text and image to video, Extend Video for Extend), and a full prefill of a job without a track clears that tab's track, as it empties image slots the params don't name (`draft.svelte.ts:680-687`).
- Retry restores an uploaded track through `uploads={...}` (`web/server.py:1425`); Copy CLI adds `--audio-file` and `--audio-start` (`frontend/src/lib/state/cliCommand.ts:181`). `EXTEND_IGNORED_FIELDS` is left alone: Extend deletes those fields from its form (`extend.ts:53`).

Done when: Reuse and Load settings both restore the track and Start at into the right tab, a job without a track clears it, and an extension's prefilled Start at equals its source's recorded end.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "Give a video your own audio track; the clip follows it, from any point in the track, and Extend carries on with it."
- Concepts: an **Audio track** section in `docs/concepts/video.md`: what it does, which models, Start at, length, Extend, and what the spike found about sync. The Web UI and CLI pages show how and link there; `--audio-file` and `--audio-start` in `docs/cli/reference.md`.

## Alternatives considered

- **A dedicated talking-head model** (InfiniteTalk, HunyuanVideo-Avatar). A new model family, download and memory budget per platform, for one use; only worth it if LTX's own mode fails the spike.
- **Re-dub an existing clip** (Dub-It IC-LoRA). LTX-2.3 only, one speaker, and it starts from a video rather than a photo. Can follow if users want to re-voice gallery clips.
- **Vocal separation in the app.** Another model to ship for one case.
- **Recording in the app.** Browsers only allow the mic over HTTPS or on localhost, so a phone reaching the app over the LAN couldn't use it; phones can already pick a voice memo as a file.
- **A "Next part" button** that queues the next stretch of the track as a separate clip. Extend with the filled-in track does the same and joins the parts.
- **Muxing in each backend.** Three code paths (Windows/Linux, macOS one and two stages) for one job; one workflow stage does it once.
- **A new H3 row set in the stored copies.** Would force every H3 user on Windows/Linux to rebuild their stored copy, track or not; assembling the track's tables from the stored Extend tables needs no rebuild.

## Open questions

- Two speakers: does the model pick who speaks from the prompt?

## Build order

Each commit leaves the app working: `audio_track` stays false in config until both platforms are in.

1. `feat(video): add the audio track request, capability and length rule`: `AudioTrack`, `VideoCapabilities.audio_track` (false everywhere) and `audio_track_upscale` on every family, `audio_track` in the load keys of all four backends, `utils/audio_track.py` with the frames floor and the Extend limit, the refusals.
2. `feat(video): put the track on the clip and into Extend's join`: the mux stage, the Extend sound stage, the silent-source join, the one sound rule.
3. `feat(video): hold the track on Windows/Linux LTX`: the hook, encoding before the transformer is attached, the Extend window, the load flag; the LTX text to video gate run.
4. `feat(video): hold the track on macOS LTX`: mocked tests and contract pins.
5. `feat(video): hold the track on MiniMax H3 (Windows/Linux)`: the held length, row sets, `read_tables` assembly, the empty-step guard; the H3 text to video, Extend and LoRA gate runs.
6. `feat(video): hold the track on MiniMax H3 (macOS)`: packing helper, staged text-only branch, routing; mocked tests and contract pins.
7. `feat(video): offer an audio track on LTX and MiniMax H3`: capabilities on, `--audio-file`/`--audio-start` and their reference entries.
8. `feat(video): record the track in the output's settings`: provenance, gallery facts, `extend_source_response`, `reuse_params`.
9. `feat(web): add the Audio track section`: upload and Browse, form, hidden controls, draft and held uploads, chips, Extend prefill, Reuse and Load settings, Copy CLI, `make frontend-build`.
10. `docs(video): document the audio track`: Concepts, Web UI and CLI pages, CHANGELOG, the enhancer bench check, the proposal archived.
