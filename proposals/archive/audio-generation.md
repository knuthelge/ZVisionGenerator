# Generate music, speech and sound

**Status:** Done (2026-11-04, unreleased): built on `feat/audio`; the Mac gate passed on a 32 GB M5 after two fixes found by it (see [Mac results](#mac-results-2026-11-04)). Higgs parity was checked against a CPU float32 reference instead of the CUDA trajectory, and the 16 GB Mac and Safari checks were skipped by the owner's decision

**Prerequisite:** [Licences at download, and a dialog for every download](model-licences.md), built first on the same branch and done. It makes Generate ask before downloading, names restricted licences in that dialog, on the Models page and on the CLI; this proposal adds the three audio models' records and has no licence UI of its own.

## Problem

The app makes pictures and clips, but every sound in them is either the video model's own or a file from somewhere else. A song for a music video, a narrator for a clip, or rain under a scene means another tool, usually a cloud one. The video **Audio track** ([audio-to-video](audio-to-video.md)) can already follow any audio file, so audio made in the app would feed straight into it.

Three local models won the blind tests on our own prompts and fit the 10 GB card: YuE2-3B for songs and instrumentals, Higgs TTS 3 for speech, and Stable Audio 3 Medium for ambience and sound effects. None is MIT or Apache; each one's licence is named when it is downloaded.

## What the user sees

### The Audio tab

A third entry in the Workspace's workflow choice: **Image · Video · Audio** (the top bar on wide and tall screens; the workflow menu in the phone's top strip, `TopStrip.svelte:29-49`). It is shown only when the server lists an audio workflow, so it is hidden where no audio model can run (`workspaceTab.ts:12-14`).

The Workspace toolbar reads, left to right, **Kind** → **Model** → the model's licence mark → its memory badge. **Kind** is a three-way segmented control, **Music · Speech · Sound** (Sound means ambience and sound effects), and decides the boxes and settings below; the **Model** dropdown lists that kind's models: one each in v1, more later without UI changes. A restricted model shows the prerequisite's ⓘ licence mark beside the dropdown, never inside it. Presets, Quantize and LoRAs are not shown on the Audio tab.

| | Music | Speech | Sound |
|---|---|---|---|
| Model | YuE2 3B | Higgs TTS 3 | Stable Audio 3 Medium |
| Compose | **Style** (genre, mood, instruments, singer; required) over **Lyrics** (optional), two stacked boxes; Lyrics with words make a song, Lyrics empty or only section tags make an instrumental | **Text** to speak, with a character count against the cap | **Prompt**: what you hear |
| Voice | — | **Voice** inspector section, first in Settings: *Any* or *From a recording* | — |
| Length | Follows the sections: a song's lyrics (4 sections ≈ 2 min, 12 ≈ 3¾ min), an instrumental's section list (the default 8 ≈ 1½ min, 13 ≈ 2½ min); the model stops at 6:00, where a take is faded out and marked cut | Follows the text, up to 10,000 characters (about 10 minutes); long text is read in parts of up to 500 characters and joined into one take | **Length** in seconds, default 10, up to 380 s |
| Also | Seed, Batch | Seed, Batch | Seed, Batch |

- **Music makes songs and instrumentals, from the same two boxes.** Lyrics with words make a song (`cot="full"`). An empty Lyrics box makes an instrumental from a default section list (`[Intro] [Instrumental] [Solo] [Instrumental] [Solo] [Bridge] [Instrumental] [Outro]`, about 1½ min); Lyrics that hold only section tags make an instrumental from that list (`cot="melody"`), so a longer list gives a longer piece and a shorter list a shorter one. The status line says which it will be: "Song · length follows the lyrics" or "Instrumental · 8 sections · add or remove section tags to change its length". There is no length control: YuE2 has none, and the section list is the lever the long-song spike measured. In that spike `melody` instrumentals ended on their own in 4 of 4 takes (73–150 s), against 1 of 5 for `full`. **Decision (listening round 3):** the default list uses `[Instrumental]` and `[Solo]` sections instead of verse and chorus, with the style passed through unchanged (no "instrumental, no vocals" text added). That recipe gave 6 of 7 clean, good takes, against about 12% with the verse and chorus list. Removing the voice afterwards (vocal separation) was rejected: it sounded dull where the voice had carried the track. An automatic vocal check with a retry was not adopted; a take can still occasionally carry a voice, and the user generates again.
- **Speech voices.** *Any* lets the model pick a voice, which changes from take to take (Higgs has no voice design); the section says so and points to Use as → Voice. *From a recording* clones a voice from a 3–30 s recording: the Audio track's file slot (Browse, Upload, Clear, drop), a required **What it says** field (the transcript Higgs needs), and a required tick: **"This is my voice, or I have the speaker's permission to clone it."** The tick is never saved: a new file, Reuse or a reload clears it. A take made with *Any* used as the voice shows no tick. Higgs's licence forbids cloning without consent (section IV(b)(ii)). Expressive tags such as `<|sfx:laughter|>` work in the text and are listed in the docs; there is no tag picker.
- **Long text is read in parts.** Higgs reads about 600 characters faithfully in one pass (the spike: 626 characters at WER ≤ 1%, 770 at 19–37%, 1406 with only 13% spoken), so the app splits longer text at sentence ends into parts of up to 500 characters and generates them in order with one voice: the recording for every part, or with *Any*, the first part's own take as the voice for the rest (the model's own voice, so no tick). The parts are joined into one take with a 300 ms pause, their levels evened out and long silences at their edges trimmed. The job card shows "Part 2 of 4"; Stop works between and within parts. The help under Text says "Long text is read in parts of about 500 characters and joined into one take."
- **Generate explains itself.** While something is missing, Generate is disabled and one reason shows above it, in a fixed order: the style, the text or prompt, the recording, its length, What it says, the tick, then text over the ceiling ("Shorten the text: it's over 10,000 characters, about 10 minutes of speech.").
- **No enhancer, no negative prompt, no steps.** The compose tools are Recent prompts only (no prompt files, Enhance, JSON or Save prompt); steps and sampling are fixed per model in config.

### While it runs, and the result

- If the model isn't downloaded yet, Generate first asks, with its size, the free space and its licence line (prerequisite); Cancel queues nothing.
- The job card counts **takes** ("Take 1 of 2") and shows the take's progress in the preview's words with one bar that only moves forward (YuE2: Planning, Writing the song with the length so far, Rendering, Finishing; Higgs: one row, "Reading part 2 of 4" and "Finishing part 2 of 4"; Stable Audio: "Step 5 of 8", then Finishing up) and Pause, Next, Repeat and Stop. The first Sound take after the model loads takes a few seconds longer (2–19 s measured); later takes take about 0.65 s per 30 s. The preview slot shows the Aurora, a live sky of ribbons over a faint starfield that follows the take's stage (waiting, plan, write, synth, decode) and settles into the take's waveform when the take is ready; the last take's waveform shows between takes and while paused. With reduced motion it is one still per stage.
- A job that runs out of GPU memory fails with a message that names the model and says to close other apps that use the GPU; for Music it also says to shorten the lyrics.
- The stage shows the finished take as a **player panel**: the take's waveform (played part in the primary colour) with a playhead, where a press seeks and a drag scrubs, over the browser's own audio controls, which are the one set of controls for keyboard and screen readers. Reuse settings and the ⋯ menu sit in the panel's corner, as on image tiles; the caption gives the words, model, length and seed. A take that reached YuE2's 6:00 limit ends in a short fade-out and carries a **Cut at 6:00** warning badge whose tooltip says to remove sections to hear its ending. Nothing plays by itself.
- Every take is saved as **FLAC** (lossless, plays in Chrome, Firefox and Safari, accepted by the video Audio track) with its settings embedded, so the Gallery, Reuse and Copy CLI work as they do for images and videos.

### Gallery and viewer

- The Gallery's dropdown gains **Audio only** (All media · Images only · Videos only · Audio only). An audio tile shows, top to bottom, the kind icon, the take's words (Music: the style, then the lyric lines without their `[section]` tags; Speech: the text; Sound: the prompt; clamped to 4 lines, 3 on a phone) and a thin waveform line, with the length tag in the corner; it doesn't play in the grid. A file without embedded settings shows a larger icon and waveform and "audio · no settings". Today audio files are skipped (`web/gallery.py:808-814`) and `/media` refuses them (`web/server.py:228-234`).
- The viewer shows the player, larger, with the take's words under it on the stage, and the details panel listing the recorded settings (kind, model, voice, length and whether it was cut, seed; file format). Playback stops when the viewer pages to another asset or closes.
- The tile and viewer menus keep **Reuse settings**, **Download** and **Delete**, and gain under **Use as**:
  - **Audio track** (any audio asset): opens the Video tab with this file as the clip's Audio track from 0 s. Disabled with the reason when the selected video model can't follow a track, as other Use as targets are (`assetMenu.ts:104-118`).
  - **Voice** (speech takes of 30 s or less, the voice limit; longer ones show the reason): opens Speech with this take as the recording and its recorded text as **What it says**. This is how to keep one voice across separate takes: generate with *Any* until a voice fits, then use it. A take made with *Any* needs no tick, since its voice is the model's own; a take that was itself cloned from a recording still needs it.

### Licences

The three models are restricted, each in its own way. Their licence records feed the prerequisite's download dialog, the ⓘ mark beside the Model dropdown and on their Models-page rows, and the CLI's licence line. There is no acceptance step.

| Model | Licence | Summary |
|---|---|---|
| YuE2 3B | CC BY-NC 4.0 (weights) | Non-commercial, credit required |
| Higgs TTS 3 | Boson Higgs TTS 3 Research and Non-Commercial License | Non-commercial; creators may publish with credit |
| Stable Audio 3 Medium | Stability AI Community License; Gemma Terms of Use for its T5Gemma text encoder | Free under $1M revenue |

**Higgs and this app.** Higgs's licence defines a Commercial Purpose as including "(i) production use, hosted use (including via an API, SaaS offering, plug-in, or end-user application), or any use made available to end users outside Your organization", and its creator grant excludes "embedding or integrating the Higgs Materials in a product, application, or service made available to third parties" (sections III and II-A(c)(iii)). A strict reading makes any app that offers Higgs to others a commercial use. The app ships none of Higgs's materials: each user downloads the weights themselves and is the licensee, as with Ideogram 4 Instant and MiniMax H3. The user has accepted this reading and decided to ship Higgs on that basis. The licence also asks that copies carry the Agreement and a NOTICE line (section IV(a)(i)-(ii)); the download dialog and the Models page link the Agreement, and `THIRD_PARTY_LICENSES.md` carries the NOTICE line and the "Derived from Higgs TTS 3, licensed from Boson AI USA, Inc." line, since upstream ships no NOTICE file.

### Layouts

- **Wide (≥1024 px).** The toolbar holds Kind, Model, the ⓘ and the memory badge on one row. The sidebar holds Compose (Music's Style and Lyrics stacked, as video's Scene and Action are) over Settings (Voice first for Speech, then Sampling with Batch, Length for Sound, and Seed). The stage holds the player panel; the strip under it holds recent outputs, audio tiles among them. The top bar fits Image · Video · Audio from 740 px up, because at 1000 px and below it hides the wordmark (the mascot stays) and tightens its tab padding.
- **Tall (640–1023 px).** The same toolbar on one row (it fits at 800 px); Compose and Settings side by side under the stage, with 44 px touch targets on touch; the boxes stay stacked.
- **Phone (compact).** Audio is the third option in the top strip's workflow menu. The stage is the player panel with 36 px corner actions. The chip strip reads Kind (icon and name; its popover lists the three kinds with a line each), Model, Voice (Speech) or Length (Sound), Seed, Batch and the pinned All. The ⓘ can't sit on the Model chip (a button inside a button), so it ends each restricted model's row in the Model popover and follows the model in the All sheet's toolbar. The Voice popover holds the choice, the file picker, What it says and the tick. The composer shows Style · Lyrics, Text or Prompt, and the reason above Generate. The viewer is the phone pager with the player and the words; the ⋯ sheet and details sheet work as for images.

## Evidence

### The three models (10 GB GPU, 2026-11-01)

| | YuE2-3B | Higgs TTS 3 4B | Stable Audio 3 Medium |
|---|---|---|---|
| Blind test (owner, loudness-matched) | 8–0 (5–0 against ACE-Step 1.5) | 15–4 (beat Breeze 2 4–1, VoxCPM2 4–3) | 8–1 (beat Small-SFX 5–1, MOSS 6–0) |
| Peak, torch / nvidia-smi | stock kit 8.3 / 8.8 GiB at 225 s, out of memory at 360 s; with the weight swap 5.7 / 6.5 GiB at 225 s and 5.7 / 6.3 GiB at 360 s | 7.9 / 8.9 GiB at about 20 s of speech, codec on the CPU | 5.2 / 5.8 GiB, fp16 cast on the CPU before moving |
| Speed | RTF about 0.6 (225 s in 130 s with the swap); the swap adds about 6 s per process and 0.3–1 s per phase | RTF ~1.0, CPU codec decode included | 0.7 s per 30 s, 1.9 s per 120 s |
| Output | 48 kHz stereo | 24 kHz mono | 44.1 kHz stereo (real stereo) |
| Length | none; follows the sections (4 ≈ 113 s, 12 ≈ 225 s; instrumentals 4 ≈ 73 s, 13 ≈ 143–150 s); the model's limit is 9000 tokens, 360 s (`protocol.py:29`) | 25 codec frames per second (24 kHz, hop 960 = 8·5·4·2·3); the spike port's default `max_new_tokens=2048` is about 82 s | `duration` argument, model limit 380 s |
| Known faults | the first spike's ~68 s instrumental cut came from its own `max_tokens=1700`, not the model; `full` instrumentals rarely end on their own (1 of 5) and once gained vocals | default voice changes per seed; no voice design; reads only about 600 characters faithfully per pass | the library clamps its output, so loud prompts clip (+1.8 dBFS rain, +6.3 dBFS glass before the clamp); one ambience very quiet |
| Python 3.14 | ran with numpy unpinned, on torch 2.10 / transformers 4.57.6 | resolves with transformers ≥5.5; not run on 3.14 | ran on 3.14 with the torch pin dropped (CPU) |
| Download | 7.3 GB + 0.5 GB VAE | 9.3 GB + 0.8 GB codec | 9.2 GB (fp32 file) + 1.2 GB text encoder |

The app's YuE2 settings, as the long-song spike ran them: `cot="full"` for songs and `"melody"` for instrumentals, `memory_budget_gib=24` with the process memory fraction set back to 1.0, an fp32 VAE decoded in 512-frame cores, semantic `max_tokens` 25 × seconds, ABC `max_tokens` 4096, CFG 1.0 and 32 ODE steps. Higgs's KV cache grows by about 3.7 MB per second of speech (36 layers, 8 KV heads, head size 128, bf16, 25 frames per second), about 0.4 GB for two minutes, so its cap is set by the measured peak rather than a guess. Memory numbers are on an otherwise empty card; the web runner frees accelerator memory after every job (`web/web_runner.py:710-727`).

### Where the code comes from

- **YuE2.** The inference kit is the `yue2` package in [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) (code Apache-2.0; the weights' CC BY-NC is separate), also shipped as a wheel inside the model repo that pins `torch==2.10.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2` and `numpy==2.2.6`, all below ours. It runs on MPS as well as CUDA: `device="auto"` picks `mps` (`yue2/pipeline.py:134-135`), and the sampler and NAR stage have MPS branches (`sampling.py:12-13, 115`; `nar.py:75`); CUDA graphs are CUDA-only (`sampling.py:86`). `fast.py` (vLLM, `fcntl`) is not needed. Why the stock kit runs out of memory on 10 GB for long songs: it caps the whole process at the card's size less 2 GiB, at most 7.64 GiB here (`pipeline.py:161-165`); it keeps the whole 6.76 GiB model on the GPU for every stage (`pipeline.py:210-223`); NAR prefill runs before `offload_ar` takes effect (`nar.py:248` against `:251`); and the KV cache is allocated up front for the prefix plus `max_tokens`. A wrapper of about 50 lines that keeps only the running stage's weights on the GPU fixes all of it without touching the kit: songs to 360 s fit at about 6.5 GiB, with output bit-identical to the stock kit.
- **Higgs TTS 3.** The official repo has weights and no modeling code. The app runs our own model class on the official snapshot (`bosonai/higgs-tts-3-4b@239f63fb7b02b1aa085f98d9efae5e35cc5523e8`): transformers' Apache-2.0 `Qwen3Model` as the backbone plus a fused codebook embedding and head and the delay pattern, sampler, frame rule and prompt, adapted from sglang-omni (Apache-2.0, pinned commit; the files are named in `THIRD_PARTY_LICENSES.md`). No remote code runs and no third-party repository is downloaded. The weights load from the official `model.safetensors` through a strict name map, one tensor at a time, which brought the load from about 8 GB of system memory to about 0.8 GB. The spike's third-party `trust_remote_code` port was the oracle only: our class matches it bit for bit on all 15 parity cases, on all 7 layers of the proof (weights and buffers, in-process repeatability, logits, greedy codes, sampled codes, waves, and WER and voice similarity). The codec, `higgs_audio_v2_tokenizer`, is native in our transformers 5.21 and loaded without remote code. Every file of the app's Higgs code is ours or adapted from Apache-2.0 sources, so it is in the repository.
- **Stable Audio 3.** Stability's own package, [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3) (MIT), pins `torch==2.7.1`; without the pin it ran on 3.14. Its inference path needs `einops`, `safetensors` and `torchaudio` (`Resample`, imported at module level); `flash_attn` is optional and Medium ran without it. torchaudio 2.11.1 is a pure-Python wheel requiring `torch>=2.11`. The official weights repo is gated; `Comfy-Org/stable-audio-3` has the same checkpoint (sha256 `48d9c65e…`) ungated, and the T5Gemma folder from `mlx-community/stable-audio-3-small-sfx` matches the original's sha256. The ungated `cocktailpeanut/stable-audio-3-medium` config differs from another mirror only in the T5Gemma conditioner's `repo_id`, which the loader replaces with the local folder anyway (as the spike did). The same repo's `optimized/mlx` code and the ungated `stabilityai/stable-audio-3-optimized` MLX weights are the official Apple path.
- Our pinned diffusers 0.42 also has `StableAudio3Pipeline`, but it needs a diffusers-layout repo, and none exists ungated (see Alternatives).

### Codebase fit

- Reusable as is: the job queue, events and job controls (`job_controls.py`), `preflight.plan_iterations` for seeds and `{a|b}` choices (`preflight.py:57`), the enhancer release before a model loads (`prompt_enhancer_session.release_resident_enhancer`, as `preflight.py:176-183` does), downloads with progress (`downloads/ensure.py:201`), the model-folder builder pattern (`built_models.py`, `moss_builder.py`), uploads and Browse of audio files (`web/path_browser.py:85, 94`), the `/api/audio-track/inspect` probe (`web/server.py:380`), `write_pcm16_wav` (`utils/video_splice.py:350`), polyphase resampling (`utils/audio_resample.py`).
- Settings in the file: the MP4 path writes `-metadata zvisiongenerator.config=<json>` with ffmpeg and reads ffprobe's format tags (`utils/provenance.py:276-322`). The same commands round-trip on FLAC unchanged (checked: a Vorbis comment with that key, JSON and non-ASCII text intact, also after a `-c copy` rewrite).
- The gallery walk skips files whose kind is unknown (`web/gallery.py:185`), so audio files were never indexed; adding the kind needs no index version bump.
- **Binary image/video checks that would treat audio as an image** (13 call sites): `/api/generate` sends every mode that isn't `video` to the image submit (`server.py:750-753`); `workflow_mode` and `default_workflow_for_mode` (`workspace_contract.py:339-346`); a reused asset's workflow on a media mismatch (`gallery.py:222-228`); the workspace tab of a workflow (`workspaceTab.ts:17-20`); Copy CLI's command (`cliCommand.ts:119-122`); Load settings (`jobSettings.ts:104-105`); the gallery filter (`gallery.svelte.ts:5`); memory badges (`web/model_status.py:75-93`); recent prompts (`web/recent_prompts.py:59-69`); the enhance route (`server.py:583-599`) and its presets lookup (`utils/prompt_enhance.py:817`); `itemNoun` (`jobView.ts:147`); and the `WorkflowMode`, `media_type`, `PresetMode` and job `mode` types (`lib/types/index.ts:13, 16, 67, 267, 990, 1265`).

## Proposed change

### Spike first: the models in the app's environment

The CUDA spike ran each model in its own throwaway environment. Before app work, a standalone script runs each one from the vendored code (or, for Higgs, the spike's pinned third-party port, which the app does not ship) **inside the project's virtual environment** (Python 3.14, torch 2.14, transformers 5.21) on the 10 GB card, with exactly the settings the presets will hold (step 1). Results are written here before step 2.

| Question | Pass |
|---|---|
| YuE2 runs on transformers 5.21 and torch 2.14 | The long-song spike's 90-label song, seed 1, through the app's backend (weight swap, preset settings) is bit-identical to the stock kit's take on the same inputs. If a fix needs edits to vendored code, stop and ask |
| YuE2 at its limit on 10 GB | The 13-section instrumental and a 12-section song, plus a run to the 360 s cap; record each phase's peak. Pass: ≤ 7.0 GiB nvidia-smi at 360 s, so `max_seconds` is 360 |
| YuE2's memory cap is undone | After `close()` sets the fraction back to 1.0 and the swapped weights are dropped, an image job in the same process peaks as it does in a fresh one |
| YuE2 instrumentals | Empty Lyrics and an 8- and a 13-section list, `cot="melody"`, 3 seeds each. Pass: at least 7 of 9 end on their own, and the owner hears no vocals in at least 7 of 9. If vocals creep in, add "instrumental, no vocals" to the style for instrumental runs and measure again |
| Higgs from the pinned port, codec ours, on the CPU | `_audio_codec` set to our codec before generate; s1 and s4 sound as before. Superseded: the app runs our own class and calls the codec itself (see Higgs under Where the code comes from) |
| Higgs's longest speech | `max_new_tokens` of 1500, 2250 and 3000 (60, 90 and 120 s; seconds = tokens / 25), on long text; record the peak. `max_new_tokens` is the largest that peaks ≤ 9.0 GiB, and `max_text_chars` is its seconds × 12 (720, 1080 or 1440; s1's 268 characters gave 18.4–21.6 s) |
| Higgs progress and Stop | A forward hook on `model.audio_head` counts frames, and raising from it while cancelled ends `generate_speech` without leaking GPU memory. Superseded: our generation loop calls back once per frame and the backend raises from that callback |
| Stable Audio 3 on torch 2.14 with torchaudio 2.11.1 | The spike's a1 prompt, 30 s, on CUDA, within 10% of the spike's time and peak |
| Stable Audio 3 at 380 s | Peak recorded; `max_seconds` is 380 if it fits, else the longest that does |
| Stable Audio 3 load RAM | Peak system memory while loading. Over about 10 GB: the backend builds the model, calls `.half()` and loads the checkpoint tensor by tensor in fp16 (`utils/safetensors_stream.py`), with no vendored edits |
| Stable Audio 3 clipping | Whether the waveform before the library's clamp is reachable from our code. If it is, the save step scales a take that peaks above 0 dBFS down to −1 dBFS; if not, leave it |
| **R0b** (separate spike): Higgs long text in parts | A 3-part narration (about 1,400 characters) split by `split_speech_text`, once with *Any* (part 1's take as the voice for parts 2–3) and once from a recording. Pass: WER ≤ 3% per part, part levels within 1.5 dB, and the owner hears no seam and one voice. It also picks the join: a pause of 150, 200 or 300 ms, or a crossfade |

**Results (spike run, 2026-11-01).** 10 GB card, Python 3.14.4, torch 2.14.1 (CUDA 13.2), transformers 5.21.0, torchaudio 2.11.1, tiktoken 0.14.0, in a copy of the project's locked environment with only those two packages added. The kits ran unmodified: YuE2 0.1.5 (the wheel in the model repo) and `Stability-AI/stable-audio-3` at `5f5e6ac2`. Higgs ran from the pinned port revision with the three sha256 values matching, the codec loaded by transformers' own `HiggsAudioV2TokenizerModel` from `815a03f1…` without remote code, set as `_audio_codec` after the model moved to the GPU, with `get_audio_codec` replaced by a function that raises (never reached). Peaks are torch allocated / nvidia-smi for the process, in GiB, on an otherwise empty card. `uv lock` with `tiktoken` and `torchaudio>=2.11.1` added resolves to exactly those two new entries (0.14.0 and 2.11.1) and changes nothing else.

| Check | Result | Peak | Time | Verdict |
|---|---|---|---|---|
| YuE2 bit-identity (90-label song, seed 1, weight swap with the preset settings) | Swap and stock kit, both in this environment: 2836 tokens, latents and wav bit-identical (equal sha256). Against the long-song spike's takes (torch 2.10, transformers 4.57.6): tokens and latents identical; the wav differs by 1 in 16-bit on 0.46% of samples, which the stock kit shows as well (VAE decode arithmetic between torch versions). No vendored edit needed | swap 5.35 / 6.28; stock kit 7.98 / 8.58 | 66 s; stock 73 s | Pass |
| YuE2 limit: 12-section song (225 s) | 5624 tokens, ended on its own | 5.55 / 6.53 | 132 s | Pass |
| YuE2 limit: 13-section instrumental (121 s, `melody`) | Ended on its own | 5.30 / 5.62 | 63 s | Pass |
| YuE2 limit: run to the 360 s cap (4 section tags, `full`, ABC and song tokens both at their caps) | 9000 tokens, 360.0 s. Phase peaks, torch / nvidia-smi: plan (ar) 4.62 / 5.05, song tokens (ar) 5.79 / 6.71, prefill 5.56 / 6.23, ode 4.90 / 5.40, decode (none) 2.75 / 3.93 | 5.79 / 6.71 (≤ 7.0); as the sixth take of one process 5.82 / 6.94 | 238 s | Pass: `max_seconds: 360` |
| YuE2 instrumentals, `melody`, 3 seeds each: empty Lyrics (the default 8-section list), a typed 8-section list, a 13-section list | 9 of 9 ended on their own (81–158 s). Automatic vocal pre-screen (Whisper small on the take; the vocal 90 song gives 75 words, the earlier "no vocals" instrumental 0): words found in 7 of 9, lyric-like text in 2 of 9. Whisper invents text on music, so this is a pre-screen only | 5.58 / 6.08 over the 9 | 41–78 s each | Ends on its own: pass. Vocals: pending owner listening |
| YuE2 instrumentals with ", instrumental, no vocals" added to the style (same 9) | 8 of 9 ended on their own (one 13-section take ran to the 360 s cap). Pre-screen: words in 7 of 9, lyric-like text in 4 of 9, so the suffix did not visibly help | as above | 37–210 s each | Pending owner listening |
| YuE2 `close()` then an image job in the same process (`zit`, 512 × 512, 4 steps) | Fresh process 1.28 / 2.29; after YuE2 and `close()` 1.29 / 2.33 (+1%, +2%). After `close()` 24 MiB stays allocated and 60 MiB reserved. The kit's cap is real: with it an 8 GiB block fails on this card, with the fraction back at 1.0 it succeeds | see result | 70 s fresh, 56 s after | Pass |
| Higgs s1 (default voice) and s4 (clone), seeds 1–3, codec on the CPU, our `_decode_codes`, reference codes we encoded | s1: samples identical to the earlier spike's takes (maximum sample difference 0.0; WER 2.1, 6.4 and 2.1%). The transformers codec class encodes and decodes bit-identically to the repo's remote-code class. s4: WER 6.4% on all three, as before; level within 1.3 dB and long-term spectrum within 2.5 dB mean of the earlier takes. The clone takes differ sample by sample because the reference is resampled with the project's polyphase resampler, not torchaudio's | 7.70 / 8.04 (s4 7.70 / 8.06) | 18.4–22.2 s for 17.6–21.6 s of speech | Pass |
| Higgs longest speech, `max_new_tokens` 1500, 2250, 3000 on text long enough to reach each cap | 1500 (60 s): 7.88 / 8.16. 2250 (90 s): 7.98 / 8.36. 3000 (120 s): 8.09 / 8.59 | largest 8.09 / 8.59 (≤ 9.0) | 60, 93, 120 s | Pass on memory: 3000 fits. The model does not read that much text, see below |
| Higgs progress and Stop | The hook on `model.audio_head` counts one call per generated frame plus the 8-codebook delay (469 calls for 461 frames, 18.44 s). A cap of 100 gave 100 calls and a cut take. Raising from the hook at call 150 ended `generate_speech`; allocation before and after the cancel was identical (0.00 MiB leak), and after `close()` allocation equalled the baseline taken before the model loaded (8.12 MiB both, the cuBLAS workspace). The next take was identical to an uncancelled one | 7.70 / 8.04 | cancel after 5.6 s | Pass |
| Higgs parity: our own class against the spike port (15 cases) | All 7 layers pass: weights and buffers identical, repeatable in-process, logits bit-equal, greedy and sampled codes equal, waves equal, and the WER and voice-similarity bar met. Load RAM about 0.8 GB, down from about 8 GB | n/a | n/a | Pass |
| Stable Audio 3 a1 (30 s), torch 2.14, torchaudio 2.11.1 | Peak identical to the spike (5.03 / 5.56). Warm takes 0.65 s against the spike's 0.70 s. The first take after load took 2.3, 7.5, 15.8 and 18.7 s in four runs (the spike: 8.9 s); the difference is the first VAE decode (6.4 s in one run), which shrank with each repeated run, as a warming kernel cache would | 5.03 / 5.56 | 0.65 s warm | Pass (warm); first take is slower than 0.7 s on a cold cache |
| Stable Audio 3 at 380 s | 380.0 s of audio, stereo 44.1 kHz | 5.21 / 6.06 | 5.8 s | Pass: `max_seconds: 380` |
| Stable Audio 3 load RAM | Building in fp16 on the CPU raised system memory by 9.0 and 9.8 GiB in two runs (9.4 GiB anonymous in the process; its resident size reached 16.8 GiB with the memory-mapped checkpoint). Load 42–44 s | n/a | 44 s | About 10 GB (9.8 GiB is 10.5 GB): R7 uses the tensor-by-tensor path |
| Stable Audio 3 clipping | Yes. The decoder output before the library's clamp can be captured without edits, and clamping it reproduces the library's output exactly. a1 peaks at +1.8 dBFS (1.23) and the glass prompt at +6.3 dBFS (2.07) before the clamp | n/a | n/a | Yes |
| **R0b** Higgs long text in parts: a 1,449-character narration (parts of 468, 483 and 496 characters from `split_speech_text`) from a recording and as *Any*, 2 seeds each, plus a 636-character single sentence (split at a comma into 495 and 140), `max_new_tokens` 1500 | WER (Whisper small, CPU): the 12 narration parts 0–1.1%, no part cut (648–799 frames); the comma-split sentence 0% and 3.1% (one word of 32). Part levels (mean power of speech frames): from a recording 1.7 and 2.2 dB apart (over 1.5), *Any* 0.5 and 0.5 dB; the two halves of one part differ by up to 1.6 dB. Voice (ECAPA speaker embeddings, cosine): *Any* part 2 against part 1 0.91 and 0.94 and part 3 against part 1 0.93 and 0.94, against 0.88–0.94 between the two halves of one part, and 0.06 between the default voice and the recording; from a recording the parts match each other at 0.87–0.91 (and the reference at 0.64–0.74) | n/a | 29–41 s per part (26–32 s of speech) | WER and voice: pass, *Any* keeps the voice. Levels: pass for *Any*, fail from a recording, so `join_speech_parts` matches each part's speech level to the mean of the parts (gain capped at ±3 dB) |
| **R0b** seams and the join (same takes, 9 joins) | Level step and spectral jump across a join lie inside the spread of the 123 pauses within parts: level step median 2.7–4.8 dB within parts and 0.2–7.7 dB at joins, spectral jump 5.4–7.4 dB median within parts and 5.2–10.4 dB at joins; no join is more than 0.8 dB beyond its run's 90th percentile, and with 10 ms fades there is no click. The gap at a join is the parts' own edge silence plus the pause: trailing 50–310 ms from a recording (one of 1,890 ms) and 60–100 ms as *Any*, leading 0–130 ms. Pauses within parts are 370–660 ms (10th–90th percentile, median 470 ms). Gap at the joins by setting, *Any* / from a recording: pause 0 or a 30 ms crossfade 30–100 / 180–380 ms (run-on, and a crossfade mixes the next part's first word into the last); 150 ms 210–250 / 360–530; 200 ms 260–300 / 410–580; 300 ms 360–400 / 510–680 (within 20 ms of the 370–660 range; the 1,890 ms tail makes one join 2,300 ms) | n/a | n/a | **`join_pause_ms: 300`**, 10 ms fades, no crossfade. Pending owner listening to the joined takes |

**What the spike settles for the settings**

- `max_seconds`: YuE2 360, Stable Audio 3 380.
- Higgs memory allows `max_new_tokens: 3000` (120 s), and so `max_text_chars` 1440, but the model does not read long text in one pass. Prose of 626 characters (114 words, 34–39 s) was read with WER at most 1% in 3 of 3 seeds; at 770 characters WER was 19–37%, at 904 it was 46–72%, at 1023 it was 48–54% (about half the words), and at 1406 characters only 13% of the text was spoken (WER 97%). On repetitive text the model spoke about 10 s and then emitted silence to the cap. The settings are therefore `max_new_tokens: 1500` (60 s, 1.6 times the longest faithful take) and `max_text_chars: 600`; reading speed was 16–18 characters per second, not the 12 the planning rule assumed.
- Stable Audio 3 `generate()` caps `duration` at its `sample_size` argument, which defaults to 5292032 samples (120 s) (`model.py:87`, `:429`). The backend must pass the config's `sample_size` (16777216 samples, 380.4 s), as the library's own CLI does (`cli.py:270-281`); without it a 380 s request returns 120 s.
- Stable Audio 3 clipping: `generate()` clamps at `model.py:350-351`; the unclamped waveform is the output of `model.pretransform.decode`, which `sample_diffusion` calls at `inference/sampling.py:511`. Wrapping that instance method to keep its output (or calling `generate(return_latents=True)` and decoding with the same method) needs no vendored edit. The save step scales a take whose pre-clamp peak is above 1.0 to −1 dBFS.
- YuE2 weight swap: `YuE2Pipeline.decode()` calls `self._model.to("cpu")` (`pipeline.py:324`), which moves the tensors the swap keeps on the GPU, so the swap puts them back at the start of every take (the long-song spike's wrapper did not, because it ran one take per process).
- Higgs (spike port only): assigning `model._audio_codec = codec` registered the codec as a submodule. Our own class does not hold the codec, so this no longer applies.
- YuE2 instrumentals: vocals or vocalising appeared in a share of takes with the verse and chorus list and the plain style, and the "no vocals" suffix cost one cap-length take without visibly removing them. Resolved in listening round 3 by the `[Instrumental]` and `[Solo]` section list (6 of 7 clean).

### Mac verification gate (after implementation, before release)

macOS is implemented without a Mac run (mocked tests and contract pins on Linux) and checked on a Mac after the build, before the release. The steps are in [Mac check](#mac-check) below.

| Check, per model | Pass | Result on a 32 GB M5 (2026-11-04) |
|---|---|---|
| Runs | The spike prompts, 3 seeds each, through `ziv audio` and the Web UI | Pass: all three models through `ziv-audio` and the Web UI's API, no unsupported MPS op |
| Sound | Owner judges each take comparable to the CUDA take of the same prompt | Measured, not judged by ear: Whisper hears the song's lyrics; Higgs WER 0% on every text; no NaN or silent take |
| Memory | No swap on the Mac; the peak goes into the preset's `memory` for the badges | Pass after the Higgs codec fix: no swap from any take; `mps` `memory_gb` 10.5 / 11.0 / 6.1 |
| Time | Recorded; Speech and Music no slower than RTF 3, Sound no slower than 30 s per 30 s take | Pass after the YuE2 speed-ups: Music RTF 2.94 at 3:39 (4.0 before) and 3.64 at the full 6:00, Speech 2.2–2.6, Sound 4 s per 30 s |
| Workflows | Reuse, Use as → Audio track and Use as → Voice in Safari on the Mac and on an iPad | Skipped (owner: the UI is already tested) |
| Higgs parity | Teacher-forced logits on MPS along the CUDA trajectory, with a loose tolerance and at least 98% argmax agreement, and WER and voice similarity over 3 seeds against the CUDA baseline: C1 WER mean 7.9% (max 9.5%); C3 WER 0%; C3 voice similarity mean 0.908 (min 0.896) | Replaced: the CUDA trajectory is not on the Mac, and the 98% bar is out of reach for bf16 on any device (below). Against a CPU float32 reference, MPS bf16 agrees on 85.0% of codebook argmaxes and CPU bf16 on 84.6%, so MPS adds no error of its own. WER 0% on a sentence, a 477-character part and a 1,625-character text in 4 parts; clone similarity 0.76–0.86 (against a different recording than the CUDA C3 baseline, so not comparable) |

What happens to a model that fails is the user's call after the test. The candidates are `PYTORCH_ENABLE_MPS_FALLBACK=1` for a missing MPS op; the model's MLX path behind the same protocol (Higgs through `mlx-audio` with the official weights; Stable Audio through Stability's `optimized/mlx` and its ungated MLX weights; YuE2 through the community port `vanch007/mlx-Yue`); another model on both platforms; or a documented exception. The preset schema allows a different backend family and file list per platform from step 1, so any of these is a config change plus, for MLX, one backend. **Vendoring an MLX fallback is deferred until after the Mac test**; if one is chosen, it gets its own `make update-*` target, a `docs/developer/vendored.md` section and an `AGENTS.md` entry in that change.

### 1. Presets and capabilities

One new config section, `audio_model_presets`. The preset key is the model id (`-m yue2`); there are no aliases, local paths or fine-tunes in v1.

```yaml
audio_model_presets:
  yue2:
    label: "YuE2 3B"
    kind: music                 # music | speech | sound
    capabilities: { lyrics: true, duration: false, voice_clone: false }
    limits: { max_seconds: 360 }                          # the model's own limit; confirmed by the spike
    settings: { instrumental_sections: "[Intro] [Instrumental] [Solo] [Instrumental] [Solo] [Bridge] [Instrumental] [Outro]",
                abc_max_tokens: 4096, cfg: 1.0, ode_steps: 32, memory_budget_gib: 24, vae_core_frames: 512, fade_seconds: 2 }
    platforms:                                            # where it runs; each with its backend, files and measured peak
      cuda: { family: yue2, memory_gb: 6.7, files: [...], settings: { weight_swap: true } }   # pinned repo@rev with file lists
      mps:  { family: yue2, memory_gb: null, files: [...], settings: { weight_swap: false } } # Mac peak filled in by the gate
  higgs3: { kind: speech, capabilities: { …, voice_clone: true }, limits: { max_new_tokens: 1500, part_chars: 500, max_text_chars: 10000, min_voice_seconds: 3, max_voice_seconds: 30 }, settings: { temperature: 0.7, top_p: 0.95, join_pause_ms: 300, join_fade_ms: 10, part_level_match_db: 3, edge_silence_max_ms: 150, codec_device: cpu }, … }   # join set by R0b
  sa3:    { kind: sound,  capabilities: { …, duration: true },   limits: { max_seconds: 380 }, settings: { steps: 8, cfg_scale: 1.0, sample_size: 16777216 }, … }
```

- `utils/config.load_config()` type-checks the section; every capability, limit and platform entry is required, as for video (`utils/config.py:412-421`). A pure parser returns frozen `AudioPreset` values (`utils/audio_preset.py`). Settings may differ per platform (Higgs's `codec_device` is `cpu` on CUDA and `mps` on a Mac) through the platform entry.
- **Kinds come from config.** The Web UI lists the kinds that have at least one preset with an entry for this platform; a kind's Model row lists its presets in config order. A second model of an existing family is a config entry; a new family is a backend plus config. No UI change either way.
- The three models' licence records go in `utils/licences.yaml`, the prerequisite's file, which is never merged with the user's config.

### 2. Backends and dependencies

**One torch backend per model, run on CUDA on Windows/Linux and on MPS on macOS.** The background-removal chain already works this way (`backends/matte/`, created in `backends/__init__.py:296`).

- `core/audio_types.py`: frozen `AudioGenerationRequest` (preset id, kind, prompt, lyrics, seconds, `VoiceRef(path, text) | None`, seed) and `AudioClip(samples, sample_rate, cut)`. `core/audio_backend.py`: the protocol, `load(preset, files, device)`, `generate(request, on_progress, cancelled) -> AudioClip`, `close()`.
- `backends/audio/yue2.py`, `higgs3.py`, `sa3.py`: one file per family, each importing its package lazily. `backends/__init__.py` gets `get_audio_backend(family)` and `audio_platform_key()` ("cuda" or "mps"); nothing else checks the platform.
- **YuE2**: the vendored kit with the preset's settings, run stage by stage through its public `plan()`, `generate_semantic()`, `synthesize()` parts and `decode()`. On CUDA a `WeightSwap` wrapper (`backends/audio/yue2_swap.py`, about 50 lines) pins the large weights in CPU memory in two groups, AR (`embed_tokens`, `lm_head`, each layer's `self_attn` and `mlp`) and NAR (each layer's `nar_self_attn` and `nar_mlp`), keeps the rest on the GPU, and moves only the running phase's group: `ar` for planning and song tokens, `prefill` (AR without `lm_head`) for `CachedNAR`'s prefill, `ode` for `solve(32)`, `none` for the VAE decode. It hands the pipeline its model through `pipe._model` and `pipe._load_model`, which are private: the contract test pins them with `CachedNAR`, `Chunk`, `CODEC_OFFSET`, `MUSIC_END` and `generation_config.ode_steps`, so `make update-yue2` fails loudly if they move. Chunked NAR is not used (it changes the take and saves nothing), nor are int8 weights or KV tricks. Because the kit's `decode()` calls `self._model.to("cpu")` (`pipeline.py:324`), which pulls the swap's GPU-resident tensors back to the CPU, the backend puts them back on the GPU at the start of every take; without it the second take of a batch breaks. On MPS the swap is skipped: unified memory has no GPU copy to free, and moving weights would only add copies. The process memory fraction is set back to 1.0 after construction and on `close()`, also after a failure. `cancelled=` and `on_token=` drive Stop and progress (`pipeline.py:378-379`). A take that reaches 360 s gets a 2 s fade-out (a pure helper) and `AudioClip.cut`.
- **Higgs**: five modules in `backends/audio/` beside `higgs3.py`: `higgs3_config` (the official `config.json`, pinned, read as plain JSON), `higgs3_model` (the network and its strict streamed loader, built on the meta device with the RoPE buffers recomputed and every buffer checked), `higgs3_prompt` (the `Qwen2Tokenizer` and the prompt ids), `higgs3_codes` (delay pattern, sampler, frame rule) and `higgs3_generate` (the loop, with an `on_frame` callback per generated frame). Both repositories, the official one and the codec, are read from the cache only; loading never downloads, and there is no built folder. The codec is loaded from the pinned `bosonai/higgs-audio-v2-tokenizer@815a03f1b57dcb5ae918db415f22cd0c8a6f0f54` with transformers' own class, no remote code, on `codec_device`. Our code resamples the reference with the project's polyphase resampler (level and WER match), encodes it once and passes the reference codes to the loop, and decodes the codes on the codec's device. Text is split by a pure `split_speech_text(text, max_chars=500)` (below) and each part is generated with `max_new_tokens: 1500` (60 s, twice a 500-character part); with *Any*, part 1's take is encoded as the reference for the rest, with part 1's text as its transcript. The parts are joined by a pure `join_speech_parts(parts, rate, settings)`: each part's leading and trailing silence trimmed to at most 150 ms (one R0b part ended in 1.9 s of silence, which made a 2.3 s gap), its speech level matched to the mean of the parts with the gain capped at ±3 dB, 10 ms fades at each edge, then a 300 ms pause; no crossfade, which blends the words across the join. Each part is seeded `seed + index`. The per-frame callback reports one progress step per frame and the part ("Part 2 of 4"), and raises our own exception when cancelled; the cancel check runs between parts too; a take is marked `cut` only if a part reaches `max_new_tokens`.
- **Splitting speech text**: `split_speech_text` packs whole sentences (ends at `.`, `!`, `?`, `…`, `。`, `！`, `？` and line breaks) into parts of up to 500 characters; a sentence longer than that is cut into clauses at `;`, `:`, `—` and `,`, and the clauses are packed greedily in the same way, so a clause may share a part with the sentence before it; a clause still too long splits at the last space before 500 characters, or at 500 if there is none. It never splits inside a `<|…|>` tag and drops empty parts. The ceiling is 10,000 characters: at the measured 16–18 characters per second that is about 9½–10½ minutes of speech in about 20 parts and, at RTF ~1, about 10 minutes of generation, one job's worth; the joined take is about 30 MB as 24 kHz mono samples. Longer text is refused, and the user splits it across takes.
- **Stable Audio 3**: the vendored package, built empty, halved to fp16 and filled tensor by tensor from the checkpoint (`utils/safetensors_stream.py`), because building it on the CPU took about 10 GB of system memory; then moved to the GPU, SDPA attention, the T5Gemma conditioner pointed at the local folder in code. `generate()` is called with the config's `sample_size` (16777216 samples, 380.4 s), as the library's own CLI does (`cli.py:270-281`); without it the library silently caps a take at 120 s (`model.py:87, 429`). The waveform before the library's clamp (`model.py:350-351`) is kept by wrapping the instance's `model.pretransform.decode` (`inference/sampling.py:511`), with no vendored edit, and handed to the save step.

**Dependencies: vendor the two MIT/Apache packages, write our own Higgs class on transformers' `Qwen3Model`, add two small PyPI packages.**

- `packages/yue2` (`make update-yue2`, pinned `YUE2_COMMIT`): `src/yue2/` without `cli.py`, `fast.py` and the server, plus `LICENSE` and the third-party notices. `packages/stable_audio_3` (`make update-sa3`, pinned `SA3_COMMIT`): `stable_audio_3/` without `training/`, `interface/`, `data/` and `cli.py`, plus `LICENSE`. Both go in the wheel's `packages` list, `docs/developer/vendored.md`, `THIRD_PARTY_LICENSES.md` and `AGENTS.md`'s vendored list, with contract tests in `tests/contracts/` pinning the entry points and keyword arguments we call. Nothing from Stability's gated repository is shipped.
- New dependencies, every platform: `tiktoken` (YuE2's tokenizer; cp314 wheels for all three OSes) and `torchaudio>=2.11.1` (pure Python, `torch>=2.11`; Stable Audio imports it). Check that `uv lock` moves nothing else.

**Downloads.** `downloads/plans.audio_plan(preset, platform)` turns the platform entry's `files` into targets, the Higgs codec pinned with the rest. Download sizes come from a dry-run plan (about 7.8, 10.1 and 10.4 GB). The Models page and a job's download step use the existing lane and progress.

Done when: mocked tests show each backend called with its preset's settings (YuE2's fraction restored on `close()` even after an exception; the swap moving only the running phase's group, in the order ar, prefill, ode, none, and not built with `weight_swap: false`; the resident tensors back on the GPU at the start of the second take of a two-take batch; `melody` and the default section list for empty Lyrics, `full` for lyrics with words; the fade on a cut take; Higgs loading both repositories from the cache only and never downloading, and its strict loader refusing a missing, extra or doubly filled tensor; `split_speech_text` cases (sentences packed to 500, a long sentence's clauses packed greedily with the sentence before, a comma-less run split at a space, a tag kept whole, 10,001 characters refused); a three-part *Any* take using part 1 as the reference and one joined clip; the joiner trimming edge silence to 150 ms, capping the level match at ±3 dB and inserting a 300 ms pause; the per-frame callback marking `cut` and raising on cancel; Stable Audio filled tensor by tensor in fp16 before `.to(device)`, `generate()` given `sample_size`, and the pre-clamp waveform returned); `get_audio_backend` refuses an unknown family; a preset without an entry for this platform isn't runnable; the contract tests pass, including the YuE2 private names the swap uses.

### 3. Runner, saving and provenance

- `audio_runner.py`: plan with `plan_iterations`, release a resident enhancer (`release_resident_enhancer()`, as `preflight.py:176-183` does when there's nothing to rewrite), make sure the files are downloaded, load once, generate each iteration, save, close. Pause, Repeat and Next between takes through `job_controls.py`.
- **Out of memory**: `torch.OutOfMemoryError`, and on a Mac the MPS out-of-memory `RuntimeError`, become "YuE2 3B ran out of GPU memory. Close other apps that use the GPU and try again." (Music adds "or shorten the lyrics").
- **Level**: when the backend hands back a waveform from before a clamp (Stable Audio), a pure step scales a take whose peak is above 1.0 down to −1 dBFS before saving; others are saved as they are.
- **Save**: one pure step writes the clip as FLAC through ffmpeg (16-bit, the model's rate and channels) under the usual `{set}_{timestamp}` name (`utils/filename.py`), and embeds the settings with the existing metadata commands, generalised from `embed_mp4_config`/`read_mp4_config` to any container ffprobe reads tags from.
- **Recorded settings**: kind, model, prompt, lyrics or the full text (never the parts), the number of parts, requested and actual seconds, whether the take was cut, voice path and text (never the tick), seed.
- **Gallery facts**: duration, channels, rate, the settings, and 100 waveform peaks (0–255) from one ffmpeg decode to 8 kHz mono and a pure `utils/audio_peaks.waveform_peaks(samples, count)`.

Done when: a FLAC written by the save step reads back the same settings; a Reuse of it restores every recorded field; the peaks of a known sine and of silence are as expected; the runner releases a resident enhancer before load; an out-of-memory error from a mocked backend gives the message above and closes the backend.

### 4. CLI

`ziv audio` (and `ziv-audio`), dispatched from `cli.py` like `image` and `video`:

```
ziv audio -m yue2 --prompt "upbeat synth-pop, female vocal" --lyrics-file song.txt
ziv audio -m higgs3 --prompt "Welcome back." --voice me.wav --voice-text "Hello, this is me." --voice-consent
ziv audio -m sa3 --prompt "rain on a tin roof, distant thunder" --seconds 30
```

Shared flags as in `ziv video`: `-m`, `--prompt`, `-r/--runs`, `--seed`, `-o/--output`; new: `--lyrics`/`--lyrics-file`, `--seconds`, `--voice`, `--voice-text`, `--voice-consent` (required with `--voice`, except when the voice file is a take whose recorded settings show no reference voice, as in the Web UI). No prompt files and no enhance flags. Text over 10,000 characters is refused with the same message as the Web UI. A flag the model doesn't take is refused ("YuE2 3B sets the length from the lyrics; drop --seconds"). The licence line before a first download comes from the prerequisite. Copy CLI builds the same command; for a speech take it adds `--voice-consent` only when the take was cloned. A missing tick is refused with "Add --voice-consent: cloning a voice needs the speaker's permission."

Done when: each refusal is tested, Copy CLI's command for each kind (and for an *Any* and a cloned speech take) runs against a mocked backend and produces the job the form would, and the `ziv-audio` parser is in the docs reference test (`tests/common/test_docs_reference.py:102-104`).

### 5. Web: jobs, media, gallery

- `/api/generate` with `mode=audio` goes to `_submit_audio_job`, whose parse (`_parse_audio_job`) the prerequisite's `/api/generate/plan` shares to plan an audio job; any other unknown mode is a 422 on both routes instead of falling to images (`server.py:750-753`). The job is `job_type="audio"` and runs `audio_runner`; `supported_controls` as for images.
- The workflow contract gains `music`, `speech` and `sound`, listed only for kinds with a runnable preset; `workflow_mode` returns `audio` for them; their visible controls are the rows above, hidden per model by capability as video hides its controls (`workspace_contract.py`). The workspace payload carries each preset's label, kind, capabilities, limits, download state and memory fit.
- Gallery and media: `AUDIO_EXTENSIONS` (`.flac`, `.wav`, `.mp3`, `.m4a`, `.ogg`) join `_asset_kind` and `GALLERY_MEDIA_EXTENSIONS`, so any audio file in the output folder lists and plays; only files with embedded settings offer Reuse. `media_type: "audio"` and the peaks go in the asset JSON.
- The enhance route refuses `mode=audio`; recent prompts store an audio prompt with its kind; presets are not offered on the Audio tab.
- Memory badges for audio compare the preset's measured `memory` with the machine's budget; no header estimates.

Done when: an audio submit queues an audio job and an unknown mode is a 422; `/media` serves a `.flac` under the output root and still refuses other suffixes and paths outside it; the gallery lists audio with peaks under the Audio only filter.

### 6. Web UI

- **Gallery, viewer and Models page first**, so they can be tested before the tab: `AudioPlayer.svelte` (waveform from peaks with a playhead, press to seek and drag to scrub with `data-own-drag` so the stage doesn't page, over a native `<audio controls>`; the stage panel with corner actions and the cut badge, and the viewer with the words under it), `Waveform.svelte` (static, for tiles), the audio tile (icon, words, waveform line, length tag; `.media-tag` in `global.css`), `music` and `speech` icons, an audio branch in `JobPreview` and the `take` noun, the Gallery's **Audio only** option, Use as → Audio track and → Voice in `assetMenu.ts` through the draft setters Reuse uses, and an **Audio models** panel on the Models page after Video models (rows like video's, with the prerequisite's licence mark).
- **The tab**: `WorkspaceTab` gains `audio`, and `visibleTabs` shows it when the contract lists any audio workflow. Draft fields: `audioKind`, a model per kind, `audioStyle`, `lyrics`, `speechText`, `soundPrompt`, `audioSeconds`, `voicePath`/`voiceUpload`/`voiceText` and the unsaved `voiceConsent`. Reset clears the open tab; uploaded voice recordings survive a Gallery visit like images (`heldUploads.svelte.ts`). `VoiceSection.svelte`/`VoicePopover.svelte` are built on the Audio track's file slot (`AudioTrackSection.svelte`).
- **App-wide**: the top bar hides its wordmark and tightens tab padding at 1000 px and below (`TopNav.svelte`), and every checkbox gets a stronger unticked ring (one `.ui-checkbox` change in `global.css`; the consent tick is the first new checkbox in a while).
- `make frontend-build`.

Done when (Vitest): the tab is hidden with no audio workflow; the Music status line says Song or Instrumental with its section count as Lyrics changes; Kind switches the boxes and settings; Generate is disabled without a style on Music, without text on Speech, and with a voice but no tick; the tick clears on a new file and on Reuse; Use as → Audio track lands in the Video tab at 0 s and is disabled with the reason on a model without the track; Use as → Voice fills the transcript and asks for no tick on an *Any* take; the one reason above Generate follows the fixed order; the ⓘ follows the selected model in the toolbar (none for an unrestricted one) and ends each restricted row in the phone Model popover, never on the chip; Presets and LoRAs are absent on Audio; the waveform seeks, a sideways drag on it doesn't page the stage, and playback pauses on paging; the stage panel shows the cut badge for a cut take; the tile hides `[section]` lines and shows the no-settings variant; the Gallery's Audio only option lists only audio; the phone chips follow the kind; the top bar has no wordmark at 1000 px and below.

### 7. A third mode where it matters

The 13 call sites listed under Codebase fit become explicit about `audio`, the types widen to `'image' | 'video' | 'audio'`, and an unknown mode is a 422 instead of an image job. Other `mode`/`kind` comparisons stay as they are: they only run inside image or video code. This is its own commit before the audio work.

Done when: `make check` passes, and a test per call site gives `audio` its own result instead of the image one.

### Tests

- New area `tests/audio/`: presets and their parsing, the request, the runner (mocked backend), saving and provenance, peaks, the CLI parser. `AGENTS.md`, `docs/developer/conventions.md` and `project-structure.md` list the area.
- `tests/backends/test_audio_*.py` for the three backends (mocked packages), `tests/contracts/test_yue2_contract.py` and `test_stable_audio_3_contract.py`, tests for the five Higgs modules on a tiny backbone, `tests/web/` for the routes and gallery, `tests/downloads/` for `audio_plan`. Frontend tests beside their sources.
- No real model loads, as everywhere else. The spike script and the gate runs are run by hand on a GPU machine.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "Make music, speech and sound on the new Audio tab and with `ziv audio`: songs and instrumentals up to six minutes (YuE2), speech of up to about 10 minutes, read in parts, with a voice the model picks or one cloned from a recording (Higgs TTS 3), and ambience and effects (Stable Audio 3 Medium); use any take as a video's audio track." Also: "The Gallery can show audio only." and "Checkboxes have a clearer outline, and the top bar fits narrower windows by hiding its wordmark."
- Concepts: a new `docs/concepts/audio.md`: the kinds and their models, length rules, voices and consent, Higgs's tags, long text read in parts, songs and instrumentals from the Lyrics box and how the section list sets their length, FLAC and its embedded settings, using a take as an audio track; the three licences and Higgs's NOTICE line in the Licences section of `models.md` (added with the presets, so the docs test passes at every commit); `memory.md` (the peaks), `configuration.md` (`audio_model_presets`).
- Web UI: the Audio tab in `workspace.md`, audio in `gallery.md`, audio rows in `models.md`, with screenshot placeholders.
- CLI: `docs/cli/audio.md`, `ziv audio` in `reference.md` and `index.md`.
- Developer: `architecture.md` (the third runner), `backends.md` (audio families, torch on MPS), `vendored.md`, `project-structure.md`, `downloads.md`.
- No page in `docs/` links to this proposal.

## Left out on purpose

- **A Music length control.** YuE2 has none; the section list is the lever, and the 6:00 cap only stops runaway takes.
- **Audio to audio, extension, inpainting, covers.** Stable Audio 3 and YuE2 can do them, but each is its own source-file flow and UI. A later proposal.
- **Voice design** (Higgs has none), **dialogue between several voices** (generate the lines one by one with Use as → Voice) and **automatic transcripts** of a recording (another model to ship).
- **Recording in the app.** Browsers allow the microphone only over HTTPS or on localhost.
- **The prompt enhancer, prompt files, presets and LoRAs on the Audio tab.**
- **Licence acceptance.** The prerequisite's download dialog, ⓘ mark and CLI line name the terms; nothing is accepted or stored.
- **Loudness normalisation.** Only the clipping fix: Stable Audio takes that peak above 0 dBFS are scaled to −1 dBFS.
- **Reading long text in one pass, and a length control for Speech.** Higgs reads about 600 characters faithfully; the app reads in parts instead.
- **A custom play button and time readout.** The browser's controls already have both, and are the player's only keyboard controls.
- **Video-to-audio foley, stem separation, upscaling and voice changers** (decided before this proposal).
- **Playing in the grid, hover previews, scrubbing thumbnails.**

## Alternatives considered

- **MLX on macOS for every model**, as the spike's Mac scripts assumed. Three more backends to write without a Mac to test them, and for YuE2 a 22-star community port. The torch backends already target MPS (YuE2's kit does so explicitly), so macOS starts on the same code that passed on CUDA, and the MLX paths stay as options for the gate.
- **Isolated per-model environments** (a venv per model, run as a child process). Avoids every pin conflict, but means a second install path, a second torch per model (gigabytes each), and IPC for progress and Stop. Vendoring the two permissively licensed kits keeps one environment; only YuE2's transformers 4.57 assumptions are a real risk, and the spike checks them first.
- **PyPI dependencies.** Neither kit is on PyPI (YuE2 ships a wheel inside its model repo), and both pin far below us.
- **Vendoring the Higgs port, or writing our own.** Chosen: our own class. The port is under Boson's non-commercial licence, which an AGPL repository can't include, and loading it needed `trust_remote_code`, a build step and a pinned hash. Our class is adapted from sglang-omni (Apache-2.0) on transformers' `Qwen3Model`, and was proven bit-identical to the port on the GPU. The port served only as the parity oracle.
- **diffusers' `StableAudio3Pipeline`** (already in our pinned 0.42). It loads a diffusers layout that only the gated repo would give us after a conversion; vendoring Stability's MIT package is what the spike ran.
- **WAV or M4A output.** WAV has no place for our settings; M4A is lossy, which matters for a take that becomes a video's soundtrack or is mixed elsewhere.
- **Waveforms drawn in the browser.** Each tile would download its whole file; 100 peaks in the gallery facts cost one decode per file, once.

## Risks

- **YuE2 on 10 GB** peaks at about 6.5 GiB with the weight swap, so about 3 GiB stays free; the stock kit's 0.25 GiB margin is gone. The swap's private hooks into the kit are the new risk, covered by the contract test.
- **YuE2's process-wide memory cap.** Missing the reset would slow or fail the next image or video job; it's tested on `close()` and after a failure.
- **YuE2 on transformers 5.** Untested until the spike; the kit was written for 4.57.
- **Higgs's codec on the CPU.** The codec only fits on the CPU on 10 GB, so decode time counts against speed.
- **Seams in long speech.** Parts are generated separately; with *Any*, the voice of parts 2 onwards is a clone of part 1's take. R0b measured that the joins and the voice hold (the *Any* voice matched part 1 as closely as the halves of one part match each other), so no fallback is needed. From a recording the parts differ in level by up to 2.2 dB, which the join's level match corrects, and a part can end with a long silence (1.9 s once in 14 parts), which the join trims to 150 ms. The joined takes still await owner listening.
- **Hiss when cloning from a poor recording.** A voice cloned from a narrow, low-quality recording (an old audiobook clip with nothing in the top of the spectrum) comes out with added hiss. In long speech split into parts the hiss steps up from part 2 on, because later parts follow part 1's take, and the likeness to the recording is weaker. A clean, full-quality recording gives no audible hiss, no growth across parts and a good likeness. The cause is the recording's quality. Variants tested: later parts following a re-encoded copy of part 1 against reusing part 1's codes as they are, an audiobook clip against a clean recording, and a take with no clone as the baseline. The blind listening test favoured the clean recording in 4 of 4 pairs, and reusing part 1's codes made no consistent difference, so there is no code fix. Decision: a tip in the docs to clone from a clean recording, and no code change.
- **Songs cut at the cap.** Mostly resolved: the first spike's cut came from its own token limit, and with the weight swap a take only stops at the model's own 360 s. A take that still reaches it ends in a 2 s fade-out and carries the "Cut at 6:00" badge, which says to remove sections; `melody` instrumentals ended on their own in every spike take.
- **Higgs's licence reading.** See Licences; accepted by the user.
- **Untested macOS.** All three run on MPS for the first time at the gate.

## Mac check

Run on an Apple-silicon Mac with 16 GB of memory, and again on a larger one (32 GB or more) if one is available; note the chip and the memory next to every number. Update to the branch's latest commit and run `make install` first. Run one model job at a time, and quit memory-hungry apps first. Every model load on the Mac is run alone: no second job, no image or video job and no prompt enhancement beside it, so a peak is the model's own.

How to measure: open Activity Monitor, Memory tab, and watch the Python process's **Memory** column, the **Memory Pressure** graph and **Swap Used**; `sysctl vm.swapusage` before and after gives the swap in a terminal. Where a step says "MPS peak", run the take from a Python session and read `torch.mps.driver_allocated_memory()` after it. On MPS the YuE2 `memory_budget_gib` cap does not apply (the kit sets it on CUDA only), so nothing stops a take from going over the Mac's memory. Apple's recommended GPU working set is `mx.device_info()["max_recommended_working_set_size"]`; write it down too, because the badge compares against it.

Pass = the step's condition holds with **no swap growth and Memory Pressure staying green**. Stop and note it if the Mac turns unresponsive. A failure is the user's call afterwards (see the gate above).

### 1. Install and first run

1. `ziv-audio -m yue2 --prompt "calm piano, slow"` on a Mac where nothing is downloaded. Look for: the licence line before the download, then the download's live progress line. Pass: it downloads and goes on to load, with no prompt to accept anything.
2. Repeat with `-m higgs3 --prompt "Hello there."` and `-m sa3 --prompt "rain on a tin roof" --seconds 10`. Pass: each prints its own licence line (CC BY-NC 4.0; Boson research and non-commercial; Stability Community with the Gemma terms) and downloads fully, including the second repository (YuE2-Vae, the Higgs codec, the T5Gemma encoder).
3. In the Web UI, open the Models page and the Audio tab's model dropdown. Look for: all three audio models listed with their sizes and licence mark, and **no memory badge** (`memory_gb` is null for `mps`). Pass: no badge, no "Too large" or "Tight" text.
4. `ziv-audio -m sa3 --prompt "test" --seconds 5` after install. Pass: `ziv-audio` finds ffmpeg, or offers to install it with Homebrew or MacPorts.

### 2. YuE2 (music)

1. A song of about 3 to 4 minutes: 12 sections of lyrics (about 3 3/4 min), `ziv-audio -m yue2 --prompt "upbeat synth-pop, female vocal" --lyrics-file song.txt`. Look for: progress phases **Plan**, **Song tokens** (a growing count), **Synthesis N/M** (steps across chunks, counting up to M without resetting) and **Decode**, in that order, with a time left that counts down within the take and never goes negative or jumps back. Pass: all four phases reported, the take saves as FLAC and plays in QuickTime and Safari, and it sounds like a song (not noise or silence).
2. During that take, record the Python process's peak Memory in Activity Monitor and the MPS peak. Look for: the load itself (the weights are streamed in on the CPU, unpinned, then moved with `model.to("mps")`) peaking near the model size (6.8 GiB), not twice it. Pass: peak under the Mac's recommended working set on the larger Mac; on 16 GB, no swap growth. If the 16 GB Mac swaps, record the peak and the time anyway and stop there.
3. Record the take's wall time and compute RTF (take seconds over audio seconds). Pass: RTF no worse than 3 (a 3 1/2 min song in under 10 1/2 min). Note which phase takes the time; eager attention on MPS is the suspect.
4. Two takes in one batch: `--runs 2` with the same style and lyrics. The kit's decode moves the model to the CPU to make room for the VAE and the backend moves it back for the next take. Look for: Memory dropping after the Decode and rising back before the second take's Plan, with the second take's peak equal to the first's (no growth), and the second take sounding like a song. Pass: both takes save, peak within about 10% of each other. A hang or a slow second take is a finding.
5. An instrumental (no lyrics, default 8 sections, about 1 1/2 min). Pass: finishes, plays, and takes noticeably less time and memory than the song.
6. Stop mid-take (press q in the CLI, or Stop in the Web UI) once during Song tokens and once during Synthesis. Pass: it stops within a few seconds, and Memory falls after the job ends.
7. Web UI: queue the same song from the Audio tab (Music). Look for: the job card's stages (Planning, Writing the song with the length so far, Rendering, Finishing), the one bar that only moves forward, the time left, the "Take 1 of 2" count for a batch of two, and the Aurora following the stage (waiting, plan, write, synth, decode) and settling into the waveform. Pass: no stage skipped or stuck, the bar never goes back, and the take appears in the stage and the Gallery.

### 3. Higgs TTS 3 (speech)

1. Any voice, a short text (one sentence): `ziv-audio -m higgs3 --prompt "Welcome back. It's good to see you again."`. The model runs in bf16 on MPS and the codec runs in fp32 on MPS (`codec_device: mps`). Look for: a clean load (no unsupported-op error or `PYTORCH_ENABLE_MPS_FALLBACK` warning) and understandable speech. Pass: the take plays and is clearly the sentence; record the load time and the time for the take. If an op is missing, record its name and the error: this is the "missing MPS op" case the gate names.
2. A cloned voice: record 10 to 20 seconds of your own voice in Voice Memos (clean, close-miked), export it (a WAV is the safe format; try the M4A too), then `ziv-audio -m higgs3 --prompt "Welcome back." --voice me.m4a --voice-text "<what it says>" --voice-consent`. Look for: the voice resampled and encoded once before the first part (a pause before the first progress), no error on a 44.1 or 48 kHz file. Pass: the take plays in a voice that is recognisably the recording's, no added hiss.
3. Long text in parts: about 1,500 characters (3 parts). Look for: "Part 2 of 3"-style progress, the parts joined with a short pause, no volume jumps between parts, no long silences at joins. Pass: one take that reads all the text in order; the time per part is similar. Also try it with *Any* voice and check the later parts keep the first part's voice.
4. Loudness: Higgs's codec output is not limited to -1..1. Run a text with expressive tags (for example a laugh, `<|sfx:laughter|>`) or shouting text. Look for the CLI line "Peak above full scale; scaled to -1 dBFS." if the take overshoots. Pass: the saved file has no clipping (peak at or below -1 dBFS); a take that does not overshoot is saved unchanged and prints no note.
5. Record peak Memory (Activity Monitor and MPS peak) for the longest allowed part (500 characters) and RTF. Pass: RTF no worse than 3; no swap growth on 16 GB. The model is about 8 GB in bf16, so the 16 GB Mac is the tight case: record how close it comes.
6. Web UI: Speech with Any, then with a recording and the consent tick. Pass: the job card shows "Reading part 2 of 3" and "Finishing"; Stop works between and within parts.

### 4. Stable Audio 3 Medium (sound)

1. A 30 s sound: `ziv-audio -m sa3 --prompt "rain on a tin roof, distant thunder" --seconds 30`, with `--runs 3`. The model is built and run in fp16. Look for: "Step N of 8" progress, then Decode; a first take that is slower than later ones (the first decode warms up). Pass: three takes save and play; later takes no slower than 30 s per 30 s take (the CUDA card does about 0.65 s); the pitch and noise sound right (no NaN silence, no static, which is what an fp16 problem sounds like).
2. A long sound: `--seconds 380`. Pass: it saves at about 380 s and plays to the end; record the time and the MPS peak, and check the 16 GB Mac does not swap.
3. A loud prompt ("glass breaking", "thunderclap"). Pass: the CLI prints the scaled-to -1 dBFS note, or the take is already under full scale; the file never clips.
4. Record the load's system memory peak. On CUDA the load needs about 5.5 GiB of system memory; on a 16 GB Mac check the fp16 build does not push Memory Pressure to yellow or red.
5. The T5Gemma text encoder is not converted to fp16: it runs in the dtype of its files, while the DiT and decoder run in fp16 and the conditioner's small layers in fp32. Check that MPS runs the encoder in that dtype without an unsupported-op error and that the sound follows the prompt (a prompt that is ignored, or NaN silence, points at the encoder); if it does not, the encoder would need fp32 on MPS.

### 5. Web UI in Safari

Start `ziv ui --host local` and open it in Safari (the Mac's own, and Safari on an iPad if one is at hand).

1. The Audio entry shows beside Image and Video on the workflow control (the phone width too), and Kind switches Music, Speech and Sound with the right boxes: Style over Lyrics; Text; Prompt with Length. Pass: no layout break at a narrow window.
2. Voice recording: in Speech, From a recording, use Upload or the drop target with a WAV, and a Voice Memos M4A. Pass: the file loads, What it says and the consent tick are required, the tick clears on a new file, Reuse or a reload, and Generate gives its one reason until all are filled.
3. Reset and Reuse: on each of Image, Video and Audio, change several settings, press Reset, then make a take and press Reuse settings on it from the Gallery. Pass: Reset restores the defaults, and Reuse fills the settings the take recorded, for all three media types.
4. Settings survive an Audio visit: set a size and a seed on Image, visit Audio, return. Pass: the Image and Video settings are as left, and the Audio settings are as left when going back.
5. Licence hover card: hover the mark beside the Model dropdown, and on the Models page row. Pass: it names the licence and its summary.
6. Download confirm dialog: delete the Higgs model on the Models page, then press Generate on Speech. Pass: the dialog shows the size, the free space and the licence line; Cancel queues nothing; plain Enter confirms, Cmd+Enter (Generate's shortcut pressed twice) does not.
7. Players: make two takes, open one in the stage and one in the viewer. Press play on a take, then play the other. Pass: only one plays at a time, the waveform playhead follows and a press on the waveform seeks. Play a take, then open another take or close the viewer: playback stops.
8. The held take: pause a batch between takes, then play the previous take using the held take's play button. Pass: it plays, and the play button does not start the next take or reset the card.
9. Safari plays the FLAC takes in the stage and the viewer; Gallery tiles do not play. Pass: sound and waveform both appear.

### 6. Memory across jobs in the Web UI

1. Queue a song, a speech and a sound one after the other, and watch Memory in Activity Monitor. Pass: Memory drops back to near its idle level between jobs (the backend's `close()` and the release after each job), with the next job's peak no higher than when run alone.
2. Queue an audio job and then an image job in the same session. Pass: the image job's peak is the same as on a fresh start. Compare against the number from a fresh app.
3. After the last job, wait 30 seconds. Pass: Memory is close to the app's idle level, and Swap Used did not grow through the run.

### 7. Record

Write back, per model and per Mac:

- `memory_gb` for `mps` in `zvisiongenerator/config.yaml` `audio_model_presets` (peak GPU memory in GiB, from the largest take: YuE2 the 3 3/4 min song, Higgs a 500-character part, Sound the 380 s take), and the three rows of the same figures plus system memory at load in `docs/concepts/memory.md#audio` (the macOS tab, which says the peaks "have not been measured on a Mac yet").
- RTF per model, the first-take penalty of Sound, the load time of each model, and the Mac's recommended working set.
- Which op failed or fell back, which phase is slow, and whether any model swapped on 16 GB. If one does, the memory badge there reads "Tight" or "Too large" and the model is a candidate for the gate's fallbacks.

Then fill the gate table above with the results, with the Safari and iPad workflow checks, and set the status line.

### Mac results (2026-11-04)

Apple M5, 32 GB, recommended GPU working set 24.96 GiB; torch MPS peaks from `torch.mps.driver_allocated_memory()`, process peaks from the physical footprint. No take swapped.

| | YuE2 | Higgs TTS 3 | Stable Audio 3 |
|---|---|---|---|
| First run | Licence line, both repos downloaded, no prompt | Two licence lines, both repos | Two licence lines, all three repos |
| Load | about 3 s | about 3 s | about 6 s |
| Largest take | the full 6:00 (24 sections, cut at the limit) | 1,625 characters in 4 parts | 380 s |
| MPS peak / process peak | 10.5 / 12.9 GiB | 11.0 / 12.5 GiB | 6.1 / 8.5 GiB |
| Time | 3:39 in 882 s (RTF 4.0), 645 s (RTF 2.94) with the speed-ups; 6:00 in 1,309 s (RTF 3.64) | RTF 2.2–2.6, 45–62 s per part | 30 s in 4 s warm (5.4 s first); 380 s in 42 s |

- **YuE2** was slower than the CUDA card by more than the bar allowed: Plan 131 s, Song tokens 406 s (13.5 tokens/s), Synthesis 322 s, Decode 22 s. Nothing ran in the wrong dtype or on the CPU; a token reads about 3.6 GB of weights and the M5 has 153 GB/s against the 3080's 760 GB/s. Three exact speed-ups on MPS (`backends/audio/yue2_mps.py`, the NAR query tile) brought the song to Plan 113 s, Song tokens 245 s, Synthesis 265 s, with the take bit-identical to the one before. Two takes of a batch at one seed are bit-identical; the second take's MPS peak was 12% above the first's, the process peak 3%. Stop exits 0.6 s after q in Song tokens and 5.3 s in Synthesis. Measured and not taken: int8 weights (about 25–30% off the token phases, lossy) and an MLX port (bf16 only 10–20% faster; 4- and 8-bit about 2× on the token phases, lossy, and the acoustic stage would need custom kernels).
- **Higgs**: clones from a 48 kHz WAV and a 44.1 kHz M4A; *Any* keeps its voice across parts (0.95–0.96) and part levels are within 0.5 dB. A laugh take peaked at −0.73 dBFS and was saved unchanged. transformers' codec caches its conv layers in an `lru_cache` keyed by the codec, which kept 4 GiB alive in the process after any take that encoded audio (a recording, or the parts after the first); the Web UI held it until restart and an image job after it peaked 4.5 GiB higher and swapped 2.5 GB. `close()` now clears the cache.
- **Higgs parity on the Mac.** A 477-character part (721 frames) was generated on MPS and the same frames teacher-forced in one pass. Codebook argmax agreement: MPS step by step against MPS one pass 93.1%, MPS one pass against CPU one pass 93.0%, and against a CPU float32 reference MPS bf16 85.0%, CPU bf16 84.6%. The codebook argmax is often a near tie, so bf16 rounding alone moves it about 7% of the time on one device; the gate's 98% bar was not reachable this way, and MPS is as close to float32 as the CPU is.
- **Review after the gate** (two independent reviews of the fixes and of the options turned down). Found and fixed: the song-token head filled the other rows with `-inf`, which turns into NaN under a guidance (`cfg`) other than 1 and stopped the take; it now fills zeros, which the sampler's mask turns into `-inf`, and a 400-token run at `cfg` 1.5 gives the same tokens with and without the speed-ups. Also fixed: a failed head swap no longer leaves the folded attention installed, `close()` warns instead of raising when a transformers version lacks the codec cache, and the contract tests check behaviour (the song phase allows exactly the head's rows; the codec cache fills and clears). Remaining exact options were measured smaller than estimated: a top-k-only sort about 1% (and ties at the top-p cut in 2 of 3 steps make it unsafe), fused projections about 1.5% (bit-identical), a Plan-phase head and a folded acoustic attention under 1% each; sampling on the GPU about 2% and not exact. Not done; nor int8, MLX or `torch.compile`. The codec cache is the same in transformers `main` and in `xcodec`; no upstream issue was filed.
- **Stable Audio 3 decoder in fp16.** Stability's MLX port runs its decoder in fp32 because of its differential attention. The same latents decoded both ways differ by 35–43 dB SNR overall and 16–31 dB above 10 kHz (about −62 to −71 dBFS); fp32 doubles the decode (30 s in 4.9 s instead of 2.4 s, 380 s in 51.7 s instead of 25.3 s) and adds about 2 GiB. In a synced A/B of five sounds the owner heard no difference, so the decoder stays fp16.
- **MLX ports.** No YuE2 port is usable: all pin Python 3.12 and an older MLX, and the only published 32 GB M5 figure (RTF 4.6–5.0) is slower than the torch path. Stability's official MLX path for Stable Audio 3 is no faster than the torch path here (30 s in 4 s, 380 s in 42 s). mlx-audio runs Higgs TTS 3 from the pinned snapshot at RTF 1.5–2.0 against 2.2–2.6, with WER 0–1.1%, but clones measured less like the recording (0.57–0.74 against 0.76–0.86) and the peak was 10.6–11.4 GiB; in a synced A/B of three pairs the owner heard slightly different takes of the same quality, so Higgs stays on our torch code. Through our own frame loop instead of mlx-audio's `generate`, clones measured 0.78–0.85 and the owner preferred MLX, so on macOS Higgs now runs on MLX ([Speech on macOS through MLX](higgs-mlx-mac.md)).
- **Stable Audio 3**: all 8 steps reported, loud takes scaled to −1 dBFS with the note, the T5Gemma encoder runs in its own dtype without an unsupported op.

## Open questions

1. **YuE2's "creator permission for individuals"**: where is it published? The model card and `LICENSE` at the pinned revision state only CC BY-NC 4.0, so the summary says "Non-commercial, credit required" until there is a source to cite.

2. **YuE2 instrumentals and vocals**: the spike's pre-screen found words in 7 of 9 `melody` instrumentals, and adding "instrumental, no vocals" to the style didn't visibly help. Do the 18 spike takes sound instrumental enough to ship empty Lyrics as "instrumental"? Resolved: not with the verse and chorus list. The default is now the `[Instrumental]` and `[Solo]` list, which gave 6 of 7 clean takes in listening round 3.

What to do with a model that fails the Mac gate is decided by the user after the test, not here.

## Build order

The licence records and the download dialog come first, from their own proposal. Each commit leaves the app working; the Audio tab stays hidden until a preset is runnable.

0. The spike in the project environment; results written here.
1. `refactor(web): make the media mode a three-way type`: step 7.
2. `feat(audio): add audio presets, the request and the backend protocol`: config section with no presets, parser, `core/audio_*`, `backends/__init__` selection, `audio_plan`, configuration docs.
3. `build(audio): vendor YuE2 and Stable Audio 3`: the make targets, packages, contract tests, `tiktoken` and `torchaudio`, licences and notices.
4. `feat(audio): run YuE2, Higgs TTS 3 and Stable Audio 3`: the three backends, mocked tests; GPU runs of the spike prompts through a test script.
5. `feat(audio): save takes as FLAC with their settings, and add ziv audio`: runner, save, provenance, peaks, CLI.
6. `feat(web): list, play and reuse audio`: the submit route and contract, media route, gallery kind and facts, player, tiles, viewer, Use as, the Models page panel.
7. `feat(web): add the Audio tab`: the tab, Kind, the per-kind settings, Voice with the tick, chips, Copy CLI, `make frontend-build`; the presets turned on in config and their licence records added to `utils/licences.yaml`, with their entries in the Licences section of `docs/concepts/models.md`.
8. `docs(audio): document making audio`: Concepts, Web UI, CLI, developer pages, `AGENTS.md`, CHANGELOG.
9. The Mac gate on the Mac; the results written here; what to do with any failure is the user's call; then the proposal is archived with the release.
