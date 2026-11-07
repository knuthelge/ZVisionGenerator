# MiniMax Music 3 as a second song model

**Status:** Proposed (2026-11-07)

## What the user gets

Music's **Model** picker gains **MiniMax Music 3** beside YuE2 3B. The Style and Lyrics boxes, Tempo, Key, Length, Seed, the helpers and the player stay as they are; the user picks the model and generates. What differs, in the order a user notices it:

- **A licence that allows selling the music.** YuE2 is CC BY-NC (creators may monetise their own outputs, companies need a licence). MiniMax-Music3's community licence allows commercial use with conditions (below).
- **Words sung as written.** On the YuE2 paper's own benchmark Music 3 has the lowest phoneme error rate among the open systems (0.0627 against YuE2's 0.0844). A vendor test heard a cleaner mix and an eight-section lyric sheet followed to the outro.
- **More section tags** (`[Pre-Chorus]`, `[Post-Chorus]`) and tempo and key as plain words in the style.
- **Not longer:** 5 minutes (stops at 6), 44.1 kHz stereo; YuE2 already reaches 6 minutes at 48 kHz.
- **Much slower and larger** (below). A take takes tens of minutes on a 10 GB card or a 32 GB Mac, against minutes for YuE2. The model's ⓘ says so and the job card shows time left.

YuE2 stays the default. This proposal asks whether Music 3 can run on both platforms at a speed worth waiting for, and whether it sounds better than YuE2 on the same lyrics.

## What it is

Checked 2026-11-07 (sources at the end). The input contract (tags on their own lines, the three-heading caption, empty lyrics refused, length a ceiling, tempo and key as hints) is already written up in `audio-controls.md`, "Two more models to support".

- **Open weights**, released 2026-08-13; no newer MiniMax release since (no official Turbo; the repository changed once on 2026-10-26, a docs rename).
- **Licence: MiniMax-Music3 Community License** (free grant, commercial use allowed). A commercial product's interface must prominently display "MiniMax-Music3" (the app shows a model's credit through the `attribution` key of `licences.yaml`, as for Stable Audio 3). Above USD 20M yearly revenue needs MiniMax's written approval. A product that lets third parties generate outputs must keep reasonable safeguards, and an acceptable-use list applies. Output ownership and termination are not addressed. This is not legal advice.
- **Architecture:** an 8B Qwen3-based planner writes 25 frames per second one at a time (it dominates the runtime), a 0.6B depth decoder adds the other codebooks, a 2.4B flow transformer (stored fp32, 9.7 GB; run in bf16, 4.9 GB) and a vocoder make the audio, 44.1 kHz natively (MiniMax's server resamples to 32 kHz). Official modular files total about 28.5 GB; the repository holds 57 GB with duplicates.
- **Not offered:** reference audio, cover, repaint, a negative prompt. None is proposed.
- **Languages:** English and Mandarin in the demo, Spanish and Japanese in one test; ACE-Step is reported broader. No comparison with YuE2 outside English exists.
- **Runtimes:**
  - **diffusers `ModularPipeline`** (in the pinned 0.42.0): bf16, about 23 GB; about 22 GB with automatic offload; group-offloading the planner fits 8 GB, slower. MiniMax's reference server needs two GPUs and is not an option.
  - **mlx-audio** (`mlx_audio.music`, already pinned at 0.6.2; no new dependency): mlx-community copies mxfp8 13.9 GB (recommended), 8-bit 14.2 GB, 6-bit 11.7 GB, 4-bit 9.2 GB, mxfp4 8.9 GB. mxfp4 drops words.
  - **GGML runtimes** (`audio.cpp`, `minimaxmusic.cpp`; CUDA, Vulkan and CPU) with community GGUFs of the whole pipeline: `audio-cpp/` and `Serveurperso/MiniMax-Music3-GGUF`. Planner: Q4_K 7.18, Q5_K_M 6.28, Q6_K 7.05, Q8_0 9.1–9.9 GB; flow transformer 1.4–2.6 GB. `minimaxmusic.cpp` says the planner breaks below Q5, which matches mxfp4: **Q5_K_M is the floor**. `audio.cpp`'s default Q4/Q8/Q4 mix peaks at 9.8 GiB and ran 1.35 times real time on an RTX 5090 for 30 s (self-reported). The Abiray GGUF holds the flow transformer only and does not help.

## Can it run here

| | Evidence | Verdict |
|---|---|---|
| 24 GB+ GPU | Card and diffusers: about 22 GB with offload. A 5090 test: a 2½-minute song in 4–5 minutes at about 11 GB with an int8 planner | Runs |
| 10 GB GPU, Windows and Linux | **Streaming the whole planner** moves 16 GB per frame. Our estimate: this card links at PCIe gen 3 (about 12 GB/s), so about 1.4 s a frame and about 100 minutes for 3 minutes of audio. **Cheaper routes, all unmeasured:** (a) a fully resident Q5_K_M or Q6_K planner (6.3–7 GB) on a GGML runtime; (b) an int8 planner with only the layers that overflow streamed, about 2 GB per frame; (c) the KV cache, which classifier-free guidance doubles (guidance scale 1.5; estimated 2–3 GB for 3–5 minutes) and which decides whether (a) fits at all | **Unknown; the one real risk** |
| 32 GB Apple silicon | mxfp8 is 13.9 GB of weights. The only community speed is a proof-of-concept fork on bf16: 29 GB planner and 18 GB flow stage, a 3-minute song in 1,900 s on an M3 Max (about 10 minutes per minute of audio). mlx-audio's own speed is unpublished | **Unknown, probably workable** |

Reference points from our runs: YuE2 on the 10 GB card takes 132 s for a 225 s song at 6.5 GB; on the Mac it runs about 3 times slower than real time at 10.5 GB.

## Checks before building

Both platforms pass the same checks or neither ships (full platform parity). Run them as one spike, as the audio models were, on a 10 GB / 32 GB Windows or Linux machine and a 32 GB Mac, in a scratch data folder and through the GPU queue.

1. **Licence.** The owner reads the full text against how the app is distributed (see the first open question). Add the `licences.yaml` entry only after this.
2. **Memory and speed.** Peak GPU and system memory, and time per minute of audio, for a 60 s and a 5-minute take. Windows and Linux, in order: (a) a resident Q5_K_M and Q6_K planner on a GGML runtime, (b) an int8 planner with overflow streaming, (c) the model card's group offload as the baseline. Mac: mxfp8, then 8-bit and 6-bit. Pass: a take finishes without out-of-memory next to a running browser; a 3-minute song takes 30 minutes or less (10 times real time; the owner may move this); Stop ends it within a few seconds; an image job in the same process afterwards peaks as in a fresh one (as for YuE2).
3. **Words.** The same 90-line lyrics through every route that passed 2, and through YuE2: transcribe with Whisper large-v3 outside the app and compare word error. A route that drops clearly more words than bf16 is out, whatever its size.
4. **Quality against YuE2.** The owner's blind, loudness-matched listening on the same lyrics and style, at least 8 pairs, including an instrumental and a non-English song if the owner uses one. Music 3 earns its place by winning at least half. Count the takes that stop under 70 % of the requested length and the instrumentals (`[Instrumental]` tags) that gain vocals. The YuE2 paper's own metrics put YuE2 ahead on musicality, so this is not a foregone conclusion.

If check 2 fails on 10 GB, the owner decides between not shipping and gating the model to machines with enough memory, as the first-run dialog already does for the larger enhancer (`min_memory_gb`). That is a product decision, not a default.

## Proposed change

Only if the checks pass. Small, because the Music tab and the enhancer already take a model by adapter:

- **Preset `minimax_music3`** in `audio_model_presets` (`kinds: [music]`, `lyrics: true`, `limits.max_seconds` from check 2, pinned files without the duplicates). `platforms.cuda` and `platforms.mps` each name their own family, as Qwen3-TTS does with a different runtime per platform: the CUDA one is whichever runtime check 2 picks, the macOS one is mlx-audio. `memory_gb` comes from check 2.
- **Backend modules** under `backends/audio/`, one per runtime, importing their library lazily, plus one pure helper that turns the request into the model's input (tags on their own lines, `[Instrumental]` for empty lyrics, tempo and key as caption words, a seed) and is shared by both.
- **Caption rules:** a `minimax_music3` entry in `_CAPTION_RULES` with the three headings and `metadata=True`. `CaptionRules.headings` and `metadata` exist for this and no model uses them yet. Never invent a BPM or key when none is set; the Without row becomes caption words.
- **The licence entry and credit**, the model's ⓘ (slow, sings as written, length is a ceiling) and job card progress by frame count with time left.
- **Docs:** `docs/concepts` for what the model does and costs, its row in `docs/concepts/configuration.md`, and a CHANGELOG entry.

### Testing

Mocked backend tests for each runtime and a contract test for the input helper (tags, `[Instrumental]`, tempo and key words, seed). No test loads the model. The same seed gives the same take on one platform; MLX and torch or GGML differ from each other, as for other models.

### Layouts

No new control. The model is one more row in the Model dropdown on the wide toolbar, on the tall layout's single-row toolbar and on the phone's Model chip popover, each with the ⓘ line and the memory badge. The long take is the layout concern: the job card shows time left and the existing Pause, Next, Repeat and Stop on every layout.

## Not proposed

- Reference audio, cover, repaint or extend: the model has none.
- The community Turbo LoRA (30 to 8 flow steps, 245 s for a 190 s song on a 4090, self-reported): it shortens only the flow stage, about 22 % of a 4090 run, while the planner dominates; it is also experimental.
- Hosted API use: the app is local.
- Replacing YuE2 as the default, or any migration of earlier takes: a YuE2 take's settings do not apply to Music 3.
- A separate UI for the three-heading caption: the enhancer writes it and the Style box shows it as text.

## Open questions

- **Licence:** is a free open-source app a "commercial product" at all? If not, the credit may only be courtesy; if so, is the licence dialog plus the Model label "prominent", or does it need a fixed spot (the ⓘ, the player caption)?
- **Which CUDA runtime:** diffusers with streaming, or a GGML runtime (a new dependency, a vendored or pinned build, Windows and Linux builds to verify)? Decided by check 2.
- The 10 times real-time threshold.
- Whether a prepared copy (a quantised planner stored next to the model, as in the pre-quantized copies proposal) is worth it on CUDA.

## Sources

- Model card, licence and file list: https://huggingface.co/MiniMaxAI/MiniMax-Music3 and its `LICENSE`
- Reference server README (two GPUs): https://github.com/MiniMax-AI/MiniMax-Music3
- diffusers pipeline docs: https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/minimax_music3.md
- mlx-audio (upstream) and its MiniMax Music 3 page: https://github.com/Blaizzy/mlx-audio ; copies: https://huggingface.co/mlx-community/MiniMax-Music3-mxfp8 (also 8-bit, 6-bit 11.7 GB, 4-bit, mxfp4)
- Community GGUFs of the whole pipeline: https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF and https://huggingface.co/Serveurperso/MiniMax-Music3-GGUF (flow transformer only: https://huggingface.co/Abiray/MiniMax-Music3-GGUF)
- Proof-of-concept Mac figures: https://huggingface.co/MiniMaxAI/MiniMax-Music3/discussions/14
- YuE2 paper, WildSongBench Table 2: https://arxiv.org/html/2609.33757v1
- RTX 5090 test against ACE-Step (vendor blog): https://blog.sogni.ai/blogs/minimax-music3-vs-ace-step-15-xl/
- Community Turbo FP8 and LoRA: https://huggingface.co/modulsx/MiniMax-Music-3-Turbo-FP8
- ComfyUI low-memory guidance: https://docs.comfy.org/tutorials/audio/minimax/minimax-music-3
