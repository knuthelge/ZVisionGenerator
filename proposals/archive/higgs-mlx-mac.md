# Speech on macOS through MLX

**Status:** Done (2026-11-05, v0.14.0; checked on a 32 GB M5)

## Problem

Higgs TTS 3 runs on macOS through our own torch model on MPS at RTF 2.2–2.6 on a 32 GB M5. mlx-audio, already in the lockfile as a dependency of mlx-vlm, runs the same official checkpoint on MLX at RTF 1.5–2.0, about 30% less time per take. In a synced A/B of a sentence, a 477-character part and a cloned voice, the owner heard slightly different takes of the same quality (see [audio-generation](audio-generation.md#mac-results-2026-11-04)).

## Proposal

On macOS, `higgs3` runs on a new backend family, `higgs3_mlx`, built on mlx-audio's `higgs_audio_v3` model. Windows and Linux keep the torch backend.

- **Per-platform family and files.** A preset's `platforms.<key>` entry may name its own `family` and `files`. `higgs3`'s `mps` entry names `higgs3_mlx` and only the model repository at the same pinned revision: the MLX model reads its codec from the checkpoint, so the 0.8 GB tokenizer repository is not downloaded on a Mac. Every reader of a preset resolves it for this machine first (`AudioPreset.on_platform`); the Hugging Face cache view counts every platform's files (`all_files`).
- **Shared part logic.** Splitting text into parts, seeding part *i* with `seed + i`, following a recording or part 1's own voice, trimming the reference, frame progress and Stop, and joining the parts move out of the torch backend into one module both backends call. Each backend supplies only its engine: read one part from text, a seed and an optional reference; encode a waveform.
- **Own loop on the model's parts.** mlx-audio's `generate` yields once at the end, with no per-frame hook. The backend runs the same loop from the model's parts (prompt embeddings, the backbone with its prompt cache, the sampler step, the audio embedding and the codec decode), calling the progress and Stop check every frame, and leaves out mlx-audio's fades (the join does its own). A contract test pins every mlx-audio name it calls; mlx-audio is pinned to its locked version.
- **Load from the cache only**, from the pinned snapshot folder, as the torch backend does.
- The recording is read and resampled by our code, as on CUDA, and given to the model as a 24 kHz waveform.

## Checks before release (on the Mac)

- Sentence, 477-character part, the 1,625-character text in 4 parts, and cloned voices from a WAV and an M4A: WER, voice similarity, part levels, and that *Any* keeps part 1's voice.
- Progress counts frames per part; Stop acts within a part.
- Memory peak and RTF; `mps` `memory_gb` re-measured; no memory left after `close()`.
- The Web UI's Speech job end to end.
- A blind listening round: long text in parts and cloned voices over several seeds, torch against MLX.

## Alternatives

- **Keep torch.** Same quality, 30% slower.
- **Call mlx-audio's `generate`.** No progress or Stop within a part, and it adds fades our join does not expect.
- **Select the MLX backend inside `get_audio_backend` by platform.** Hides the choice from the config, and the Mac would still download the separate codec repository.

## Results (2026-11-05, M5, 32 GB)

| | Torch on MPS | MLX |
|---|---|---|
| 1,625-character text in 4 parts | 224 s | 163 s (−27%) |
| 477-character part | 62 s | 43 s |
| Process memory peak | 12.5 GiB | 11.2 GiB |
| WER, long text, 3 seeds | 0–1.3% | 0.6–1.3% |
| Voice similarity of a clone to its recording | 0.76–0.86 | 0.78–0.85 |
| Download on a Mac | 10.1 GB | 9.3 GB |

- Parts of one take stay one voice (similarity 0.95–0.96 between parts, levels within 0.2 dB).
- Stop acts within a part in 0.3 s; progress counts frames per part; the Web UI's Speech job ran end to end, and memory fell back to 0.5 GiB after it.
- MLX's free-buffer cache is off during a take: at mlx-audio's defaults the process grew to 24 GiB over the long text, and to 15 GiB at the usual 4 GB cap, for 3% more time than with the cache.
- `mps` `memory_gb` is 11.1.
- Listening round (owner, loudness-matched, not blind): MLX preferred on all three long-text pairs; the three cloned-voice pairs were equal.
