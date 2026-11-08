# Hear what was said: speech recognition with Whisper

**Status:** Proposed (2026-11-04)

## Idea

Add Whisper (OpenAI, MIT licence) so the app can turn speech into text with timestamps. Today you type what a voice recording says by hand, and nothing checks a speech take against its text.

## What the user gets

1. **What it says, filled in.** Dropping a voice recording on **Voice** (or `--voice` without `--voice-text`) fills in the transcript, which you can still edit.
2. **A check on speech takes.** A take whose transcript doesn't match its text (skipped or garbled words) gets a warning.
3. **Captions.** An SRT file next to a speech take or a video with speech.

## Evidence

In a test session making a music video and short films with the CLI, `openai/whisper-small` (244M, through `transformers`, already a dependency) on the CPU was used to check that narration stayed intelligible in a mix, to place narration lines by word timestamps, and to find speech a video model had invented in a clip's sound. It invents text over silence and music ("Thank you for watching"), so only its output on speech should be trusted.

## Open

- Model size: `whisper-small` on the CPU, or `large-v3-turbo` (809M) on the GPU through the job queue.
- macOS: the same `transformers` model on the CPU or MPS, or Whisper large-v3-turbo on MLX through mlx-audio (MIT), which the Mac already installs for Higgs TTS 3. mlx-audio also runs Parakeet TDT v3 (CC-BY-4.0). Neither has been timed here.
- Related: "Transcribe a voice recording automatically" in [Audio, the next round](archive/audio-controls.md) is item 1 here.
