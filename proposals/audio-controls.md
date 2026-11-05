# Audio, the next round: prompt help, controls and edits

**Status:** Proposed (2026-11-02)

## Idea

[The Audio tab](archive/audio-generation.md) gives each kind one or two text boxes and nothing to steer the result. Images and video have the prompt enhancer, with options like Style and Mood. Audio could get the same, plus controls and edits. Which of these to support is still to be decided.

## Findings

- **YuE2 plans every song as an editable score,** with tempo, key, chords, a vocal melody and an instrument melody. We throw it away today.
  - Feeding back an edited score sets tempo and key far more reliably than text.
  - It can also re-arrange a song in a new style, keeping its melody and lyrics.
- **YuE2 learned from captions written by Qwen,** so our own Qwen enhancer can write its style text in a form it understands.
- **Stable Audio 3's weights, which we already download, can also make variations of a take, extend it and redo a region.**
  - Stability ships its own Qwen prompt rewriter.
  - Prompts in its training-caption style adhere best.
- **Higgs TTS 3 is steered only by 43 inline tags:** emotion, whisper or shout, speed, pitch, pauses, laughs and similar. They work today but sit hidden in the docs.
- **What doesn't work:**
  - Stable Audio ignores negative prompts.
  - YuE2 can't extend a song or redo one section.
  - Higgs can't design a voice from a description or voice several speakers.
- **Faster speech models with open licences run on MLX through mlx-audio (MIT),** which the Mac already installs for Higgs TTS 3. On Windows and Linux each would need its own torch path. None has been timed or heard here.
  - Qwen3-TTS (Apache-2.0): 0.6B and 1.7B models for cloning from 3 s, nine built-in voices with emotion instructions, and voice design from a description; 10 languages.
  - VoxCPM2 (Apache-2.0): voice design and cloning in 30 languages. It lost 3–4 to Higgs in the earlier blind test.
  - Ming-3.0 TTS (MIT): a 0.5B model with cloning, built-in voices and voice design, reported as about real time on an M4 Pro; a larger "tiny" model adds a two-speaker podcast mode.
  - Dia 1.6B (Apache-2.0): English dialogue with `[S1]`/`[S2]` turns.
  - Chatterbox multilingual (MIT, 23 languages) and Kokoro-82M (Apache-2.0, 54 built-in voices, no cloning) are small and fast.
- **MiniMax Music 3 is the one other song model in mlx-audio:** an 8B model with section tags, up to 6 minutes, stereo at 44.1 kHz. Its community licence allows commercial use with conditions, unlike YuE2's CC BY-NC. MiniMax's card names CUDA only; the MLX port is young, its 4-bit copy is 9.2 GB, and its speed and quality on a Mac are unknown.
- **mlx-audio has no sound-effects model** that fits 32 GB and no multi-stem separator (only vocals against the rest, from a converted checkpoint).
- **Ruled out by licence:** MMAudio, HunyuanVideo-Foley and the LTX Foley LoRA for video sound, and SheetSage2 for covers of outside songs.
- **Others:** Suno, Udio and ElevenLabs offer AI lyric writing, tempo and key in the style, auto-tagging of speech, voice design from a description, extend, inpaint and stems.

## Possible features

- An audio enhancer for each kind:
  - Music: genre, mood, tempo, instruments, voice, and lyrics written from a theme.
  - Speech: emotion, pace, pauses.
  - Sound: type, space, perspective, suggested length.
- Speech delivery pickers that add Higgs's tags.
- Tempo and key for music, set through the score.
- Keep the score with every song take, and show its sections, bars and lyric lines under the waveform in the viewer. In a test, section starts from the score, stretched to the take's length, landed within half a second of the heard ones.
- Fewer stray voices in instrumentals, through the score.
- Sound edits: variation, extend, redo a region, seamless loops.
- Restyle a song, including an instrumental version of it.
- Even loudness across takes.
- Voice design from a description (VoxCPM2, Apache-2.0), read by Higgs.
- Dialogue scripts with a voice per speaker.
- A faster speech model beside Higgs, and a song model with a commercial-friendly licence (MiniMax Music 3), both on the Mac first.
- Transcribe a voice recording automatically.
- Clean a noisy voice recording.
- Split a song into vocals and instrumental (Demucs, MIT).
- A soundtrack for a generated video, made with Stable Audio at the clip's length.

The user's first leanings are the enhancers, speech delivery, tempo and key, cleaner instrumentals, Sound edits, Restyle, even loudness, voice design, dialogue, splitting and the video soundtrack.
