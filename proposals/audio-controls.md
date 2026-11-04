# Audio, the next round: prompt help, controls and edits

**Status:** Proposed (2026-11-02)

## Idea

[The Audio tab](audio-generation.md) gives each kind one or two text boxes and nothing to steer the result. Images and video have the prompt enhancer, with options like Style and Mood. Audio could get the same, plus controls and edits. Which of these to support is still to be decided.

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
- **Ruled out by licence:** MMAudio, HunyuanVideo-Foley and the LTX Foley LoRA for video sound, and SheetSage2 for covers of outside songs.
- **Others:** Suno, Udio and ElevenLabs offer AI lyric writing, tempo and key in the style, auto-tagging of speech, voice design from a description, extend, inpaint and stems.

## Possible features

- An audio enhancer for each kind:
  - Music: genre, mood, tempo, instruments, voice, and lyrics written from a theme.
  - Speech: emotion, pace, pauses.
  - Sound: type, space, perspective, suggested length.
- Speech delivery pickers that add Higgs's tags.
- Tempo and key for music, set through the score.
- Fewer stray voices in instrumentals, through the score.
- Sound edits: variation, extend, redo a region, seamless loops.
- Restyle a song, including an instrumental version of it.
- Even loudness across takes.
- Voice design from a description (VoxCPM2, Apache-2.0), read by Higgs.
- Dialogue scripts with a voice per speaker.
- Transcribe a voice recording automatically.
- Clean a noisy voice recording.
- Split a song into vocals and instrumental (Demucs, MIT).
- A soundtrack for a generated video, made with Stable Audio at the clip's length.

The user's first leanings are the enhancers, speech delivery, tempo and key, cleaner instrumentals, Sound edits, Restyle, even loudness, voice design, dialogue, splitting and the video soundtrack.
