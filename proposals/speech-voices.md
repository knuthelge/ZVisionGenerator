# Voices: one picker for built-in, saved and described speech voices

**Status:** Proposed (2026-11-07)

## Problem

The Speech **Voice** dropdown lists nine names and nothing else. It feels limiting, and three things in it mislead or cost the user work:

- The language beside a voice reads like a limit. Every voice reads all 10 languages; the label is only the voice's native language.
- There is no native English female voice. Ryan and Aiden, the only English entries, are both male.
- A cloned voice is not remembered. The recording, its transcript and the consent are set again in every session, and "keep one voice across separate takes" means finding the old take again.

## What the research found

- **Nine is all Qwen has released.** The open release has exactly 9 CustomVoice speakers and no new open checkpoint since 2026-01. The "49 voices" Qwen advertises are API-only. The README gives one line per speaker (https://github.com/QwenLM/Qwen3-TTS), which the app can show.
- **VoiceDesign makes a voice from a description.** Checkpoint `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign):
  - Apache-2.0, about 4.5 GB.
  - Same architecture as CustomVoice, so the VRAM should be similar (inferred, not measured).
  - Best on InstructTTSEval among the voice-design models Qwen lists, by Qwen's own numbers.
  - Already supported by the vendored `faster_qwen3_tts` (`generate_voice_design`) and by mlx-audio (`generate_voice_design`, with `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`), so both platforms are covered.
  - It re-creates the voice on every call, so two calls with the same description give two different voices. Qwen recommends "design, then clone" for a consistent voice: design once, then clone the result.
- **Cloning takes no instruction upstream.** The Base checkpoint has no `instruct`. faster-qwen3-tts injects one experimentally; whether it works is unproven.
- **Other models were checked and rejected.**
  - Kokoro's 54 voices are mostly graded weak.
  - Fish Drama-3 is too big and too restrictive.
  - Maya 2, Breeze-TTS-2, OmniVoice and Voxtral fail on licence or size.
  - VoxCPM2 already lost to Higgs, which lost 0–12 to Qwen.
- **Other apps** split it the same way: ElevenLabs (below) and the ComfyUI Qwen TTS nodes, which do "design, then save voice".

So the answer to "I want more voices" is not more built-in names; it is a library you build, from recordings and from descriptions, plus a clearer built-in list.

## Prior art: ElevenLabs

ElevenLabs has the most developed voice picker (https://elevenlabs.io/docs/product-guides/voices/voice-library, https://elevenlabs.io/docs/product-guides/voices/voice-design):

- **Voice Library:** a shared catalogue with filters for language then accent, category, gender (male, female, neutral) and age (young, middle, old), sorted by trending, latest or most users. Every voice has a click-to-preview, with previews in several languages.
- **Using voices:** "Use voice" works without saving. "+" saves a voice to My Voices, which then shows in every voice picker and can be grouped into Collections.
- **Voice Design:** a description plus preview text gives 3 candidates; you pick one and save it.

What we take: a one-line description per voice, a preview, saved voices in the same picker as the built-in ones, and Describe giving 3 candidates to pick, name and save.

What we leave out, because we have nine voices plus the user's own saved ones, not thousands: a shared catalogue, filters, trending and collections.

## Proposed change

One **Voice** control in the Speech settings, opening one picker with three parts: Built-in, Saved, and Describe. The picker is a popover on desktop, a narrower panel on a tablet, and a full sheet on a phone, as the other settings pickers are. In each layout the voice previews play inside it.

### 1. Built-in voices, made clear

- Each voice shows its name, the README's one-line description, its native language and a **Play** button for a short preview clip.
- One line at the top: *Every voice reads all 10 languages; the language shown is the voice's native one.*
- The nine clips are generated once with the app and shipped, so a preview needs no model load and no GPU. They are listed in the picker only; nothing else uses them.
- No English female voice exists, so none is shown. Describe (3) is how to get one, and the picker says so under the English voices.

### 2. Saved voices: a personal library

- **Save as voice** appears on a voice recording (in Voice) and on a speech take (**Use as → Voice** in the Gallery). The user names it.
- A saved voice is a reference WAV, its transcript and its name, kept in the app's data folder, not in the output folder, so it never shows up as a take. It plugs into the existing clone path (`VoiceRef`), so no backend changes.
- Saved voices list in the picker with a preview (the stored recording), **Rename** and **Delete**.
- **Consent (owner decision):** saving a voice made from a recording asks for the consent tick once, and the saved voice remembers it; using it later asks nothing. A voice saved from a built-in voice take needs none, as today. This reverses the current rule that consent is never stored, so the Concepts page and the Gallery note change with it.
- A saved voice is a clone, so Delivery stays hidden for it, with the line it has now.
- CLI: `--voice` accepts a saved voice's name as well as a path. A name needs no `--voice-text` or `--voice-consent`. A new `--save-voice NAME` saves the recording given to `--voice`, with the same consent rule. A name that matches no saved voice and no file fails with the list of saved names.

### 3. Describe a voice (VoiceDesign)

- A description box ("a warm British woman in her thirties, calm") and a preview line (default: a sentence in the chosen language, editable). **Make 3** generates three candidates; the user plays them, picks one, names it, and it goes to Saved.
- The preview line must be long enough for a reference (3 to 30 seconds today), and it becomes the saved voice's transcript, since it is known exactly.
- The picked candidate is saved as a recording and cloned from then on ("design, then clone"), so it sounds the same on every take. The three candidates differ from each other, which is the point of making three.
- No consent tick: a designed voice is the model's own, not anyone's voice.
- VoiceDesign downloads the first time it is used, with the usual size and free-space dialog, like the cloning files. It is a separate model, so it is not loaded for ordinary takes.
- The box hints at delivery words ("a calm narrator, speaking slowly") because a designed voice carries its delivery inside the description.

### 4. Delivery

Delivery stays for built-in voices only. Nothing changes until Base accepts an instruction (see the check below).

### Docs and changelog

- CHANGELOG `[Unreleased]`: the picker, saved voices, Describe a voice.
- Concepts (Audio, Voices): the picker, the all-languages line, the library and the changed consent rule; the Web UI and CLI pages link there. `docs/cli/reference.md` gets `--save-voice` and the new meaning of `--voice`.
- `docs/concepts/configuration.md`: the voice descriptions in `config.yaml` and the VoiceDesign files.

## Scope: what this leaves out

- **A filter or search** in the picker. Nine built-in voices and a personal library do not need one; the research puts the need at dozens. Revisit when a library grows past what fits on a screen.
- **Delivery on saved voices.** Wait for a proven instruction on Base.
- **Importing or sharing voices** between machines, a public voice library, tags and folders.
- **A language per voice** or any limit by language. There is none to enforce.
- **Voice mixing**, several voices in one take, and designing from an example recording.

The work is two steps that can ship apart: 1 and 2 (clearer built-ins, the library) need no new model; 3 needs the checks below first.

## Open questions and checks before building

- **VoiceDesign on the card:** VRAM and quality on 10 GB (it should be about the 6 GB of CustomVoice, which is only inferred), and on the Mac through mlx-audio, run through the GPU queue. Also how often "make 3" gives one usable candidate for a plain description.
- **Experimental instruct on Base:** a blind A/B of faster-qwen3-tts's injected instruction on a cloned voice, against no instruction. If it holds, Delivery can come to saved voices.
- **Mockups** for the picker in all three layouts from the UI designer, before any frontend work.
- **Filter at nine-plus voices:** only if the Saved list grows long in real use.
- **Where the library lives** and how a saved voice's files are named, settled at build with the data-folder conventions.
