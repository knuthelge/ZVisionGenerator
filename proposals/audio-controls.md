# Steer audio before it's generated: settings and an enhancer

**Status:** In progress (2026-11-07): the build is done on branch `feat/audio-controls`; the Mac checks and the owner's listening backlog are still open (see below), and the proposal is archived after the Mac checks

## Still open after the build

**Mac checks:**

- **M-S1, Qwen3-TTS on MLX:** the model loads, the built-in voices and cloning work, the same seed gives the same take, and the memory figure for `qwen3_tts.platforms.mps.memory_gb` is measured.
- **M1:** the YuE2 score steering (tempo, key, silenced Vocal line) on MPS.
- **M2:** Delivery on MLX.
- **M3:** Stable Audio 3 on MPS.
- **M4:** the helpers (Describe it, Fill in, Lyrics and Script helpers) on the MLX 4-bit enhancer.

**The owner's listening backlog:** the `step1`, `qwen-gs` and `qwen-eager-smoke` takes decide which Delivery values are cut; the same listening settles Laugh, the Sound shell (`TrackType: SFX`) and the clone quality.

**After the Mac checks:** archive this proposal, and write the MiniMax Music 3 follow-up as a proposal of its own.

## Scope

The Audio tab gets what Image and Video already have: **settings that shape a take and an enhancer that writes the prompt**, so the user says what they want before Generate and the model follows it more reliably. Every take is still made from scratch.

**In**

- An enhancer per kind, built like the image and video one: pickable options plus a rewrite in the caption style each model follows best.
- Settings per kind that act on the model directly where it allows it: delivery for Speech (Higgs's tags), tempo and key for Music (YuE2's score).
- Fewer stray voices in instrumentals, if steering the score does it.
- A helper model is allowed if it is the enhancer's size or smaller (today Qwen3.5 4B) and works like it: settings in, the main model's input out.
- Stable Audio 3 as a second Music model, for instrumentals at an exact length (weights already downloaded for Sound).
- A lyrics helper for Music: write lyrics from an idea, or improve your own, with versions.
- First for our three models: YuE2 3B (Music), Higgs TTS 3 (Speech), Stable Audio 3 Medium (Sound). What is specific to each model is kept apart, so another model later means adding one adapter.

**Out**

- Anything that starts from an existing take: variations, extend, redo a region, loops, restyle, an instrumental version of a song.
- Anything that processes audio afterwards: even loudness, stems, noise cleanup.
- New large models: voice design (VoxCPM2), stem splitting (Demucs), transcription.
- Other or faster models. Each would be one more adapter later; none has been timed or heard here:
  - Speech models with open licences on MLX through mlx-audio (MIT), which the Mac already installs for Higgs TTS 3; on Windows and Linux each would need its own torch path. Qwen3-TTS (Apache-2.0; 0.6B and 1.7B, cloning from 3 s, nine built-in voices with emotion instructions, voice design, 10 languages), VoxCPM2 (Apache-2.0; voice design and cloning in 30 languages; lost 3–4 to Higgs in the earlier blind test), Ming-3.0 TTS (MIT; 0.5B, about real time on an M4 Pro), Dia 1.6B (Apache-2.0; English dialogue with `[S1]`/`[S2]` turns), Chatterbox multilingual (MIT, 23 languages) and Kokoro-82M (Apache-2.0, 54 voices, no cloning).
  - MiniMax Music 3, the one other song model in mlx-audio: 8B, section tags, up to 6 minutes, stereo at 44.1 kHz, a community licence that allows commercial use with conditions (YuE2 is CC BY-NC). CUDA only on MiniMax's card; the MLX port is young and its 4-bit copy is 9.2 GB. mlx-audio has no sound-effects model that fits 32 GB.
- Showing a take's score in the viewer (sections, bars and lyric lines under the waveform). Planning from a written header keeps the score, so this becomes possible: in a test, section starts from the score, stretched to the take's length, landed within half a second of the heard ones.
- Dialogue with several voices, and a soundtrack for a video.

## Findings

Researched 2026-11-04 across current open models of each kind and the tools users already use. Sources are listed per section; claims marked *unverified* came from secondary sources.

### Settings are interchangeable, the model's input is not

The same answer for all three kinds. What a user wants to say is shared across models of a kind and can be one settings vocabulary (and the enhancer's options). How it has to be written is different for every model: prose, comma lists or tag lists; each speech model's own tag syntax; tempo as a real field, a caption word or a score line; trained-in prefixes, suffixes and word limits; unknown speech tags read aloud. So: **shared settings per kind, and a small adapter per model** (roughly 50–100 lines) that renders them in the model's format, drops what it doesn't support and tells the UI which settings to hide or mark as approximate.

| | Shared settings (any model of the kind) | Specific to our model |
|---|---|---|
| Music | Genre, mood, instruments, singer (gender, timbre), song or instrumental, language, section list with lyrics. Tempo and key are real inputs only in some models (ACE-Step, MiniMax Music 3); elsewhere caption words with unmeasured adherence | **YuE2**: style is a comma descriptor list; tempo and key live in the ABC score header (`Q:`, `K:`), not the prompt; the planning mode (`full` for songs, `melody` for instrumentals); the kit accepts a finished score (`SongRequest.abc`) and skips planning; no length input, length follows the sections |
| Speech | Emotion (a small set) and intensity, pace, pauses, whisper or shout, common non-verbal sounds (laugh, sigh, breath, cough) | **Higgs TTS 3**: its exact 43-tag vocabulary in `<\|cat:value\|>` form (21 emotions, style, 4 speeds, pitch, pauses, expressiveness, 9 sounds); emotion, style and prosody tags go at the start, pauses and sounds inline; a sound tag must be followed by its onomatopoeia (`<\|sfx:laughter\|>Haha`); no free-text instruction, no voice design, no multi-speaker |
| Sound | A plain description of source, material, space, distance, motion and density; length in seconds | **Stable Audio 3**: caption shell `TrackType: SFX, …` with a `Length: X seconds` suffix and at most 45 words; length up to 380 s; distilled checkpoints run at CFG 1, so a negative prompt does nothing |

Our backends today pass only the raw text: YuE2 gets style and lyrics and discards the score it plans (`backends/audio/yue2.py` `_take`); Higgs gets the text as typed; Stable Audio gets prompt and length (`backends/audio/sa3.py`). `AudioGenerationRequest` has no settings beyond prompt, lyrics, length, voice and seed.

### Music models

- **YuE2-3B** (CC BY-NC 4.0): prompt is `instruction / [Tags] style / [Lyrics] lyrics`; style examples on the model card are comma descriptors ("Jazz, expressive lead vocal, piano, tenor saxophone, upright bass, brushed drums, no guitar"). No published caption provenance: **the earlier claim that YuE2 learned from captions written by Qwen is unconfirmed** (that is documented for ACE-Step). Until a source turns up, the enhancer copies the model card's style. https://huggingface.co/m-a-p/YuE2-3B
- **ACE-Step 1.5** (MIT): caption, lyrics, explicit BPM, key, time signature and duration; ships its own Qwen3 planner (0.6/1.7/4B) that writes metadata, caption and lyrics. Training captions: Gemini 2.5 Pro on a seed set, then a fine-tuned Qwen2.5-Omni captioner. https://github.com/ace-step/ACE-Step-1.5, https://arxiv.org/html/2602.00744
- **SongGeneration / LeVo 2** (Tencent): "comma-separated tags, not sentences", BPM written as "the bpm is 110", typed sections with `;`.
- **MiniMax Music 3** (open weights, Aug 2026, ~11B): structured caption (genre, BPM, key, arrangement), optional caption rewriter. Too big for 10 GB without slow streaming.
- **HeartMuLa 3B** (Apache-2.0): tags without spaces; **DiffRhythm** needs timestamped LRC; **JAM** word timings; **SongBloom** needs an audio prompt.
- No model publishes tempo or key adherence numbers.

### Speech models

- **Higgs TTS 3**: tags as above; EmergentTTS win rates 53.8% emotions, 68.6% paralinguistics. https://huggingface.co/bosonai/higgs-tts-3-4b
- Four unrelated control styles elsewhere: inline tags with clashing syntax (`[laughter]` CosyVoice, `(laughs)` Dia, `[laugh]` Chatterbox Turbo, free-form `[whispers]` Fish S2), free-text instructions (Qwen3-TTS, VoxCPM2 `(cheerful, slightly faster)Text`, Breeze TTS 2, Ming-3.0), emotion vectors (IndexTTS2, Zonos), reference audio only (F5, Kokoro, Dia2).
- Newer models converge on free-text instructions plus free-form tags, like OpenAI's `instructions` and ElevenLabs v3 audio tags.
- Adherence is partial everywhere: CosyVoice 3 hits sigh about 4% and emotion about 49% (https://huggingface.co/adventists-ai/tag-tts); Ming-3.0 claims about 73% emotion accuracy.
- Unknown tags are often spoken aloud, so an adapter must whitelist.

### Sound models

- **Stability's own SA3 prompt rewriter is public** (`stable_audio_3/interface/reprompt.py`, https://github.com/Stability-AI/stable-audio-3): Qwen3.5 2B, thinking off. A classifier picks music genre, instrument, SFX or one-shot; each has its own system prompt. The SFX prompt asks for source, material, space (indoor/outdoor, dry/reverberant, close/distant), evolution over time and motion; output is 1–2 sentences, at most 45 words, no brackets or markdown, no vocals, ending in `Length: X seconds` (impacts 1–3 s, actions 3–6 s, ambience 6–15 s); retried up to 5 times on failure; prefixed `TrackType: SFX, ` (an AudioSparx metadata tag). Trained on AudioSparx and Freesound metadata.
- Our Sound settings (type, space, perspective, length) map almost one to one onto that prompt; the SA3 adapter can be close to Stability's prompt with the settings filled in.
- No open model has a loop control (only post-processing or inpainting the seam, both out of scope).
- Others (TangoFlux, AudioLDM 2, MMAudio, MOSS-SoundEffect, MiDashengLM-Gen) use AudioCaps/Freesound-style English with their own shells and length ranges.

### What users meet in other tools

- **One shell, separate compose per kind** everywhere (Suno, ElevenLabs, Stable Audio, Adobe): a prompt with Enhance, kind-specific settings, several takes per press.
- **Baselines users expect:**
  - Music: Style, Lyrics with section tags, "write lyrics for me", Instrumental on/off, Duration with Auto, reference audio.
  - Speech: a script, a voice with preview, a few delivery settings or a free-text instruction (OpenAI.fm, Hume, Google AI Studio), inline tags (ElevenLabs v3).
  - Sound: a prompt, Duration with Auto, Loop, about 4 variations (ElevenLabs, Adobe).
- **Worth considering:**
  - Vibe presets that show and let you edit their instruction text (OpenAI.fm); matches our Style and Mood pickers.
  - An enhancer that fills only empty fields and never overwrites what the user typed (ACE-Step's Format).
  - A tag picker for speech, so nobody memorises syntax; an Enhance that inserts tags (ElevenLabs v3, *unverified*).
  - Auto defaults with pro settings one click away.
  - Lyric help on a selection: alternate lines, rhymes, "make this funnier" (Suno, *unverified*).
- **Anti-patterns users complain about:**
  - Settings that silently do nothing (Suno: exact BPM, gender tags, "no drums" in the style). Say when a setting is approximate.
  - Rewriting the prompt without showing what changed (Udio).
  - Lyrics or tags quietly overriding the style.
  - Auto length giving 1-second ambience.
  - Sliders whose extremes break the output; a consistency-versus-expressiveness trade-off left unexplained (ElevenLabs Stability).
  - A different voice on every regenerate (ElevenLabs v3); our *Any* voice has the same behaviour.

## Decisions (2026-11-04)

1. **Direct settings in Settings, caption choices in Enhance.** Settings that act on the model without a language model (Higgs's delivery tags, YuE2's tempo and key) live in the Audio tab's Settings panel and apply on every Generate, with or without the enhancer. Choices that only reach the model as caption words (genre, mood, instruments and singer for Music; space, distance and density for Sound) are Enhance options, as Style, Mood and Shot are for images and video.
2. **Best effort is enough.** A setting ships if it lands some of the time (50% is fine); it is cut only if it practically never lands or damages the take. No reliability badges or extra help text: users know generation is best effort, and the enhancer's rewrite is always visible.
3. **Tempo and key are both supported, after a GPU check.** The check tries the ways of feeding them in: as words in the style (the model card's way), by editing the planned score's `Q:` and `K:` header lines (with the notes transposed for key), and both, giving most weight to how YuE2's authors intend it. It also checks that instrumentals stay clean through an edited score, and how well Stable Audio follows tempo and key. Only a total failure drops one.
4. **Stable Audio 3 becomes a second Music model, and all four of Stability's prompt types are used.** Under Music, Stable Audio makes instrumentals at an exact length (up to about 380 s), fast, including loops and beds at a set tempo; choosing it hides Lyrics and shows Length. The Music Enhance options are shared by both models, and each model's adapter renders them. Sound uses the effect and one-shot types, with a sound type option (Auto, Ambience, Effect, One-shot) that also suggests a length when none is set. Stability's published rewriter is a starting point and the baseline to beat, not a limit: we write our own instructions and compare on the enhancer bench (`bench/enhance`).
5. **Music gets two helpers, one per box.** Enhance on the Style box works on the sound (genre, mood, instruments, singer) and never touches Lyrics; tempo and key are settings (decision 1), not Enhance options. A lyrics helper on the Lyrics box can **write from an idea** (lyrics with section tags, or an instrumental's section list on request) or **improve these lyrics** (tighten, fit to sections, smooth rhymes), with a version stepper like the prompt box's rewrite history. An empty Lyrics box still means instrumental. Line-by-line suggestions or a rhyme finder are a possible later proposal. The concept phase decides whether Speech gets the same split (write a script from an idea, add delivery to a script).

## Concept (2026-11-04)

Three concepts were written from different starting points (the user's goal, the models' levers, the app's existing patterns) and agreed on the core. This is the combination: the app's patterns as the skeleton, the model levers as the contents, and the best ideas from the user-goal concept. Values and groupings are a starting point for the design round, not final.

### The three layers, as in Image and Video

- **Direct settings** in a Settings section per kind, with a header summary ("Auto" or "92 BPM · A minor"), reset marks and Auto as the unset state, as Seed shows "random". They apply on every Generate. The enhanced text never contains them: the adapter adds them at Generate, so changing a setting never makes a rewrite out of date (Sound's Length is passed to Enhance as context only).
- **Enhance** on the main text box (Style for Music, Prompt for Sound; Speech has the Script helper instead): the existing popover with per-kind options from the server's option contract, Keep defaults, Direction, Reset and Fast/Smart; the rewrite lands in the Enhanced tab with Use mine, Re-enhance, Out of date and the five-rewrite stepper. Enhance rewrites the user's text but keeps facts exact: names, dates, places and quoted text are copied word for word.
- **A helper on the second box** (Lyrics for Music, the script for Speech), with the same tabs, stepper and status line as the Enhanced tab.

### Music (YuE2 3B and Stable Audio 3)

- **Settings:** **Tempo** (Auto, or 40–220 BPM), **Key** (Auto, or the 24 keys grouped major and minor), **Length** in seconds for Stable Audio only (default 30, up to 380). Song or instrumental still follows the Lyrics box; choosing Stable Audio hides Lyrics.
- **Enhance options on Style:** **Genre** (one, grouped: Pop & rock, Electronic, Hip-hop & R&B, Jazz & blues, Folk & country, Classical & cinematic, World, Lo-fi & ambient), **Mood** (one, reusing the image Mood options where they fit), **Instruments** (any) with a **Without** row ("no guitar", as YuE2's model card writes it), **Singer** (Female, Male, Duet, Choir), hidden for Stable Audio and instrumentals, and **Production** (Polished, Lo-fi, Live, Vintage, Cinematic).
- **Lyrics helper:** **Improve lyrics** (Tighten, Fit the sections, Smooth rhymes, Simpler words, plus Direction), on the whole song or one section; a section's **Write** fills an empty section to fit the rest. Writing lyrics from an idea is Describe it's job (see Design). The Style and its Enhance options are context, so the lyrics match the style.
- **Switching the Music model marks the style rewrite out of date**, since each model wants a different caption shape.
- **Adapters:** YuE2 sends a comma descriptor list and writes Tempo and Key at the top of the score before YuE2 plans it, so it composes in that tempo and key; for an instrumental it also silences the score's Vocal line (see the GPU check); Stable Audio sends one sentence of at most 45 words with its `TrackType` prefix and `Length` line, plus "no vocals".

### Speech (Higgs TTS 3)

- **Settings, a Delivery section after Voice:** a **Vibe** preset (Calm narrator, Storyteller, Excited host, Announcer, Sad confession, Whispered secret) fills five values shown under it, which stay editable: **Emotion** (Keep, then the 21 emotions in groups), **Manner** (Natural, Whisper, Shout), **Pace** (five steps), **Pitch** and **Expressiveness** (Low, Normal, High). The adapter applies them to every part of a long text; a tag typed in the text wins for its sentence, and the adapter puts the Delivery tags back at the next one.
- **Insert menu** at the caret, reusing the LoRA trigger-word menu: Pause, Long pause, Cough, Sigh and Laughter (each inserted with the model card's written-out sound, `<|sfx:laughter|>Hehe`), and an emotion or manner for one sentence. ++alt+enter++ opens the same menu at the caret, as it does for `{a|b}` choices in prompts. Only Higgs's 43 tags can be inserted.
- **Script helper:** **Fit to N seconds** and **Add delivery** (marks up a script with tags from its meaning, never changes a word). Output is checked against the tag whitelist, because an unknown tag is read aloud.
- **Reading time** in the status line ("1,180 characters · about 1:20"), measured on Higgs at each Pace.

### Sound (Stable Audio 3)

- **Settings:** **Type** (Auto, Ambience, Effect, One-shot) and **Length** (Auto shows the number it will use from the type: about 2 s for a one-shot, 5 s for an effect, 12 s for ambience; typing locks a number).
- **Enhance options:** **Space** (Small room, Large hall, Street, Open country, Forest, Underwater), **Distance** (Close, Medium, Far), **Over time** (Steady, Builds, Hit then tail), **Motion** (Still, Passing, Approaching, Receding).

### Layouts

- **Desktop:** sidebar Compose over Settings, as today. Each text box has a tools row (Style: Enhance, Recents; Lyrics: Lyrics helper; Text: Script helper, Insert); popovers anchor to their button.
- **Tablet:** Compose and Settings side by side under the stage; popovers become sheets that show the box's text at the top.
- **Phone:** the chip strip becomes Kind, Model, then Music: Tempo, Key (Stable Audio: Length); Speech: Voice, Delivery; Sound: Type, Length; then Seed, Batch, All. Enhance, the helpers and Insert open as full sheets.

### Left out

Start from goal chips, a "Sends" line saying what goes to the model (the settings and the Enhanced tab already show it), tags drawn as chips inside the text box, a rhyme finder and line-by-line suggestions, reliability badges, a tap-tempo button, Timbre, Stable Audio's two music prompt types as a choice, Sound's Material, Density and Character, a batch that changes with the sound type, a negative prompt, enhancing each take separately, reference audio, dialogue with several voices, saved user presets.

## GPU check (2026-11-05)

Every take was listened to in one sitting; tempo and key were also measured from the audio.

- **YuE2 tempo and key are written into the score header before it plans.** Three ways were tried. Style words got near the tempo (90 asked, 96 measured) but not the key (A minor asked, E measured). Editing a finished plan's `Q:` and `K:` lines with the notes transposed landed both (89 BPM, G), but transposing keeps major or minor, so a major plan can't become A minor. Writing the header (`Q:1/4=90`, the voice lines, `K:Am`) and letting YuE2 plan the rest landed the tempo and the key in either mode, on two seeds, and sounded good by ear; it needs no transposer. It sometimes plans without an intro or outro. With Key set and Tempo on Auto, YuE2 plans once to pick a tempo, then again with the header.
- **The same seed gives the same take.** A song made twice with the same settings and seed came out identical, so Reuse reproduces a take.
- **Redoing one verse makes a new song.** With one verse rewritten and the same seed, a new plan changes the song from the first note; reusing the old plan keeps tempo, key and melody, but the performance drifts after a few seconds. By ear the feel, vibe and voice stay, not the build-up. So lyric versions simply make a new take; keeping the other sections exactly would mean continuing from the old take, which is editing a take (out of scope).
- **YuE2 instrumentals need the Vocal line silenced.** The planned score of an instrumental still has a Vocal melody, and the take had a voice in it. With that line replaced by rests (the bar count kept), the voice was gone.
- **Stable Audio follows tempo as words** (80 asked, 81 measured; 140 asked, 144 measured); `TrackType` captions follow too. Key lands about half the time, which is enough under decision 2.
- **Higgs:** all 21 emotions, Whisper, Shout, all prosody tags and both pauses work; Sing is weak and is left out. Of the nine sounds, cough, sigh and laughter work and go in the Insert menu; laughter lands best written the model card's way (`<|sfx:laughter|>Hehe`) or after an amused emotion, and never as the tag alone. Burping, humming, sniff, sneeze, crying and screaming are left out: a second try wrote each four ways (the model card's spelling, the tag alone, mid-sentence, after a fitting emotion) on two seeds, and none of the 48 takes landed. Pace is clearly audible even though the total length barely changes (26.7 s very fast, 27.9 s very slow), so the reading time uses one rate, about 15 characters a second.
- **Higgs over a long text:** delivery tags at the start of each part only and tags on every sentence both work; the start of each part wins, since the text stays easier to read.

## Design (2026-11-05)

Eight live design pages were reviewed side by side. The choices:

- **Direction: Brief first**, with Studio's song blocks for lyrics. Music and Speech start with a **Describe it** box ("a sad folk song about leaving the coast"). **Fill in** sends it to the enhancer and fills only the fields left empty (Music: Style, Lyrics, Tempo, Key; Speech: Text and Vibe), marking each filled field with ✦; then the user adjusts as in the concept above. Sound has no Describe it box: its Prompt already is the brief, and Enhance covers it.
- **Lyrics as section blocks:** a song is a list of blocks (Verse, Chorus, Bridge…) that can be dragged, duplicated, added and removed; the block list is written out as the tagged lyrics the model gets. The **Lyrics helper is a popover** (as Enhance), opened on the whole song or on one block, with a version stepper on whatever it changed.
- **Tempo and Key in Settings**, in a Music section, as Steps and Seed are for images.
- **Speech delivery as Vibe presets** with their five values shown and editable (see Speech above).
- **Insert menu** button on the Text box, plus ++alt+enter++ at the caret.
- **Sound Type as a segmented control** (Auto, Ambience, Effect, One-shot); Length shows the number Auto will use.
- **One Enhance per box**, as on Image and Video.
- **The writing tools have one job each** (a second design round, after the plan review found them overlapping):
  - **Describe it with Fill in is the only place that writes from an idea.** It fills the empty fields (Music: Style, Lyrics, Tempo, Key; Speech: Text and Vibe) and marks them ✦, with Undo fill. New lyrics from a different idea mean editing Describe it and pressing Fill in again.
  - **The Style Fill in writes lands in Style's Enhanced tab as a version**, so Enhance adds the next version beside it instead of rewriting a rewrite; Fill in and Enhance share one version history, and Generate uses Enhanced, as on Image.
  - **The Lyrics and Script helpers only work on what's there** (Improve lyrics; Add delivery and Fit to time). On an empty song or script they say so and offer Fill in.
  - **Each tools row says what its tool does** in a short line ("rewords Style for the model", "polishes your words; new lyrics come from Describe it").
  - **Describe it folds to one line** after Fill in, and from the start when the user begins with their own lyrics or script.
- **No "Sends" line.** The user picks settings and generates; the settings and the Enhanced tab already say what goes in.

Layouts: on desktop the Describe it box sits at the top of Compose and the song blocks fill the Lyrics area; on tablet the same column sits beside Settings; on phone the blocks stack full width, reordering moves to each block's menu (Move up, Move down), and the helpers, Enhance and Insert open as full sheets.

## Plan review (2026-11-05)

A peer review, a code review and a simulated user test checked this plan against the code. What it adds to the build:

- **Settings are saved and restored with the take.** The request gains the new values (tempo, key, the five delivery values actually used, sound type and the length Auto picked), and they go into the FLAC's settings, Gallery Reuse, Retry, Load settings and Copy CLI together; otherwise reusing a take silently drops them. Audio takes also get the `prompt_input` and `enhance` record images have. The CLI gets `--tempo`, `--key`, the delivery values and `--sound-type`.
- **One model can serve two kinds.** A preset lists its kinds (Stable Audio: sound and music); the request carries the kind the user picked, and lyrics and length rules come from the model and the kind together, in the backend, the CLI (`--kind`), the model picker and Reuse. Stable Audio under Music ships with its UI, so the picker never shows a half-built entry.
- **Audio gets its own enhancer module**, as video has: its own option axes and token budgets (a long song's lyrics or a two-minute script don't fit today's 300-word cap), several fields per answer for Fill in, and an output cleaner that keeps Higgs tags, which today's cleaner strips. The enhancer and the audio models never share the GPU (a running job blocks the enhancer), so the helpers are unavailable while a take is made, and the UI says so.
- **Higgs: split first, then tag.** One function makes the parts for the backend, the request check and the runner: it splits the plain text, adds the Delivery tags to each part, and keeps tags out of the reference transcript the later parts use and out of the length estimate. Tags typed by the user are checked against the 43 known ones at Generate, since an unknown tag is read aloud. Higgs now runs on two engines (torch on Windows and Linux, mlx-audio on macOS) that share the part logic in `backends/audio/higgs3_takes.py`, so split-then-tag lives there, and whether the tags land on the MLX engine is a macOS check.
- **The YuE2 score code is its own pure module** (`backends/audio/yue2_score.py`): it writes the header to plan from and silences the Vocal line of an instrumental, tested against real planned scores; when a score can't be read or was cut short, the take uses the unedited score with a warning. Planning from a written header calls the kit's token functions from our backend; the vendored package is not edited.
- **Stable Audio:** the user's caption is never cut to 45 words (the enhancer aims for it instead), a `TrackType:` or `Length:` already in the text isn't added twice, and Sound Type Auto uses a fixed rule when the enhancer isn't used. The new caption shape changes takes made from older Sound settings; the changelog says so.
- **Song blocks are a view over the tagged lyrics text**, with Edit as text, so lyrics from Recents, Reuse, `--lyrics-file` or a paste (text before the first tag, unknown sections such as `[Pre-Chorus]`) are never lost. Reordering uses Move up and Move down on every layout first; drag can come later.
- **The Insert menu and ++alt+enter++** come out of the prompt box into a shared part, since the audio text boxes are plain text areas today.
- **macOS:** the enhancer bench runs for each audio mode on both platforms (the Mac enhancer is a 4-bit MLX model), and one score-edited YuE2 take runs on MPS before step 1 closes.

## Two more models to support (2026-11-05)

The owner listened to MiniMax Music 3 and Qwen3-TTS and wants both supported: everything this plan adds must work for them too, and each gets a test after the build (Qwen first, see below). The settings stay shared; what differs goes in each model's adapter.

**MiniMax Music 3 (Music): the plan fits as it is.** Sources: the [model card](https://huggingface.co/MiniMaxAI/MiniMax-Music3), the official [caption-rewriter skill](https://github.com/MiniMax-AI/MiniMax-Music3/blob/main/skills/music-caption-rewriter/SKILL.md), and mlx-audio's port (in its PyPI releases from v0.6.0).

- **Tempo and Key** go into the caption's metadata ("BPM: 96", "Key: A minor"): a hint, not a hard setting as in YuE2's score, which decision 2 accepts. The official rules say never to invent a BPM or key, so Auto writes none.
- **The caption** has three headings (Global Metadata, Vocal Details, Arrangement), about 250–450 English words; the enhancer writes this shape for MiniMax. Singer and the Without row become caption words; there is no negative prompt.
- **Lyrics:** every section tag on its own line (mlx-audio drops the words on a line that starts with a tag). It knows `[Intro] [Verse] [Pre-Chorus] [Chorus] [Post-Chorus] [Bridge] [Instrumental] [Solo] [Outro]`, and song blocks offer these. Empty lyrics are refused, so an instrumental is sent as `[instrumental]` with "instrumental" and a lead instrument in the caption.
- **Length** is a maximum (up to 6 minutes); the song ends where the lyrics end.
- Seed works (MLX and torch don't match each other). Caption and lyrics together are capped at 5,000 tokens.
- **Licence:** the MiniMax-Music3 community licence allows commercial use; "MiniMax-Music3" must be shown in the interface, and revenue over USD 20M a year needs permission.
- **For its test:** on a 10 GB card it only runs with the model streamed from system memory (guides call 12 GB the floor; nobody has timed it); on a 32 GB Mac the 8-bit copy (13.9 GB) is the one that keeps the lyrics accurate. Output is 32 kHz stereo on torch and 44.1 kHz through mlx-audio.

**Qwen3-TTS (Speech): it steers very differently from Higgs.** Sources: the [repository](https://github.com/QwenLM/Qwen3-TTS) and its issues, mlx-audio's port.

- One free-text instruction per call for the whole text ("Speak calmly, slowly, a little sad"); no inline tags, no SSML, no sound effects (laughter only "implicitly" through "haha" in the text). Delivery changing mid-text means a new call.
- Instructions work only on the 1.7B model with its nine built-in voices (English: Ryan, Aiden); the 0.6B model drops them. Cloning is a separate model that takes no instruction; a clone-plus-instruction model is announced but unreleased. A voice-design model makes a new voice from a description.
- No seed parameter (the global seed is set, reproducibility to be checked). Reference code is slow on a 10 GB card (about 4–5× slower than real time); the faster-qwen3-tts runtime (MIT) reaches about 1.8× real time. 1.7B in bf16 is about 8 GB.
- **Licence:** Apache-2.0, free for any use. Higgs TTS 3 is non-commercial plus a creator grant (published content may be monetised with a credit), and embedding it in an app for others needs Boson's commercial licence.

**Decided:** Sing is dropped from Manner (it was weak on Higgs too).

### Qwen3-TTS replaces Higgs (blind test, 2026-11-06)

The owner rated 17 loudness-matched pairs blind: **Qwen3-TTS won 12, Higgs 0, and 5 were the same.**

| | Qwen | Higgs | Same |
|---|---|---|---|
| Narration (~45–65 s) | 4 | 0 | 0 |
| Delivery (sad, angry, excited, whisper, slow) | 5 | 0 | 0 |
| Long text (~2 min, read in parts) | 2 | 0 | 0 |
| Pause | 1 | 0 | 0 |
| Laugh | 0 | 0 | 1 |
| Voice copy (both models cloning both references) | 0 | 0 | 4 |

Qwen ran through faster-qwen3-tts (CUDA graphs, bf16) at 0.33–0.41 s per second of audio against Higgs's 1.05–1.5 s, peaked at 4.4–5.9 GB on the card against 8.0–8.4 GB, and gave bit-identical audio for the same seed. Its tokenizer failed to build on transformers 5.22 and worked on 5.15; the app needs 5.19 or later, so the build has to solve that.

**Decided:**
- **Qwen3-TTS replaces Higgs, first, on this branch** (build step S): a Qwen backend on both platforms (faster-qwen3-tts on Windows and Linux, mlx-audio on macOS), the nine built-in voices and cloning, long texts in parts; Higgs's code, config, licence entries and docs go.
- **Delivery is one instruction sentence per call**, built from the Vibe's values (Emotion, Manner Natural/Whisper/Shout, Pace, Pitch, Expressiveness), for the built-in voices.
- **A cloned voice reads as recorded:** Delivery is hidden for it, with one line saying so. Revisit when Qwen releases its steerable cloning model.
- **The Insert menu** puts readable marks in the text: Pause and Long pause (rendered as "…" and a line break), Laugh ("haha"), and Delivery from here (`[sad]`, `[whisper]`…), which splits the text there and starts a new call with that instruction. Alt+Enter opens it at the caret. The Script helper's Add delivery writes the same marks.
- Higgs's tag findings above stay as a record; the 43-tag whitelist and split-then-tag no longer apply.


### Qwen setup check (2026-11-06)

Spike S0: how Qwen3-TTS runs on the app's transformers. **Decision: option (b), a three-part shim in our backend, on the lock's transformers (5.21.0, also 5.22.1). No pin; option (a) does not work.** The spike used a scratch project on Python 3.14.4 (torch 2.15.0+cu132, `faster-qwen3-tts==0.5.4`, `qwen-tts-hf==0.1.1.post1`, `transformers==5.21.0`, accelerate 1.15.0), scripts kept outside the repo, and the two cached official snapshots (CustomVoice, Base).

**Dependencies.** `qwen-tts-hf` pins `accelerate==1.12.0` and requires gradio; the override lifts accelerate and drops gradio with a marker that is never true. In the app's `pyproject.toml`:

```toml
dependencies = [
    ...
    "faster-qwen3-tts==0.5.4; sys_platform != 'darwin'",
    "qwen-tts-hf==0.1.1.post1; sys_platform != 'darwin'",
]

[tool.uv]
override-dependencies = [
    "accelerate>=1.15.0",
    "gradio; python_version < '0'",
]
```

The app's own pyproject with these lines locks (149 packages, transformers stays in the existing 5.21 lock). Resolved wheel-only for Linux x86_64 and Windows amd64 on Python 3.14: librosa 1.0.0 (pure), numba 0.68.1 and llvmlite 0.50.0 (cp314 wheels on both), onnxruntime 1.31.1 (cp314 on both), torch/torchaudio from the cu132 index. `sox` has no wheel for 1.5.0 (sdist only, pure Python; 1.4.1 has one); uv builds it. `import qwen_tts, faster_qwen3_tts` works; it prints a harmless "SoX could not be found" (the `sox` binary is not used) and a flash-attn notice.

**What breaks without the shim, and the shim.** Three separate faults, each shown by switching its part off (5.21.0, CustomVoice):

| Parts applied | Result |
|---|---|
| none | `AttributeError: property 'upsample' of 'Qwen3TTSTokenizerV2Encoder' object has no setter` |
| 1 + 2 | `AttributeError: 'MimiEncoderConfig' object has no attribute 'rope_theta'` |
| 1 + 3 | loads without error but the encoder is untrained: 125 missing keys, encode codes 100% different |
| 2 + 3 | the `upsample` error again |
| 1 + 2 + 3 | loads; 0 missing, unexpected, mismatched |

1. **Read-only Mimi properties.** transformers 5.21 made `MimiModel.upsample` and `.decoder_transformer` read-only; class attributes `upsample = None` and `decoder_transformer = None` on `Qwen3TTSTokenizerV2Encoder` shadow them.
2. **Key renames.** The encoder is custom code and skips the built-in `MimiModel` renames, so `register_checkpoint_conversion_mapping("Qwen3TTSTokenizerV2Encoder", [...])` takes MimiModel's six `WeightRenaming` rules (`conversion_mapping.py:778-785`). It must be registered once (a second call raises unless `overwrite=True`).
3. **The "MimiConfig.rope_theta" error, which the plan did not have.** Importing `qwen_tts` adds a `"default"` entry reading `config.rope_theta` to transformers' global `ROPE_INIT_FUNCTIONS`. From 5.20.2 on (the earliest version tried above 5.15.1), `PreTrainedModel._init_weights` lets the registry win over each model's own `compute_default_rope_parameters`, and the configs keep `rope_theta` inside `rope_parameters`. So the error is not a 5.21 effect: it appears on **5.20.2 too** (`'MimiConfig' object has no attribute 'rope_theta'`) and is why the research notes pinned 5.15.1. It is also not limited to Qwen: **after `import qwen_tts`, loading an unrelated default-RoPE model in the same process fails the same way** (a tiny Llama, on 5.20.2, 5.21.0 and 5.22.1; fine on 5.15.1 and when the entry is removed). The enhancer and other transformers models in the app process would break after the first Qwen load. The shim removes the entry from the global registry and gives `qwen_tts`'s own rotary classes (and a private registry copy in its two modeling modules) the function instead; the Llama check passes afterwards on 5.21.0 and 5.22.1.

```python
def apply_qwen_shim() -> None:  # once, after `import qwen_tts`, before any from_pretrained
    import sys
    from qwen_tts._transformers_compat import _default_rope_parameters
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Encoder
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightRenaming
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    Qwen3TTSTokenizerV2Encoder.upsample = None
    Qwen3TTSTokenizerV2Encoder.decoder_transformer = None
    register_checkpoint_conversion_mapping(
        "Qwen3TTSTokenizerV2Encoder",
        [
            WeightRenaming(source_patterns=r"^encoder\.layers", target_patterns="encoder.encoder.layers"),
            WeightRenaming(source_patterns=r"^encoder_transformer\.", target_patterns="encoder.encoder_transformer."),
            WeightRenaming(source_patterns=r"^downsample\.", target_patterns="encoder.downsample."),
            WeightRenaming(source_patterns=r"^upsample\.", target_patterns="decoder.upsample."),
            WeightRenaming(source_patterns=r"^decoder_transformer\.", target_patterns="decoder.decoder_transformer."),
            WeightRenaming(source_patterns=r"^decoder\.layers", target_patterns="decoder.decoder.layers"),
        ],
    )
    if ROPE_INIT_FUNCTIONS.get("default") is _default_rope_parameters:
        del ROPE_INIT_FUNCTIONS["default"]
    private = {**ROPE_INIT_FUNCTIONS, "default": _default_rope_parameters}
    for name in ("qwen_tts.core.models.modeling_qwen3_tts", "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2"):
        module = sys.modules[name]
        module.ROPE_INIT_FUNCTIONS = private
        for cls_name, cls in vars(module).items():
            if isinstance(cls, type) and cls.__module__ == name and ("Rotary" in cls_name or "Rotatory" in cls_name):
                cls.compute_default_rope_parameters = staticmethod(_default_rope_parameters)
```

The private `qwen_tts._transformers_compat._default_rope_parameters` import is safe because `qwen-tts-hf` is pinned with `==`; a contract test should fail if it moves.

**Loading info.** `from_pretrained(..., output_loading_info=True)` (recorded for every nested load by wrapping `PreTrainedModel.from_pretrained`, since `qwen_tts`'s own wrapper cannot return the tuple), CPU float32, full shim:

| transformers | Snapshot | Whole model | Speech tokenizer |
|---|---|---|---|
| 5.21.0 | CustomVoice | 0 / 0 / 0 | 0 / 0 / 0 |
| 5.21.0 | Base | 0 / 0 / 0 | 0 / 0 / 0 |
| 5.22.1 | CustomVoice | 0 / 0 / 0 | 0 / 0 / 0 |
| 5.22.1 | Base | 0 / 0 / 0 | 0 / 0 / 0 |

(missing / unexpected / mismatched, no error messages; the Base whole-model load includes the speaker encoder.) The loader reads `speech_tokenizer/` from inside each snapshot (`<snapshot>/speech_tokenizer`, `local_files_only=True`), so the separate Tokenizer-12Hz repository is not needed. All submodels run `sdpa` in every venv.

**Parity on CPU float32** against the blind-test venv (transformers 5.15.1, same `sdpa` everywhere): seeded 100 x 16 codes decoded, a synthetic 5 s 24 kHz wave encoded, greedy `generate_custom_voice` (Ryan, English, no instruct, `do_sample=False`, `subtalker_dosample=False`; 59 frames).

| | Decode max abs diff | Encode codes differing | Talker codes | Talker wave max abs diff |
|---|---|---|---|---|
| 5.21.0 + shim, CustomVoice | 0 | 0% | equal | 0 |
| 5.22.1 + shim, CustomVoice | 0 | 0% | equal | 0 |
| 5.21.0 + shim, Base | 0 | 0% | n/a | speaker embedding diff 0 |
| 5.22.1 + shim, Base | 0 | 0% | n/a | speaker embedding diff 0 |

Bit-identical, far inside the limits (1e-4; 0.5%). The control without part 2 gives 100% different encode codes, so the comparison does detect an untrained encoder.

**GPU smoke** (one queued job, 5.21.0 + full shim, `FasterQwen3TTS.from_pretrained(..., device="cuda", dtype=bfloat16, local_files_only=True)`, CustomVoice): loads in 2.5 s, warms up the CUDA graphs in 2.9 s, reads one 5.6 s line (Ryan, seed 1) in 2.1 s (0.38 s per second of audio), no error, audible level (RMS 0.06). Peak 4,379 MiB allocated, 4,538 MiB reserved, 4,882 MiB seen by `nvidia-smi` (the blind test measured 4,638 to 5,936 MiB in `nvidia-smi`). Loading info on the GPU path again 0 / 0 / 0 for both models.

**Option (a) checked and rejected.** No locked or resolved dependency needs transformers 5.21 or later: the app's pyproject resolves for Linux, Windows and macOS with `transformers>=5.19.0,<5.21` (5.20.2; mlx-lm, mlx-vlm and mlx-audio included). But 5.20.2 without the shim fails to load both snapshots with the `rope_theta` error above, so a pin below 5.21 fixes nothing by itself; only 5.15.1 (below the app's floor) loads unpatched. Part 3 alone makes 5.20.2 load with no missing keys, so the shim is needed on any version from 5.20 up.

**For S1.** The backend applies the shim once before the first load (the loading-info hook is spike tooling, not part of the backend). A contract test asserts that, after the shim, `"default"` is absent from `ROPE_INIT_FUNCTIONS`, `Qwen3TTSTokenizerV2Encoder` carries both class attributes, and the conversion mapping is registered exactly once (the backend guards the call, because the snippet itself raises on a second registration). Not covered by the spike: bf16 on 5.22.1 and CUDA (the smoke ran on 5.21.0 only), the cloning path on the GPU, and the app's test suite under 5.20.


### GPU checks for step S (2026-11-06)

- **G-S1 parity: pass for built-in voices.** The app's `qwen3_tts` backend (transformers 5.21 with the shim, language English) re-made three blind-test takes (12.3 s, 66.8 s in 2 parts, 138.9 s in 4 parts) bit-identical to the files the owner rated, so part splitting, per-part seeds, sampling and the join match exactly. The cloned take differs by design: the app reads the recording itself (read at twice the rate, resampled, 0.5 s of silence appended) where the blind test passed the file to the library, so it is a different sample of the same voice (6.64 s against 6.40 s, no word errors in either). A loudness-matched blind pair is with the owner.
- **G-S2 language: with the owner.** Two lines, Ryan, seed 1: `English` reads 13–15 % faster than `auto` (5.4 against 6.2 s, 7.8 against 9.0 s), both word-perfect; one `English` take came out 8.5 dB quieter than the rest.
- **G-S3 checkpoint switch: pass.** Built-in voice, recording, built-in voice, built-in voice in one session: peaks 4.44, 4.83, 4.48 and 4.48 GB in torch, 4.98, 5.65, 4.97 and 4.97 GB for the process. The held checkpoint is freed before the next one loads (no growth), and the loading step shows only on the three real loads. The highest process peak across the checks was 6.05 GB on a 4-part take, so the preset's figure is raised to 6.0 GB.
- The same seed gives the same codec codes every time, but CUDA's codec decoder is not bit-exact: repeats differ at about −50 dB, which is inaudible.

### GPU checks for step 1 (2026-11-06)

Run through the app's own backends (YuE2, Qwen3-TTS with the built-in voice Ryan, Stable Audio 3) on CUDA with purpose-written texts; tempo and key measured with librosa (beat tracker, Krumhansl key profile). About 20 minutes of GPU time; memory peaks YuE2 6.3 GB, Qwen3-TTS 5.9 GB, Stable Audio 5.7 GB.

- **YuE2 tempo header: pass.** With tempo only (90, 120) YuE2 writes the `K:` line itself (A both times) and the takes measure 89 and 117 BPM. Tempo and key: 90 A minor measures 89 BPM (key reads as C, the same notes); 128 D major measures 129 BPM in D.
- **YuE2 instrumentals: pass on the score, the owner listens for a voice.** 100 D minor measures 103 BPM in D minor; 112 F measures 112 BPM in F. Every Vocal line of the score is rests.
- **YuE2 fallbacks: pass.** A key with tempo on Auto plans twice (E minor at the planned 142 BPM, B-flat at 108, both landing). A header plan forced to be cut off, and one forced to be unreadable, both fall back to YuE2's own plan with the note.
- **Delivery sweep: every value makes a take, none cut; the owner listens for values that never land.** Emotion alone moves the length of the same line from 6.6 s to 10.8 s (neutral 9.1 s). Pace, measured on a 329-character paragraph over six seeds against Normal (13.2 characters a second): very slow 0.72, slow 0.81, fast 1.17, very fast 1.22 (spread about ±0.15). These replace the provisional factors; 14 characters a second at Normal stays. The "very" steps add little.
- **Marks and pauses: marks split as designed; Long pause fails the measurement.** `[sad]`, `[whisper]` and `[as set]` give four parts with one instruction each. A line break adds a pause, but not a longer one than "…" (longest pause 0.46 against 0.56 s and 0.31 against 0.29 s on two sentences; longer in only 6 of 14 seeds). Unless the owner's blind listen disagrees, Long pause leaves the Insert menu. Laugh ("haha") waits for the owner's ears.
- **Stable Audio Sound: the owner's blind pairs decide** whether the `TrackType: SFX` shell helps (ambience, effect, one-shot, two seeds).
- **Stable Audio Music: tempo passes, key mixed.** 80 BPM measures 81, 140 measures 136 and 144, 120 measures 117; the key is right or its relative in 3 of 6. "No vocals" by ear.

## Build order

Each step ships with its docs, changelog entry and tests, on every platform.

S. **Qwen3-TTS in, Higgs out** (see above), so the speech steps build on Qwen.
0. **The request and saved-settings contract** and **models with two kinds**, as above.
1. **Adapters and settings in the backend:** the YuE2 score module (tempo, key, silenced Vocal line for instrumentals), Qwen's Delivery instruction and Insert marks, the Stable Audio caption shape, Sound Type and Length; the CLI flags.
2. **Settings in the Web UI:** Music Tempo, Key and Length, Stable Audio under Music, Speech Delivery with Vibes, Sound Type and Length, and the phone chips.
3. **The audio enhancer:** first the module, the draft state and the saved record, done with Sound; then Music on both models, compared on the enhancer bench.
4. **Insert menu** and ++alt+enter++.
5. **Song blocks**, then the Lyrics helper and the Script helper.
6. **Describe it and Fill in.**
