# Prompt enhancer: audio, image awareness and model-aware rewrites

**Status:** Done (2026-10-11, v0.13.0b19) for items 1, 2, 3a and 5; items 3b, 4, 6 and 7 rejected

## Decision

- **Shipped:** 1 (Sound axis), 2 (clip length and framing), 3a (text-only reference-image rule), 5 (Direction).
- **Sound default:** no option picked. The enhancer keeps the user's own sounds and quoted speech and adds none.
- **Dialogue:** writes short lines when the user wrote none.
- **Direction UI:** one input with placeholder text at the bottom of the options, above the "Enhance each…" toggle. It has no label or help text.
- **Rejected:** 3b (vision model), too much for the gain. 4 (model-aware hints and the quoted-text check), not for now. 7 (previous rewrites).
- **Rejected:** 6. Length stays guidance and the enhancer never cuts output; going about 20 % over the target is fine. That is already the case: nothing trims a rewrite after generation.

## Problem

The enhancer (`utils/prompt_enhance.py`) builds one system prompt from five axes (Style, Mood, Details, Length, Motion) and the mode (`image` or `video`). It knows nothing else about the job. That leaves several gaps:

1. **It removes sound from video prompts.** LTX-2.3 generates audio with the video, and `--audio` is on by default (Web UI too). But the rules in `build_messages` say *"Describe only what is visible: no sounds, smells, …"*. So the enhancer never adds audio cues, and it may delete sounds the user wrote ("a dog barking", dialogue in quotes). That contradicts its own first rule, *keep every element the user mentions*. The audio track is then left to guess.
2. **It doesn't know the clip length.** A 97-frame clip is about 4 seconds, but with *Action / sequence* the enhancer often writes a multi-beat story that cannot fit. LTX then rushes, skips, or morphs between beats.
3. **It doesn't see the reference image.** Image-to-video and img2img rewrite the prompt without looking at the input (the video guide says so). For image-to-video the rewrite often re-describes the subject differently from the first frame, which works against the image.
4. **It treats all models the same.** Families want different prompts: LTX wants one chronological paragraph that starts with the main action; Ideogram 4, Klein and Z-Image render text, so quoted text must survive verbatim; FLUX.1 only gets a word ceiling (`enhance_max_words`). Nothing stops the enhancer from paraphrasing `"OPEN 24 HOURS"` on a sign.
5. **It has no free-form direction.** The only way to say "make it night" or "focus on the dog" is to edit the prompt and enhance it again.
6. **Length is not checked afterwards.** `plan_length` tells the model a range, but `enhance_prompt` accepts any usable output. A rewrite over the text-encoder ceiling is silently truncated by the encoder, usually cutting the end of the prompt.

## Proposed changes

Ordered by value for effort. Each item can ship alone.

### 1. Audio for video (Sound axis)

A new video-only, multi-select axis, used only when the job has audio:

| Slug | Label | Instruction to the model |
|---|---|---|
| `ambient` | Ambience | the ambient soundscape of the setting |
| `effects` | Sound effects | sound effects synced to the visible action |
| `dialogue` | Dialogue | speech: keep the user's quoted lines verbatim and say who speaks them, in what voice and tone; never invent lines unless the user's prompt implies speech |
| `music` | Music | music: whether it is in the scene or a score, its genre and tempo |

- **Rules change in `build_messages`.** In video mode with audio, replace *no sounds* with *"Describe what is seen and heard, in time order; no smells, thoughts or backstory."* Always keep sounds and quoted speech the user wrote, whether or not an option is picked. With `--no-audio`, keep today's rule (audio words would be wasted tokens).
- **Plumbing.** `EnhanceSettings.sound: tuple[str, ...]`; `build_messages(..., audio: bool)`; the CLI spec gains `sound=ambient+effects`, prompt files `enhance: {sound: [...]}`. The Web UI hides the axis when the Audio toggle is off, the same way Motion is hidden for images. `is_noop` counts Sound as a change in video mode with audio.
- **Default:** `ambient` + `effects` for video. Silence and generic noise are the usual LTX audio failure, and both options only describe sounds the scene already implies. This changes the output of existing video prompts that use `enhance: true`; see Open questions.
- **Cleanup:** `clean_output` only strips quotes that wrap the whole output and already skips text with inner quotes, so quoted dialogue survives. Add a test for it.

### 2. Clip length and framing in the prompt

Pass the job's shape into `build_messages`:

- **Video:** duration in seconds (`frames / fps`). A rule like *"The clip is about 4 seconds: describe one continuous action, at most two beats."*, scaled by duration (one beat under 5 s, two or three up to 10 s).
- **Image and video:** orientation from the ratio (portrait, landscape, square), so composition advice fits the frame ("full-length figure" in 9:16, "wide vista" in 21:9).

No new UI. Both values are already in the request.

### 3. Image-to-video and img2img: describe change, not appearance

Two steps, the first needs no new model:

- **3a. Text-only (cheap).** When the job has a reference image, tell the model: *"The first frame is supplied as an image. Describe the motion, camera and changes over time; refer to the subject briefly and don't restyle it or add appearance details."* For img2img, Style and Mood still apply, but the subject and layout come from the image. `enhance_prompt` gets `has_reference: bool`.
- **3b. Vision (spike first).** Qwen3.5 is natively multimodal. Feed the reference image so the rewrite can name what is actually in the frame. Needs `mlx-vlm` on macOS and the processor or VLM class in transformers on CUDA, and a vision-capable checkpoint. The current pinned `heretic` 4-bit MLX conversion may be text-only (the `*_text` fallback in `prompt_enhancer_mac._load` suggests it). Spike: find an uncensored Qwen3.5-4B checkpoint that keeps the vision tower on both platforms, and measure memory and the time per rewrite. Same machinery would serve Klein edits in [Edit images](../image-editing.md).

### 4. Model-aware rewrites

- **Preserve quoted text.** A permanent rule: *"Text in quotes is rendered literally: keep it verbatim, including case and punctuation."* After generation, check that every quoted span of the input appears in the output, and retry once if not (the same retry path `is_unusable` uses).
- **Per-family hints.** A preset key next to `enhance_max_words`, e.g.:

  ```yaml
  video_model_presets:
    ltx:
      enhance_hints: "Write one flowing paragraph in present tense. Start with the main action, then movement, then camera, then sound."
  model_presets:
    ideogram4:
      enhance_hints: "Describe typography explicitly: font style, placement and color of any text."
  ```

  `resolve_enhance_hints(config, family=, mode=)` mirrors `resolve_enhance_ceiling`. Users can override it in their config like any preset key. The Web UI enhancer endpoint already receives the model's `max_words`; it would receive the family instead and resolve both server-side.

### 5. A free-text Direction field

An optional one-line note added to the rules as *"User direction: …"* ("at night, in the rain", "focus on the dog"). It works together with the axes and counts as a change for `is_noop`.

- Web UI: one text field at the top of the Enhance popover.
- CLI: `--enhance-note "…"` (the comma-separated spec grammar can't hold free text safely).
- Prompt files: `enhance: {note: "…"}`.

### 6. Check length after generation

After cleaning, if the word count is above the ceiling (not just above `plan.hi`), retry once with the seed bumped and a firmer instruction; if it is still over, cut at the last sentence boundary under the ceiling and set a `trimmed` flag that the Web UI shows like the existing `clamped` note. That keeps the end of the prompt (often the camera or sound) from being silently dropped by the encoder.

### 7. Previous rewrites in the Enhanced tab (small)

Keep the last few rewrites of the current prompt in the draft, with ‹ › to step between them. **Enhance again** today replaces the previous result, so a good rewrite is lost when trying for a better one. No new panel; two small buttons next to the existing **used** badge. Needs a visual preview before building.

## Considered and not proposed

- **Generate several candidates side by side.** Adds a panel and multiplies the time per rewrite; item 7 covers the same need with less surface.
- **Negative-prompt generation.** The default video and most image models run distilled at guidance 1, where negatives do little.
- **User-defined Style and Mood options in config.** Possible later (the axes are data already), but item 5 covers most one-off needs without a config format.
- **Prompt caching across a batch.** The shared system prompt is about 200 tokens; prefill is a small part of each rewrite.
- **Stop and skip for video jobs while enhancing.** Already covered by [Controls for Web UI video jobs](../video-job-controls.md).

## Testing

All pure logic in `utils/prompt_enhance.py`, tested with a fake enhancer as today:

- `build_messages`: sound rules present only for video with audio; user sounds kept even with no Sound option; duration and orientation lines; the reference-image line; hints from presets.
- `settings_from_mapping` / `parse_enhance_spec`: `sound` and `note` round-trip; `sound` in image mode is ignored with a warning, like `motion`.
- `enhance_prompt`: retry when a quoted span is lost; retry, then trim, when over the ceiling; `trimmed` flag.
- Web contract: the Sound axis in `matrix_contract()` with a flag the UI uses to hide it without audio.
- Manual: LTX-2.3 with and without Sound options on the same seeds, listening for whether audio follows the action.

## Open questions

- Should the `ambient` + `effects` default apply to existing `enhance: true` video entries, or only to new settings (with `enhance: true` keeping today's output)?
- Dialogue: should the enhancer ever write lines when the user wrote none ("two friends chatting")? The proposal says only when the prompt implies speech.
- Is a vision-capable, uncensored Qwen3.5-4B available on both platforms, and how much memory does it add (item 3b)?
- Hints in item 4: ship defaults for every family, or only LTX and Ideogram 4 where the gain is clear?
