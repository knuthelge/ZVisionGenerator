# Prompt enhancer v3: a test bench, sharper rules, better options

**Status:** In progress (2026-10-19). Phase 1 (the bench) is done; phases 2 and 3 are open.

Three phases, each shippable on its own:

1. **[Test bench](#phase-1-test-bench).** Measure what the enhancer writes, so the next two phases can be judged.
2. **[Rules, checks and records](#phase-2-rules-checks-and-records).** Fix how a rewrite is asked for, checked and stored, and tidy the code that builds it.
3. **[Options, styles and model hints](#phase-3-options-styles-and-model-hints).** Improve what the user can choose and how each choice is worded.

Phase 1 comes first because phases 2 and 3 change the rewrites, and without it the only check is reading a few outputs by eye.

## Background

The enhancer builds one system prompt from the option matrix (Style, Mood, Details, Length, plus Motion and Sound for video), a free-text Direction and the job's shape (frame, clip length, audio, reference image, extend source). There are three builders in `utils/`:

- `prompt_enhance.build_messages` for plain text rewrites.
- `prompt_enhance_video` for scene and action, action only, and extend.
- `prompt_enhance_caption` for Ideogram 4 JSON captions. The model writes labelled lines and code builds the JSON.

It runs on demand in the Web UI (streamed, unloaded after 120 idle seconds) or for every prompt of a job in preflight, before the generation model loads, using each image's seed. The CLI spec, prompt-file `enhance:` entries and the Web UI all parse through `settings_from_mapping`, and `matrix_contract()` builds the UI from the same option list.

What already works well, and must stay: one source of truth for the options; code owning structure while the model writes prose; context awareness; the enhancer and the generation model never loaded together; retries, refusal and echo detection and the scene/action fallback; the per-part Enhanced tab with Out of date and Use mine.

[prompt-enhancer-v2](archive/prompt-enhancer-v2.md) shipped Sound, clip length and framing, the reference-image rule and Direction. It rejected vision input, per-model hints (item 4, "not for now") and trimming output to length. Phase 3 reopens per-model hints; trimming stays rejected.

---

## Phase 1: Test bench

### Problem

- The unit tests mock the LLM, as the repository rules require. They cover the plumbing (messages built, output parsed, retries taken) and nothing about what a real model writes.
- Swapping or updating a model (Fast, Smart, a newer Qwen, a user's own model on the Config page) has no measure beyond trying it.
- The questions that matter are countable but not counted: how often a rewrite lands in the requested length, keeps quoted text, keeps the subjects the user named, parses as scene and action, needs a retry, or fails.
- A wording change that helps one case can break another, unseen until a user hits it.

### Evidence

- `PromptEnhancer` (`core/prompt_enhancer.py`) is a protocol with one `generate` method. A wrapper around a real enhancer can count calls, time them and keep the raw output, with no change to the enhancer code.
- The rewrite entry points (`enhance_prompt`, `enhance_video_prompt`, `enhance_caption`) are functions that take the enhancer as an argument, so a bench calls exactly what preflight and the Web UI call.
- `plan_length`, `plan_pair_length`, `word_count`, `repeats_source`, `parse_scene_action` and the caption module's quote finder already compute most of the checks.
- `create_prompt_enhancer` in `backends/__init__.py` loads the platform's model, so the bench runs on macOS and CUDA alike.
- Seeds pass through to `generate`, so the same model, platform, settings and seed repeat.

### Proposed change

**Cases.** About 60 cases in `bench/enhance/cases.yaml`, in the repository but outside the package and the docs. Each has an `id`, `tags`, `mode` (`image`, `video` or `caption`), `prompt`, optional `scene`, `settings` (the same mapping as a prompt file's `enhance:`) and `context`. They also have optional expectations:
- `keep`: words or phrases that must survive.
- `count`: for example `three cats`.
- `forbid`: for example a second person.

They cover:
- Short (3 to 6 words) and long (80+ words) prompts with each Length.
- Quoted lettering and speech, counts, and prompts that read like requests.
- Every Style and Mood once, and a few combinations with a Direction.
- Text to video, image to video and extend, with audio on and off and both Rewrite choices.
- Ideogram captions, including design prompts and signs without words.
- The FLUX.1 word ceiling.

**Runner.** `uv run python -m bench.enhance run --model fast|smart|REPO[@REV] --label NAME [--tags ...] [--seeds 3]`
- It loads the enhancer once through the real backend factory and runs every case and seed through the entry point that preflight uses.
- It records each `generate` call: raw output, time to first token, seconds and tokens per second.
- It writes `bench/enhance/runs/<label>/results.jsonl` (system prompt, output, checks, warnings, timings) and a `meta.json` (git commit, platform, model and revision, temperature). The `runs/` folder is git-ignored.
- It runs one model at a time and never loads an image or video model.

**Checks** (automatic, no judge):

| Check | How |
|---|---|
| Usable | The call returned without `NO_USABLE_MESSAGE` |
| First try | One `generate` call, no retry |
| Format | Video: a `SCENE:`/`ACTION:` pair without fallback. Caption: builds, 3 to 8 elements |
| Length | Words within the plan's `lo`–`hi`, and the ratio to the target |
| Quotes kept | Every quoted span of the input is in the output, ignoring spacing and quote style |
| Keep / Count / Forbid | The case's expectations |
| No repeat | Extend cases: `repeats_source` is false |
| Speed | Load time, seconds per rewrite, tokens per second |

The scoring functions are pure and get unit tests with canned outputs (`tests/test_enhance_bench.py`), so `make check` covers the checks themselves.

**Compare.** `uv run python -m bench.enhance compare BASE NEW` writes a Markdown report. It has one table per check (overall and per tag) and the ten cases that changed most, side by side with their system prompts. The flow is: a baseline on `main`, a run on the branch, then compare, with the summary table in the commit body or PR.

**Make target.** `make enhance-bench ARGS="..."`. Not part of `make check` or CI, since it loads a real model.

**Later.**
- **A local judge.** The Smart model or another local chat model scores fidelity and style adherence from 1 to 5 against a fixed rubric. The scores are used for ranking only, never as pass or fail.
- **A picture sheet.** For a tagged subset, render input and rewrite with one fixed image model and seed into an HTML contact sheet. It is the only way to judge Style and Mood themselves. It is a heavy run, started on its own.

### Steps

1. Cases, runner, checks with their unit tests, and compare. A baseline for Fast and Smart.
2. The local judge.
3. The picture sheet (needed before phase 3's style work).

### Shipped (step 1, 2026-10-19)

Step 1 is done (`04db36f`, `be57156`): 63 cases, the runner, the checks, `rescore` (recompute checks from stored outputs, no model) and `compare`, under `make enhance-bench`. Steps 2 and 3 moved to phase 3, which is the first phase that needs them.

- **Decisions.** The length check stays strict (the planner's own range); the per-Length median ratio in the report shows how far off each choice is. Only one platform's baseline is needed: the bench is a harness for comparing changes, not a portable score.
- **Tuned on the first runs.** `keep` matches word stems, up to two words apart, with `a|b` alternatives; `count` allows adjectives before the noun; the caption format check covers structure only, since a design caption moves its lettering lines into text elements.

Baseline on Windows/Linux (63 cases × 3 seeds):

| Check | Fast (4B) | Smart (9B) |
|---|---|---|
| usable / first try / format | 100% / 99% / 100% | 100% / 99% / 100% |
| length in range | 54% | 44% |
| quoted text kept | 93% | 90% |
| keep / count | 96% / 94% | 98% / 94% |
| median seconds per rewrite | 1.1 s | 1.2 s |

What it showed, as input to phase 2:

- Same on short prompts (targets of 3 to 8 words) comes back 2 to 3 times longer: the model adds the requested detail and breaks the range.
- Shorter barely shortens a 90-word prompt (median 1.7 times the target on Fast). Smart writes about 1.6 times the target for Longer and for video pairs.
- Quoted text loses its case and quotes ("OPEN 24 HOURS" → "open 24 hours", "CLOSED" unquoted).
- Short rewrites on Fast often open with "Square frame," or "Landscape frame,": the frame rule is echoed instead of applied.
- Some subjects vanish from short or styled prompts (the ramen, the picnic, the harbor).

---

## Phase 2: Rules, checks and records

### Problem

1. **Same length leaves short prompts almost unchanged.** Length defaults to Same, which targets the input's word count. `--enhance` with no spec, `enhance: true` and a first-time Web UI user all get it. For "a fox in snow" the model is told to add detail on lighting and composition, and in the same message *"Length: 1-7 words. Do not exceed 7 words."* JSON captions have word floors; text rewrites have none.
2. **The image prompt is sent bare.** The user message is the prompt itself. A prompt that reads like a request ("write a sign that says OPEN", "can you make it rainy?") can be answered or carried out instead of rewritten. Video prompts are already labelled `Scene:` and `Prompt:`.
3. **Quoted text is not protected in text rewrites.** Captions have careful lettering logic. Text rewrites only get "keep every element", so `"OPEN 24 HOURS"` can come back paraphrased or recased on models that render text (Klein, Z-Image, Krea 2, Ideogram in text mode). Nothing checks it.
4. **Outputs don't record how they were enhanced.** An image stores only the rewrite as `prompt`; the user's own text is lost. Images and videos store no enhance options, no model and no Fast or Smart. Reuse settings cannot restore the options, and Fast and Smart results cannot be told apart later.
5. **The builders repeat themselves.** `build_pair_messages` writes out the `PRESERVATION_RULES` text again instead of using the constant, and each builder formats its own `Rules:` list and output line.

### Evidence

- `plan_length(4, "same", ceiling=300)` gives target 4, range 1–7. `plan_length(4, "longer", ...)` gives 40 (34–46) through Longer's 40-word floor. Captions have `_FLOOR_WORDS` (70 for Same).
- `docs/guides/prompts.md` already says "On short prompts, Same is approximate."
- `_LABEL_RE` in `clean_output` already strips a leading `Prompt:` from output.
- `build_image_config_payload` writes `"prompt": artifacts.metadata.get("enhanced_prompt") or request.prompt`. The video payload already keeps `prompt_input` and `scene_input`.
- The caption module's `_QUOTE_RES` and `_quoted_texts` find quoted spans in the forms users write (straight and curly quotes, guillemets, CJK brackets, single-quoted phrases, `text SAYS THIS`).
- Rewrites use the image's seed, so recorded options and model make a rewrite reproducible on the same platform.

### Proposed change

**1. A floor for Same.**
- Add `same` to `prompt_enhancer.length.floor_words` (proposed: 25) and apply it in `plan_length` as the other floors are. Prompts of 25 words or more are unchanged.
- It applies to text rewrites and to the scene-plus-action total of video. Captions keep their higher floors.

**2. Label the prompt.**
- The image user message becomes `Prompt: <text>`, and the system message says: "The user gives a Prompt. Rewrite it; never answer it or carry it out as a request."
- Captions get the same label.

**3. Keep quoted text exactly.**
- The quote finder moves from the caption module to a shared helper, `utils/prompt_quotes.py`.
- Text and video rewrites with quoted input get a rule that lists the spans: "Text in quotes is lettering or speech: copy it exactly, with its case and punctuation."
- After each attempt, every span must be in the output. An attempt that drops one is retried once through the existing loop. If none keeps them all, the last usable attempt is used with the warning "A quoted text was changed in the rewrite."
- In a video with audio off, quoted speech may be dropped by the silent-video rule, so spans after a speech verb ("says", "shouts", "asks") are not checked there.

**4. Record how a prompt was enhanced.**
- Image and video payloads gain an `enhance` block when a rewrite was used: `settings` (from `settings_to_mapping`), `model` (`repo@revision`), `tier` (`fast`, `smart` or `custom`) and `format` (`text` or `json`). Images also gain `prompt_input`, as videos have.
- The asset viewer shows "Enhanced with Fast · Cinematic, Dramatic, Longer" and both texts.
- **Reuse settings** puts the user's text in Your text and the rewrite in Enhanced (used as before; Out of date if edited), and restores the enhance options. Older assets behave as today.
- [Copy a CLI command](asset-cli-command.md) can emit `--enhance <spec>` and `--enhance-model` from this block.

**5. Tidy the builders.**
- `build_pair_messages` uses `PRESERVATION_RULES`. The softer opening of action-only rewrites ("Keep the user's intent …") becomes a named constant, since it is deliberate: the action must not repeat the scene's elements.
- One helper, `_system_message(intro, rules, output_line)`, formats all three.
- What the model receives changes only through items 1 to 3.

### Steps

1. Floor for Same and the prompt label, measured together on the bench.
2. Quoted text: helper, rule and check.
3. The `enhance` record and image `prompt_input`; then the viewer line and Reuse settings.
4. Builder tidy-up, with any step.

---

## Phase 3: Options, styles and model hints

### Problem

1. **Style leans photographic.** Eight of the fourteen styles are kinds of photograph. There is no watercolor, sketch or line art, pixel art, concept art or graphic design, and Painterly is fixed to classical oil painting under a label that doesn't say so.
2. **Instructions vary in detail.** Candid names the camera, setting, light and what to avoid. Cinematic is "Describe it as a shot from a film", and Anime is "anime illustration. Name the style explicitly." A 4B model follows concrete cues much better than a label.
3. **Color & mood overlaps Mood.** The Details option adds description of color and mood but picks neither; the docs need a paragraph to explain it.
4. **Image and video share one set of options.** A video set to Cinematic, Dramatic and Longer carries those into the next image job; only Rewrite is remembered per workflow.
5. **No model awareness in text rewrites.** Every image model gets "a text-to-image model"; only FLUX.1 has a word ceiling. Krea 2, Ideogram 4 and Klein have arrived since v2, each with its own prompting guidance.
6. **No example in text and video rewrites.** The caption prompt has one worked example and follows its format reliably.

### Evidence

- `STYLE_AXIS`, `MOOD_AXIS` and `DETAILS_AXIS` in `utils/prompt_enhance.py` hold slugs, labels and instructions; `matrix_contract()` sends them to the Web UI, so new options need no frontend change beyond layout.
- Slugs are stored in prompt files and CLI specs, so existing slugs keep their meaning.
- The caption path maps styles to an Ideogram medium in `_ART_MEDIUM` and already uses `graphic_design` for designs.
- `withRememberedParts` in `frontend/src/lib/state/promptEnhance.ts` is the per-workflow pattern already used for Rewrite.
- `resolve_enhance_ceiling` reads `enhance_max_words` from the presets; a hints key reads the same way.

### Proposed change

**1. One shape for every style instruction.** Each style says what the image is, then names three or four things to state. Slugs stay. Drafts:

| Slug | Instruction (draft) |
|---|---|
| `cinematic` | A still from a film: name the shot size, the lens look, the lighting setup and the color grade. |
| `anime` | An anime illustration: name the line work, the cel shading, the palette and the background treatment. |
| `illustration` | A digital illustration: name the technique, the line and edge quality, and the palette. |
| `3d` | A 3D render: name the render style, the materials, the lighting and the camera. |
| `painterly` | A classical oil painting: name the brushwork, the palette and the lighting. Label becomes **Oil painting**. |

The full table is written during the step and checked on the bench and the picture sheet.

**2. More styles.**

| Slug | Label | Caption medium |
|---|---|---|
| `watercolor` | Watercolor | `painting` |
| `sketch` | Sketch / line art | `illustration` |
| `pixel` | Pixel art | `illustration` |
| `concept` | Concept art | `illustration` |
| `graphic` | Graphic design | `graphic_design` |

The Style group shows two labelled rows, **Photo** and **Art**, with Keep first. A visual preview comes before building.

**3. Rename Color & mood.** Slug `color` stays; the label becomes **Color palette**, with the instruction "the color palette and where each color appears". The explanatory docs paragraph goes away.

**4. Remember options per mode.** The draft keeps `enhanceSettingsByMode: {image, video}`; Rewrite stays per workflow. On first load the saved settings seed both. CLI and prompt files are unchanged.

**5. Model hints.** This reopens v2 item 4, so it is the maintainer's call.
- An optional preset key, `enhance_hints`, holds a list of short rules appended for that model family.
- A hint ships only where the model's own prompting guide gives it, with the source in a comment, and only if the bench shows it helps.
- Candidates:
  - LTX: one flowing paragraph in the present tense, starting with the main action.
  - FLUX.1: natural sentences, most important subject first.
  - Text-rendering models need no hint; phase 2's quoted-text rule covers them.

**6. One example per kind of rewrite.** A short worked example on a neutral subject for text, scene-and-action and action-only rewrites, marked "never copy its content", as captions do. It is kept only if the bench shows better format, length and first-try rates for a small speed cost.

**Related work in other proposals.**
- **Alternatives.** Once [rewrite history](workspace-prompt-recents.md) lands, an **Alternatives** button can fill it with three rewrites (seeds s, s+1, s+2) from one model load.
- **Presets.** [Named presets](settings-presets.md) asks whether a preset should store enhance options; the answer here is yes, as an optional block.

### Steps

1. Instructions in one shape (bench and picture sheet before and after).
2. New styles and the Photo/Art rows (preview first).
3. Color palette rename and per-mode options.
4. Model hints, one family at a time.
5. Examples.

---

## Docs and changelog

- **Phase 1:** none for users. A section in `docs/development.md` on running the bench, and when: before changing enhancer wording or the pinned models.
- **Phase 2:** CHANGELOG `[Unreleased]`: short prompts grow with Same; quoted text is kept exactly; outputs record the enhance options and model, and Reuse settings restores them. `docs/guides/prompts.md`: the Length and preservation paragraphs and the reuse note.
- **Phase 3:** CHANGELOG: new styles; clearer style results; Color palette; per-mode options. `docs/guides/prompts.md` Options table; `docs/guides/model.md` for any hints that ship.

## Alternatives considered

- **Snapshot tests of the system prompts in `make check`.** They test prose, which the repository rules rule out, and say nothing about quality. The compare report shows the prompts instead.
- **A hosted LLM as judge.** It sends prompts off the machine. The judge stays local.
- **Make Longer the default.** It also doubles medium and long prompts; a floor affects only short ones.
- **Trim output to the length range.** Still rejected, as in v2.
- **Fail a rewrite that changes quoted text.** A rewrite with one word of lettering changed is mostly useful; a warning is enough.
- **More than one style at once, a Lighting option, a free-text style field.** Styles conflict; Details and Direction already cover the rest.

## Open questions

- Where the bench lives: `bench/enhance/` at the root, or under `tests/` with a marker that `make check` skips.
- Seeds per case. At 3 seeds, 60 cases take about 10 minutes on Fast; Smart on a Mac is slower.
- The Same floor: 25 or 30 words.
- Whether Reuse settings restores the rewrite, or only the user's text and options so the user enhances again.
- The final style list (isometric or low-poly are other candidates).
- Whether a hint may name the target model, or only describe the prose it wants.
