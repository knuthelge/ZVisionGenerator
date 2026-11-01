# Prompt enhancer: shot, angle and colour palette

**Status:** In progress (2026-11-01): built on feat/enhancer-camera-color; left: the real-device colour-picker check.

Three new enhancer options:

- **Shot** (how much of the subject the frame holds).
- **Angle** (where the camera is).
- **Palette** (a set of colours the image uses), either a preset or up to five colours you pick.

## Problem

- **You can't choose an angle or a shot size.** Focus on → Camera & viewpoint asks the enhancer to write *more* about the camera, but it picks the angle itself. A low or high angle can only be asked for in Direction or typed into the prompt.
- **A bare keyword does little.** Typed into the prompt, "low angle shot" or "Dutch angle" mostly gives an eye-level picture, and "worm's-eye view" gave a *high* angle on Z-Image (bench rows `angle-bare-*`). Community tests point the same way (for example [this 17-model low-angle test](https://briefdock.blogspot.com/2025/06/comprehensive-camera-angle-performance.html)): models follow an angle far more often when the prompt opens with it, says where the camera sits, and shows what that view sees (sky behind a low-angle subject).
- **You can't choose colours.** Focus on → Color palette describes whatever palette the enhancer picks, and Mood and Theme steer colour only loosely.
- **Hex codes don't work in prompts.** The GenColorBench colour benchmark ([arXiv 2510.20586](https://arxiv.org/abs/2510.20586)) found hex and RGB prompts are the hardest colour task for open models. Even named colours were followed less than half the time. FLUX.2 documents hex support ([BFL guide](https://docs.bfl.ai/guides/usecases_t2i_hex_color_prompting)), but users report Klein drifts.

## What you see

In the Enhance popover, two new sections sit between **Look** and **Focus on**:

- **Framing**, with two pickers like Style:
  - **Shot:** Keep, Extreme close-up, Close-up, Medium, Full body, Wide, Extreme wide.
  - **Angle:** Keep, Eye level, High angle, Low angle, Overhead, Worm's-eye, Dutch angle, Over the shoulder, From behind.
- **Color**, with one picker, **Palette**:
  - Keep.
  - Thirteen presets: Teal & orange, Pastel, Earth tones, Film, Neutrals, Forest, Cool blues, Warm gold, Neon, Jewel tones, Primary, Sunset, Retro. Film replaced Muted; Neutrals, Forest and Retro were added. The set was chosen from vendor presets, Color Hunt tags and common prompt looks, and benched on Z-Image and Klein 4B.
  - **Custom.**

**How the Palette picker looks and behaves:**

- Each palette row in the picker shows its five swatches, as display-only 12 px squares beside the label. The closed picker shows the chosen palette's swatches.
- **Custom** is an ordinary option of the picker. When it is picked, a row of colour pills appears *under* the closed picker, not inside the list.
  - Each pill is a colour dot with a **×** to remove it; tapping the dot opens the system colour picker for that colour.
  - A **+** at the end of the row adds a colour and is disabled at five.
  - On a phone, dots, **×** and **+** have 44 px touch targets.
  - With no colours yet, a muted hint next to the **+** reads "Add up to 5 colors".
  - The closed picker shows your dots.
- **The colours you pick belong to Custom.** Picking any other palette drops them; there is no hidden list waiting behind a preset.

**Where the options apply:**

- Both sections apply to images and to a video's scene, like Style.
- They have no effect with Rewrite: Action only.
- The rewrite opens with the framing ("A low-angle view of …"), and colours each thing already in the prompt from the palette ("a terracotta sofa on an olive green rug").

**Layouts:**

- **Desktop and tablet:** the popover grows by two short sections. Where it no longer fits, as on an iPad in landscape, it scrolls with **Enhance prompt** pinned at the bottom. Its positioning is unchanged.
- **Phone:** the popover is the existing bottom sheet, with the same sections in the same order. The colour dots wrap to a second line. The system colour picker is the native one (an iOS or Android sheet).
- **Prompt builder:** uses the same component in its entry editor (Prompts page), so it gets both sections unchanged.

**CLI and prompt files:**

```text
--enhance shot=wide,angle=low,palette=earth
--enhance palette=custom,colors=1b3a4b+e07a5f   # Custom colours; # is optional
```

```yaml
enhance:
  angle: overhead
  colors: ["#1B3A4B", "#E07A5F"]          # quoted; needs palette: custom
```

- `colors` without `palette: custom` (a preset palette, or none) is an error ("Colors need palette=custom").
- **Records:** each output records the options, and Reuse restores them. The summary reads, for example, `Fast · Wide, Low angle, Earth tones` or `… Custom`.
- **Copy CLI:** the copied command writes the new keys.

## Evidence

The bench gained 47 cases (tags `framing`, `angle`, `shot`, `palette`, `colors`). The cases cover:

- Every angle on a person and the hard angles on an object.
- Every shot size, and combinations with Photo and Art styles.
- Every palette on a room and on a portrait, and three sets of your own colours.
- A colour the prompt names, and a palette together with a Mood.
- Rows with the angle typed into the prompt and no option set.

Rewrites ran with Fast and Smart, and picture sheets were rendered on Z-Image Turbo, FLUX.2 Klein 4B and Krea 2 (q8). Klein 9B was too slow for a full sheet on the test card. Every rewrite was usable and kept the prompt's subjects.

| Option | Z-Image Turbo | Klein 4B | Krea 2 q8 | Decision |
|---|---|---|---|---|
| Palette presets, Custom colours | strong | strong | strong | Ship |
| Shot sizes | all six land | five of six land; Wide reads as a high-angle view | Medium, Full and Extreme wide land; Extreme close-up and Close-up land loosely; Wide failed (a close view from behind) | Ship; Wide is weak on Klein 4B and Krea 2 |
| High angle, Overhead, From behind | yes | yes | yes | Ship |
| Dutch angle | yes | yes | strong | Ship |
| Over the shoulder | yes | yes | yes | Ship |
| Low angle, Worm's-eye | varies by seed; Low angle on a person stays weak | mostly | strong | Ship; depends on the model |
| Eye level | yes | yes | yes (was: drew the subject holding a camera) | Ship, worded as a view |
| Point of view | no | no | no | **Cut** |

**Video (LTX-2.3 and LTX-2.5, scene and action rewrites with Fast):**

- A low angle and the Sunset palette land on both.
- A wide high angle lands on LTX-2.5; LTX-2.3 rolled the camera mid-clip.
- For Overhead, Fast wrote "a high-angle view", so the clip ends looking down instead of starting straight down.

**The new presets** Film, Neutrals, Forest and Retro land visibly on Z-Image and Klein 4B.

**What the bench showed about wording:**

- **Describing the view is what makes an angle land.** "Towering skyscrapers against the pale sky" made a low angle land; a wording that told the enhancer to keep the prompt's setting dropped those cues and lost the angle. The rule now asks the enhancer to describe what the view shows behind and around the subject, in the prompt's own setting.
- **"The camera tilts" reads as a camera move,** so Dutch is worded as "the whole image is rotated about 20 degrees, so the horizon … slant[s] diagonally".
- **Over the shoulder needs one side of the frame:** "the subject's own shoulder and the back of their head fill one side of the frame, out of focus". Before that wording it came out as a plain back view.
- **Opening with the framing can push out the style.** A watercolour row dropped "watercolor", so the rule ends with "Still name the style."
- **The palette rule must forbid new things.** Without that, Fast added a burgundy scarf and a gold watch to carry a palette. It now says to colour only what is already in the prompt and its surroundings.
- **Point of view never lands.** Fast never wrote the viewer's hands. Smart wrote the *subject's* hands. No model gave a first-person view.

**Smart is not better for angles.** It equals Fast on most angles and flattens worm's-eye and Dutch. Its longer prose dilutes the cue, as the research on prompt length predicted. Both models do palettes well.

**Ideogram 4 differs from what its [prompting docs](https://github.com/ideogram-oss/ideogram4/blob/main/docs/prompting.md) suggest:**

- Each caption was rendered with and without its `color_palette` field. The field barely changes the image; the colour words in the object lines do the work.
- A low angle at the start of the caption's camera line still rendered at eye level.
- So for Ideogram, the palette field stays (it is exact and costs nothing), but the colour names in the object lines carry the palette. Angles are only asked for in the caption's camera line; code doesn't rewrite that line.

**Custom colours are sent as names, not hex codes:**

- Code, not the language model, maps each colour to the nearest of about 55 plain names, by OKLab distance with hue and chroma weighted double, so a dark blue-green reads "petrol blue", not "charcoal grey".
- `#1B3A4B` → petrol blue, `#E07A5F` → coral, `#2A9D8F` → teal.
- Ideogram 4 captions get the exact codes in `color_palette`.

## Proposed change

**Enhancer** (done in the prototype):

- `SHOT_AXIS`, `ANGLE_AXIS` and `PALETTE_AXIS` in `utils/prompt_enhance.py`, single choice, not video-only. Palette options carry `swatches`; the last option, `custom` (Custom), has none.
- A `colors` setting key: up to five hex codes, parsed in the spec (`+`-separated), in prompt files and over the API.
  - It is used only with `palette: custom`, and implies it when the palette is left out.
  - With a preset it is an error.
  - `custom` with no colours adds no rule.
- `framing_rules` and `palette_rules`, used by the text rewrite, the video scene (with "For the SCENE: ") and the caption builder.
- `utils/color_names.py`: `parse_hex` and `color_name`.
- **Caption builder:** the palette's exact colours go into `style_description.color_palette`; Black & white still has none. The framing is asked for in the camera or art-style line.
- The bench `sheet` command takes `--quantize`.

**To build:**

- **Contract:** the matrix already serves the new axes, each option's `swatches`.
- **Frontend:**
  - `EnhanceAxisKey` and `EnhanceSettings` get the new keys, and `DEFAULT_ENHANCE_SETTINGS` (`lib/state/promptEnhance.ts`) gets all four, so Reset and the "options picked" state are right.
  - `EnhanceOptions.svelte` gets the Framing and Color sections. `EnhancePicker` shows swatches when an option has them.
  - A small `ColorSwatches` molecule for Custom. Picking another palette sends `colors: []`.
  - These places drop keys they don't list, so each needs `colors` and the new axes:
    - `cliCommand.ts`, so Copy CLI keeps them;
    - `enhanceSettingsFor` in `features/prompts/document.ts`, so saving a prompt-builder entry keeps them.
  - Rebuild the packaged SPA.
- **Records:** no migration. Records without the new keys read as Keep. The summary names Custom through the option's label.
- **Tests:** add tests for:
  - the matrix contract (`swatches`, `custom`);
  - `colors` through records and Reuse;
  - Copy CLI;
  - saving a prompt-builder entry.

## Before it ships

1. ~~**Re-measure eye level.**~~ Done: the "write it as a view" rule fixed Krea 2 drawing the subject holding a camera; Z-Image was fine.
2. ~~**LTX scene check.**~~ Done: low angle and Sunset land on LTX-2.3 and LTX-2.5; a wide high angle lands on LTX-2.5 only; Overhead ends looking down (see Evidence).
3. ~~**Shot sizes on Klein 4B and Krea 2.**~~ Done: most sizes land; Wide is weak on both (see Evidence).
4. **The colour picker keeps the popover open.** The system colour picker is a floating panel in macOS Safari and a sheet on iOS. A blur or outside-click dismissal must not close the popover.
   - Done in Chrome on desktop, a short window, a touch tablet and a phone: Enhance prompt stays pinned, targets are 44 px, and the popover stays open under the colour picker.
   - Done on a Mac in Chromium and WebKit with the colour inputs set by script (the system panel doesn't open under automation): black adds, a repeated colour is ignored, a dot changes, **×** removes, **+** hides at five, and the popover stays open. The Fast enhancer on the Mac applies Framing and Color: an overhead angle with `1b3a4b+e07a5f` opens with the view and names petrol blue and coral.
   - Left for real devices: macOS Safari, iOS Safari and Android Chrome. Check that the popover or sheet stays open under the system colour picker (if not, set `followViewport: true`), that one **+** adds exactly one colour (black, and the same colour twice, included), that **×** removes it, and that a dot can be changed.
5. ~~**Custom colour names render as their colour.**~~ Done. The name table was first hand-picked; it now has 77 names. The 22 added names come from the xkcd colour survey (the names people most often gave a colour), chosen to close the gaps a coverage check found: vivid colours fell to duller names (pure blue was "cobalt blue", every neon green was "lime green"). Across 4,096 sample colours the typical distance to the nearest name dropped by about a fifth, and the worst 10% by about a third.
   - Tested a range, not every name: the bench tag `color-names` puts one Custom colour on a plain ceramic vase for vivid names, less common ones and two controls, plus a living room with three vivid colours. Text run with 2 seeds, image sheets on Z-Image and Klein 4B with one seed.
   - The enhancer used the exact name in every rewrite.
   - Klein 4B rendered every new name as its swatch. Z-Image got electric blue, bright orange, periwinkle and cerulean, but softened neon green, fuchsia and chartreuse (pale mint, plum, pale yellow); kept, since they still land in the right colour family.
   - "Eggplant" drew an actual eggplant on both models, so every name now carries a colour word: eggplant became deep purple, and wine, rose and jade became wine red, dusty rose and jade green. "Mauve" read as lilac rather than its dusty-pink swatch and was dropped. Rechecked on both models: deep purple, wine red and jade green land with no objects drawn (jade green comes out softer, closer to celadon); dusty rose lands, though one rewrite put it on the walls instead of the vase.
   - Petrol blue, an original name, came out as a beige vase on a blue table on Klein 4B.
   - Rerun: `make enhance-bench ARGS="run --label color-names --tags color-names --seeds 2"`, then `sheet --tags color-names` per image model.

## Docs and changelog

- **CHANGELOG** `[Unreleased]`: "Enhance options for the shot, the camera angle and a colour palette, from thirteen presets or up to five colours you pick."
- **`docs/concepts/prompt-enhancer.md`:**
  - Add the three rows to the options table.
  - Add a short paragraph each: angles land more often on some models than others; a palette colours what is in the prompt and replaces Mood and Theme colours; Custom colours are matched to plain colour names.
  - Update the lines these options make wrong:
    - "Style, Mood, Theme and Focus on have no effect with Action only";
    - the scene and action "Shapes" table;
    - "The popover groups the options as Look, Focus on …";
    - the JSON captions section (a palette's exact colours go into the caption).
- **`docs/cli/enhance.md` and the prompt-file docs:** the spec values, `palette=custom` and `colors`, with quoted hex codes in YAML.
- **`docs/web-ui/`:** the popover sections. Take a new screenshot of the Enhance popover.

## Alternatives considered

- **Hex codes in the prompt for FLUX.2 Klein.** Klein drifts on hex, and plain names already land on Klein. Not worth a per-model branch.
- **Recolouring the finished image to the palette** (a colour transfer with a strength slider). This guarantees the colours on any model. It is a processing step, not an enhancer option, and the palettes already land well. A separate proposal if users ask for exact colours.
- **An orbit diagram for the angle,** as in camera-angle LoRA demos. Text gets at most eight dependable angles, so a list is honest and fits the existing pickers.
- **Lens and subject placement** (rule of thirds, telephoto). The Photo styles already name a lens; placement words are followed weakly.
- **Camera-angle LoRAs.** They exist for edit models (Qwen-Image-Edit, FLUX.2 dev), not for Z-Image or Klein text to image. They are a separate feature.
- **Point of view.** Measured and cut (above).

## Decided

- **Focus on keeps Camera & viewpoint and Color palette as they are.** They ask for more description of whatever camera or colours the image has, which the new pickers don't do.
- **A palette decides the colours over Mood and Theme.** Pastel with Eerie stays pastel; the mood still shows through light, setting and expressions. The concept page says so.
