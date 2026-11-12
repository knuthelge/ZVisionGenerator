# One Image and one Video workflow

**Status:** Done (2026-10-30, v0.14.0; branch feat/merge-workflows)

## Problem

The top bar has five workflows: Text to Image, Image to Image, Image to Video, Text to Video and Extend Video. The two pairs differ only in whether a start image (a first frame for video) is used, so the user makes the same choice twice: once by picking the tab, and again by adding the image. Picking Image to Image and forgetting the image is a job that can't start. Five entries also crowd the tablet top bar, which wraps below 1000 px, and make the phone's workflow select longer.

The start image is also hard to see. On a computer it is a collapsible section in the Settings pane, with its **Img strength** in a different section (Sampling); on a phone it is a chip halfway along the settings strip. Once the start image decides what runs, it has to sit next to the prompt.

The backend and the CLI already work the merged way. A saved job is recorded as `img2img` when `image_path` is set (`provenance.py`), the video stage is picked from whether there is a first frame (`workflows/__init__.py`), and `ziv` and `ziv-video` are one command with an optional `--image`.

## What the user sees

The top bar shows **Image · Video · Extend Video**. The phone's workflow select shows the same three.

- **Image:** with no start image, Generate makes an image from the prompt. With one, it starts from that image, at the strength set on the image.
- **Video:** with no first frame, Generate makes a clip from the prompt. With one, the clip starts on it, and an optional last frame can be added.
- **Extend Video:** unchanged.

The images the run uses are in one **Images** section of Settings, whose summary line says what the run uses even when it is collapsed. The Gallery, the job log and the job card keep saying what actually ran (*Image to image*, *Text to video*), from the recorded workflow.

### Image inputs

One collapsible **Images** section sits in the Settings pane third, after **Size** and **Sampling**, the two sections every run uses; it takes the place of the Start image / Frames section, and the sections after it keep their order. The section holds every image the run uses, each with one line saying what it does. Collapsed, its summary says what Generate will use: in Image *None*, *pond.png*, *pond.png · 2 references* or *2 references*; in Video *None*, *dawn.png* or *dawn.png → dusk.png*. It replaces the Start image / Frames section, the References section and the Img strength row, so no image input shows in two places.

**Image mode**

- **Start image:** an *Image* row with the thumbnail, its name, browse, upload and ×, and a *Path* row under it to type a path. The empty row reads *Browse or drop*. **Strength** sits directly under it and exists only while a start image is set.
- **References:** rows labelled *Reference 1* to *Reference 4*, then an *Add reference* row, and under them the line *Refer to them in the prompt as image 1, image 2…*. They behave as they do today (add, drop several, Replace, the missing-file mark).
- **Images the model can't use are hidden:** a start image on a model without image-to-image (Ideogram 4), references on a model without references, and a last frame on a video model without one stay in the draft but are not shown, not sent and not counted in the summary, so a Klein start image moved to Ideogram 4 reads *None*. They show again on a model that can use them, as references already do. When the model can use no image at all, the Images section is hidden. The run is from text.

**Video mode**

- **First frame:** its *Image* row and *Path* row, as for a start image. Video has no strength.
- **Last frame:** the same two rows, marked *Optional*, on models that take one, with the line *The clip ends on this image; different scenes dissolve into each other.* Without a first frame it is greyed and reads *Not used without a first frame*.
- The prompt enhancer's Rewrite choice is remembered separately for runs with and without a first frame, as it is today for Image to Video and Text to Video. With a first frame it starts as **Action only**, since the frame sets the scene.

### What stays where the user put it

- An image the user adds stays until the user removes it with × or presses Settings **Reset**, including across reloads. That includes images uploaded from the device, which today are lost on reload; the browser keeps them now.
- **Reset** clears the open tab's images along with the other settings: in Image the start image, the references and Strength, in Video the first and last frame, in Extend Video none. The other tab keeps its images.
- Image and Video each keep their own images. Adding a first frame in Video doesn't put it into Image, and switching between Image, Video and Extend Video clears none of them. Neither does changing the model.
- Removing the first frame keeps the last frame. Its row stays, greyed, with *Not used without a first frame*, and it is used again when a first frame is added.
- **Reuse settings** and **Load settings** replace the images of the mode they open with the asset's: a text-made asset leaves that mode's slots empty.

### Gallery and viewer

- **Use as** offers *Start image*, *Reference*, *First frame* and *Last frame*. Each opens Image or Video with the image in that slot.
- *Start image* loads the image's prompt, as it does today. *Reference* adds the next number without touching the prompt and confirms with a toast, *Added as image 2*. Each target shows a hint beside it (*as image 2*, *all 4 in use*), and a target the model it opens on can't use is disabled with its reason (*takes none*, *can't use a start image*, *has no last frame*).
- **Reuse settings** opens the mode the asset was made in, with its images in their slots.

### Layouts

- **Desktop and landscape tablet:** the Images section in the Settings pane, after Size and Sampling. The Compose pane is unchanged.
- **Portrait tablet:** the same section in the sidebar's Settings. With three workflow entries the top bar fits on one row.
- **Phone:** one **Images** chip in the settings strip shows the same summary as the section header. Tapping it opens one popover with the same groups as the Settings pane: the start image or first frame with its thumbnail, name and the sources (this device, browse the server, latest output) and **Strength** while a start image is set; the references, each with its number, × and, when its image is gone, **Replace**; and the last frame. A missing image is marked in red in its group and the chip shows a warning mark. **All** holds the same Images section as a computer. The `reference`, `strength`, `references` and `endReference` chips leave the strip.

## Technical design

### Draft state

- `Workflow` keeps its five values. `draft.state.workflow` stays a stored field and is still what every reader uses (`memoryFit`, `promptEnhance`, `capabilities`, presets), so those don't change.
- A **tab** (`image`, `video`, `extend`; `WorkspaceTab` in `lib/state/workspaceTab.ts`) is what the top bar shows and sets. `workspaceTab()` maps a workflow to its tab and `workflowFor()` maps a tab and its slots back to a workflow. One action, `draft.setTab(tab)`, writes the workflow derived from that tab's slot, and the Workspace calls `draft.onTabChange(ctx)` to re-hydrate when the mode changes; `TopNav.svelte` and `TopStrip.svelte` call `setTab` and mark the active entry by tab.
- The draft keeps the start image per tab: `referenceImagePath` for Image and a new `firstFramePath` for Video, plus `endImagePath`. An uploaded file has no path, so each slot also stores a marker (`referenceImageUpload`, `firstFrameUpload`, `endImageUpload`: the file's name) that survives a reload; the workflow is derived from path or marker.
- One action, `draft.setStartImage(tab, value, ctx)` (with `draft.setLastFrame(value)` for the last frame), writes the slot, its marker and the stored workflow together. Every writer goes through it: the upload and drop handlers, a browsed path, Use as, Reuse and Load settings, and deleting the asset the slot points at (`ImagesSection.svelte`, `ImagesPopover.svelte`, `WorkspacePage.svelte`, `GalleryPage.svelte`). `hydrateFromContext` derives the workflow from the slot, so a saved draft corrects itself.
- The workflow effect in `WorkspacePage.svelte` re-hydrates (`onTabChange`) only when the **mode** changes, so adding or removing a start image never re-hydrates and the settings strip doesn't flash placeholders.
- Nothing clears an image except ×, Reset and a prefill:
  - `image_path` and `end_image_path` leave the `clear_fields` of every workflow, including `extend` and `img2img`, and their entries leave `_CLEAR_FIELD_STATE_UPDATES`.
  - The capability gate that drops the last frame on a model without one goes, and so does the effect that drops the held last-frame upload when its slot is hidden.
  - `resetSelections` clears only the open tab's images, markers and held files (and Image's Strength); the other tab keeps its own.
- A prefill (Reuse, Load settings, a `#/workspace?…` link) sets every slot of the mode it opens, `null` when absent, routes `image_path` to `referenceImagePath` or `firstFramePath` by mode, and drops only that mode's held uploads.

### Uploads

`heldUploads` (`lib/state/heldUploads.svelte.ts`) stores its files in IndexedDB through an `UploadStore` (`lib/state/uploadStore.ts`), one record per tab, so uploaded start images, frames and references survive a reload. The Workspace keeps `authorityReady` false until the restore finishes, so reference rows don't flash *Choose this image again* and Generate doesn't run on a half-restored draft.

### Submit

The plan moves the named inputs submit reads today (`image_path`, `end_image_path` and `image_strength` in `SettingsPane.svelte`), so `buildFormData` sets these three fields from the draft instead, through `submittedImages()` in `lib/state/imageSlots.ts`, by tab and capability:

- `image_path` and `image_strength` only when the mode's start image is set and the model takes one (Ideogram 4 gets neither).
- `end_image_path` only together with a first frame.

The CLI copy (`cliCommand.ts`) reads the same values, so it stays right.

### Components

- A new `ImagesSection.svelte` is the Images `InspectorSection` in `SettingsPane`, placed after Size and Sampling and built from the existing `ReferenceSlot` and the reference rows; `imagesSummary.ts` words its summary line. On a phone the Images chip opens `ImagesPopover.svelte` in the setting popover, with the same groups (its sources in `ImageSources.svelte`), since the settings sheet is out of view there.
- `capabilities.ts` keeps hiding the start image row on `supports_img2img: false`, hides the whole Images section when the model can use no image (`ImagesView.visible`) and no longer ties the last-frame row to whether a first frame is set.
- `settingsChips.ts` drops the `reference`, `references`, `endReference` and `strength` chips for one `images` chip. `SettingPopover.svelte` hosts the image popover (start image, frames and references) with the strength stepper, Replace and ×.
- `useAsState()` in `assetActions.ts` gives the Use as menu each target's hint or reason, read from the model each tab opens on (`lastImageModel` for Image); `assetMenu.ts` builds the items, including the Reference target worded from `max_reference_images`. A Reference added from the Gallery or viewer reaches the Workspace through `pendingReference.svelte.ts`. `assetActions.ts` keeps `WORKFLOW_LABELS` for history. `discoverView.ts` labels workflow-tagged tools as *Image, with a start image* and so on.
- `ActionMenu.svelte` lets the arrow keys reach disabled items, so a Use as target's reason can be read; Enter and click on them do nothing.

### Server

- `txt2img` and `txt2vid` in `workspace_contract.py` list the start image controls (and `txt2vid` the last frame controls), so the slot shows in both variants.
- The job's `workflow` is derived from the resolved image (`server.py`, where it falls back to the posted value); `job_type` already is. The checks that reject a posted `img2img` or `img2vid` without an image stay. In the browser, a missing image the run uses already blocks Generate with its reason, as a missing reference does.
- Reuse URLs keep carrying `workflow=img2img` and so on; they are current values, so nothing needs mapping.

### Tests

- Mode and slots: adding and removing a start image switches the stored workflow in both modes without re-hydrating; Video with a last frame → Image → Video keeps both frames; Image → Extend Video → Image keeps the start image; a model change keeps every image.
- Last frame: removing the first frame keeps the last frame, marks it unused and submits no `end_image_path`.
- Unsupported model: a start image on Ideogram 4 stays in the draft, the workflow is `txt2img`, and no `image_path` or `image_strength` is submitted.
- Reset clears only the open tab's images (and Image's Strength) and leaves the other tab's images and held uploads; in Extend Video it clears none.
- Prefill: Reuse of a text-made asset empties that mode's slots; Reuse of an `img2vid` asset fills `firstFramePath`, not `referenceImagePath`; only the opened mode's held uploads are dropped.
- Uploads: held uploads and their markers come back after a reload, and Generate stays off until the restore finishes.
- Use as: each target opens the right mode and slot; Reference is disabled with its reason when `lastImageModel` takes no references.
- Server: a submit with an image path is recorded as `img2img` / `img2vid` whatever workflow was posted; `img2img` without an image is still refused.
- Existing workflow tests keep their values; the fixture in `test-utils/workspace.ts` gains the start image controls on the text variants.

### Docs and changelog

- CHANGELOG `[Unreleased]`: *The Workspace has one Image and one Video workflow* (Changed › Web UI) and *Use as → Reference* (Added › Web UI).
- `docs/web-ui/workspace.md` (the Settings table, the References and upload paragraphs, the "can't start" example), `first-image.md`, `first-video.md`, `docs/concepts/images.md` and `video.md` describe the Images group.

## Alternatives considered

- **Keep five tabs and only move the start image near the prompt.** Leaves the double choice and the job that can't start.
- **One "+ Image" tray where each image gets a role** (start or reference), as in Midjourney. Only Klein has references, video's frames are fixed slots, and mixing the start image into numbered references muddies "image 1" in the prompt.
- **Fold Extend Video into Video too.** Extend takes a source video, has no size or ratio and continues a clip; it is a different task.
- **A slot for the planned edit source.** `image-editing.md` opens editing in its own editor from the viewer, not in the Workspace.
- **Clear the start image on reload.** Rejected: what the user puts in stays until they remove it.

## Open questions

- None.
