# A scene for video prompts

**Status:** In progress (2026-10-11)

## Problem

A video prompt holds two kinds of text in one box: the scene (who and where, their look, the style, the camera setup) and the action (what happens, in time order, with camera moves, sound and quoted dialogue). Retyping the scene for every prompt is tedious, and the two mix badly in a prompt file, in a batch and in the enhancer, which rewrites the whole text as one block. [Extending a video](video-extend.md) needs the split most of all: the next clip must keep the scene of the last one and say only what happens next.

## Proposed change

Video workflows (text to video, image to video, and later extend) get one optional **Scene** field above the **Prompt**. Image workflows are unchanged.

- **Sending.** The model receives `"{scene} {prompt}"`, scene first. A blank scene sends the prompt alone, byte for byte as today, so nobody who ignores the field sees a change.
- **Scene or prompt.** Text to video and image to video accept a scene with no prompt; a prompt-file entry still needs its `prompt`.
- **Entry points.** A `--scene` flag on `ziv-video`; an optional `scene:` on a prompt-file entry (the flag and the Workspace box fill entries that have none); a Scene box in the Web UI, the Prompts page and the enhancer's Enhanced tab.
- **Enhancer.** One call returns both parts, marked `SCENE:` and `ACTION:`, parsed and checked in code. Style, mood and details shape the scene; motion and sound shape the action; the direction applies to both. Text to video splits and expands, so it also works when everything sits in the prompt; image to video writes a short scene plus the change. The length setting covers the total, split in proportion to the inputs (a third scene, two thirds action when one is empty) within the encoder ceiling, with a floor of 12 words for the action. If the output has no valid pair, it retries once; then the action alone is taken and the user's scene is kept.
- **No-op rule.** A video with a blank scene and a prompt always counts as a change, since splitting it is the change.
- **Provenance and Reuse.** The recorded settings keep the scene and prompt as generated, plus the user's inputs, so Reuse can offer either.

## Alternatives considered

- **Keep one box and let the enhancer split internally.** Nothing would persist between clips, so extend would have to guess the scene from the old prompt every time.
- **A scene on the set or the whole prompt file.** One entry-level key does the same through a shared `$snippet` and avoids a second place to look.
- **Two model calls (scene, then action).** Doubles the load on a local model for no gain over one call with a checked format.

## Decisions

- `scene:` lives on entries only; the Workspace default scene and `--scene` fill entries that have none.
- Scene alone is enough to generate; both blank is an error.
- Delivered in steps: the enhancer first (this step), then prompt files, the core and CLI, the Web API, and the Web UI.
