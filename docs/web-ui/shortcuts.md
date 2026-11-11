# Keyboard shortcuts

Every key the Web UI answers to is in the tables below, grouped by where it works. ++cmd++ on macOS is ++ctrl++ on Windows and Linux, and ++alt++ is ++option++ on a Mac.

## Anywhere

| Key | Action |
|---|---|
| <kbd>?</kbd> | Show the shortcuts of the page you are on (outside a text field) |
| ++g++ then ++w++ / ++p++ / ++g++ / ++m++ / ++c++ | Go to Workspace / Prompts / Gallery / Models / Config |
| ++g++ then ++j++ | Open the job log |

## Workspace

| Key | Action |
|---|---|
| ++cmd+enter++ | Generate |
| ++cmd+shift+enter++ | Generate with a new seed (re-rolls a locked seed) |
| ++cmd+e++ | Enhance the prompt |
| ++cmd+enter++ in **Describe it** | Fill what is empty from the idea |
| ++alt+up++ | Open the recent prompts (in an empty prompt box); ++up++ and ++down++ move, ++enter++ picks, ++esc++ closes |
| ++alt+enter++ | Open the Insert menu at the caret (in the Speech text of the Audio tab) |
| ++alt+1++ / ++alt+2++ | Focus the prompt / the settings (expands the sidebar) |
| ++alt+3++ | Focus the newest Gallery strip tile (expands the strip; ++left++ / ++right++ move between tiles) |
| ++alt+shift+1++ / ++alt+shift+2++ | Show or hide the sidebar (focuses the prompt / the settings when shown) |
| ++alt+shift+3++ | Show or hide the Gallery strip (focuses the newest tile when shown) |
| ++left++ / ++right++ | Step through earlier outputs in the preview |
| ++esc++ ++esc++ | Stop the running job (press twice) |
| ++p++ | Pause or resume the running job |
| ++n++ | Skip to the next image or video; during a review, discard the small render |
| ++r++ | Repeat the current image or video |
| ++y++ | Approve the small render for upscaling (during a review) |

### The Continue from bar

On the **Continue from** bar of **Video**'s **Extend** kind, once it has focus:

| Key | Action |
|---|---|
| ++left++ / ++down++ | Move the handle back one frame |
| ++right++ / ++up++ | Move the handle forward one frame |
| ++shift++ with an arrow | Five frames at a time |
| ++home++ | The earliest frame an extension can continue from (frame 25) |
| ++end++ | The end of the clip |

A held key moves the handle without asking; the point is set when the key lifts.

## Prompts page

| Key | Action |
|---|---|
| ++cmd+s++ | Save the prompt file |
| ++cmd+z++ | Undo (outside text fields) |
| ++enter++ | Edit the focused prompt text or random choice |
| `$` | Suggest snippets while editing a prompt |
| ++alt+enter++ | Edit the `{a\|b}` choice at the caret, make one from the selected words, or start a new one (while editing) |
| ++cmd+enter++ | In a choice's options box, apply the options; elsewhere, finish editing a prompt |
| ++esc++ | In a choice's options box, cancel; elsewhere, finish editing the prompt |

## Gallery

| Key | Action |
|---|---|
| Arrow keys | Move between assets |
| ++enter++ | Open the viewer |
| ++space++ / ++x++ | Select or deselect the focused asset |
| ++cmd+a++ | Select all loaded assets |
| ++delete++ / ++backspace++ | Delete the selection, or the focused asset (asks first) |
| ++esc++ | Clear the selection |

## Viewer

| Key | Action |
|---|---|
| ++left++ / ++right++ | Previous / next asset |
| ++home++ / ++end++ | First / last asset |
| ++r++ | Reuse settings |
| ++e++ | Use as (image or audio) or Extend (video) |
| ++m++ | [Edit an image](workspace.md#edit-an-image) |
| ++x++ then ++2++ / ++4++ | Plain upscale 2× / 4× of an image (a creative upscale where the plain one isn't offered); ++x++ then ++2++ upscales a video where its **Upscale** menu offers one |
| ++d++ | Download |
| ++c++ | Copy the prompt |
| ++delete++ / ++backspace++ | Delete (asks first: ++enter++ confirms, ++esc++ cancels) |
| ++shift+delete++ | Delete without asking |
| ++i++ | Show or hide the details |
| ++esc++ | Close the shortcut list, then the viewer |

Which videos offer **Upscale** is in [Upscaling a finished video](../concepts/upscaling.md#upscaling-an-existing-video).

## Painter

The painter is the full-screen layer of the [Edit kind](workspace.md#the-painter).

| Key | Action |
|---|---|
| ++1++ / ++2++ | Paint / Extend |
| ++b++ / ++e++ | Brush / Erase |
| <kbd>,</kbd> / <kbd>.</kbd> | Smaller / larger brush |
| <kbd>+</kbd> / <kbd>-</kbd> / <kbd>0</kbd> | Zoom in / zoom out / fit the picture |
| ++space++ and drag | Pan (mouse) |
| ++cmd+z++ | Undo |

## Player

These work in a video or audio player, in the viewer, the Workspace preview and the file browser's preview. Click the picture or the timeline first; until then, ++left++ and ++right++ page through files.

| Key | Action |
|---|---|
| ++space++ | Play or pause |
| ++left++ / ++right++ | One frame while a video is paused, 1 s while it plays (always 1 s for audio) |
| ++shift+left++ / ++shift+right++ | Five frames, or 5 s |
| ++up++ / ++down++ | Volume, where the player has a volume slider |
| ++home++ / ++end++ | The first frame / the last frame |
| ++0++ to ++9++ | Jump to 0% to 90% of the clip |

## File browser and paths

Any field that takes a path on the server, such as a start image, a prompt file or the output folder, completes as you type:

| Key | Action |
|---|---|
| ++tab++ | Take the first suggestion, or the highlighted one |
| ++up++ / ++down++ | Move through the suggestions |
| ++enter++ | Pick the highlighted suggestion |
| ++down++ in an empty field | Suggest recent and usual folders |

The **Browse…** dialog these fields open has its own keys:

| Key | Action |
|---|---|
| ++up++ / ++down++ | Move through the list (all four arrows in the grid view) |
| ++home++ / ++end++ | First / last entry |
| ++enter++ | Open the folder, or choose the file |
| ++backspace++ | Up one folder |
| `/` or ++cmd+l++ | Type a path: ++enter++ goes there, ++esc++ cancels |
| A letter | Jump to the next name that starts with it |
| ++alt+p++ | Show or hide the preview (where the dialog has one) |

In the **Filter** box, ++down++ moves to the list and ++enter++ opens the first match. In the **New folder** name box, ++enter++ creates the folder and ++esc++ cancels.

## Command line { #command-line }

The terminal commands read single keys while a batch runs; see [Pause, skip or stop a run](../guides/batches.md#keys-during-a-run).
