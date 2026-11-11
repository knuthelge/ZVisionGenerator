# Keyboard shortcuts

Press <kbd>?</kbd> anywhere outside a text field to see every shortcut. ++cmd++ on macOS is ++ctrl++ on Windows and Linux.

## Anywhere

| Key | Action |
|---|---|
| <kbd>?</kbd> | Show all keyboard shortcuts |
| ++g++ then ++w++ / ++p++ / ++g++ / ++m++ / ++c++ | Go to Workspace / Prompts / Gallery / Models / Config |
| ++g++ then ++j++ | Open the job log |

## Workspace

| Key | Action |
|---|---|
| ++cmd+enter++ | Generate |
| ++cmd+shift+enter++ | Generate with a new seed (re-rolls a locked seed) |
| ++cmd+e++ | Enhance the prompt |
| ++alt+up++ | Open the recent prompts (in an empty prompt box) |
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

## Prompts page

| Key | Action |
|---|---|
| ++cmd+s++ | Save the prompt file |
| ++cmd+z++ | Undo (outside text fields) |
| ++enter++ | Edit the focused prompt text or random choice |
| `$` | Suggest snippets while editing a prompt |
| ++alt+enter++ | Edit the `{a\|b}` choice at the caret, make one from the selected words, or start a new one (while editing) |
| ++esc++ / ++cmd+enter++ | Finish editing a prompt |

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
| ++m++ | Edit an image |
| ++x++ then ++2++ / ++4++ | Plain upscale 2× / 4× of an image (a creative upscale where the plain one isn't offered); ++x++ then ++2++ upscales a video (Windows and Linux) |
| ++d++ | Download |
| ++c++ | Copy the prompt |
| ++delete++ / ++backspace++ | Delete (asks first: ++enter++ confirms, ++esc++ cancels) |
| ++shift+delete++ | Delete without asking |
| ++i++ | Show or hide the details |
| <kbd>?</kbd> | Show the viewer's shortcuts |
| ++esc++ | Close the shortcut list, then the viewer |

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

## Painter

| Key | Action |
|---|---|
| ++b++ / ++e++ | Brush / Erase |
| <kbd>,</kbd> / <kbd>.</kbd> | Smaller / larger brush |
| <kbd>+</kbd> / <kbd>-</kbd> / <kbd>0</kbd> | Zoom in / zoom out / fit the picture |
| ++space++ and drag | Pan (mouse) |
| ++cmd+z++ | Undo |

## Command line

`ziv-image` and `ziv-video` read keys while a batch runs. See [Controls during a run](../cli/batches.md#controls-during-a-run).
