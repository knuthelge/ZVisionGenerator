# Toasts as a bottom snackbar

**Status:** Done (2026-10-09, unreleased)

## Problem

The Web UI raises toasts from 52 `addToast()` calls. They looked and behaved oddly:

- **They covered the navigation.** `ToastContainer` was fixed at `top-4 right-4`, which is on the 48 px top bar, over Gallery, Models and Config. Switching page meant waiting for the toast to fade or dismissing it.
- **They piled up.** The stack had no cap and no merging. A busy run (queue, copy, upscale, complete, fail) grew five toasts down over the canvas.
- **One timeout for every tone.** Everything stayed 5 s (upscale errors 8 s), so errors could vanish before they were read while "Prompt copied" outstayed its welcome.
- **Hidden behind the asset viewer.** The container was `z-50` and the viewer `z-100`, so "Prompt copied" and upscale toasts raised from the viewer were not visible.
- **Two toasts for one action.** Queuing a prompt from a file raised "Queued portrait from the prompt file." and "Added to the queue as #2." together.
- **Tone barely showed.** Every toast was the same near-black box; only a 16 px icon was tinted, so success and error looked alike at a glance. `role="alert"` with `aria-live="polite"` contradicted itself.
- **No follow-up.** A failed save or partial delete could not be retried from the message.

## Alternatives considered

Three were mocked up with the real messages and tokens:

| | A · Bottom snackbar | B · Status strip + activity log | C · Feedback in place |
|---|---|---|---|
| Idea | One toast at a time at the bottom centre of the content | No floating messages; a 26 px strip shows the latest message and the queue, and a bell opens the last 50 events | Controls confirm themselves; only background events notify, with tab badges |
| Covers | One line over the content | Nothing | Rarely |
| History | No | Yes | No |
| Change | Small: the toast components, the store and a few call sites | Medium: new strip organism and layout row | Large: every call site |

**A was chosen** as the smallest change that fixes the overlap and the pile-up. B and C remain options if notifications need history or should become rarer.

## Design

| Topic | Decision |
|---|---|
| Position | Fixed, bottom centre of the page content, 16 px up. A page with a fixed left sidebar sets `--toast-inset-left` on `:root` (Workspace: 360 px, 56 px collapsed, none below 640 px) so the toast stays clear of the sidebar's Generate button. |
| Swap | The outgoing and incoming toast share one grid cell, so a swap animates in place. |
| Layer | `z-[150]`: above the asset viewer (`z-100`) and normal modals (`z-50`), below elevated modals (`z-[200]`). |
| One at a time | `toasts[0]` is shown; the rest wait in order, and the count shows as "+N". |
| Merging | A message with the same text and tone as a waiting or showing toast bumps its count ("×2") and restarts its timer instead of queueing again. It takes the newer action, if one is given. |
| Errors | Go ahead of every non-error, including the one showing, and stay until dismissed while nothing waits behind them. Once other toasts are waiting, an error gives way after 8 s so it can't hold them back indefinitely. Errors keep their arrival order among themselves. |
| Timeouts | Success 2.5 s, info 4 s, warning 7 s, error 0 (until dismissed); `{ timeout }` overrides. Hover or focus pauses the countdown, which restarts when the pointer or focus leaves; a toast that appears under a resting pointer starts paused. |
| Look | `ui-overlay` surface tinted 12 % with the tone, with a 1 px tone ring and no drop shadow; info uses teal. One line, at least 36 px tall, at most 520 px wide. |
| Action | `{ action: { label, run } }` adds one button in the tone colour; running it dismisses the toast. Used for **Retry** on a failed Config save and on a failed or partial Gallery delete. An action acts on its page, so the page dismisses that toast when it unmounts. |
| Accessibility | The container is a persistent, labelled `aria-live="polite"` region, so each new toast is announced; errors also carry `role="alert"`. |

## Implementation

- `lib/state/toastQueue.ts`: the pure rules (`enqueueToast`, `findTwin`, `TOAST_TIMEOUTS`) with unit tests.
- `lib/state/toasts.svelte.ts`: `addToast(message, type, options)` with `options: { timeout?, action? }` instead of a numeric timeout, applied through `enqueueToast`.
- `Toast.svelte` takes the toast item and a `pending` count, runs its own countdown and pauses on hover or focus. `ToastContainer.svelte` renders only the first toast, with a short fly-in.
- Call sites: one queue toast in `WorkspacePage` ("Queued portrait as #2."), no fixed timeouts in `upscale.ts` and `PromptsPage`, and Retry actions in `ConfigPage` and `GalleryPage`.
