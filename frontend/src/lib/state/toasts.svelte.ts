// Module-level toast singleton using Svelte 5 runes.
// Import { addToast } from this module to push toasts from any page or component.
// The queue rules (one shown at a time, repeats merge, errors first) live in toastQueue.ts.

import { enqueueToast, TOAST_TIMEOUTS, type ToastAction, type ToastItem, type ToastType } from './toastQueue';

export interface ToastOptions {
  /** Milliseconds to show the toast once it is on screen; defaults by tone, 0 keeps it until dismissed. */
  timeout?: number;
  /** One button on the toast, such as Retry; running it also dismisses the toast. */
  action?: ToastAction;
}

// Use in-place mutations only so the exported $state reference is never reassigned.
// toasts[0] is on screen; the rest wait their turn.
const toasts = $state<ToastItem[]>([]);
let _nextId = 0;

export function addToast(message: string, type: ToastType = 'info', options: ToastOptions = {}): string {
  const item: ToastItem = { id: String(++_nextId), type, message, timeout: options.timeout ?? TOAST_TIMEOUTS[type], count: 1, action: options.action };
  const next = enqueueToast(toasts, item);
  toasts.splice(0, toasts.length, ...next.queue);
  return next.id;
}

export function dismissToast(id: string): void {
  const idx = toasts.findIndex((t) => t.id === id);
  if (idx >= 0) toasts.splice(idx, 1);
}

export { toasts };
