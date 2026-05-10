// Module-level toast singleton using Svelte 5 runes.
// Import { addToast } from this module to push toasts from any page or component.

interface ToastItem {
  id: string;
  type: 'info' | 'success' | 'error' | 'warning';
  message: string;
  timeout: number;
}

// Use in-place mutations only so the exported $state reference is never reassigned.
const toasts = $state<ToastItem[]>([]);
let _nextId = 0;

export function addToast(
  message: string,
  type: ToastItem['type'] = 'info',
  timeout = 5000
): string {
  const id = String(++_nextId);
  toasts.push({ id, type, message, timeout });
  return id;
}

export function dismissToast(id: string): void {
  const idx = toasts.findIndex((t) => t.id === id);
  if (idx >= 0) toasts.splice(idx, 1);
}

export { toasts };
