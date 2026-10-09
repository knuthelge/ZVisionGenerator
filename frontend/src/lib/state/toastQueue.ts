// Pure queue rules for toasts: one is shown at a time, repeats merge, errors go first.

export type ToastType = 'info' | 'success' | 'error' | 'warning';

export interface ToastAction {
  label: string;
  run: () => void;
}

export interface ToastItem {
  id: string;
  type: ToastType;
  message: string;
  /** Milliseconds the toast stays once shown; 0 keeps it until dismissed. */
  timeout: number;
  /** How many times this message was raised while it was waiting or showing. */
  count: number;
  action?: ToastAction;
}

/** Default display time per tone: short for confirmations, until dismissed for errors. */
export const TOAST_TIMEOUTS: Record<ToastType, number> = {
  success: 2500,
  info: 4000,
  warning: 7000,
  error: 0
};

/** How long an error that stays until dismissed still holds the slot once other toasts are waiting. */
export const ERROR_YIELD_MS = 8000;

/**
 * Return the queue with *item* added, and the id the caller should track.
 *
 * A repeat of a queued message bumps that toast's count (taking the newer action, if any) instead of
 * queueing again. An error goes ahead of every non-error, including the one showing, so it is seen at once;
 * other toasts wait at the end.
 */
export function enqueueToast(queue: readonly ToastItem[], item: ToastItem): { queue: ToastItem[]; id: string } {
  const twin = queue.findIndex((t) => t.type === item.type && t.message === item.message);
  if (twin >= 0) {
    const merged = queue.map((t, i) => (i === twin ? { ...t, count: t.count + 1, action: item.action ?? t.action } : t));
    return { queue: merged, id: queue[twin].id };
  }
  if (item.type !== 'error') return { queue: [...queue, item], id: item.id };
  const firstNonError = queue.findIndex((t) => t.type !== 'error');
  const at = firstNonError < 0 ? queue.length : firstNonError;
  return { queue: [...queue.slice(0, at), item, ...queue.slice(at)], id: item.id };
}
