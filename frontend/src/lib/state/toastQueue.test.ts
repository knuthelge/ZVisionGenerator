import { describe, expect, it } from 'vitest';

import { enqueueToast, TOAST_TIMEOUTS, type ToastItem, type ToastType } from './toastQueue';

function item(id: string, type: ToastType = 'info', message = `message ${id}`): ToastItem {
  return { id, type, message, timeout: TOAST_TIMEOUTS[type], count: 1 };
}

describe('enqueueToast', () => {
  it('queues non-errors behind the one on screen', () => {
    const queue = enqueueToast([item('1', 'success')], item('2', 'info')).queue;
    expect(queue.map((t) => t.id)).toEqual(['1', '2']);
  });

  it('merges a repeat into the queued toast instead of adding another', () => {
    const first = item('1', 'success', 'Deleted');
    const { queue, id } = enqueueToast([first, item('2')], item('3', 'success', 'Deleted'));
    expect(id).toBe('1');
    expect(queue.map((t) => t.id)).toEqual(['1', '2']);
    expect(queue[0].count).toBe(2);
    expect(first.count).toBe(1);
  });

  it('gives a merged repeat the newer action', () => {
    const run = (): void => {};
    const { queue } = enqueueToast([item('1', 'error', 'Delete failed')], { ...item('2', 'error', 'Delete failed'), action: { label: 'Retry', run } });
    expect(queue[0].action?.label).toBe('Retry');
  });

  it('does not merge the same text with a different tone', () => {
    const queue = enqueueToast([item('1', 'info', 'Saved')], item('2', 'success', 'Saved')).queue;
    expect(queue).toHaveLength(2);
  });

  it('puts an error ahead of the non-error on screen', () => {
    const queue = enqueueToast([item('1', 'success'), item('2', 'info')], item('3', 'error')).queue;
    expect(queue.map((t) => t.id)).toEqual(['3', '1', '2']);
  });

  it('keeps errors in the order they arrived', () => {
    const queue = enqueueToast([item('1', 'error'), item('2', 'info')], item('3', 'error')).queue;
    expect(queue.map((t) => t.id)).toEqual(['1', '3', '2']);
  });

  it('keeps errors until dismissed and the rest for a limited time', () => {
    expect(TOAST_TIMEOUTS.error).toBe(0);
    for (const type of ['success', 'info', 'warning'] as const) expect(TOAST_TIMEOUTS[type]).toBeGreaterThan(0);
  });
});
