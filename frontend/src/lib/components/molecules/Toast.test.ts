// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import Toast from './Toast.svelte';
import { ERROR_YIELD_MS, type ToastItem } from '$lib/state/toastQueue';

function toastItem(overrides: Partial<ToastItem> = {}): ToastItem {
  return { id: '1', type: 'success', message: 'Settings saved', timeout: 2500, count: 1, ...overrides };
}

describe('Toast', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    vi.useFakeTimers();
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    vi.useRealTimers();
    target.remove();
    document.body.innerHTML = '';
  });

  function mountToast(toast: ToastItem, pending = 0): ReturnType<typeof vi.fn> {
    const ondismiss = vi.fn();
    app = flushSync(() => mount(Toast, { target, props: { toast, pending, ondismiss } }));
    return ondismiss;
  }

  function root(): HTMLElement {
    return target.querySelector('[data-tone]') as HTMLElement;
  }

  it('announces errors as alerts', () => {
    mountToast(toastItem({ type: 'error', timeout: 0 }));
    expect(root().getAttribute('role')).toBe('alert');
    expect(root().dataset.tone).toBe('error');
  });

  it('leaves non-errors to the container live region', () => {
    mountToast(toastItem({ type: 'info' }));
    expect(root().getAttribute('role')).toBeNull();
  });

  it('lets an error that stays give way once others are waiting', () => {
    const ondismiss = mountToast(toastItem({ type: 'error', timeout: 0 }), 1);
    vi.advanceTimersByTime(ERROR_YIELD_MS - 1);
    expect(ondismiss).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(ondismiss).toHaveBeenCalledWith('1');
  });

  it('dismisses itself after its timeout', () => {
    const ondismiss = mountToast(toastItem());
    vi.advanceTimersByTime(2499);
    expect(ondismiss).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(ondismiss).toHaveBeenCalledWith('1');
  });

  it('stays until dismissed when the timeout is 0', () => {
    const ondismiss = mountToast(toastItem({ type: 'error', timeout: 0 }));
    vi.advanceTimersByTime(60_000);
    expect(ondismiss).not.toHaveBeenCalled();
    (target.querySelector('[aria-label="Dismiss notification"]') as HTMLButtonElement).click();
    expect(ondismiss).toHaveBeenCalledWith('1');
  });

  it('pauses the countdown while hovered', () => {
    const ondismiss = mountToast(toastItem());
    root().dispatchEvent(new MouseEvent('mouseenter'));
    flushSync();
    vi.advanceTimersByTime(10_000);
    expect(ondismiss).not.toHaveBeenCalled();
    root().dispatchEvent(new MouseEvent('mouseleave'));
    flushSync();
    vi.advanceTimersByTime(2500);
    expect(ondismiss).toHaveBeenCalledWith('1');
  });

  it('runs its action and then dismisses', () => {
    const run = vi.fn();
    const ondismiss = mountToast(toastItem({ type: 'error', timeout: 0, action: { label: 'Retry', run } }));
    const button = [...target.querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Retry') as HTMLButtonElement;
    button.click();
    expect(run).toHaveBeenCalledOnce();
    expect(ondismiss).toHaveBeenCalledWith('1');
  });

  it('shows how many repeats merged and how many toasts wait', () => {
    mountToast(toastItem({ count: 3 }), 2);
    expect(target.querySelector('[data-testid="toast-count"]')?.textContent).toBe('×3');
    expect(target.querySelector('[data-testid="toast-pending"]')?.textContent?.trim()).toBe('+2');
  });

  it('shows no counters for a single toast with nothing waiting', () => {
    mountToast(toastItem());
    expect(target.querySelector('[data-testid="toast-count"]')).toBeNull();
    expect(target.querySelector('[data-testid="toast-pending"]')).toBeNull();
  });
});
