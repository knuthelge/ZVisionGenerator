// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { hasOpenModal } from '$lib/keyboard';
import { SHEET_SETTLE_MS } from '$lib/actions/sheetDrag';
import SheetHost from './SheetHost.test.svelte';

describe('Sheet', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  const sheet = () => document.querySelector<HTMLElement>('[data-testid="sheet"]');

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
    target.remove();
    vi.useRealTimers();
  });

  it('mounts its content only while open by default, and counts as an open modal', () => {
    app = flushSync(() => mount(SheetHost, { target, props: { open: false } }));
    expect(sheet()).toBeNull();
    expect(hasOpenModal()).toBe(false);

    (target.querySelector('[data-testid="open"]') as HTMLButtonElement).click();
    flushSync();
    expect(sheet()).not.toBeNull();
    expect(document.querySelector('input[name="steps"]')).not.toBeNull();
    expect(hasOpenModal()).toBe(true);
  });

  it('keeps form fields in the page while closed when asked, hidden, inert and not modal', () => {
    app = flushSync(() => mount(SheetHost, { target, props: { open: false, keepMounted: true } }));
    const root = target.querySelector('[data-testid="sheet"]') as HTMLElement;
    expect(root).not.toBeNull();
    expect(root.hidden).toBe(true);
    expect(root.inert).toBe(true);
    expect(target.querySelector('input[name="steps"]')).not.toBeNull();
    expect(hasOpenModal()).toBe(false);
  });

  it('closes on Escape and on its Close button, reporting it, and leaves once it has slid away', () => {
    const onclose = vi.fn();
    app = flushSync(() => mount(SheetHost, { target, props: { open: true, onclose } }));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    flushSync();
    expect(onclose).toHaveBeenCalledTimes(1);
    expect(hasOpenModal()).toBe(false);
    expect(sheet()?.inert).toBe(true);
    vi.advanceTimersByTime(SHEET_SETTLE_MS);
    flushSync();
    expect(sheet()).toBeNull();

    (target.querySelector('[data-testid="open"]') as HTMLButtonElement).click();
    flushSync();
    (document.querySelector('[data-testid="sheet"] .sheet-head button[aria-label="Close"]') as HTMLButtonElement).click();
    flushSync();
    expect(onclose).toHaveBeenCalledTimes(2);
  });
});
