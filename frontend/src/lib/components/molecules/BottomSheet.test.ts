// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { hasOpenModal } from '$lib/keyboard';
import BottomSheetHost from './BottomSheetHost.test.svelte';

describe('BottomSheet', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
  });

  it('mounts its content only while open by default, and counts as an open modal', () => {
    app = flushSync(() => mount(BottomSheetHost, { target, props: { open: false } }));
    expect(target.querySelector('[data-testid="bottom-sheet"]')).toBeNull();
    expect(hasOpenModal()).toBe(false);

    (target.querySelector('[data-testid="open"]') as HTMLButtonElement).click();
    flushSync();
    expect(target.querySelector('[data-testid="bottom-sheet"]')).not.toBeNull();
    expect(target.querySelector('input[name="steps"]')).not.toBeNull();
    expect(hasOpenModal()).toBe(true);
  });

  it('keeps form fields in the page while closed when asked, hidden and not modal', () => {
    app = flushSync(() => mount(BottomSheetHost, { target, props: { open: false, keepMounted: true } }));
    const root = target.querySelector('[data-testid="bottom-sheet"]') as HTMLElement;
    expect(root).not.toBeNull();
    expect(root.hidden).toBe(true);
    expect(target.querySelector('input[name="steps"]')).not.toBeNull();
    expect(hasOpenModal()).toBe(false);
  });

  it('closes on Escape and on the scrim, reporting it', () => {
    const onclose = vi.fn();
    app = flushSync(() => mount(BottomSheetHost, { target, props: { open: true, onclose } }));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    flushSync();
    expect(onclose).toHaveBeenCalledTimes(1);
    expect(target.querySelector('[data-testid="bottom-sheet"]')).toBeNull();

    (target.querySelector('[data-testid="open"]') as HTMLButtonElement).click();
    flushSync();
    (target.querySelector('button[aria-label="Close"]') as HTMLButtonElement).click();
    flushSync();
    expect(onclose).toHaveBeenCalledTimes(2);
  });
});
