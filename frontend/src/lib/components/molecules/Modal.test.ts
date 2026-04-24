// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import Modal from './Modal.svelte';

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await Promise.resolve();
  await Promise.resolve();
  flushSync();
}

describe('Modal', () => {
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
    document.body.innerHTML = '';
  });

  it('dismisses from the backdrop and restores focus to the opener', async () => {
    const opener = document.createElement('button');
    opener.textContent = 'Open modal';
    document.body.appendChild(opener);
    opener.focus();

    const onclose = vi.fn();
    app = flushSync(() => mount(Modal, {
      target,
      props: { open: true, title: 'Preview asset', onclose },
    }));
    await settle();

    const closeButtons = target.querySelectorAll('button[aria-label="Close dialog"]');
    expect(closeButtons.length).toBeGreaterThan(0);

    (closeButtons[0] as HTMLButtonElement).click();
    await settle();

    expect(onclose).toHaveBeenCalledTimes(1);
    expect(target.querySelector('[role="dialog"]')).toBeNull();
    expect(document.activeElement).toBe(opener);

    opener.remove();
  });

  it('dismisses from the header close button', async () => {
    const onclose = vi.fn();
    app = flushSync(() => mount(Modal, {
      target,
      props: { open: true, title: 'Preview asset', onclose },
    }));
    await settle();

    const closeButtons = target.querySelectorAll('button[aria-label="Close dialog"]');
    expect(closeButtons.length).toBeGreaterThan(1);

    (closeButtons[1] as HTMLButtonElement).click();
    await settle();

    expect(onclose).toHaveBeenCalledTimes(1);
    expect(target.querySelector('[role="dialog"]')).toBeNull();
  });
});