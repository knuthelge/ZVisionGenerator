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

    const closeButtons = document.body.querySelectorAll('button[aria-label="Close dialog"]');
    expect(closeButtons.length).toBeGreaterThan(0);

    (closeButtons[0] as HTMLButtonElement).click();
    await settle();

    expect(onclose).toHaveBeenCalledTimes(1);
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
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

    const closeButtons = document.body.querySelectorAll('button[aria-label="Close dialog"]');
    expect(closeButtons.length).toBeGreaterThan(1);

    (closeButtons[1] as HTMLButtonElement).click();
    await settle();

    expect(onclose).toHaveBeenCalledTimes(1);
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
  });

  it('dismisses from Escape and restores focus to the opener', async () => {
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

    const escapeEvent = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    document.dispatchEvent(escapeEvent);
    await settle();

    expect(escapeEvent.defaultPrevented).toBe(true);
    expect(onclose).toHaveBeenCalledTimes(1);
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
    expect(document.activeElement).toBe(opener);

    opener.remove();
  });

  it('closes only the innermost of two open modals on Escape', async () => {
    const outerClose = vi.fn();
    const innerClose = vi.fn();
    const innerTarget = document.createElement('div');
    document.body.appendChild(innerTarget);
    app = flushSync(() => mount(Modal, { target, props: { open: true, title: 'New file', onclose: outerClose } }));
    const inner = flushSync(() => mount(Modal, { target: innerTarget, props: { open: true, title: 'Choose folder', onclose: innerClose } }));
    await settle();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    await settle();

    expect(innerClose).toHaveBeenCalledTimes(1);
    expect(outerClose).not.toHaveBeenCalled();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    await settle();
    expect(outerClose).toHaveBeenCalledTimes(1);
    await unmount(inner);
  });

  it('traps Tab focus within the modal, cycling from last to first focusable element', async () => {
    app = flushSync(() => mount(Modal, {
      target,
      props: { open: true, title: 'Trap test' },
    }));
    await settle();

    const dialog = document.body.querySelector('[role="dialog"]') as HTMLElement;
    const focusable = Array.from(dialog.querySelectorAll<HTMLElement>('button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'));
    expect(focusable.length).toBeGreaterThan(0);

    const last = focusable[focusable.length - 1];
    last.focus();

    const tabEvent = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    document.dispatchEvent(tabEvent);

    expect(tabEvent.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(focusable[0]);
  });

  it('traps Shift+Tab focus within the modal, cycling from first to last focusable element', async () => {
    app = flushSync(() => mount(Modal, {
      target,
      props: { open: true, title: 'Trap test' },
    }));
    await settle();

    const dialog = document.body.querySelector('[role="dialog"]') as HTMLElement;
    const focusable = Array.from(dialog.querySelectorAll<HTMLElement>('button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'));
    expect(focusable.length).toBeGreaterThan(0);

    focusable[0].focus();

    const shiftTabEvent = new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, bubbles: true, cancelable: true });
    document.dispatchEvent(shiftTabEvent);

    expect(shiftTabEvent.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(focusable[focusable.length - 1]);
  });

  it('does not trap Tab when the modal is closed', async () => {
    app = flushSync(() => mount(Modal, {
      target,
      props: { open: false, title: 'Closed modal' },
    }));
    await settle();

    const tabEvent = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    document.dispatchEvent(tabEvent);

    expect(tabEvent.defaultPrevented).toBe(false);
  });

  it('focusin safety net returns focus inside the dialog when it escapes to a background element', async () => {
    // Simulate the real-browser failure: focus leaks to a background element
    // (the case the keydown boundary check failed to catch). The capture-phase
    // focusin listener must detect the escape and redirect focus back inside.
    const background = document.createElement('button');
    background.textContent = 'Background Ratio';
    document.body.appendChild(background);

    app = flushSync(() => mount(Modal, {
      target,
      props: { open: true, title: 'Focus trap' },
    }));
    await settle();

    // Directly move focus to a background element (bypasses the keydown handler,
    // exercising the focusin safety net exclusively).
    background.focus();

    const dialog = document.body.querySelector('[role="dialog"]') as HTMLElement;
    // Focus must be back inside the dialog after the capture-phase handler fires.
    expect(dialog.contains(document.activeElement)).toBe(true);
    expect(document.activeElement).not.toBe(background);

    background.remove();
  });

  it('focusin safety net is inactive when the modal is closed', async () => {
    const background = document.createElement('button');
    background.textContent = 'Background Clear';
    document.body.appendChild(background);

    app = flushSync(() => mount(Modal, {
      target,
      props: { open: false, title: 'Closed modal' },
    }));
    await settle();

    background.focus();

    // With the modal closed the safety net must not redirect focus.
    expect(document.activeElement).toBe(background);

    background.remove();
  });

  it('docks a side dialog to the right edge and still closes from the backdrop and Escape', async () => {
    const onclose = vi.fn();
    app = flushSync(() => mount(Modal, {
      target,
      props: { open: true, title: 'Job log', placement: 'side', onclose },
    }));
    await settle();

    const dialog = document.body.querySelector('[role="dialog"]') as HTMLElement;
    expect(dialog.dataset.placement).toBe('side');

    (document.body.querySelector('button[aria-label="Close dialog"]') as HTMLButtonElement).click();
    await settle();
    expect(onclose).toHaveBeenCalledTimes(1);
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
  });

  it('centres the dialog by default', async () => {
    app = flushSync(() => mount(Modal, { target, props: { open: true, title: 'Preview asset' } }));
    await settle();
    expect((document.body.querySelector('[role="dialog"]') as HTMLElement).dataset.placement).toBe('center');
  });
});
