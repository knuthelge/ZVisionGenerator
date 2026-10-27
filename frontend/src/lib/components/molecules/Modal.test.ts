import { beforeEach, describe, expect, it, vi } from 'vitest';

const layoutMock = vi.hoisted(() => ({ compact: false, mode: 'wide', coarse: false }));

vi.mock('$lib/state/layoutMode.svelte', () => ({ layout: layoutMock }));

import { mountComponent, settle } from '../../../test-utils/mount';
import Modal from './Modal.svelte';

const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

function dialog(): HTMLElement {
  return document.body.querySelector('[role="dialog"]') as HTMLElement;
}

function focusableIn(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE));
}

describe('Modal', () => {
  beforeEach(() => {
    Object.assign(layoutMock, { compact: false, mode: 'wide', coarse: false });
  });

  it('dismisses from the backdrop and from the header close button, restoring focus to the opener', async () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);

    // The backdrop is the first close control, the header button the second.
    for (const index of [0, 1]) {
      opener.focus();
      const onclose = vi.fn();
      const modal = mountComponent(Modal, { open: true, title: 'Preview asset', onclose });
      await settle();

      const closeButtons = document.body.querySelectorAll('button[aria-label="Close dialog"]');
      expect(closeButtons.length).toBeGreaterThan(index);
      (closeButtons[index] as HTMLButtonElement).click();
      await settle();

      expect(onclose).toHaveBeenCalledTimes(1);
      expect(dialog()).toBeNull();
      expect(document.activeElement).toBe(opener);
      await modal.unmount();
    }
  });

  it('dismisses from Escape and restores focus to the opener', async () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();

    const onclose = vi.fn();
    mountComponent(Modal, { open: true, title: 'Preview asset', onclose });
    await settle();

    const escapeEvent = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    document.dispatchEvent(escapeEvent);
    await settle();

    expect(escapeEvent.defaultPrevented).toBe(true);
    expect(onclose).toHaveBeenCalledTimes(1);
    expect(dialog()).toBeNull();
    expect(document.activeElement).toBe(opener);
  });

  it('closes only the innermost of two open modals on Escape', async () => {
    const outerClose = vi.fn();
    const innerClose = vi.fn();
    mountComponent(Modal, { open: true, title: 'New file', onclose: outerClose });
    mountComponent(Modal, { open: true, title: 'Choose folder', onclose: innerClose });
    await settle();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    await settle();

    expect(innerClose).toHaveBeenCalledTimes(1);
    expect(outerClose).not.toHaveBeenCalled();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    await settle();
    expect(outerClose).toHaveBeenCalledTimes(1);
  });

  it.each([
    { name: 'Tab cycles from the last focusable element to the first', shiftKey: false, from: -1, to: 0 },
    { name: 'Shift+Tab cycles from the first focusable element to the last', shiftKey: true, from: 0, to: -1 },
  ])('traps focus within the modal: $name', async ({ shiftKey, from, to }) => {
    mountComponent(Modal, { open: true, title: 'Trap test' });
    await settle();

    const focusable = focusableIn(dialog());
    expect(focusable.length).toBeGreaterThan(0);
    focusable.at(from)!.focus();

    const event = new KeyboardEvent('keydown', { key: 'Tab', shiftKey, bubbles: true, cancelable: true });
    document.dispatchEvent(event);

    expect(event.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(focusable.at(to));
  });

  it('returns focus inside the dialog when it escapes to a background element', async () => {
    // Focus leaks past the keydown boundary check; the capture-phase focusin listener must redirect it.
    const background = document.createElement('button');
    document.body.appendChild(background);

    mountComponent(Modal, { open: true, title: 'Focus trap' });
    await settle();
    background.focus();

    expect(dialog().contains(document.activeElement)).toBe(true);
    expect(document.activeElement).not.toBe(background);
  });

  it('neither traps Tab nor redirects focus while closed', async () => {
    const background = document.createElement('button');
    document.body.appendChild(background);
    mountComponent(Modal, { open: false, title: 'Closed modal' });
    await settle();

    const tabEvent = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    document.dispatchEvent(tabEvent);
    background.focus();

    expect(tabEvent.defaultPrevented).toBe(false);
    expect(document.activeElement).toBe(background);
  });

  it('docks a side dialog to the right edge and still closes from the backdrop', async () => {
    const onclose = vi.fn();
    mountComponent(Modal, { open: true, title: 'Job log', placement: 'side', onclose });
    await settle();

    expect(dialog().dataset.placement).toBe('side');

    (document.body.querySelector('button[aria-label="Close dialog"]') as HTMLButtonElement).click();
    await settle();
    expect(onclose).toHaveBeenCalledTimes(1);
    expect(dialog()).toBeNull();
  });

  it('renders a side dialog as the same bottom sheet as a centred one on a compact layout', async () => {
    Object.assign(layoutMock, { compact: true, mode: 'compact', coarse: true });
    mountComponent(Modal, { open: true, title: 'Job log', placement: 'side', fill: true });
    await settle();

    expect(dialog()).not.toBeNull();
    expect(document.body.querySelector('[data-placement]')).toBeNull();
  });
});
