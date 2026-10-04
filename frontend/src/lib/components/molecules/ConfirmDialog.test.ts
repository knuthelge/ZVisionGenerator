// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import ConfirmDialog from './ConfirmDialog.svelte';
import { requestConfirm } from './confirm.svelte';

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await Promise.resolve();
  flushSync();
}

function press(key: string, init: KeyboardEventInit = {}, target: EventTarget = document): KeyboardEvent {
  const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
  target.dispatchEvent(event);
  flushSync();
  return event;
}

function dialog(): HTMLElement | null {
  return document.querySelector('[role="alertdialog"]');
}

function actionButton(action: 'confirm' | 'cancel'): HTMLButtonElement {
  return document.querySelector(`[role="alertdialog"] [data-action="${action}"]`) as HTMLButtonElement;
}

describe('ConfirmDialog', () => {
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

  function mountDialog(props: Record<string, unknown> = {}): { onconfirm: ReturnType<typeof vi.fn>; oncancel: ReturnType<typeof vi.fn> } {
    const onconfirm = vi.fn();
    const oncancel = vi.fn();
    app = flushSync(() => mount(ConfirmDialog, {
      target,
      props: { open: true, question: 'Remove 3 queued jobs?', info: 'The running job keeps going.', confirmLabel: 'Clear queue', cancelLabel: 'Keep', onconfirm, oncancel, ...props },
    }));
    return { onconfirm, oncancel };
  }

  it('is an alert dialog named by its question and described by its information', () => {
    mountDialog();
    const el = dialog()!;
    expect(el.getAttribute('aria-modal')).toBe('true');
    expect(document.getElementById(el.getAttribute('aria-labelledby')!)?.textContent).toBe('Remove 3 queued jobs?');
    expect(document.getElementById(el.getAttribute('aria-describedby')!)?.textContent).toBe('The running job keeps going.');
    expect(el.querySelector('h2')).toBeNull();
    expect(actionButton('confirm').textContent).toContain('Clear queue');
    expect(actionButton('cancel').textContent).toContain('Keep');
  });

  it('confirms with Enter and cancels with Escape', () => {
    const first = mountDialog();
    expect(press('Enter').defaultPrevented).toBe(true);
    expect(first.onconfirm).toHaveBeenCalledTimes(1);

    press('Escape');
    expect(first.oncancel).toHaveBeenCalledTimes(1);
    expect(dialog()).toBeNull();
  });

  it('leaves Enter on the cancel button to the button and ignores a held key', () => {
    const { onconfirm } = mountDialog();
    const cancel = actionButton('cancel');
    cancel.focus();
    expect(press('Enter', {}, cancel).defaultPrevented).toBe(false);
    expect(press('Enter', { repeat: true }).defaultPrevented).toBe(true);
    expect(onconfirm).not.toHaveBeenCalled();
  });

  it('keeps keys from reaching the page but lets Tab move focus inside the dialog', () => {
    mountDialog();
    const pageKeys = vi.fn();
    document.addEventListener('keydown', pageKeys);
    press('ArrowRight');
    press('Delete');
    expect(pageKeys).not.toHaveBeenCalled();

    actionButton('confirm').focus();
    expect(press('Tab').defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(actionButton('cancel'));
    document.removeEventListener('keydown', pageKeys);
  });

  it('disables the buttons while the action is pending', () => {
    const { onconfirm } = mountDialog({ pending: true });
    expect(actionButton('confirm').disabled).toBe(true);
    expect(actionButton('cancel').disabled).toBe(true);
    press('Enter');
    expect(onconfirm).not.toHaveBeenCalled();
  });
});

describe('requestConfirm', () => {
  afterEach(() => {
    document.body.innerHTML = '';
  });

  it('resolves true on confirm and false on cancel, then removes the dialog', async () => {
    const approved = requestConfirm({ question: 'Delete "a.png"?', confirmLabel: 'Delete' });
    await settle();
    expect(dialog()?.textContent).toContain('Delete "a.png"?');
    actionButton('confirm').click();
    await expect(approved).resolves.toBe(true);
    await settle();
    expect(dialog()).toBeNull();

    const declined = requestConfirm({ question: 'Delete "b.png"?', confirmLabel: 'Delete' });
    await settle();
    press('Escape');
    await expect(declined).resolves.toBe(false);
  });

  it('returns focus to the element that opened it', async () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    const answer = requestConfirm({ question: 'Delete?', confirmLabel: 'Delete' });
    await settle();
    actionButton('cancel').click();
    await answer;
    await settle();
    expect(document.activeElement).toBe(opener);
  });
});
