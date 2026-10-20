// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import ShortcutsDialog from './ShortcutsDialog.svelte';

describe('ShortcutsDialog', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;
  let onnavigate: ReturnType<typeof vi.fn>;
  let onopenjoblog: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
    onnavigate = vi.fn();
    onopenjoblog = vi.fn();
    app = flushSync(() => mount(ShortcutsDialog, { target, props: { currentPage: 'workspace', onnavigate, onopenjoblog } }));
  });

  afterEach(async () => {
    if (app) await unmount(app);
    app = null;
    document.body.innerHTML = '';
  });

  function press(key: string, from: EventTarget = document): KeyboardEvent {
    const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true });
    from.dispatchEvent(event);
    flushSync();
    return event;
  }

  const groups = (): HTMLElement | null => document.querySelector('[data-testid="shortcut-groups"]');

  it('opens the shortcut list with ?', () => {
    expect(groups()).toBeNull();
    expect(press('?').defaultPrevented).toBe(true);
    expect(groups()?.querySelectorAll('section').length).toBeGreaterThan(1);
  });

  it('navigates with G then a page key, but not to the current page', () => {
    press('g');
    press('m');
    expect(onnavigate).toHaveBeenCalledWith('models');

    onnavigate.mockClear();
    press('g');
    press('w');
    expect(onnavigate).not.toHaveBeenCalled();
  });

  it('opens the job log with G then J without navigating', () => {
    press('g');
    expect(press('j').defaultPrevented).toBe(true);
    expect(onopenjoblog).toHaveBeenCalledTimes(1);
    expect(onnavigate).not.toHaveBeenCalled();
  });

  it('does not open the job log while another dialog is open', () => {
    const dialog = document.createElement('div');
    dialog.setAttribute('aria-modal', 'true');
    document.body.appendChild(dialog);
    press('g');
    press('j');
    expect(onopenjoblog).not.toHaveBeenCalled();
  });

  it('ignores the keys while typing in a field', () => {
    const field = document.createElement('input');
    document.body.appendChild(field);
    press('?', field);
    press('g', field);
    press('c', field);
    press('j', field);
    expect(groups()).toBeNull();
    expect(onnavigate).not.toHaveBeenCalled();
    expect(onopenjoblog).not.toHaveBeenCalled();
  });
});
