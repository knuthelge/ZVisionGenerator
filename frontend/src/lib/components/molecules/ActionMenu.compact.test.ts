// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/state/layoutMode.svelte', () => ({
  layout: { compact: true, mode: 'compact', coarse: true },
}));

import ActionMenu from './ActionMenu.svelte';

describe('ActionMenu on a compact layout', () => {
  let app: Record<string, unknown> | null = null;

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    document.body.innerHTML = '';
  });

  it('shows groups, radios and hints in the action sheet and toggles a group in place', async () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const onclose = vi.fn();
    app = flushSync(() => mount(ActionMenu, {
      target,
      props: {
        open: true,
        anchor: null,
        label: 'Available LoRAs',
        onclose,
        items: [
          { kind: 'item', id: 'a', label: 'A', checked: true },
          { kind: 'group', id: 'more', label: 'More (1)', items: [{ kind: 'item', id: 'g1', label: 'G1', hint: 'LTX' }] },
        ],
      },
    }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    flushSync();

    const sheet = document.querySelector('[data-testid="action-sheet"]') as HTMLElement;
    expect(sheet).not.toBeNull();
    expect(sheet.querySelector('[data-action="a"]')?.getAttribute('role')).toBe('menuitemradio');
    const group = sheet.querySelector('[data-action="more"]') as HTMLElement;
    expect(group.getAttribute('aria-expanded')).toBe('false');

    flushSync(() => group.click());
    expect(group.getAttribute('aria-expanded')).toBe('true');
    expect(sheet.querySelector('[data-action="g1"]')?.textContent).toContain('LTX');
    expect(onclose).not.toHaveBeenCalled();

    const panel = document.querySelector('[role="menu"]') as HTMLElement;
    (sheet.querySelector('[data-action="a"]') as HTMLElement).focus();
    flushSync(() => panel.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true })));
    expect(document.activeElement).toBe(group);
    flushSync(() => panel.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true })));
    expect(document.activeElement).toBe(sheet.querySelector('[data-action="g1"]'));
  });
});
