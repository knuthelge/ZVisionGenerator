// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';
import ActionMenu, { type ActionMenuEntry } from './ActionMenu.svelte';

describe('ActionMenu', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  const menu = () => document.querySelector<HTMLElement>('[role="menu"]');
  const row = (id: string) => document.querySelector<HTMLElement>(`[data-action="${id}"]`);

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

  function mountMenu(items: ActionMenuEntry[], onclose = vi.fn()) {
    const props = reactiveProps({ open: true, anchor: null as HTMLElement | null, items, label: 'Test menu', onclose });
    app = flushSync(() => mount(ActionMenu, { target, props }));
    return props;
  }

  it('keeps a plain item a menuitem', () => {
    mountMenu([{ kind: 'item', id: 'a', label: 'A' }]);
    expect(row('a')?.getAttribute('role')).toBe('menuitem');
    expect(row('a')?.hasAttribute('aria-checked')).toBe(false);
  });

  it('renders a checked item as a menuitemradio with aria-checked', () => {
    mountMenu([
      { kind: 'item', id: 'on', label: 'On', checked: true },
      { kind: 'item', id: 'off', label: 'Off', checked: false },
    ]);
    expect(row('on')?.getAttribute('role')).toBe('menuitemradio');
    expect(row('on')?.getAttribute('aria-checked')).toBe('true');
    expect(row('off')?.getAttribute('role')).toBe('menuitemradio');
    expect(row('off')?.getAttribute('aria-checked')).toBe('false');
    expect(row('on')?.querySelector('svg')).not.toBeNull();
    expect(row('off')?.querySelector('svg')).toBeNull();
  });

  it('shows a hint after the label', () => {
    mountMenu([{ kind: 'item', id: 'a', label: 'Alpha', hint: 'Klein 9B' }]);
    expect(row('a')?.textContent).toContain('Alpha');
    expect(row('a')?.textContent).toContain('Klein 9B');
  });

  it('selects an item by closing the menu and calling onselect', () => {
    const onselect = vi.fn();
    const onclose = vi.fn();
    mountMenu([{ kind: 'item', id: 'a', label: 'A', onselect }], onclose);
    flushSync(() => row('a')?.click());
    expect(onclose).toHaveBeenCalledTimes(1);
    expect(onselect).toHaveBeenCalledTimes(1);
  });

  describe('group', () => {
    const entries = (defaultOpen?: boolean): ActionMenuEntry[] => [
      { kind: 'item', id: 'first', label: 'First' },
      { kind: 'group', id: 'more', label: 'More (2)', defaultOpen, items: [
        { kind: 'item', id: 'g1', label: 'G1', hint: 'LTX' },
        { kind: 'item', id: 'g2', label: 'G2', checked: false },
      ] },
    ];

    it('starts collapsed and expands in place without closing the menu', () => {
      const onclose = vi.fn();
      mountMenu(entries(), onclose);
      expect(row('more')?.getAttribute('aria-expanded')).toBe('false');
      expect(row('g1')).toBeNull();

      flushSync(() => row('more')?.click());
      expect(row('more')?.getAttribute('aria-expanded')).toBe('true');
      expect(row('g1')).not.toBeNull();
      expect(onclose).not.toHaveBeenCalled();
      expect(menu()).not.toBeNull();

      flushSync(() => row('more')?.click());
      expect(row('more')?.getAttribute('aria-expanded')).toBe('false');
      expect(row('g1')).toBeNull();
      expect(onclose).not.toHaveBeenCalled();
    });

    it('starts expanded with defaultOpen', () => {
      mountMenu(entries(true));
      expect(row('more')?.getAttribute('aria-expanded')).toBe('true');
      expect(row('g1')).not.toBeNull();
    });

    it('closes the menu when a grouped item is chosen', () => {
      const onselect = vi.fn();
      const onclose = vi.fn();
      mountMenu([{ kind: 'group', id: 'more', label: 'More', defaultOpen: true, items: [{ kind: 'item', id: 'g1', label: 'G1', onselect }] }], onclose);
      flushSync(() => row('g1')?.click());
      expect(onclose).toHaveBeenCalledTimes(1);
      expect(onselect).toHaveBeenCalledTimes(1);
    });

    it('resets to the defaults each time the menu opens', () => {
      const props = mountMenu(entries());
      flushSync(() => row('more')?.click());
      expect(row('g1')).not.toBeNull();

      props.open = false;
      flushSync();
      props.open = true;
      flushSync();
      expect(row('more')?.getAttribute('aria-expanded')).toBe('false');
      expect(row('g1')).toBeNull();
    });

    it('reaches the group row and its visible children with the arrow keys', () => {
      mountMenu(entries(true));
      const keydown = (key: string) => flushSync(() => menu()?.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true })));
      expect(document.activeElement).toBe(row('first'));
      keydown('ArrowDown');
      expect(document.activeElement).toBe(row('more'));
      keydown('ArrowDown');
      expect(document.activeElement).toBe(row('g1'));
      keydown('ArrowDown');
      expect(document.activeElement).toBe(row('g2'));
      keydown('ArrowDown');
      expect(document.activeElement).toBe(row('first'));
      keydown('End');
      expect(document.activeElement).toBe(row('g2'));
    });

    it('skips the children of a collapsed group', () => {
      mountMenu(entries());
      const keydown = (key: string) => flushSync(() => menu()?.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true })));
      keydown('End');
      expect(document.activeElement).toBe(row('more'));
    });
  });
});
