import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const layoutMock = vi.hoisted(() => ({ compact: false, mode: 'wide', coarse: false }));

vi.mock('$lib/state/layoutMode.svelte', () => ({ layout: layoutMock }));

import { mountComponent, settle } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';
import ActionMenu, { type ActionMenuEntry } from './ActionMenu.svelte';

describe('ActionMenu', () => {
  const menu = () => document.querySelector<HTMLElement>('[role="menu"]');
  const row = (id: string) => document.querySelector<HTMLElement>(`[data-action="${id}"]`);

  beforeEach(() => {
    Object.assign(layoutMock, { compact: false, mode: 'wide', coarse: false });
  });

  function mountMenu(items: ActionMenuEntry[], onclose = vi.fn()) {
    const props = reactiveProps({ open: true, anchor: null as HTMLElement | null, items, label: 'Test menu', onclose });
    mountComponent(ActionMenu, props);
    return props;
  }

  it('renders a plain item as a menuitem and a checked item as a menuitemradio with aria-checked', () => {
    mountMenu([
      { kind: 'item', id: 'plain', label: 'Plain' },
      { kind: 'item', id: 'on', label: 'On', checked: true },
      { kind: 'item', id: 'off', label: 'Off', checked: false },
    ]);
    expect(row('on')?.getAttribute('role')).toBe('menuitemradio');
    expect(row('on')?.getAttribute('aria-checked')).toBe('true');
    expect(row('off')?.getAttribute('role')).toBe('menuitemradio');
    expect(row('off')?.getAttribute('aria-checked')).toBe('false');
    expect(row('plain')?.getAttribute('role')).toBe('menuitem');
    expect(row('plain')?.hasAttribute('aria-checked')).toBe(false);
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

  it('shows groups, radios and hints in the action sheet on a compact layout and toggles a group in place', async () => {
    Object.assign(layoutMock, { compact: true, mode: 'compact', coarse: true });
    const onclose = vi.fn();
    mountMenu([
      { kind: 'item', id: 'a', label: 'A', checked: true },
      { kind: 'group', id: 'more', label: 'More (1)', items: [{ kind: 'item', id: 'g1', label: 'G1', hint: 'LTX' }] },
    ], onclose);
    await settle();

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
