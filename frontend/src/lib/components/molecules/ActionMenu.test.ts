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

  it('says what a choice does in a description under its label', () => {
    mountMenu([{ kind: 'item', id: 'smart', label: 'Smart', description: 'Follows your idea more closely.', checked: false }]);

    expect(row('smart')?.querySelector('.action-menu-name')?.textContent).toBe('Smart');
    expect(row('smart')?.querySelector('.action-menu-description')?.textContent).toBe('Follows your idea more closely.');
  });

  it('shows a hint after the label', () => {
    mountMenu([{ kind: 'item', id: 'a', label: 'Alpha', hint: 'Klein 9B' }]);
    expect(row('a')?.textContent).toContain('Alpha');
    expect(row('a')?.textContent).toContain('Klein 9B');
  });

  it('shows a note as plain text that is not an action, and an icon before an item label', () => {
    mountMenu([
      { kind: 'note', text: 'The connection dropped.', tone: 'error' },
      { kind: 'item', id: 'retry', label: 'Retry download', icon: 'reset' },
    ]);

    expect(menu()?.querySelectorAll('[role="menuitem"]')).toHaveLength(1);
    expect(menu()?.textContent).toContain('The connection dropped.');
    expect(menu()?.querySelector('.action-menu-note')?.getAttribute('role')).toBe('presentation');
    expect(row('retry')?.querySelector('svg')).not.toBeNull();
  });

  it('selects an item by closing the menu and calling onselect', () => {
    const onselect = vi.fn();
    const onclose = vi.fn();
    mountMenu([{ kind: 'item', id: 'a', label: 'A', onselect }], onclose);
    flushSync(() => row('a')?.click());
    expect(onclose).toHaveBeenCalledTimes(1);
    expect(onselect).toHaveBeenCalledTimes(1);
  });

  describe('disabled items', () => {
    const entries = (onselect = vi.fn()): ActionMenuEntry[] => [
      { kind: 'item', id: 'first', label: 'First' },
      { kind: 'item', id: 'off', label: 'Off', disabled: true, hint: 'No last frame', onselect },
      { kind: 'item', id: 'last', label: 'Last', disabled: true, onselect },
    ];
    const keydown = (key: string) => flushSync(() => menu()?.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true })));

    it('stay in reach of the arrow keys, Home and End so their reason can be read', () => {
      mountMenu(entries());
      expect(document.activeElement).toBe(row('first'));
      keydown('ArrowDown');
      expect(document.activeElement).toBe(row('off'));
      keydown('ArrowDown');
      expect(document.activeElement).toBe(row('last'));
      keydown('ArrowDown');
      expect(document.activeElement).toBe(row('first'));
      keydown('ArrowUp');
      expect(document.activeElement).toBe(row('last'));
      keydown('Home');
      expect(document.activeElement).toBe(row('first'));
      keydown('End');
      expect(document.activeElement).toBe(row('last'));
    });

    it('do nothing on click or Enter, and leave the menu open', () => {
      const onselect = vi.fn();
      const onclose = vi.fn();
      mountMenu(entries(onselect), onclose);
      keydown('ArrowDown');
      flushSync(() => row('off')?.click());
      expect(onselect).not.toHaveBeenCalled();
      expect(onclose).not.toHaveBeenCalled();
    });

    it('keep their menuitem role and aria-disabled, and stay in the menu when every item is off', () => {
      const onselect = vi.fn();
      mountMenu([{ kind: 'item', id: 'a', label: 'A', disabled: true, onselect }, { kind: 'item', id: 'b', label: 'B', checked: false, disabled: true, onselect }]);
      expect(row('a')?.getAttribute('role')).toBe('menuitem');
      expect(row('a')?.getAttribute('aria-disabled')).toBe('true');
      expect(row('b')?.getAttribute('role')).toBe('menuitemradio');
      expect(row('b')?.getAttribute('aria-disabled')).toBe('true');
      // Nothing can be chosen, yet the keys still have a place to start and to read the reasons from.
      expect(document.activeElement).toBe(row('a'));
      keydown('ArrowDown');
      expect(document.activeElement).toBe(row('b'));
      flushSync(() => row('b')?.click());
      expect(onselect).not.toHaveBeenCalled();
    });

    it('open with the first item that can be chosen focused', () => {
      mountMenu([{ kind: 'item', id: 'off', label: 'Off', disabled: true }, { kind: 'item', id: 'on', label: 'On' }]);
      expect(document.activeElement).toBe(row('on'));
    });
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

    it('shows a heading inside a group that is not a row', () => {
      mountMenu([{ kind: 'group', id: 'more', label: 'More', defaultOpen: true, items: [{ kind: 'heading', label: 'Warm' }, { kind: 'item', id: 'g1', label: 'G1' }] }]);
      expect(Array.from(menu()?.querySelectorAll('.action-menu-heading') ?? []).map((node) => node.textContent)).toEqual(['Warm']);
      expect(menu()?.querySelectorAll('[role="menuitem"]')).toHaveLength(2);
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
