import { afterEach, describe, expect, it } from 'vitest';

import { acceptsPageShortcut, APP_SHORTCUTS, CHORD_TIMEOUT_MS, hasOpenModal, inMenu, isPlainKey, isTyping, isUnfocusedTarget, stepChord, stepGoChord } from './keyboard';

function el(html: string): HTMLElement {
  const host = document.createElement('div');
  host.innerHTML = html;
  return host.firstElementChild as HTMLElement;
}

describe('keyboard', () => {
  afterEach(() => { document.body.innerHTML = ''; });

  it.each([
    ['<input type="text">', true],
    ['<input>', true],
    ['<textarea></textarea>', true],
    ['<select></select>', true],
    ['<button role="combobox"></button>', true],
    ['<input type="checkbox">', false],
    ['<button></button>', false],
  ])('treats %s as typing: %s', (html, expected) => {
    expect(isTyping(el(html))).toBe(expected);
  });

  it.each([
    ['<button></button>', false],
    ['<a href="#top"></a>', false],
    ['<input type="checkbox" role="switch">', false],
    ['<input type="checkbox">', false],
    ['<input type="radio">', false],
    ['<div role="slider" tabindex="0"></div>', false],
    ['<div tabindex="-1"></div>', true],
  ])('treats %s as a target with no tab stop: %s', (html, expected) => {
    const node = el(html);
    document.body.appendChild(node);
    expect(isUnfocusedTarget(node)).toBe(expected);
  });

  it('treats the page body as no target at all', () => {
    expect(isUnfocusedTarget(document.body)).toBe(true);
    expect(isUnfocusedTarget(document)).toBe(false);
    expect(isUnfocusedTarget(null)).toBe(false);
  });

  it('allows Shift but not ⌘, Ctrl or Alt in plain keys', () => {
    const base = { key: '?', metaKey: false, ctrlKey: false, altKey: false };
    expect(isPlainKey(base)).toBe(true);
    expect(isPlainKey({ ...base, ctrlKey: true })).toBe(false);
    expect(isPlainKey({ ...base, metaKey: true })).toBe(false);
    expect(isPlainKey({ ...base, altKey: true })).toBe(false);
  });

  describe('menus', () => {
    it('sees a key from inside a menu, which sits outside the page it was opened from', () => {
      document.body.innerHTML = '<div role="menu"><button role="menuitem">Extend</button></div><button id="outside">Close</button>';

      expect(inMenu(document.querySelector('[role="menuitem"]'))).toBe(true);
      expect(inMenu(document.getElementById('outside'))).toBe(false);
      expect(inMenu(document)).toBe(false);
    });

    it('lets no page shortcut run from a menu', () => {
      document.body.innerHTML = '<div role="menu"><button role="menuitem">Extend</button></div><button id="outside">Close</button>';
      const from = (selector: string) => {
        const event = new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true });
        document.querySelector(selector)!.dispatchEvent(event);
        return event;
      };

      expect(acceptsPageShortcut(from('#outside'))).toBe(true);
      expect(acceptsPageShortcut(from('[role="menuitem"]'))).toBe(false);
    });
  });

  it('detects an open modal dialog', () => {
    expect(hasOpenModal()).toBe(false);
    document.body.innerHTML = '<div role="dialog" aria-modal="true"></div>';
    expect(hasOpenModal()).toBe(true);
  });

  describe('APP_SHORTCUTS', () => {
    const entriesOf = (title: string) => APP_SHORTCUTS.find((group) => group.title === title)?.entries ?? [];

    it('lists Alt+Enter for the Speech text under Workspace, beside its use for random choices under Prompts', () => {
      expect(entriesOf('Workspace').some((entry) => entry.keys.includes('⌥/Alt+↵') && entry.label.includes('Speech text'))).toBe(true);
      expect(entriesOf('Prompts').some((entry) => entry.keys.includes('⌥/Alt+↵'))).toBe(true);
    });

    it('lists the Gallery shortcuts', () => {
      expect(entriesOf('Gallery').length).toBeGreaterThan(0);
    });
  });

  describe('stepGoChord', () => {
    it('opens a page when G is followed by its key in time', () => {
      const first = stepGoChord(null, 'g', 1000);
      expect(first).toEqual({ waitingSince: 1000, target: null });
      expect(stepGoChord(first.waitingSince, 'm', 1500)).toEqual({ waitingSince: null, target: 'models' });
      expect(stepGoChord(first.waitingSince, 'G', 1500).target).toBe('gallery');
    });

    it('opens the job log when G is followed by J', () => {
      expect(stepGoChord(1000, 'j', 1500)).toEqual({ waitingSince: null, target: 'job-log' });
    });

    it('drops an unknown second key and a late one', () => {
      expect(stepGoChord(1000, 'z', 1100)).toEqual({ waitingSince: null, target: null });
      expect(stepGoChord(1000, 'w', 1000 + CHORD_TIMEOUT_MS + 1)).toEqual({ waitingSince: null, target: null });
    });

    it('ignores keys other than G when nothing is pending', () => {
      expect(stepGoChord(null, 'w', 1000)).toEqual({ waitingSince: null, target: null });
    });
  });
});

describe('stepChord', () => {
  const values = { '2': 2, '4': 4 } as const;

  it('waits after the start key and completes with a value key', () => {
    const started = stepChord('x', values, null, 'x', 1000);
    expect(started).toEqual({ waitingSince: 1000, value: null, consumed: true });
    expect(stepChord('x', values, started.waitingSince, '4', 1500)).toEqual({ waitingSince: null, value: 4, consumed: true });
  });

  it('drops a pending chord on another key or after the timeout', () => {
    expect(stepChord('x', values, 1000, 'r', 1200)).toEqual({ waitingSince: null, value: null, consumed: false });
    expect(stepChord('x', values, 1000, '2', 1000 + CHORD_TIMEOUT_MS + 1).value).toBeNull();
  });
});
