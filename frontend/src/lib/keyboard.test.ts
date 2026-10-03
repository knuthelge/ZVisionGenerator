import { afterEach, describe, expect, it } from 'vitest';

import { CHORD_TIMEOUT_MS, hasOpenModal, isPlainKey, isTyping, stepGoChord } from './keyboard';

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
    ['<input type="checkbox">', false],
    ['<button></button>', false],
  ])('treats %s as typing: %s', (html, expected) => {
    expect(isTyping(el(html))).toBe(expected);
  });

  it('allows Shift but not ⌘, Ctrl or Alt in plain keys', () => {
    const base = { key: '?', metaKey: false, ctrlKey: false, altKey: false };
    expect(isPlainKey(base)).toBe(true);
    expect(isPlainKey({ ...base, ctrlKey: true })).toBe(false);
    expect(isPlainKey({ ...base, metaKey: true })).toBe(false);
    expect(isPlainKey({ ...base, altKey: true })).toBe(false);
  });

  it('detects an open modal dialog', () => {
    expect(hasOpenModal()).toBe(false);
    document.body.innerHTML = '<div role="dialog" aria-modal="true"></div>';
    expect(hasOpenModal()).toBe(true);
  });

  describe('stepGoChord', () => {
    it('opens a page when G is followed by its key in time', () => {
      const first = stepGoChord(null, 'g', 1000);
      expect(first).toEqual({ waitingSince: 1000, page: null });
      expect(stepGoChord(first.waitingSince, 'm', 1500)).toEqual({ waitingSince: null, page: 'models' });
      expect(stepGoChord(first.waitingSince, 'G', 1500).page).toBe('gallery');
    });

    it('drops an unknown second key and a late one', () => {
      expect(stepGoChord(1000, 'z', 1100)).toEqual({ waitingSince: null, page: null });
      expect(stepGoChord(1000, 'w', 1000 + CHORD_TIMEOUT_MS + 1)).toEqual({ waitingSince: null, page: null });
    });

    it('ignores keys other than G when nothing is pending', () => {
      expect(stepGoChord(null, 'w', 1000)).toEqual({ waitingSince: null, page: null });
    });
  });
});
