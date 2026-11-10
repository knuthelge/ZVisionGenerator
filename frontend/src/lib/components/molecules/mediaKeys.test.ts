import { describe, expect, it } from 'vitest';

import { navigatesByKey, playerKeyCommand, type KeyContext, type PlayerCommand } from './mediaKeys';

const plain = { shiftKey: false, metaKey: false, ctrlKey: false, altKey: false };
const press = (key: string, context: Partial<KeyContext> = {}, modifiers: Partial<typeof plain> = {}): PlayerCommand | null =>
  playerKeyCommand({ key, ...plain, ...modifiers }, { kind: 'video', paused: true, target: 'timeline', volume: true, ...context });

describe('playerKeyCommand', () => {
  it('plays and pauses with Space, except on a button, which activates itself', () => {
    expect(press(' ')).toEqual({ type: 'toggle' });
    expect(press(' ', { target: 'volume' })).toEqual({ type: 'toggle' });
    expect(press(' ', { target: 'button' })).toBeNull();
  });

  it('steps a paused video by one frame, and five with Shift', () => {
    expect(press('ArrowRight')).toEqual({ type: 'step', frames: 1 });
    expect(press('ArrowLeft')).toEqual({ type: 'step', frames: -1 });
    expect(press('ArrowRight', {}, { shiftKey: true })).toEqual({ type: 'step', frames: 5 });
    expect(press('ArrowLeft', {}, { shiftKey: true })).toEqual({ type: 'step', frames: -5 });
  });

  it('seeks a playing video by one second, and five with Shift', () => {
    expect(press('ArrowRight', { paused: false })).toEqual({ type: 'seek', seconds: 1 });
    expect(press('ArrowLeft', { paused: false }, { shiftKey: true })).toEqual({ type: 'seek', seconds: -5 });
  });

  it('always seeks a take by seconds, paused or not', () => {
    expect(press('ArrowRight', { kind: 'audio', paused: true })).toEqual({ type: 'seek', seconds: 1 });
    expect(press('ArrowLeft', { kind: 'audio', paused: true }, { shiftKey: true })).toEqual({ type: 'seek', seconds: -5 });
  });

  it('changes the volume with the up and down arrows, and with left and right on the volume slider', () => {
    expect(press('ArrowUp')).toEqual({ type: 'volume', delta: 0.1 });
    expect(press('ArrowDown')).toEqual({ type: 'volume', delta: -0.1 });
    expect(press('ArrowRight', { target: 'volume' })).toEqual({ type: 'volume', delta: 0.1 });
    expect(press('ArrowLeft', { target: 'volume', paused: false })).toEqual({ type: 'volume', delta: -0.1 });
  });

  it.each([0, 1, 5, 9])('jumps to %i0% of the clip with the digit', (digit) => {
    expect(press(String(digit))).toEqual({ type: 'jump', fraction: digit / 10 });
    expect(press(String(digit), { kind: 'audio', paused: false })).toEqual({ type: 'jump', fraction: digit / 10 });
  });

  it('leaves a digit with ⌘, Ctrl or Alt to the browser', () => {
    for (const modifier of ['altKey', 'ctrlKey', 'metaKey'] as const) expect(press('5', {}, { [modifier]: true })).toBeNull();
  });

  it('leaves the up and down arrows to the page in a player with no volume control', () => {
    expect(press('ArrowUp', { volume: false })).toBeNull();
    expect(press('ArrowDown', { volume: false })).toBeNull();
    expect(press('ArrowRight', { volume: false })).toEqual({ type: 'step', frames: 1 });
  });

  it('goes to the start and the end of the clip with Home and End', () => {
    expect(press('Home')).toEqual({ type: 'start' });
    expect(press('End')).toEqual({ type: 'end' });
  });

  it.each(['k', 'm', 'f', 'e', 'i', 'PageUp', 'PageDown', 'Escape', 'Enter', 'Tab'])('leaves %s to the page', (key) => {
    expect(press(key)).toBeNull();
  });

  it('leaves a key with ⌘, Ctrl or Alt to the browser', () => {
    expect(press('ArrowLeft', {}, { altKey: true })).toBeNull();
    expect(press('ArrowRight', {}, { metaKey: true })).toBeNull();
    expect(press(' ', {}, { ctrlKey: true })).toBeNull();
  });
});

describe('navigatesByKey', () => {
  it('is true for Tab, the arrows, Home, End and the page keys, with or without Shift', () => {
    for (const key of ['Tab', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown']) {
      expect(navigatesByKey({ key, ...plain })).toBe(true);
      expect(navigatesByKey({ key, ...plain, shiftKey: true })).toBe(true);
    }
  });

  it('is false for Space, Enter and letters, which act rather than navigate', () => {
    for (const key of [' ', 'Enter', 'Escape', 'e', 'm']) expect(navigatesByKey({ key, ...plain })).toBe(false);
  });

  it('is false with a command modifier held', () => {
    expect(navigatesByKey({ key: 'Tab', ...plain, ctrlKey: true })).toBe(false);
    expect(navigatesByKey({ key: 'ArrowLeft', ...plain, metaKey: true })).toBe(false);
  });
});
