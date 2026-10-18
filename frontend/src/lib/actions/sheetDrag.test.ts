import { describe, expect, it } from 'vitest';
import { sheetDragOutcome } from './sheetDrag';

describe('sheetDragOutcome', () => {
  it('closes after a long pull down', () => {
    expect(sheetDragOutcome({ dy: 160, ms: 900, height: 400 })).toBe('close');
  });

  it('springs back after a short slow pull', () => {
    expect(sheetDragOutcome({ dy: 40, ms: 600, height: 400 })).toBe('stay');
  });

  it('closes after a quick flick', () => {
    expect(sheetDragOutcome({ dy: 40, ms: 50, height: 400 })).toBe('close');
  });

  it('needs a quarter of a tall sheet', () => {
    expect(sheetDragOutcome({ dy: 120, ms: 900, height: 800 })).toBe('stay');
    expect(sheetDragOutcome({ dy: 200, ms: 900, height: 800 })).toBe('close');
  });

  it('never closes on a pull up', () => {
    expect(sheetDragOutcome({ dy: -200, ms: 50, height: 400 })).toBe('stay');
  });
});
