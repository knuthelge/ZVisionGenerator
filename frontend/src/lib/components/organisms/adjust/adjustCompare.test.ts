import { describe, expect, it } from 'vitest';

import { createCompare, RELEASE_KEYS } from './adjustCompare.svelte';

describe('createCompare', () => {
  it('starts off', () => {
    const compare = createCompare();
    expect(compare.showing).toBe(false);
    expect(compare.toggled).toBe(false);
    expect(compare.held).toBeNull();
  });

  it('shows the original while toggled, until the toggle is cleared', () => {
    const compare = createCompare();
    compare.toggle();
    expect(compare.showing).toBe(true);
    compare.clearToggle();
    expect(compare.showing).toBe(false);
    compare.toggle();
    compare.toggle();
    expect(compare.showing).toBe(false);
  });

  it('shows the original only while held, and a hold leaves the toggle alone', () => {
    const compare = createCompare();
    compare.hold('key');
    expect(compare.showing).toBe(true);
    expect(compare.toggled).toBe(false);
    compare.release('key');
    expect(compare.showing).toBe(false);
  });

  it('is released only by the source that holds it', () => {
    const compare = createCompare();
    compare.hold('button');
    compare.hold('key');
    compare.release('key');
    expect(compare.held).toBe('button');
    compare.release('button');
    expect(compare.held).toBeNull();
  });

  it('releases every hold at once', () => {
    const compare = createCompare();
    compare.hold('picture');
    compare.releaseAll();
    expect(compare.showing).toBe(false);
  });

  it('ends a keyboard hold on any modifier key', () => {
    expect([...RELEASE_KEYS].sort()).toEqual(['Alt', 'AltGraph', 'Control', 'Meta', 'Shift']);
  });
});
