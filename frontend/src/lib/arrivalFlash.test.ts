import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ARRIVAL_MS, arrivalFor, flashArrival } from './arrivalFlash';

describe('flashArrival', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    document.body.innerHTML = '<div id="first" aria-labelledby="ws-start-group"></div><div id="last" aria-labelledby="ws-last-group"></div><section id="source" data-section="source-video"></section><button id="chip" data-chip="images"></button>';
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    document.body.innerHTML = '';
  });

  const marked = (id: string): boolean => document.getElementById(id)!.hasAttribute('data-arrived');

  it('rings the slot a frame filled and the chip that opens it on a phone, then lets go', () => {
    flashArrival('first-frame');

    expect([marked('first'), marked('chip'), marked('last'), marked('source')]).toEqual([true, true, false, false]);
    vi.advanceTimersByTime(ARRIVAL_MS);
    expect(marked('first')).toBe(false);
  });

  it('rings the last frame slot and the source section on their own', () => {
    flashArrival('last-frame');
    expect(marked('last') && !marked('first')).toBe(true);
    flashArrival('source-video');
    expect(marked('source')).toBe(true);
  });

  it('shows nothing under reduced motion', () => {
    vi.stubGlobal('matchMedia', () => ({ matches: true }));

    flashArrival('first-frame');

    expect(marked('first')).toBe(false);
  });
});

describe('arrivalFor', () => {
  it('knows what a Use as for a frame or an Extend fills, and nothing else', () => {
    expect(arrivalFor({ workflow: 'img2vid', use_as: 'start', image_path: '/a.png' })).toBe('first-frame');
    expect(arrivalFor({ workflow: 'img2vid', use_as: 'last', end_image_path: '/a.png' })).toBe('last-frame');
    expect(arrivalFor({ workflow: 'extend', source_video: '/clip.mp4' })).toBe('source-video');
    expect(arrivalFor({ workflow: 'img2img', use_as: 'start' })).toBeNull();
    expect(arrivalFor({ workflow: 'txt2img' })).toBeNull();
  });
});
