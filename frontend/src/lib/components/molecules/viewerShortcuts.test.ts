import { describe, expect, it } from 'vitest';

import { VIEWER_SHORTCUTS, viewerActionFor, viewerShortcutsFor } from './viewerShortcuts';

describe('viewerShortcutsFor', () => {
  const labels = (media: 'image' | 'video' | 'audio'): string[] => viewerShortcutsFor(media).map((entry) => entry.label);
  const framesLine = (list: readonly { label: string }[]): boolean => list.some((entry) => entry.label.includes('step a frame'));

  it('lists the frame-stepping keys for a video, whose player takes them after a click', () => {
    expect(framesLine(viewerShortcutsFor('video'))).toBe(true);
    expect(labels('video')).toContain('Upscale 2×');
  });

  it('leaves them out for an image and for a take', () => {
    expect(framesLine(viewerShortcutsFor('image'))).toBe(false);
    expect(framesLine(viewerShortcutsFor('audio'))).toBe(false);
  });

  it('gives a take no upscale chord', () => {
    expect(labels('audio')).not.toContain('Upscale 2×');
    expect(labels('image')).toContain('Upscale 2×');
  });

  it('shows the frame-stepping line in the full list of the shortcuts dialog', () => {
    expect(framesLine(VIEWER_SHORTCUTS)).toBe(true);
  });
});

describe('the star shortcut', () => {
  const key = (k: string, init: Partial<Parameters<typeof viewerActionFor>[0]> = {}) => viewerActionFor({ key: k, shiftKey: false, metaKey: false, ctrlKey: false, altKey: false, ...init });

  it('maps S in either case to star, and leaves modified S alone', () => {
    expect(key('s')).toBe('star');
    expect(key('S', { shiftKey: true })).toBe('star');
    expect(key('s', { metaKey: true })).toBeNull();
  });

  it('lists S for every kind of media', () => {
    for (const media of ['image', 'video', 'audio'] as const) {
      expect(viewerShortcutsFor(media).some((entry) => entry.keys.includes('S'))).toBe(true);
    }
  });
});

describe('the adjust shortcut', () => {
  const key = (k: string, init: Partial<Parameters<typeof viewerActionFor>[0]> = {}) => viewerActionFor({ key: k, shiftKey: false, metaKey: false, ctrlKey: false, altKey: false, ...init });
  const rows = (media: 'image' | 'video' | 'audio') => viewerShortcutsFor(media).filter((entry) => entry.keys.includes('A'));

  it('maps A in either case to adjust, and leaves modified A alone', () => {
    expect(key('a')).toBe('adjust');
    expect(key('A', { shiftKey: true })).toBe('adjust');
    expect(key('a', { ctrlKey: true })).toBeNull();
    expect(key('a', { metaKey: true })).toBeNull();
    expect(key('a', { altKey: true })).toBeNull();
  });

  it('lists one A row labelled Adjust for an image, and none for a video or a take', () => {
    expect(rows('image')).toEqual([{ keys: ['A'], label: 'Adjust' }]);
    expect(rows('video')).toEqual([]);
    expect(rows('audio')).toEqual([]);
  });

  it('keeps the row in the full list of the shortcuts dialog', () => {
    expect(VIEWER_SHORTCUTS.filter((entry) => entry.keys.includes('A'))).toEqual([{ keys: ['A'], label: 'Adjust' }]);
  });
});
