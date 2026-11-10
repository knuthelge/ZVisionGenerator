import { beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { sceneSource } from '$lib/state/extend.fixtures';
import { makeContext } from '../../test-utils/workspace';
import { mountComponent, settle } from '../../test-utils/mount';
import { flushSync } from 'svelte';
import SourceVideoSection from './SourceVideoSection.svelte';

vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);
vi.mock('$lib/api/extend', () => ({ inspectExtendSource: vi.fn(async () => { throw new Error('not inspected here'); }) }));

describe('SourceVideoSection', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  function show(): HTMLElement {
    mountComponent(SourceVideoSection, { context: makeContext(), busy: false, guard: { kind: 'none', blocked: true, note: null } });
    return document.body.querySelector('[data-testid="slot-thumb"]') as HTMLElement;
  }

  it('shows the server\'s last frame of a clip outside the output folder, with no hidden video element', async () => {
    draft.setExtendSourcePath('/mnt/clips/walk.mp4');
    const thumb = show();
    await settle();

    expect(thumb.querySelector('img')?.getAttribute('src')).toBe('/preview/source_video/mnt/clips/walk.mp4?frame=last');
    expect(thumb.querySelector('video')).toBeNull();
    expect(thumb.querySelector('[data-testid="play-badge"]')).toBeNull();
  });

  it('keeps the film icon while no clip is chosen', async () => {
    const thumb = show();
    await settle();

    expect(thumb.querySelector('img')).toBeNull();
    expect(thumb.querySelector('svg')).not.toBeNull();
  });

  describe('the point to continue from', () => {
    const SOURCE = '/mnt/clips/walk.mp4';
    const described = (overrides = {}) => sceneSource({ path: SOURCE, fps: 24, frame_count: 144, video_start: 0, extend_frames: 48, ...overrides });
    const bar = () => document.body.querySelector<HTMLElement>('[data-testid="extend-cut-bar"]');
    const keeps = () => document.body.querySelector('[data-testid="extend-keeps"]')?.textContent;

    function showReady(): void {
      draft.setExtendSourcePath(SOURCE);
      draft.applyExtendSource(described());
      mountComponent(SourceVideoSection, { context: makeContext(), busy: false, guard: { kind: 'ok', blocked: false, note: null } });
    }

    it('shows the clip as a bar with the whole clip kept', () => {
      showReady();

      expect(bar()).not.toBeNull();
      expect(keeps()).toBe('Keeps the whole clip');
    });

    it('has no bar before the source is described, or when it is too short to cut', () => {
      draft.setExtendSourcePath(SOURCE);
      mountComponent(SourceVideoSection, { context: makeContext(), busy: false, guard: { kind: 'checking', blocked: true, note: null } });
      expect(bar()).toBeNull();

      draft.applyExtendSource(described({ frame_count: 25 }));
      flushSync();
      expect(bar()).toBeNull();
    });

    it('has no bar for a source that cannot be extended', () => {
      draft.setExtendSourcePath(SOURCE);
      draft.applyExtendSource(described({ problems: [{ code: 'too_short', message: 'Too short.' }] }));
      mountComponent(SourceVideoSection, { context: makeContext(), busy: false, guard: { kind: 'rejected', blocked: true, note: null } });

      expect(bar()).toBeNull();
    });

    it('shows the frame at the point in the thumb, and the point in the bar', () => {
      draft.setExtendSourcePath(SOURCE);
      draft.applyExtendSource(described());
      draft.setExtendFrom(76 / 24);
      mountComponent(SourceVideoSection, { context: makeContext(), busy: false, guard: { kind: 'ok', blocked: false, note: null } });

      expect(keeps()).toBe('Keeps 3.2 s of 6.0 s');
      expect(document.body.querySelector('[data-testid="slot-thumb"] img')?.getAttribute('src')).toBe('/preview/source_video/mnt/clips/walk.mp4?at=3.166667');
    });

    it('sets the point in the draft when the handle is released', () => {
      showReady();
      const slider = bar()!;
      slider.getBoundingClientRect = () => ({ left: 0, top: 0, width: 143, height: 48, right: 143, bottom: 48, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;

      slider.dispatchEvent(new PointerEvent('pointerdown', { clientX: 76, pointerType: 'mouse', pointerId: 1, button: 0, bubbles: true }));
      flushSync();
      expect(draft.state.extendFrom).toBeNull();
      slider.dispatchEvent(new PointerEvent('pointerup', { clientX: 76, pointerType: 'mouse', pointerId: 1, bubbles: true }));
      flushSync();

      expect(draft.state.extendFrom).toBeCloseTo(76 / 24, 6);
      expect(keeps()).toBe('Keeps 3.2 s of 6.0 s');
    });
  });
});
