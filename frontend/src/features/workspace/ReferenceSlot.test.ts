// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { settle } from '../../test-utils/mount';
import ReferenceSlot from './ReferenceSlot.svelte';

vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);

const NAMES = { browse: 'Browse for a start image', upload: 'Upload a start image from this device', clear: 'Remove start image', preview: 'Start image preview' };

describe('ReferenceSlot', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  function show(props: Record<string, unknown>): void {
    app = flushSync(() => mount(ReferenceSlot, {
      target,
      props: { idBase: 'ws-image', label: 'Start image', names: NAMES, emptyText: '', path: null, file: null, onfilechange: vi.fn(), onpathchange: vi.fn(), ...props },
    }));
  }

  const group = () => target.querySelector('[role="group"]') as HTMLElement;
  const nameText = () => target.querySelector('.reference-name')?.textContent;

  beforeEach(() => {
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:test-image') });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() });
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
    document.body.innerHTML = '';
  });

  it('names its group by the label alone while it holds no image', () => {
    show({});
    expect(group().getAttribute('aria-label')).toBe('Start image');
  });

  describe('the thumbnail of an image', () => {
    const thumbImage = () => target.querySelector('[data-testid="slot-thumb"] img');

    it('shows the picture of a path outside the output folder through the server', () => {
      show({ path: '/mnt/photos/fox.png', name: 'fox.png' });

      expect(thumbImage()?.getAttribute('src')).toBe('/preview/reference_image/mnt/photos/fox.png');
      expect(target.querySelector('[data-testid="slot-thumb"] [role="img"]')?.getAttribute('aria-label')).toBe('Start image preview');
    });

    it('shows an uploaded file from its object URL, ahead of a path', () => {
      show({ path: '/mnt/photos/fox.png', file: new File(['x'], 'up.png', { type: 'image/png' }), name: 'up.png' });

      expect(thumbImage()?.getAttribute('src')).toBe('blob:test-image');
    });

    it('keeps the empty icon while the slot holds nothing', () => {
      show({ emptyIcon: 'plus' });
      expect(thumbImage()).toBeNull();
      expect(target.querySelector('[data-testid="slot-thumb"] svg')).not.toBeNull();
    });

    it('prefers the alert state over the picture of a missing file', () => {
      show({ path: '/mnt/photos/gone.png', name: 'gone.png', missing: true });

      expect(thumbImage()).toBeNull();
      expect(target.querySelector('[data-testid="slot-thumb"][data-missing="true"] svg')).not.toBeNull();
    });
  });

  describe('the thumbnail of an audio track', () => {
    it('keeps the alert state for a missing track', async () => {
      show({ kind: 'audio', path: '/mnt/music/gone.flac', name: 'gone.flac', missing: true });
      await settle();

      expect(target.querySelector('[data-testid="slot-thumb"][data-missing="true"] svg')).not.toBeNull();
    });
  });
});
