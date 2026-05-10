// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryAsset } from '$lib/types';

import Lightbox from './Lightbox.svelte';

function makeAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return {
    id: 'out/asset-a.png',
    url: '/media/out/asset-a.png',
    thumbnail_url: '/media/out/asset-a.png',
    filename: 'asset-a.png',
    created_at: '2026-04-30T12:00:00Z',
    workflow: 'txt2img',
    prompt: 'Test asset',
    model: 'zit',
    media_type: 'image',
    reuse_workspace_url: '#/workspace?workflow=txt2img',
    ...overrides,
  };
}

describe('Lightbox boundary controls', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
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

  it('keeps previous rendered but disabled on the first asset', () => {
    const onclose = vi.fn();
    const onnavigate = vi.fn();

    app = flushSync(() => mount(Lightbox, {
      target,
      props: {
        assets: [makeAsset(), makeAsset({ id: 'out/asset-b.png', url: '/media/out/asset-b.png', filename: 'asset-b.png' })],
        currentIndex: 0,
        open: true,
        onclose,
        onnavigate,
      },
    }));

    const previousButton = document.querySelector('button[aria-label="Previous asset"]') as HTMLButtonElement | null;
    const nextButton = document.querySelector('button[aria-label="Next asset"]') as HTMLButtonElement | null;

    expect(previousButton).not.toBeNull();
    expect(previousButton?.disabled).toBe(true);
    expect(nextButton).not.toBeNull();
    expect(nextButton?.disabled).toBe(false);
  });

  it('keeps next rendered but disabled on the last asset', () => {
    const onclose = vi.fn();
    const onnavigate = vi.fn();
    const assets = [
      makeAsset(),
      makeAsset({ id: 'out/asset-b.png', url: '/media/out/asset-b.png', filename: 'asset-b.png' }),
    ];

    app = flushSync(() => mount(Lightbox, {
      target,
      props: { assets, currentIndex: 1, open: true, onclose, onnavigate },
    }));

    const previousButton = document.querySelector('button[aria-label="Previous asset"]') as HTMLButtonElement | null;
    const nextButton = document.querySelector('button[aria-label="Next asset"]') as HTMLButtonElement | null;

    expect(previousButton).not.toBeNull();
    expect(previousButton?.disabled).toBe(false);
    expect(nextButton).not.toBeNull();
    expect(nextButton?.disabled).toBe(true);
  });

  it('renders both buttons enabled on a middle asset', () => {
    const onclose = vi.fn();
    const onnavigate = vi.fn();
    const assets = [
      makeAsset(),
      makeAsset({ id: 'out/asset-b.png', url: '/media/out/asset-b.png', filename: 'asset-b.png' }),
      makeAsset({ id: 'out/asset-c.png', url: '/media/out/asset-c.png', filename: 'asset-c.png' }),
    ];

    app = flushSync(() => mount(Lightbox, {
      target,
      props: { assets, currentIndex: 1, open: true, onclose, onnavigate },
    }));

    const previousButton = document.querySelector('button[aria-label="Previous asset"]') as HTMLButtonElement | null;
    const nextButton = document.querySelector('button[aria-label="Next asset"]') as HTMLButtonElement | null;

    expect(previousButton).not.toBeNull();
    expect(previousButton?.disabled).toBe(false);
    expect(nextButton).not.toBeNull();
    expect(nextButton?.disabled).toBe(false);
  });

  it('ArrowLeft does not call onnavigate when already at the first asset', () => {
    const onclose = vi.fn();
    const onnavigate = vi.fn();
    const assets = [makeAsset(), makeAsset({ id: 'out/asset-b.png', url: '/media/out/asset-b.png', filename: 'asset-b.png' })];

    app = flushSync(() => mount(Lightbox, {
      target,
      props: { assets, currentIndex: 0, open: true, onclose, onnavigate },
    }));

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    expect(onnavigate).not.toHaveBeenCalled();
  });

  it('ArrowRight does not call onnavigate when already at the last asset', () => {
    const onclose = vi.fn();
    const onnavigate = vi.fn();
    const assets = [makeAsset(), makeAsset({ id: 'out/asset-b.png', url: '/media/out/asset-b.png', filename: 'asset-b.png' })];

    app = flushSync(() => mount(Lightbox, {
      target,
      props: { assets, currentIndex: 1, open: true, onclose, onnavigate },
    }));

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    expect(onnavigate).not.toHaveBeenCalled();
  });

  it('closes when Escape is pressed while open', () => {
    const onclose = vi.fn();
    const onnavigate = vi.fn();

    app = flushSync(() => mount(Lightbox, {
      target,
      props: {
        assets: [makeAsset()],
        currentIndex: 0,
        open: true,
        onclose,
        onnavigate,
      },
    }));

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

    expect(onclose).toHaveBeenCalledTimes(1);
  });

  it('closes when the backdrop is clicked', () => {
    const onclose = vi.fn();
    const onnavigate = vi.fn();

    app = flushSync(() => mount(Lightbox, {
      target,
      props: {
        assets: [makeAsset()],
        currentIndex: 0,
        open: true,
        onclose,
        onnavigate,
      },
    }));

    const backdrop = document.querySelector('button[aria-label="Close fullscreen viewer"]') as HTMLButtonElement | null;

    expect(backdrop).not.toBeNull();
    backdrop?.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(onclose).toHaveBeenCalledTimes(1);
  });
});