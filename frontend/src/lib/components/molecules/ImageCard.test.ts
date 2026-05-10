// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryAsset } from '$lib/types';

import ImageCard from './ImageCard.svelte';

function makeAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return {
    id: 'nested/asset.png',
    url: '/media/asset.png',
    thumbnail_url: '/media/thumb.png',
    filename: 'asset.png',
    created_at: '2026-04-24T12:00:00Z',
    workflow: 'txt2img',
    prompt: 'A calm shoreline at dusk',
    model: 'zit',
    reuse_workspace_url: '#/workspace?workflow=txt2img&prompt=A%20calm%20shoreline%20at%20dusk',
    has_reusable_config: true,
    media_type: 'image',
    ...overrides,
  };
}

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await Promise.resolve();
  await Promise.resolve();
  flushSync();
}

describe('ImageCard', () => {
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

  it('opens the asset from pointer and keyboard activation', async () => {
    const asset = makeAsset();
    const onactivate = vi.fn();

    app = flushSync(() => mount(ImageCard, {
      target,
      props: { asset, onactivate },
    }));
    await settle();

    const card = target.querySelector('[role="button"]');
    expect(card).not.toBeNull();

    card!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    card!.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter' }));
    card!.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: ' ' }));

    expect(onactivate).toHaveBeenCalledTimes(3);
    expect(onactivate).toHaveBeenNthCalledWith(1, asset);
    expect(onactivate).toHaveBeenNthCalledWith(2, asset);
    expect(onactivate).toHaveBeenNthCalledWith(3, asset);
  });

  it('prevents default scroll on Space activation', async () => {
    const asset = makeAsset();
    const onactivate = vi.fn();

    app = flushSync(() => mount(ImageCard, {
      target,
      props: { asset, onactivate },
    }));
    await settle();

    const card = target.querySelector('[role="button"]') as HTMLElement;
    const event = new KeyboardEvent('keydown', { bubbles: true, key: ' ', cancelable: true });
    card.dispatchEvent(event);

    expect(event.defaultPrevented).toBe(true);
    expect(onactivate).toHaveBeenCalledWith(asset);
  });

  it('toggles selection without bubbling into the card view action', async () => {
    const asset = makeAsset();
    const onactivate = vi.fn();
    const onselect = vi.fn();

    app = flushSync(() => mount(ImageCard, {
      target,
      props: { asset, onactivate, onselect, selected: false },
    }));
    await settle();

    const checkbox = target.querySelector('input[type="checkbox"]') as HTMLInputElement | null;
    expect(checkbox).not.toBeNull();

    checkbox!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(onactivate).not.toHaveBeenCalled();

    expect(onselect).toHaveBeenCalledTimes(1);
    expect(onselect).toHaveBeenCalledWith(asset, true);
  });

  it('keeps hover actions and checkbox clickable without bubbling into card activation', async () => {
    const asset = makeAsset();
    const onactivate = vi.fn();
    const onselect = vi.fn();
    const onopenlightbox = vi.fn();
    const onreuse = vi.fn();
    const ondelete = vi.fn();

    app = flushSync(() => mount(ImageCard, {
      target,
      props: { asset, onactivate, onselect, onopenlightbox, onreuse, ondelete },
    }));
    await settle();

    const card = target.querySelector('[role="button"]') as HTMLElement | null;
    card!.dispatchEvent(new MouseEvent('mouseenter', { bubbles: true }));
    await settle();

    const checkbox = target.querySelector('input[type="checkbox"]') as HTMLInputElement | null;
    checkbox!.checked = true;
    checkbox!.dispatchEvent(new Event('change', { bubbles: true }));

    const fullscreenButton = target.querySelector('button[aria-label="View fullscreen"]') as HTMLButtonElement | null;
    const reuseLink = target.querySelector('a[aria-label="Reuse settings in workspace"]') as HTMLAnchorElement | null;
    const deleteButton = target.querySelector(`button[aria-label="Delete ${asset.filename}"]`) as HTMLButtonElement | null;

    fullscreenButton!.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    reuseLink!.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    deleteButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(onactivate).not.toHaveBeenCalled();
    expect(onselect).toHaveBeenLastCalledWith(asset, true);
    expect(onopenlightbox).toHaveBeenCalledWith(asset);
    expect(onreuse).toHaveBeenCalledWith(asset);
    expect(ondelete).toHaveBeenCalledWith(asset);
  });

  it('disables reuse action when reusable settings are unavailable', async () => {
    const asset = makeAsset({ has_reusable_config: false, reuse_workspace_url: '#/workspace?workflow=txt2img' });
    const onreuse = vi.fn();

    app = flushSync(() => mount(ImageCard, {
      target,
      props: { asset, onreuse },
    }));
    await settle();

    const card = target.querySelector('[role="button"]') as HTMLElement | null;
    card!.dispatchEvent(new MouseEvent('mouseenter', { bubbles: true }));
    await settle();

    expect(target.querySelector('a[aria-label="Reuse settings in workspace"]')).toBeNull();

    const unavailableButton = target.querySelector('button[aria-label="Reusable settings unavailable"]') as HTMLButtonElement | null;
    expect(unavailableButton).not.toBeNull();
    expect(unavailableButton?.disabled).toBe(true);
    unavailableButton!.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));

    expect(onreuse).not.toHaveBeenCalled();
  });
});