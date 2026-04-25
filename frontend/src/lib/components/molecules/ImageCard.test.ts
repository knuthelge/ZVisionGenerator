import { readFileSync } from 'node:fs';

// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryAsset } from '$lib/types';

import ImageCard from './ImageCard.svelte';

function makeAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return {
    path: '/tmp/output/asset.png',
    url: '/media/asset.png',
    thumbnail_url: '/media/thumb.png',
    filename: 'asset.png',
    created_at: '2026-04-24T12:00:00Z',
    workflow: 'txt2img',
    prompt: 'A calm shoreline at dusk',
    model: 'zit',
    reuse_workspace_url: '#/workspace?workflow=txt2img&prompt=A%20calm%20shoreline%20at%20dusk',
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

function readSource(relativePath: string): string {
  return readFileSync(new URL(relativePath, import.meta.url), 'utf-8');
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

    expect(onactivate).toHaveBeenCalledTimes(2);
    expect(onactivate).toHaveBeenNthCalledWith(1, asset);
    expect(onactivate).toHaveBeenNthCalledWith(2, asset);
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

  it('uses the shared token-backed gallery chrome classes for shell and overlay actions', async () => {
    const asset = makeAsset();

    app = flushSync(() => mount(ImageCard, {
      target,
      props: { asset, selected: true },
    }));
    await settle();

    const card = target.firstElementChild as HTMLElement | null;
    expect(card?.className).toContain('surface-gallery-card-selected');

    card?.dispatchEvent(new MouseEvent('mouseenter', { bubbles: true }));
    await settle();

    const checkbox = target.querySelector('input[type="checkbox"]') as HTMLInputElement | null;
    expect(checkbox?.className).toContain('surface-checkbox');

    const reuseLink = target.querySelector('a[href^="#/workspace"]') as HTMLAnchorElement | null;
    expect(reuseLink?.className).toContain('surface-overlay-action-primary');

    const deleteButton = Array.from(target.querySelectorAll('button')).find((button) => button.getAttribute('aria-label') === `Delete ${asset.filename}`);
    expect(deleteButton?.className).toContain('surface-overlay-action-danger');
  });

  it('renders the active detail state as a separate non-selected ring', async () => {
    const asset = makeAsset();

    app = flushSync(() => mount(ImageCard, {
      target,
      props: { asset, active: true, selected: false },
    }));
    await settle();

    const card = target.firstElementChild as HTMLElement | null;
    const checkbox = target.querySelector('input[type="checkbox"]') as HTMLInputElement | null;

    expect(card?.className).toContain('surface-gallery-card');
    expect(card?.className).toContain('ring-white/30');
    expect(card?.className).not.toContain('surface-gallery-card-selected');
    expect(checkbox?.checked).toBe(false);
  });

  it('suppresses the active ring once the card becomes batch-selected', async () => {
    const asset = makeAsset();

    app = flushSync(() => mount(ImageCard, {
      target,
      props: { asset, active: true, selected: true },
    }));
    await settle();

    const card = target.firstElementChild as HTMLElement | null;
    const checkbox = target.querySelector('input[type="checkbox"]') as HTMLInputElement | null;

    expect(card?.className).toContain('surface-gallery-card-selected');
    expect(card?.className).not.toContain('ring-white/30');
    expect(checkbox?.checked).toBe(true);
  });

  it('keeps the selection checkbox reachable when the hover action overlay is shown', () => {
    const source = readSource('./ImageCard.svelte');

    expect(source).toContain('Selection checkbox (top-left)');
    expect(source).toContain('Hover overlay with action buttons');
    expect(source).toContain('pointer-events-none');
    expect(source).toContain('surface-overlay-action pointer-events-auto');
    expect(source).toContain('surface-overlay-action-primary pointer-events-auto');
    expect(source).toContain('surface-overlay-action-danger pointer-events-auto');
  });
});