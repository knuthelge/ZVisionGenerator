// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryAsset } from '$lib/types';

import AssetTile from './AssetTile.svelte';

function makeAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return {
    id: 'out/a.png',
    url: '/media/out/a.png',
    thumbnail_url: '/media/out/a-thumb.png',
    filename: 'a.png',
    created_at: '2026-10-03T12:00:00Z',
    workflow: 'txt2img',
    prompt: 'a fox',
    model: 'zit',
    media_type: 'image',
    file_path: '/outputs/out/a.png',
    has_reusable_config: true,
    reuse_workspace_url: '#/workspace?workflow=txt2img',
    ...overrides,
  };
}

function menuActions(): string[] {
  return Array.from(document.querySelectorAll('[role="menu"] [role="menuitem"]')).map((item) => item.getAttribute('data-action') ?? '');
}

describe('AssetTile', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.remove();
    document.body.innerHTML = '';
  });

  function mountTile(props: Record<string, unknown>): void {
    app = flushSync(() => mount(AssetTile, { target, props }));
  }

  function openMenu(asset: GalleryAsset): void {
    (target.querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
    flushSync();
  }

  it('previews from the media and the expand action with the clicked trigger', () => {
    const asset = makeAsset();
    const onpreview = vi.fn();
    mountTile({ asset, onpreview });

    const media = target.querySelector(`button[aria-label="View ${asset.filename}"]`) as HTMLButtonElement;
    media.click();
    expect(onpreview).toHaveBeenLastCalledWith(asset, media);

    (target.querySelector(`button[aria-label="View ${asset.filename} fullscreen"]`) as HTMLButtonElement).click();
    expect(onpreview).toHaveBeenCalledTimes(2);
  });

  it('reuses settings, and disables reuse without a reusable config', async () => {
    const onreuse = vi.fn();
    const asset = makeAsset();
    mountTile({ asset, onreuse });
    (target.querySelector(`button[aria-label="Reuse settings from ${asset.filename}"]`) as HTMLButtonElement).click();
    expect(onreuse).toHaveBeenCalledWith(asset);

    await unmount(app!);
    mountTile({ asset: makeAsset({ has_reusable_config: false }), onreuse });
    expect((target.querySelector('button[aria-label="Reusable settings unavailable"]') as HTMLButtonElement).disabled).toBe(true);
  });

  it('offers reference, download and delete for images in the overflow menu', () => {
    const asset = makeAsset();
    const onreference = vi.fn();
    const ondelete = vi.fn();
    mountTile({ asset, onreference, ondelete });

    openMenu(asset);
    expect(menuActions()).toEqual(['reference-image', 'reference-video', 'download', 'delete']);
    expect(document.querySelector('[data-action="download"]')?.getAttribute('href')).toBe(asset.url);

    (document.querySelector('[data-action="reference-video"]') as HTMLButtonElement).click();
    flushSync();
    expect(onreference).toHaveBeenCalledWith(asset, 'video');
    expect(document.querySelector('[role="menu"]')).toBeNull();

    openMenu(asset);
    (document.querySelector('[data-action="delete"]') as HTMLButtonElement).click();
    expect(ondelete).toHaveBeenCalledWith(asset);
  });

  it('offers upscale factors after the reference actions, disabling one over a limit', () => {
    const asset = makeAsset({
      upscale: {
        factors: [
          { factor: 2, width: 1664, height: 2432, allowed: true, reason: null },
          { factor: 4, width: 3328, height: 4864, allowed: false, reason: 'Too large.' },
        ],
      },
    });
    const onupscale = vi.fn();
    mountTile({ asset, onreference: vi.fn(), onupscale, ondelete: vi.fn() });

    openMenu(asset);
    expect(menuActions()).toEqual(['reference-image', 'reference-video', 'upscale-2', 'upscale-4', 'download', 'delete']);
    expect(document.querySelector('[data-action="upscale-4"]')?.getAttribute('aria-disabled')).toBe('true');

    (document.querySelector('[data-action="upscale-2"]') as HTMLButtonElement).click();
    flushSync();
    expect(onupscale).toHaveBeenCalledWith(asset, 2);
  });

  it('hides the upscale actions without upscale options', () => {
    const asset = makeAsset();
    mountTile({ asset, onupscale: vi.fn() });
    openMenu(asset);
    expect(menuActions()).toEqual(['download']);
  });

  it('hides the reference actions for videos', () => {
    const asset = makeAsset({ media_type: 'video', filename: 'a.mp4', url: '/media/out/a.mp4' });
    mountTile({ asset, onreference: vi.fn(), ondelete: vi.fn() });
    openMenu(asset);
    expect(menuActions()).toEqual(['download', 'delete']);
  });

  it('closes the menu with Escape and returns focus to its button', async () => {
    const asset = makeAsset();
    mountTile({ asset, ondelete: vi.fn() });
    openMenu(asset);
    await Promise.resolve();

    document.querySelector('[role="menu"]')!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    flushSync();
    await Promise.resolve();

    expect(document.querySelector('[role="menu"]')).toBeNull();
    expect(document.activeElement?.getAttribute('aria-label')).toBe(`More actions for ${asset.filename}`);
  });

  it('shows a selection checkbox only when selection is supported', async () => {
    const asset = makeAsset();
    const onselect = vi.fn();
    mountTile({ asset });
    expect(target.querySelector('input[type="checkbox"]')).toBeNull();

    await unmount(app!);
    mountTile({ asset, onselect });
    const checkbox = target.querySelector(`input[aria-label="Select ${asset.filename}"]`) as HTMLInputElement;
    checkbox.checked = true;
    checkbox.dispatchEvent(new Event('change', { bubbles: true }));
    expect(onselect).toHaveBeenCalledWith(asset, true);
  });

  it('shows the full-size file on the stage and the thumbnail elsewhere', async () => {
    const asset = makeAsset();
    mountTile({ asset, density: 'compact' });
    expect(target.querySelector('img')?.getAttribute('src')).toBe(asset.thumbnail_url);

    await unmount(app!);
    mountTile({ asset, density: 'stage' });
    expect(target.querySelector('img')?.getAttribute('src')).toBe(asset.url);
  });
});
