import { flushSync } from 'svelte';
import { describe, expect, it, vi } from 'vitest';

import type { GalleryAsset } from '$lib/types';

import AssetTile from './AssetTile.svelte';
import { mountComponent } from '../../../test-utils/mount';

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
  let page: ReturnType<typeof mountComponent>;
  let target: HTMLElement;

  function mountTile(props: Record<string, unknown>): void {
    page = mountComponent(AssetTile, props);
    target = page.target;
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

    await page.unmount();
    mountTile({ asset: makeAsset({ has_reusable_config: false }), onreuse });
    expect((target.querySelector('button[aria-label="Reusable settings unavailable"]') as HTMLButtonElement).disabled).toBe(true);
  });

  it('leads the menu with Reuse settings when reuse is offered, disabled without a reusable config', async () => {
    const onreuse = vi.fn();
    const asset = makeAsset();
    mountTile({ asset, onreuse, ondelete: vi.fn() });
    openMenu(asset);
    expect(menuActions()).toEqual(['reuse', 'download', 'delete']);
    (document.querySelector('[data-action="reuse"]') as HTMLButtonElement).click();
    expect(onreuse).toHaveBeenCalledWith(asset);
    flushSync();

    const plain = makeAsset({ id: 'out/b.png', filename: 'b.png', has_reusable_config: false });
    await page.unmount();
    mountTile({ asset: plain, onreuse });
    openMenu(plain);
    expect(document.querySelector('[data-action="reuse"]')?.getAttribute('aria-disabled')).toBe('true');
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

  it('runs an upscale from the menu, disabling a factor over a limit', () => {
    const asset = makeAsset({
      upscale: {
        factors: [
          { factor: 2, width: 1664, height: 2432, allowed: true, reason: null },
          { factor: 4, width: 3328, height: 4864, allowed: false, reason: 'Too large.' },
        ],
        seedvr2: null,
      },
    });
    const onupscale = vi.fn();
    mountTile({ asset, onreference: vi.fn(), onupscale, ondelete: vi.fn() });

    openMenu(asset);
    expect(document.querySelector('[data-action="upscale-creative-4"]')?.getAttribute('aria-disabled')).toBe('true');

    (document.querySelector('[data-action="upscale-creative-2"]') as HTMLButtonElement).click();
    flushSync();
    expect(onupscale).toHaveBeenCalledWith(asset, 2, 'creative');
  });

  it('hides the reference actions for videos', () => {
    const asset = makeAsset({ media_type: 'video', filename: 'a.mp4', url: '/media/out/a.mp4' });
    mountTile({ asset, onreference: vi.fn(), ondelete: vi.fn() });
    openMenu(asset);
    expect(menuActions()).toEqual(['download', 'delete']);
  });

  describe('extend', () => {
    const video = (overrides: Partial<GalleryAsset> = {}): GalleryAsset =>
      makeAsset({ media_type: 'video', filename: 'a.mp4', url: '/media/out/a.mp4', extend: { available: true, reason: null }, ...overrides });

    it('lists Extend first in a video menu, ahead of Download and Delete', () => {
      const asset = video();
      const onextend = vi.fn();
      mountTile({ asset, onextend, ondelete: vi.fn(), onreference: vi.fn() });
      openMenu(asset);
      expect(menuActions()).toEqual(['extend', 'download', 'delete']);

      (document.querySelector('[data-action="extend"]') as HTMLButtonElement).click();
      flushSync();
      expect(onextend).toHaveBeenCalledWith(asset);
      expect(document.querySelector('[role="menu"]')).toBeNull();
    });

    it('disables Extend with the reason when the video cannot be extended', () => {
      const asset = video({ extend: { available: false, reason: "Upscaled videos can't be extended yet." } });
      const onextend = vi.fn();
      mountTile({ asset, onextend });
      openMenu(asset);

      const item = document.querySelector('[data-action="extend"]') as HTMLElement;
      expect(item.getAttribute('aria-disabled')).toBe('true');
      expect(item.getAttribute('title')).toBe("Upscaled videos can't be extended yet.");
      item.click();
      expect(onextend).not.toHaveBeenCalled();
    });

    it('has no Extend item for an image, an unavailable video, or without a handler', async () => {
      const cases: Array<[GalleryAsset, boolean]> = [
        [makeAsset({ extend: { available: true, reason: null } }), true],
        [video({ extend: null }), true],
        [video(), false],
      ];
      for (const [asset, withHandler] of cases) {
        mountTile({ asset, onextend: withHandler ? vi.fn() : undefined });
        openMenu(asset);
        expect(menuActions()).not.toContain('extend');
        await page.unmount();
        document.body.innerHTML = '';
      }
    });

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

    await page.unmount();
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

    await page.unmount();
    mountTile({ asset, density: 'stage' });
    expect(target.querySelector('img')?.getAttribute('src')).toBe(asset.url);
  });

  it('opens the menu on a right-click, but leaves a video its own menu', async () => {
    const asset = makeAsset();
    mountTile({ asset, ondelete: vi.fn() });
    const rightClick = (el: Element) => { const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 40, clientY: 50 }); el.dispatchEvent(event); flushSync(); return event; };
    expect(rightClick(target.querySelector('img')!).defaultPrevented).toBe(true);
    expect(document.querySelector('[role="menu"]')).not.toBeNull();
    await page.unmount();
    document.body.innerHTML = '';

    const video = makeAsset({ id: 'out/clip.mp4', filename: 'clip.mp4', media_type: 'video' });
    mountTile({ asset: video, density: 'stage', ondelete: vi.fn() });
    expect(rightClick(target.querySelector('video')!).defaultPrevented).toBe(false);
    expect(document.querySelector('[role="menu"]')).toBeNull();
  });
});
