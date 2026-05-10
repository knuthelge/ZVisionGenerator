// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryAsset, GalleryPage as GalleryPageResponse } from '$lib/types';

const galleryApiMocks = vi.hoisted(() => ({
  getGallery: vi.fn<() => Promise<GalleryPageResponse>>(),
  deleteAsset: vi.fn<(assetId: string) => Promise<void>>(),
}));

const routerMocks = vi.hoisted(() => ({
  params: {} as Record<string, string>,
  replace: vi.fn<(page: string, params?: Record<string, string>) => void>(),
  navigate: vi.fn<(page: string, params?: Record<string, string>) => void>(),
}));

vi.mock('$lib/api/gallery', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/gallery')>();
  return {
    ...actual,
    getGallery: galleryApiMocks.getGallery,
    deleteAsset: galleryApiMocks.deleteAsset,
  };
});

vi.mock('$lib/state/router.svelte', () => ({
  router: {
    get params(): Record<string, string> {
      return routerMocks.params;
    },
    replace: routerMocks.replace,
    navigate: routerMocks.navigate,
  },
}));

vi.mock('$lib/state/toasts.svelte', () => ({
  addToast: vi.fn(),
}));

import GalleryPage from './GalleryPage.svelte';

class MockIntersectionObserver {
  static instances: MockIntersectionObserver[] = [];

  observe = vi.fn();
  disconnect = vi.fn();

  constructor(private readonly callback: IntersectionObserverCallback) {
    MockIntersectionObserver.instances.push(this);
  }

  trigger(isIntersecting: boolean): void {
    this.callback([{ isIntersecting } as IntersectionObserverEntry], this as unknown as IntersectionObserver);
  }
}

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

function getSelectedAssetViewerButton(container: ParentNode): HTMLButtonElement | null {
  return container.querySelector('#gallery-details button[type="button"]') as HTMLButtonElement | null;
}

function queryButtonByName(container: ParentNode, name: string): HTMLButtonElement | null {
  return Array.from(container.querySelectorAll('button')).find((button) => button.textContent?.trim() === name) ?? null;
}

let target: HTMLDivElement;
let app: Record<string, unknown> | null = null;

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
  galleryApiMocks.getGallery.mockReset();
  galleryApiMocks.deleteAsset.mockReset();
  routerMocks.params = {};
  routerMocks.replace.mockReset();
  routerMocks.navigate.mockReset();
  MockIntersectionObserver.instances = [];
  globalThis.IntersectionObserver = MockIntersectionObserver as unknown as typeof IntersectionObserver;
});

afterEach(async () => {
  if (app) {
    await unmount(app);
    app = null;
  }
  target.remove();
  document.body.innerHTML = '';
});

describe('GalleryPage active detail selection behavior', () => {
  it('keeps active detail separate from batch selection until the checkbox is toggled', async () => {
    const asset = makeAsset();
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [asset],
      page: 1,
      total_pages: 1,
      total_count: 1,
    });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    const card = target.querySelector(`[aria-label="Asset: ${asset.filename}"]`) as HTMLElement | null;
    const checkbox = target.querySelector(`input[aria-label="Select ${asset.filename}"]`) as HTMLInputElement | null;

    expect(card).not.toBeNull();
    expect(checkbox).not.toBeNull();
    expect(checkbox?.checked).toBe(false);

    card!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(routerMocks.replace).toHaveBeenCalledWith('gallery', { selected: asset.id });
    expect(checkbox?.checked).toBe(false);
    expect(target.textContent).toContain(asset.prompt);
  });

  it('updates the checkbox state and batch count only for true batch selections', async () => {
    const asset = makeAsset();
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [asset],
      page: 1,
      total_pages: 1,
      total_count: 1,
    });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    const card = target.querySelector(`[aria-label="Asset: ${asset.filename}"]`) as HTMLElement | null;
    const checkbox = target.querySelector(`input[aria-label="Select ${asset.filename}"]`) as HTMLInputElement | null;

    expect(card).not.toBeNull();
    expect(checkbox).not.toBeNull();

    card!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    checkbox!.checked = true;
    checkbox!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();

    expect(checkbox?.checked).toBe(true);
  });
});

describe('GalleryPage regressions', () => {
  it('threads filter and sort state through backend gallery loads and pagination', async () => {
    const asset = makeAsset();
    const filteredAsset = makeAsset({ id: 'nested/video-1.mp4', filename: 'video-1.mp4', media_type: 'video' });
    const pagedAsset = makeAsset({ id: 'nested/video-2.mp4', filename: 'video-2.mp4', media_type: 'video' });
    galleryApiMocks.getGallery
      .mockResolvedValueOnce({ assets: [asset], page: 1, total_pages: 2, total_count: 3 })
      .mockResolvedValueOnce({ assets: [filteredAsset], page: 1, total_pages: 1, total_count: 1 })
      .mockResolvedValueOnce({ assets: [filteredAsset], page: 1, total_pages: 2, total_count: 2 })
      .mockResolvedValueOnce({ assets: [pagedAsset], page: 2, total_pages: 2, total_count: 2 });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    const filterSelect = target.querySelector('select[aria-label="Filter gallery media"]') as HTMLSelectElement | null;
    const sortSelect = target.querySelector('select[aria-label="Sort gallery assets"]') as HTMLSelectElement | null;
    expect(filterSelect).not.toBeNull();
    expect(sortSelect).not.toBeNull();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(1, 1, 'all', 'newest');

    filterSelect!.value = 'video';
    filterSelect!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(2, 1, 'video', 'newest');
    expect(filterSelect!.value).toBe('video');

    sortSelect!.value = 'oldest';
    sortSelect!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(3, 1, 'video', 'oldest');
    expect(sortSelect!.value).toBe('oldest');

    MockIntersectionObserver.instances.at(-1)?.trigger(true);
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(4, 2, 'video', 'oldest');
    expect(target.textContent).toContain('video-2.mp4');
  });

  it('navigates to workspace reuse through parsed router params', async () => {
    const asset = makeAsset({ reuse_workspace_url: '#/workspace?workflow=img2img&prompt=Reuse%20me&model=zit&seed=9876' });
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [asset],
      page: 1,
      total_pages: 1,
      total_count: 1,
    });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    const card = target.querySelector(`[aria-label="Asset: ${asset.filename}"]`) as HTMLElement | null;
    card!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    const reuseButton = queryButtonByName(target, 'Reuse in Workspace');
    expect(reuseButton).not.toBeNull();

    reuseButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(routerMocks.navigate).toHaveBeenCalledWith('workspace', {
      workflow: 'img2img',
      prompt: 'Reuse me',
      model: 'zit',
      seed: '9876',
    });
  });

  it('shows a recovery action when the current filter has no results', async () => {
    const asset = makeAsset();
    galleryApiMocks.getGallery
      .mockResolvedValueOnce({ assets: [asset], page: 1, total_pages: 1, total_count: 1 })
      .mockResolvedValueOnce({ assets: [], page: 1, total_pages: 1, total_count: 0 })
      .mockResolvedValueOnce({ assets: [asset], page: 1, total_pages: 1, total_count: 1 });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    const filterSelect = target.querySelector('select[aria-label="Filter gallery media"]') as HTMLSelectElement | null;
    filterSelect!.value = 'video';
    filterSelect!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();

    expect(target.textContent).toContain('No matching assets');

    const clearButton = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Show All Media');
    expect(clearButton).not.toBeUndefined();
    clearButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(3, 1, 'all', 'newest');
    expect(filterSelect!.value).toBe('all');
  });

  it('shows a workspace recovery action when the gallery is genuinely empty', async () => {
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [], page: 1, total_pages: 1, total_count: 0 });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    expect(target.textContent).toContain('No generated assets yet');

    const workspaceButton = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Open Workspace');
    expect(workspaceButton).not.toBeUndefined();
    workspaceButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(routerMocks.navigate).toHaveBeenCalledWith('workspace');
  });

  it('restores the selected asset from router params after the initial page load', async () => {
    const asset = makeAsset();
    routerMocks.params = { selected: asset.id };
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [asset],
      page: 1,
      total_pages: 1,
      total_count: 1,
    });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    expect(target.textContent).toContain(asset.prompt);
  });

  it('restores the selected asset after remount when the router keeps the selected id', async () => {
    const asset = makeAsset();
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [asset],
      page: 1,
      total_pages: 1,
      total_count: 1,
    });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    const card = target.querySelector(`[aria-label="Asset: ${asset.filename}"]`) as HTMLElement | null;
    card!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(routerMocks.replace).toHaveBeenCalledWith('gallery', { selected: asset.id });
    routerMocks.params = { selected: asset.id };

    await unmount(app!);
    app = null;

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    expect(target.textContent).toContain(asset.prompt);
  });

  it('disconnects the infinite-scroll observer when the page unmounts', async () => {
    const asset = makeAsset();
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [asset],
      page: 1,
      total_pages: 2,
      total_count: 2,
    });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    const observer = MockIntersectionObserver.instances.at(-1);
    expect(observer).toBeDefined();
    expect(observer?.observe).toHaveBeenCalled();

    await unmount(app!);
    app = null;

    expect(observer?.disconnect).toHaveBeenCalled();
  });

  it('shows backend reuse reasons in the selected asset panel', async () => {
    const asset = makeAsset({
      reuse_state: {
        requested_workflow: 'img2img',
        resolved_workflow: 'txt2img',
        workflow_available: false,
        requested_model: 'missing-model',
        resolved_model: 'zit',
        model_available: false,
        fallback_reasons: ['workflow_media_mismatch', 'model_not_configured'],
      },
    });
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [asset],
      page: 1,
      total_pages: 1,
      total_count: 1,
    });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    const card = target.querySelector(`[aria-label="Asset: ${asset.filename}"]`) as HTMLElement | null;
    card!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    const alert = target.querySelector('[role="alert"]');
    expect(alert).not.toBeNull();
    expect(alert?.textContent).toContain('workflow_media_mismatch');
    expect(alert?.textContent).toContain('model_not_configured');
  });

  it('shows a display prompt without offering reuse when embedded config is missing', async () => {
    const asset = makeAsset({
      model: 'Unavailable',
      prompt: 'Display-only prompt',
      reuse_workspace_url: '#/workspace?workflow=txt2img',
      has_reusable_config: false,
      reuse_state: {
        requested_workflow: 'txt2img',
        resolved_workflow: 'txt2img',
        workflow_available: true,
        requested_model: null,
        resolved_model: null,
        model_available: true,
        fallback_reasons: [],
      },
    });
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [asset],
      page: 1,
      total_pages: 1,
      total_count: 1,
    });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    const card = target.querySelector(`[aria-label="Asset: ${asset.filename}"]`) as HTMLElement | null;
    card!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(target.querySelector('[role="alert"]')).toBeNull();
    expect(target.textContent).toContain('Unavailable');
    expect(target.textContent).toContain('Display-only prompt');
    expect(target.textContent).toContain('Reusable settings unavailable');

    expect(queryButtonByName(target, 'Reuse in Workspace')).toBeNull();
    expect(routerMocks.navigate).not.toHaveBeenCalled();
  });

  it('opens the fullscreen viewer and closes it with Escape', async () => {
    const asset = makeAsset();
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [asset],
      page: 1,
      total_pages: 1,
      total_count: 1,
    });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    const card = target.querySelector(`[aria-label="Asset: ${asset.filename}"]`) as HTMLElement | null;
    card!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    const openButton = getSelectedAssetViewerButton(target);
    expect(openButton).not.toBeNull();
    openButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(document.querySelector('[role="dialog"]')).not.toBeNull();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await settle();

    expect(document.querySelector('[role="dialog"]')).toBeNull();
  });
});

describe('GalleryPage lightbox navigation (REC-UX-003)', () => {
  it('renders Previous button as disabled and Next as enabled on the first asset', async () => {
    const assetA = makeAsset({ id: 'out/a.png', url: '/media/out/a.png', filename: 'a.png' });
    const assetB = makeAsset({ id: 'out/b.png', url: '/media/out/b.png', filename: 'b.png' });
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [assetA, assetB],
      page: 1,
      total_pages: 1,
      total_count: 2,
    });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    // Select first asset and open lightbox
    const cardA = target.querySelector(`[aria-label="Asset: ${assetA.filename}"]`) as HTMLElement | null;
    cardA!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    const openBtn = getSelectedAssetViewerButton(target);
    expect(openBtn).not.toBeNull();
    openBtn!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    // First asset: Prev rendered but disabled, Next rendered and enabled
    const prevBtn = document.querySelector('button[aria-label="Previous asset"]') as HTMLButtonElement | null;
    const nextBtn = document.querySelector('button[aria-label="Next asset"]') as HTMLButtonElement | null;
    expect(prevBtn).not.toBeNull();
    expect(prevBtn?.disabled).toBe(true);
    expect(nextBtn).not.toBeNull();
    expect(nextBtn?.disabled).toBe(false);
  });

  it('clicking Next navigates to the next asset and syncs gallery selection and URL', async () => {
    const assetA = makeAsset({ id: 'out/a.png', url: '/media/out/a.png', filename: 'a.png', prompt: 'first' });
    const assetB = makeAsset({ id: 'out/b.png', url: '/media/out/b.png', filename: 'b.png', prompt: 'second' });
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [assetA, assetB],
      page: 1,
      total_pages: 1,
      total_count: 2,
    });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    const cardA = target.querySelector(`[aria-label="Asset: ${assetA.filename}"]`) as HTMLElement | null;
    cardA!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    routerMocks.replace.mockReset();

    const openBtn = getSelectedAssetViewerButton(target);
    expect(openBtn).not.toBeNull();
    openBtn!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    const nextBtn = document.querySelector('button[aria-label="Next asset"]') as HTMLButtonElement | null;
    expect(nextBtn).not.toBeNull();
    nextBtn!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    // Gallery route is updated to reflect the new selection
    expect(routerMocks.replace).toHaveBeenCalledWith('gallery', { selected: assetB.id });
  });

  it('ArrowRight keyboard event navigates to the next asset in the lightbox', async () => {
    const assetA = makeAsset({ id: 'out/a.png', url: '/media/out/a.png', filename: 'a.png' });
    const assetB = makeAsset({ id: 'out/b.png', url: '/media/out/b.png', filename: 'b.png' });
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [assetA, assetB],
      page: 1,
      total_pages: 1,
      total_count: 2,
    });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    const cardA = target.querySelector(`[aria-label="Asset: ${assetA.filename}"]`) as HTMLElement | null;
    cardA!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    routerMocks.replace.mockReset();

    const openBtn = getSelectedAssetViewerButton(target);
    expect(openBtn).not.toBeNull();
    openBtn!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    await settle();

    expect(routerMocks.replace).toHaveBeenCalledWith('gallery', { selected: assetB.id });
  });

  it('ArrowLeft keyboard event navigates to the previous asset in the lightbox', async () => {
    const assetA = makeAsset({ id: 'out/a.png', url: '/media/out/a.png', filename: 'a.png' });
    const assetB = makeAsset({ id: 'out/b.png', url: '/media/out/b.png', filename: 'b.png' });
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [assetA, assetB],
      page: 1,
      total_pages: 1,
      total_count: 2,
    });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    // Select second asset first
    const cardB = target.querySelector(`[aria-label="Asset: ${assetB.filename}"]`) as HTMLElement | null;
    cardB!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    routerMocks.replace.mockReset();

    const openBtn = getSelectedAssetViewerButton(target);
    expect(openBtn).not.toBeNull();
    openBtn!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    await settle();

    expect(routerMocks.replace).toHaveBeenCalledWith('gallery', { selected: assetA.id });
  });

  it('shows both Prev and Next buttons when navigated to a middle asset', async () => {
    const assetA = makeAsset({ id: 'out/a.png', url: '/media/out/a.png', filename: 'a.png' });
    const assetB = makeAsset({ id: 'out/b.png', url: '/media/out/b.png', filename: 'b.png' });
    const assetC = makeAsset({ id: 'out/c.png', url: '/media/out/c.png', filename: 'c.png' });
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [assetA, assetB, assetC],
      page: 1,
      total_pages: 1,
      total_count: 3,
    });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    // Open lightbox on asset B (middle)
    const cardB = target.querySelector(`[aria-label="Asset: ${assetB.filename}"]`) as HTMLElement | null;
    cardB!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    const openBtn = getSelectedAssetViewerButton(target);
    expect(openBtn).not.toBeNull();
    openBtn!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(document.querySelector('button[aria-label="Previous asset"]')).not.toBeNull();
    expect(document.querySelector('button[aria-label="Next asset"]')).not.toBeNull();
  });

  it('disables Next button at last asset and keeps Previous enabled', async () => {
    const assetA = makeAsset({ id: 'out/a.png', url: '/media/out/a.png', filename: 'a.png' });
    const assetB = makeAsset({ id: 'out/b.png', url: '/media/out/b.png', filename: 'b.png' });
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [assetA, assetB],
      page: 1,
      total_pages: 1,
      total_count: 2,
    });

    app = flushSync(() => mount(GalleryPage, { target }));
    await settle();

    // Open lightbox on last asset
    const cardB = target.querySelector(`[aria-label="Asset: ${assetB.filename}"]`) as HTMLElement | null;
    cardB!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    const openBtn = getSelectedAssetViewerButton(target);
    expect(openBtn).not.toBeNull();
    openBtn!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    const prevBtnLast = document.querySelector('button[aria-label="Previous asset"]') as HTMLButtonElement | null;
    const nextBtnLast = document.querySelector('button[aria-label="Next asset"]') as HTMLButtonElement | null;
    expect(prevBtnLast).not.toBeNull();
    expect(prevBtnLast?.disabled).toBe(false);
    expect(nextBtnLast).not.toBeNull();
    expect(nextBtnLast?.disabled).toBe(true);
  });
});

