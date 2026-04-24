// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { readFileSync } from 'node:fs';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryAsset, GalleryPage as GalleryPageResponse } from '$lib/types';

const galleryApiMocks = vi.hoisted(() => ({
  getGallery: vi.fn<() => Promise<GalleryPageResponse>>(),
  deleteAsset: vi.fn<(path: string) => Promise<void>>(),
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

function readSource(relativePath: string): string {
  return readFileSync(new URL(relativePath, import.meta.url), 'utf-8');
}

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

describe('GalleryPage active detail selection behavior', () => {
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
    globalThis.IntersectionObserver = class {
      observe(): void {}
      disconnect(): void {}
    } as typeof IntersectionObserver;
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
    document.body.innerHTML = '';
  });

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
    expect(target.textContent).toContain('0 selected');
    expect(checkbox?.checked).toBe(false);
    expect(card?.className).not.toContain('surface-gallery-card-selected');
    expect(card?.className).not.toContain('ring-white/30');

    card!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(routerMocks.replace).toHaveBeenCalledWith('gallery', { selected: asset.path });
    expect(target.textContent).toContain('0 selected');
    expect(checkbox?.checked).toBe(false);
    expect(card?.className).toContain('surface-gallery-card');
    expect(card?.className).toContain('ring-white/30');
    expect(card?.className).not.toContain('surface-gallery-card-selected');
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

    expect(target.textContent).toContain('1 selected');
    expect(checkbox?.checked).toBe(true);
    expect(card?.className).toContain('surface-gallery-card-selected');
    expect(card?.className).not.toContain('ring-white/30');
  });
});

describe('GalleryPage regressions', () => {
  it('threads filter and sort state through backend gallery loads instead of filtering the loaded subset client-side', () => {
    const source = readSource('./GalleryPage.svelte');

    expect(source).toContain('const result = await getGallery(p, filter, sort);');
    expect(source).toContain('const result = await getGallery(page + 1, mediaFilter, sortOrder);');
    expect(source).toContain('loadPage(1, value, sortOrder);');
    expect(source).toContain('loadPage(1, mediaFilter, value);');
    expect(source).not.toMatch(/\.filter\([^\n]*media_type/);
  });

  it('uses bind:value on filter and sort selects so the selected option always reflects reactive state', () => {
    const source = readSource('./GalleryPage.svelte');

    // Both selects must use bind:value, not selected={} attributes on options
    expect(source).toContain('bind:value={mediaFilter}');
    expect(source).toContain('bind:value={sortOrder}');
    // No option-level selected-attribute expressions for these two selects
    expect(source).not.toContain('selected={mediaFilter ===');
    expect(source).not.toContain('selected={sortOrder ===');
  });

  it('uses router.navigate for workspace reuse so URL params reach WorkspacePage.onMount', () => {
    const source = readSource('./GalleryPage.svelte');

    // Must NOT use direct hash assignment (bypasses router and drops params)
    expect(source).not.toContain("window.location.hash = asset.reuse_workspace_url.replace(/^#/, '');");
    // Must extract params from the reuse URL and navigate via the router
    expect(source).toContain("router.navigate('workspace', params)");
    expect(source).toContain('new URLSearchParams(queryStr)');
  });

  it('restores selectedAsset from URL params after the initial page load', () => {
    const source = readSource('./GalleryPage.svelte');

    expect(source).toContain('_pendingSelected');
    expect(source).toContain('router.params');
    // loadPage must check _pendingSelected and match assets
    expect(source).toContain("assets.find((a) => a.path === _pendingSelected)");
  });

  it('returns an IntersectionObserver cleanup function from the sentinel $effect', () => {
    const source = readSource('./GalleryPage.svelte');

    // The $effect must return the disconnect cleanup so old observers are torn down
    expect(source).toContain('return () => io.disconnect();');
    // Old pattern of storing observer in a module-level variable must be gone
    expect(source).not.toContain('observer?.disconnect()');
  });

  it('keeps the fallback reuse notice wired to backend reuse metadata', () => {
    const source = readSource('./GalleryPage.svelte');

    expect(source).toContain('selectedAsset.reuse_state?.fallback_reasons');
    expect(source).toContain('fallback_reasons.length > 0');
    expect(source).toContain('Reuse notice');
    expect(source).toContain('role="alert"');
    expect(source).toContain('Reuse in Workspace');
  });

  it('uses shared token-backed shell and action classes for the live gallery chrome', () => {
    const source = readSource('./GalleryPage.svelte');

    expect(source).toContain('panel-scroll-surface');
    expect(source).toContain('panel-shell panel-shell-right');
    expect(source).toContain('surface-card');
    expect(source).toContain('surface-button-primary');
    expect(source).toContain('surface-button-secondary');
    expect(source).toContain('surface-button-danger');
    expect(source).toContain('surface-overlay-action');
  });

  it('does not conflate the active detail card with batch-selected state', () => {
    const source = readSource('./GalleryPage.svelte');

    expect(source).toContain('@const isSelected = selected.has(asset.path)');
    expect(source).toContain('@const isActive = selectedAsset?.path === asset.path');
    expect(source).toContain('selected={isSelected}');
    expect(source).not.toContain('selected={isSelected || isActive}');
  });

  it('does not keep inline static-element accessibility suppressions in the live gallery path', () => {
    const source = readSource('./GalleryPage.svelte');

    expect(source).not.toContain('a11y_click_events_have_key_events');
    expect(source).not.toContain('a11y_no_static_element_interactions');
    expect(source).not.toContain('a11y_no_noninteractive_element_to_interactive_role');
    expect(source).not.toContain('role="button"');
    expect(source).not.toContain('onclick={() => selectAsset(asset)}');
    expect(source).not.toContain("onclick={(e) => { if (e.target === e.currentTarget) closeLightbox(); }}");
  });

  it('wires fullscreen viewer dismissal through Escape and native close controls', () => {
    const source = readSource('./GalleryPage.svelte');

    expect(source).toContain("if (e.key === 'Escape') { e.preventDefault(); closeLightbox(); }");
    expect(source).toContain("document.addEventListener('keydown', handleEsc);");
    expect(source).toContain('aria-label="Close fullscreen viewer"');
    expect(source).toContain('role="dialog"');
    expect(source).toContain('aria-label="Fullscreen viewer"');
    expect(source.match(/onclick=\{closeLightbox\}/g)?.length ?? 0).toBeGreaterThanOrEqual(2);
  });

  it('ships a captions track for fullscreen gallery video playback', () => {
    const source = readSource('./GalleryPage.svelte');

    expect(source).toContain('<track kind="captions" src="/app-static/empty.vtt" default srclang="en" label="No captions available">');
  });
});