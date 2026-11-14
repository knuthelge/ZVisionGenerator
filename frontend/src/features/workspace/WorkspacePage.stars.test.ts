// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { reactiveProps } from '../../test-utils/reactiveProps.svelte';
import { draft } from '$lib/state/draft.svelte';
import { galleryStore } from '$lib/state/gallery.svelte';
import { jobStore } from '$lib/state/job.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import type { GalleryAsset, GalleryPage, PromptEnhancerContract, WorkspaceContext } from '$lib/types';

import WorkspacePage from './WorkspacePage.svelte';
import { chooseInBrowser, listingWith, placesFor } from '../../test-utils/pathBrowser';
import { lastEventSource } from '../../test-utils/eventSource';
import { EMPTY_AUDIO_MATRIX, makeAsset, makeContext, makeGalleryPage } from '../../test-utils/workspace';
import { createHost, mockClipboard, mountWorkspacePage, resetWorkspaceTest, settle, submitForm } from '../../test-utils/workspaceHarness';
import { browseApiMocks, extendApiMocks, galleryApiMocks, layoutProbe, workspaceApiMocks } from '../../test-utils/workspaceMocks';

vi.mock('$lib/state/layoutMode.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/state/layoutMode.svelte')>();
  const { layoutProbe } = await import('../../test-utils/workspaceMocks');
  return {
    ...actual,
    layout: {
      get mode() { return layoutProbe.compact ? 'compact' : 'wide'; },
      get compact() { return layoutProbe.compact; },
      get coarse() { return layoutProbe.compact; },
    },
  };
});

vi.mock('$lib/api/config', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/config')>();
  const { configApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, updateConfig: configApiMocks.updateConfig };
});

vi.mock('$lib/api/extend', async () => {
  const { extendApiMocks } = await import('../../test-utils/workspaceMocks');
  return { inspectExtendSource: extendApiMocks.inspectExtendSource };
});

vi.mock('$lib/api/videoMemory', async () => {
  const { videoMemoryApiMocks } = await import('../../test-utils/workspaceMocks');
  return { checkVideoMemoryFit: videoMemoryApiMocks.checkVideoMemoryFit };
});

vi.mock('$lib/api/workspace', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/workspace')>();
  const { workspaceApiMocks: mocks } = await import('../../test-utils/workspaceMocks');
  return {
    ...actual,
    getWorkspaceContext: mocks.getWorkspaceContext,
    getWorkspaceCoreContext: mocks.getWorkspaceCoreContext,
    submitGenerate: mocks.submitGenerate,
    planGenerate: mocks.planGenerate,
    getJobSnapshot: mocks.getJobSnapshot,
    parseUrlPrefill: mocks.parseUrlPrefill,
    checkReferencePaths: mocks.checkReferencePaths,
    listJobs: mocks.listJobs,
    clearQueue: mocks.clearQueue,
    cancelJob: mocks.cancelJob,
    skipFirstRun: mocks.skipFirstRun,
  };
});

vi.mock('$lib/api/gallery', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/gallery')>();
  const { galleryApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, getGallery: galleryApiMocks.getGallery, deleteAssets: galleryApiMocks.deleteAssets, setStarred: galleryApiMocks.setStarred };
});

vi.mock('$lib/api/promptEnhance', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/promptEnhance')>();
  const { enhanceApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, enhancePrompt: enhanceApiMocks.enhancePrompt };
});

vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);

vi.mock('$lib/api/presets', async () => (await import('../../test-utils/workspaceMocks')).presetApiMocks);

vi.mock('$lib/api/models', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/models')>();
  const { modelApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, getDownloads: modelApiMocks.getDownloads };
});

vi.mock('$lib/api/promptFiles', async () => {
  const { promptFileApiMocks } = await import('../../test-utils/workspaceMocks');
  return {
    inspectPromptFile: promptFileApiMocks.inspectPromptFile,
    writePromptFile: promptFileApiMocks.writePromptFile,
    appendPromptEntry: promptFileApiMocks.appendPromptEntry,
    createPromptFile: promptFileApiMocks.createPromptFile,
  };
});

let target: HTMLDivElement;
let app: Record<string, unknown> | null = null;

beforeEach(() => {
  resetWorkspaceTest();
  target = createHost();
  galleryApiMocks.setStarred.mockImplementation(async (id, starred) => ({ id, starred }));
  workspaceApiMocks.submitGenerate.mockResolvedValue({ job_id: 'job-s', workflow: 'txt2img', prompt: 'Test prompt', model: 'zit', runs: 1, created_at: '2026-11-08T10:00:00Z' });
});

afterEach(async () => {
  if (app) { await unmount(app); app = null; }
  target.remove();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

const asset = (name: string, overrides: Partial<GalleryAsset> = {}) =>
  makeAsset({ id: `out/${name}`, filename: name, url: `/media/out/${name}`, thumbnail_url: `/media/out/${name}`, ...overrides });

async function mountWithHistory(assets: GalleryAsset[]): Promise<void> {
  galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage(assets));
  app = await mountWorkspacePage(makeContext({ history_assets: assets }), target);
}

async function finishJob(outputs: GalleryAsset[]): Promise<void> {
  target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  await settle();
  lastEventSource().emit('job_completed', { type: 'job_completed', job_id: 'job-s', total_runs: outputs.length, outputs });
  await settle();
}

const starButton = (scope: ParentNode, filename: string) => scope.querySelector<HTMLButtonElement>(`button[aria-label="Star ${filename}"]`);
const pressed = (button: Element | null) => button?.getAttribute('aria-pressed');

describe('WorkspacePage stars', () => {
  describe('stage', () => {
    it('stars the file shown on the stage from its Star button', async () => {
      const [a, b] = [asset('a.png'), asset('b.png')];
      await mountWithHistory([a, b]);
      const button = starButton(target.querySelector('.latest-region')!, 'a.png');
      expect(pressed(button)).toBe('false');

      button!.click();
      await settle();

      expect(galleryApiMocks.setStarred).toHaveBeenCalledWith('out/a.png', true);
      expect(pressed(starButton(target.querySelector('.latest-region')!, 'a.png'))).toBe('true');
    });

    it('stars a completed output of the run, and the star stays when the grid re-renders', async () => {
      await mountWithHistory([]);
      const [a, b] = [asset('a.png'), asset('b.png')];
      await finishJob([a, b]);
      const grid = target.querySelector('.completed-output-grid')!;
      expect(pressed(starButton(grid, 'b.png'))).toBe('false');

      starButton(grid, 'b.png')!.click();
      await settle();

      expect(galleryApiMocks.setStarred).toHaveBeenCalledWith('out/b.png', true);
      expect(pressed(starButton(grid, 'b.png'))).toBe('true');
      expect(pressed(starButton(grid, 'a.png'))).toBe('false');
      expect(jobStore.current?.outputs?.find((output) => output.id === 'out/b.png')?.starred).toBe(true);
    });
  });

  describe('strip', () => {
    const strip = () => target.querySelector('#ws-history-shell')!;
    const stripTile = (filename: string) => strip().querySelector(`.asset-tile[aria-label="${filename}"]`)!;

    it('keeps the tile in the strip when it is unstarred from its menu or starred from the viewer', async () => {
      await mountWithHistory([asset('a.png', { starred: true }), asset('b.png')]);
      const tileIds = () => Array.from(strip().querySelectorAll('.asset-tile')).map((tile) => tile.getAttribute('data-asset-id'));

      stripTile('a.png').querySelector<HTMLButtonElement>('button[aria-label="More actions for a.png"]')!.click();
      flushSync();
      document.querySelector<HTMLElement>('[role="menu"] [data-action="star"]')!.click();
      await settle();

      expect(galleryApiMocks.setStarred).toHaveBeenLastCalledWith('out/a.png', false);
      expect(tileIds()).toEqual(['out/a.png', 'out/b.png']);

      strip().querySelector<HTMLButtonElement>('button[aria-label="View b.png"]')!.click();
      await settle();
      document.querySelector<HTMLButtonElement>('[data-testid="asset-viewer"] button[data-action="star"]')!.click();
      await settle();

      expect(galleryApiMocks.setStarred).toHaveBeenLastCalledWith('out/b.png', true);
      expect(tileIds()).toEqual(['out/a.png', 'out/b.png']);
      expect(document.querySelector('.viewer-title small')?.textContent).toContain('2 of 2');
    });
  });

  describe('viewer', () => {
    const viewerToggle = () => document.querySelector<HTMLButtonElement>('[data-testid="asset-viewer"] button[data-action="star"]');

    it('stars a job output from the viewer toggle and its S key', async () => {
      await mountWithHistory([]);
      await finishJob([asset('a.png'), asset('b.png')]);
      target.querySelector(`button[aria-label="View b.png fullscreen"]`)!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      await settle();
      expect(pressed(viewerToggle())).toBe('false');

      viewerToggle()!.click();
      await settle();

      expect(galleryApiMocks.setStarred).toHaveBeenLastCalledWith('out/b.png', true);
      expect(pressed(viewerToggle())).toBe('true');

      document.dispatchEvent(new KeyboardEvent('keydown', { key: 's', bubbles: true }));
      await settle();

      expect(galleryApiMocks.setStarred).toHaveBeenLastCalledWith('out/b.png', false);
      expect(pressed(viewerToggle())).toBe('false');
    });
  });
});
