// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { projectsStore } from '$lib/state/projects.svelte';
import type { GalleryAsset } from '$lib/types';

import { lastEventSource } from '../../test-utils/eventSource';
import { makeAsset, makeContext, makeGalleryPage } from '../../test-utils/workspace';
import { answerConfirm, createHost, mountWorkspacePage, resetWorkspaceTest, settle } from '../../test-utils/workspaceHarness';
import { galleryApiMocks, workspaceApiMocks } from '../../test-utils/workspaceMocks';

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

const adjustApiMocks = vi.hoisted(() => ({ saveAdjusted: vi.fn<(body: unknown) => Promise<GalleryAsset>>() }));
vi.mock('$lib/api/adjust', () => adjustApiMocks);

vi.mock('$lib/api/config', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/config')>();
  const { configApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, updateConfig: configApiMocks.updateConfig };
});

vi.mock('$lib/api/extend', async () => {
  const { extendApiMocks } = await import('../../test-utils/workspaceMocks');
  return { inspectExtendSource: extendApiMocks.inspectExtendSource };
});

vi.mock('$lib/api/memory', async () => {
  const { memoryApiMocks } = await import('../../test-utils/workspaceMocks');
  return { checkMemory: memoryApiMocks.checkMemory };
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
  return { ...actual, getGallery: galleryApiMocks.getGallery, deleteAssets: galleryApiMocks.deleteAssets, moveAssets: galleryApiMocks.moveAssets, setStarred: galleryApiMocks.setStarred };
});

const projectApiMocks = vi.hoisted(() => ({ listProjects: vi.fn(), createProject: vi.fn() }));
vi.mock('$lib/api/projects', () => projectApiMocks);

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

beforeEach(async () => {
  resetWorkspaceTest();
  projectApiMocks.listProjects.mockResolvedValue([{ name: 'Night Bus', path: '/tmp/output/Night Bus' }]);
  await projectsStore.load();
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
  makeAsset({ id: `out/${name}`, filename: name, url: `/media/out/${name}`, thumbnail_url: `/media/out/${name}`, file_path: `/outputs/${name}`, ...overrides });

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

const editor = (): HTMLElement | null => document.querySelector('[role="dialog"][aria-label^="Adjust "]');
const shown = (): string | null | undefined => document.querySelector('#asset-viewer-title .file-name')?.getAttribute('title');
const position = (): string | null | undefined => document.querySelector('.vb-name small')?.textContent;
const viewer = (): HTMLElement | null => document.querySelector('[data-testid="asset-viewer"]');
const press = (key: string): void => {
  document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
  flushSync();
};

/** Choose Adjust in the ⋯ menu of a tile inside *scope*. */
async function adjustFrom(scope: ParentNode, filename: string): Promise<void> {
  scope.querySelector<HTMLButtonElement>(`button[aria-label="More actions for ${filename}"]`)!.click();
  flushSync();
  document.querySelector<HTMLElement>('[role="menu"] [data-action="adjust"]')!.click();
  await settle();
}

/** Change a slider in the open editor and press the save button. */
async function save(label: string): Promise<void> {
  const slider = editor()!.querySelector<HTMLInputElement>('input[type="range"]')!;
  slider.value = '30';
  slider.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
  Array.from(editor()!.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent?.trim() === label)!.click();
  await settle();
  await settle();
}

describe('WorkspacePage adjust', () => {
  const copy = asset('b_adjusted.png');

  beforeEach(() => {
    adjustApiMocks.saveAdjusted.mockReset().mockResolvedValue(copy);
    vi.stubGlobal('Image', class { src = ''; naturalWidth = 800; naturalHeight = 600; decode = (): Promise<void> => Promise.resolve(); });
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe('the strip', () => {
    const strip = () => target.querySelector('#ws-history-shell')!;

    it('opens the viewer in History on the tile\'s picture with the editor open, and Escape closes the editor before the viewer', async () => {
      await mountWithHistory([asset('a.png'), asset('b.png'), asset('c.png')]);
      await adjustFrom(strip(), 'b.png');

      expect(shown()).toBe('b.png');
      expect(document.querySelector('.vb-name small')?.textContent).toContain('Gallery');
      expect(editor()!.getAttribute('aria-label')).toBe('Adjust b.png');

      press('Escape');
      await settle();
      expect(editor()).toBeNull();
      expect(viewer()).not.toBeNull();

      press('Escape');
      await settle();
      expect(viewer()).toBeNull();
    });

    it('shows the copy first in History, with the source behind it', async () => {
      await mountWithHistory([asset('a.png'), asset('b.png'), asset('c.png')]);
      await adjustFrom(strip(), 'b.png');
      await save('Save as copy');

      expect(editor()).toBeNull();
      expect(shown()).toBe('b_adjusted.png');
      expect(position()).toContain('1 of 4');
      expect(Array.from(strip().querySelectorAll('.asset-tile')).map((tile) => tile.getAttribute('data-asset-id'))).toEqual(['out/b_adjusted.png', 'out/a.png', 'out/b.png', 'out/c.png']);
    });

    it('keeps the asset after Replace, with the fresh URL', async () => {
      localStorage.setItem('ziv-adjust-replace-asked-v1', '1');
      adjustApiMocks.saveAdjusted.mockResolvedValue({ ...asset('b.png'), url: '/media/out/b.png?v=2' });
      await mountWithHistory([asset('a.png'), asset('b.png')]);
      await adjustFrom(strip(), 'b.png');
      await save('Replace');

      expect(shown()).toBe('b.png');
      expect(position()).toContain('2 of 2');
      expect(document.querySelector('[data-testid="viewer-image"]')?.getAttribute('src')).toBe('/media/out/b.png?v=2');
    });
  });

  describe('the stage', () => {
    it('opens the viewer on the latest output from its menu, with the editor open', async () => {
      await mountWithHistory([asset('a.png'), asset('b.png')]);
      await adjustFrom(target.querySelector('.latest-region')!, 'a.png');
      expect(shown()).toBe('a.png');
      expect(editor()).not.toBeNull();
    });
  });

  describe('This run', () => {
    it('opens the viewer on This run with the editor open', async () => {
      await mountWithHistory([]);
      await finishJob([asset('a.png'), asset('b.png')]);
      await adjustFrom(target.querySelector('.completed-output-grid')!, 'b.png');

      expect(shown()).toBe('b.png');
      expect(document.querySelector('.vb-name small')?.textContent).toContain('This run');
      expect(editor()).not.toBeNull();
    });

    it('opens a saved copy in History, since the run\'s list holds the job\'s outputs only', async () => {
      await mountWithHistory([]);
      // The finished job is in History by the time the editor saves.
      galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([asset('a.png'), asset('b.png')]));
      await finishJob([asset('a.png'), asset('b.png')]);
      await adjustFrom(target.querySelector('.completed-output-grid')!, 'b.png');
      await save('Save as copy');

      expect(editor()).toBeNull();
      expect(shown()).toBe('b_adjusted.png');
      expect(position()).toContain('1 of 3');
      expect(document.querySelector('.vb-name small')?.textContent).not.toContain('This run');
    });

    it('offers Adjust from the viewer of a completed output, and A opens the editor', async () => {
      await mountWithHistory([]);
      await finishJob([asset('a.png'), asset('b.png')]);
      target.querySelector<HTMLButtonElement>('button[aria-label="View b.png fullscreen"]')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      await settle();
      expect(editor()).toBeNull();

      press('a');
      await settle();
      expect(editor()!.getAttribute('aria-label')).toBe('Adjust b.png');
    });
  });
});
