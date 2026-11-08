// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { reactiveProps } from '../../test-utils/reactiveProps.svelte';
import { draft } from '$lib/state/draft.svelte';
import { sceneSource } from '$lib/state/extend.fixtures';
import { galleryStore } from '$lib/state/gallery.svelte';
import { jobStore } from '$lib/state/job.svelte';
import { projectPick } from '$lib/state/projectPick.svelte';
import { projectsStore } from '$lib/state/projects.svelte';
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
  return { ...actual, getGallery: galleryApiMocks.getGallery, deleteAsset: galleryApiMocks.deleteAsset, moveAssets: galleryApiMocks.moveAssets };
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

const projectApiMocks = vi.hoisted(() => ({ listProjects: vi.fn(), createProject: vi.fn() }));

vi.mock('$lib/api/projects', () => projectApiMocks);


const ROOT = '/tmp/output';
const NIGHT_BUS = { name: 'Night Bus', path: `${ROOT}/Night Bus` };
const FEATURE_REEL = { name: 'Feature Reel', path: `${ROOT}/Feature Reel` };

let target: HTMLDivElement;
let app: Record<string, unknown> | null = null;

beforeEach(async () => {
  resetWorkspaceTest();
  projectPick.set(null);
  target = createHost();
  projectApiMocks.listProjects.mockResolvedValue([FEATURE_REEL, NIGHT_BUS]);
  await projectsStore.load();
  workspaceApiMocks.submitGenerate.mockResolvedValue({ job_id: 'job-p', workflow: 'txt2img', prompt: 'Test prompt', model: 'zit', runs: 1, created_at: '2026-11-08T10:00:00Z' });
});

afterEach(async () => {
  if (app) { await unmount(app); app = null; }
  target.remove();
  document.body.innerHTML = '';
  projectPick.set(null);
  vi.restoreAllMocks();
});

async function mountWorkspace(context: WorkspaceContext = makeContext()): Promise<void> {
  app = await mountWorkspacePage(context, target);
}

/** Answer the Gallery by project: a project's own files for a pick, the whole Gallery for none. */
function serveGallery(byProject: Record<string, GalleryAsset[]>, all: GalleryAsset[]): void {
  // The mock's type has three parameters; the real call passes the project as the fourth.
  const answer = async (...args: unknown[]): Promise<GalleryPage> => {
    const project = args[3] as string | null | undefined;
    return makeGalleryPage(project ? (byProject[project] ?? []) : all);
  };
  galleryApiMocks.getGallery.mockImplementation(answer);
}

const generateLines = (id: string) => Array.from(target.querySelectorAll(`${id} .generate-stack > span`)).map((el) => el.textContent);

describe('WorkspacePage in a project', () => {
  describe('submit', () => {
    it('sends the picked project, and nothing with All files', async () => {
      await mountWorkspace();
      expect((await submitForm(target)).has('project')).toBe(false);

      projectPick.set('Night Bus');
      await settle();
      expect((await submitForm(target)).get('project')).toBe('Night Bus');
    });
  });

  describe('Copy CLI', () => {
    const copied = async (): Promise<string> => {
      const writeText = mockClipboard();
      (target.querySelector('#ws-copy-cli') as HTMLButtonElement).click();
      await settle();
      return writeText.mock.calls.at(-1)?.[0] ?? '';
    };

    beforeEach(() => {
      draft.update('prompt', 'a fox');
    });

    it('saves to the project\'s folder, and to the root with All files', async () => {
      await mountWorkspace();
      expect(await copied()).toMatch(/--output \/tmp\/output$/);

      projectPick.set('Night Bus');
      await settle();
      expect(await copied()).toMatch(/--output '\/tmp\/output\/Night Bus'$/);
    });

    it('keeps an Extend source\'s folder out of other workflows', async () => {
      const context = makeContext();
      context.workflow_contract.values = [...context.workflow_contract.values, 'extend'];
      context.workflow_contract.definitions.extend.visible_controls = ['workflow', 'model', 'prompt_scene', 'prompt_inline', 'source_video', 'source_video_path', 'extend_length', 'runs', 'steps', 'seed'];
      context.prompt_enhancer = { matrix: { axes: [], defaults: { style: 'keep', mood: 'keep', theme: 'keep', shot: 'keep', angle: 'keep', palette: 'keep', colors: [], camera: 'keep', pace: 'keep', details: [], length: 'same', motion: [], sound: [], note: '' } }, audio_matrix: EMPTY_AUDIO_MATRIX, model: 'owner/llm', revision: null, downloaded: true, download_size_label: null, default_max_words: 300, error: null } satisfies PromptEnhancerContract;
      browseApiMocks.browsePlaces.mockResolvedValue(placesFor('source_video'));
      extendApiMocks.inspectExtendSource.mockResolvedValue(sceneSource({ output_folder: FEATURE_REEL.path }));
      projectPick.set('Night Bus');
      draft.update('workflow', 'extend');
      await mountWorkspace(context);

      browseApiMocks.listDirectory.mockResolvedValue(listingWith('/clips/walk.mp4'));
      (target.querySelector('#ws-source-browse') as HTMLButtonElement).click();
      await settle();
      chooseInBrowser('walk.mp4');
      await settle();
      expect(draft.extendSource?.output_folder).toBe(FEATURE_REEL.path);
      expect(await copied()).toMatch(/--output '\/tmp\/output\/Feature Reel'$/);

      draft.patch({ workflow: 'txt2img', prompt: 'a fox' });
      await settle();
      expect(draft.extendSource?.output_folder).toBe(FEATURE_REEL.path);
      expect(await copied()).toMatch(/--output '\/tmp\/output\/Night Bus'$/);
    });
  });

  describe('the stage', () => {
    it('says what an empty project holds, and keeps the old words for All files', async () => {
      await mountWorkspace();
      expect(target.querySelector('.ui-empty-title')?.textContent).toBe('No generated assets yet');

      projectPick.set('Night Bus');
      await settle();
      expect(target.querySelector('.ui-empty-title')?.textContent).toBe('Nothing in Night Bus yet');
    });

    it('does not start with another project\'s history', async () => {
      const other = makeAsset({ id: 'out/other.png', filename: 'other.png', project: 'Feature Reel' });
      const mine = makeAsset({ id: 'Night Bus/mine.png', filename: 'mine.png', project: 'Night Bus' });
      projectPick.set('Night Bus');
      serveGallery({ 'Night Bus': [mine] }, [other, mine]);
      await mountWorkspace(makeContext({ history_assets: [other, mine] }));

      expect(galleryStore.assets.map((asset) => asset.id)).toEqual(['Night Bus/mine.png']);
    });

    async function finishJob(outputs: GalleryAsset[]): Promise<void> {
      target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
      await settle();
      lastEventSource().emit('job_completed', { type: 'job_completed', job_id: 'job-p', total_runs: outputs.length, outputs });
      await settle();
    }

    it('shows a finished output of the pick, with the plain toast', async () => {
      projectPick.set('Night Bus');
      await mountWorkspace();
      await finishJob([makeAsset({ id: 'Night Bus/a.png', filename: 'a.png', project: 'Night Bus' })]);

      expect(target.textContent).toContain('Completed outputs');
      expect(toasts.at(-1)).toMatchObject({ type: 'success' });
      expect(toasts.at(-1)?.message).not.toContain('Night Bus');
    });

    it('leaves a finished output of another project off the stage, and says where it went', async () => {
      projectPick.set('Night Bus');
      await mountWorkspace();
      await finishJob([makeAsset({ id: 'Feature Reel/a.png', filename: 'a.png', project: 'Feature Reel' })]);

      expect(target.textContent).not.toContain('Completed outputs');
      expect(target.querySelector('button[aria-label="View a.png fullscreen"]')).toBeNull();
      expect(toasts.at(-1)).toMatchObject({ type: 'success' });
      expect(toasts.at(-1)?.message).toContain('Feature Reel');
    });

    it('names No project for a file at the top level', async () => {
      projectPick.set('Night Bus');
      await mountWorkspace();
      await finishJob([makeAsset({ id: 'out/a.png', filename: 'a.png', project: null })]);

      expect(toasts.at(-1)?.message).toContain('No project');
    });

    it('keeps the plain toast with All files, whatever the output\'s project', async () => {
      await mountWorkspace();
      await finishJob([makeAsset({ id: 'Feature Reel/a.png', filename: 'a.png', project: 'Feature Reel' })]);

      expect(toasts.at(-1)).toMatchObject({ type: 'success' });
      expect(toasts.at(-1)?.message).not.toContain('Feature Reel');
    });

    it('shows one toast for a finished job', async () => {
      projectPick.set('Night Bus');
      await mountWorkspace();
      await finishJob([makeAsset({ id: 'Feature Reel/a.png', filename: 'a.png', project: 'Feature Reel' })]);

      expect(toasts.filter((toast) => toast.message.includes('Feature Reel'))).toHaveLength(1);
      expect(toasts).toHaveLength(1);
    });

    it('still shows a running job from another project', async () => {
      projectPick.set('Night Bus');
      await mountWorkspace();
      target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
      await settle();
      lastEventSource().emit('job_started', { type: 'job_started', job_id: 'job-p', total_runs: 1 });
      await settle();

      expect(target.querySelector('.job-view')).not.toBeNull();
    });
  });

  describe('what stays global', () => {
    it('takes the last seed from all work, not from the pick', async () => {
      const mine = makeAsset({ id: 'Night Bus/mine.png', filename: 'mine.png', project: 'Night Bus', seed: 1 });
      const latest = makeAsset({ id: 'Feature Reel/new.png', filename: 'new.png', project: 'Feature Reel', seed: 99 });
      serveGallery({ 'Night Bus': [mine] }, [latest, mine]);
      await mountWorkspace();
      projectsStore.choose('Night Bus');
      await settle();
      expect(galleryStore.assets.map((asset) => asset.id)).toEqual(['Night Bus/mine.png']);

      (target.querySelector('button[aria-label^="Lock seed"]') as HTMLButtonElement).click();
      await settle();

      expect(draft.state.seed).toBe(99);
    });
  });

  describe('Generate on a phone', () => {
    beforeEach(() => {
      layoutProbe.compact = true;
    });

    it('names the project on both buttons', async () => {
      projectPick.set('Night Bus');
      await mountWorkspace();

      expect(generateLines('#ws-submit-compact')).toEqual(['Generate', 'Night Bus']);
      expect(target.querySelectorAll('.composer-foot .generate-project')).toHaveLength(1);
      expect(target.querySelector('.composer-foot .generate-project')?.textContent).toBe('Night Bus');
    });

    it('is one line with All files', async () => {
      await mountWorkspace();

      expect(target.querySelector('#ws-submit-compact')?.textContent?.trim()).toBe('Generate');
      expect(target.querySelector('.generate-project')).toBeNull();
    });
  });
});

describe('WorkspacePage Move', () => {
  const top = (name: string, overrides: Partial<GalleryAsset> = {}) =>
    makeAsset({ id: `out/${name}`, filename: name, file_path: `/outputs/out/${name}`, ...overrides });
  const inBus = (name: string, overrides: Partial<GalleryAsset> = {}) =>
    makeAsset({ id: `Night Bus/${name}`, filename: name, project: 'Night Bus', file_path: `/outputs/Night Bus/${name}`, ...overrides });
  const stripIds = () => Array.from(target.querySelectorAll('#ws-history-shell .asset-tile')).map((tile) => tile.getAttribute('data-asset-id'));
  const stageId = () => target.querySelector('.latest-media')?.getAttribute('data-asset-id');
  const rows = () => Array.from(document.querySelectorAll<HTMLElement>('[role="menu"] [role="menuitemradio"]'));
  const row = (label: string) => rows().find((item) => item.textContent?.trim() === label)!;

  async function mountWithHistory(assets: GalleryAsset[]): Promise<void> {
    galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage(assets));
    await mountWorkspace(makeContext({ history_assets: assets }));
  }

  async function moveFrom(scope: ParentNode, asset: GalleryAsset, label: string): Promise<void> {
    (scope.querySelector(`.asset-tile[aria-label="${asset.filename}"] button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
    flushSync();
    (document.querySelector('[role="menu"] [data-action="move"]') as HTMLElement).click();
    await settle();
    row(label).click();
    await settle();
  }

  it('moves a strip tile, keeping it in place under its new ID, and says where it went', async () => {
    const [a, b] = [top('a.png'), top('b.png')];
    await mountWithHistory([a, b]);
    galleryApiMocks.moveAssets.mockResolvedValue({ moved: [{ from: a.id, asset: inBus('a.png') }], failed: [] });

    await moveFrom(target.querySelector('#ws-history-shell')!, a, 'Night Bus');

    expect(galleryApiMocks.moveAssets).toHaveBeenCalledWith(['out/a.png'], 'Night Bus');
    expect(stripIds()).toEqual(['Night Bus/a.png', 'out/b.png']);
    expect(toasts.at(-1)).toMatchObject({ type: 'success' });
    expect(toasts.at(-1)?.message).toContain('Night Bus');
  });

  it('points the slots that held the file at its new path, reference rows included', async () => {
    const [a, b] = [top('a.png'), top('b.png')];
    draft.setStartImage('image', { path: '/outputs/out/a.png' }, null);
    draft.setLastFrame({ path: '/outputs/out/a.png' });
    draft.patch({ referenceImagePaths: ['/outputs/out/a.png', '/outputs/out/b.png'] });
    await mountWithHistory([a, b]);
    galleryApiMocks.moveAssets.mockResolvedValue({ moved: [{ from: a.id, asset: inBus('a.png') }], failed: [] });

    await moveFrom(target.querySelector('#ws-history-shell')!, a, 'Night Bus');

    expect(draft.state).toMatchObject({ referenceImagePath: '/outputs/Night Bus/a.png', endImagePath: '/outputs/Night Bus/a.png' });
    // The row keeps its place, so the prompt's "image 1" keeps its meaning.
    expect(draft.state.referenceImagePaths).toEqual(['/outputs/Night Bus/a.png', '/outputs/out/b.png']);
  });

  it('keeps the moved file among the job\'s outputs under its new ID', async () => {
    const [a, b] = [top('a.png'), top('b.png')];
    await mountWithHistory([a, b]);
    const replaceOutputs = vi.spyOn(jobStore, 'replaceOutputs');
    galleryApiMocks.moveAssets.mockResolvedValue({ moved: [{ from: a.id, asset: inBus('a.png') }], failed: [] });

    await moveFrom(target.querySelector('#ws-history-shell')!, a, 'Night Bus');

    expect(replaceOutputs).toHaveBeenCalledWith([{ from: 'out/a.png', asset: inBus('a.png') }]);
    replaceOutputs.mockRestore();
  });

  it('leaves the slots of another file alone', async () => {
    const [a, b] = [top('a.png'), top('b.png')];
    draft.setStartImage('image', { path: '/outputs/out/b.png' }, null);
    await mountWithHistory([a, b]);
    galleryApiMocks.moveAssets.mockResolvedValue({ moved: [{ from: a.id, asset: inBus('a.png') }], failed: [] });

    await moveFrom(target.querySelector('#ws-history-shell')!, a, 'Night Bus');

    expect(draft.state.referenceImagePath).toBe('/outputs/out/b.png');
  });

  it('keeps the stage on the older output it showed when that file moves', async () => {
    const [a, b] = [top('a.png'), top('b.png')];
    await mountWithHistory([a, b]);
    (target.querySelector('button[aria-label="Older output"]') as HTMLButtonElement).click();
    await settle();
    expect(stageId()).toBe('out/b.png');
    galleryApiMocks.moveAssets.mockResolvedValue({ moved: [{ from: b.id, asset: inBus('b.png') }], failed: [] });

    await moveFrom(target.querySelector('.latest-region')!, b, 'Night Bus');

    expect(stageId()).toBe('Night Bus/b.png');
  });

  it('keeps the open viewer on a file that stays, under its new ID', async () => {
    const [a, b] = [top('a.png'), top('b.png')];
    await mountWithHistory([a, b]);
    (target.querySelector('#ws-history-shell button[aria-label="View b.png"]') as HTMLButtonElement).click();
    await settle();
    galleryApiMocks.moveAssets.mockResolvedValue({ moved: [{ from: b.id, asset: inBus('b.png') }], failed: [] });

    (document.querySelector('[data-testid="asset-viewer"] button[data-action="move"]') as HTMLButtonElement).click();
    await settle();
    row('Night Bus').click();
    await settle();

    const viewer = document.querySelector('[data-testid="asset-viewer"]')!;
    expect(viewer.textContent).toContain('b.png');
    expect(viewer.textContent).toContain('2 of 2');
  });

  it('leaves the file in place and reports Move failed when the server refuses it', async () => {
    const [a, b] = [top('a.png'), top('b.png')];
    await mountWithHistory([a, b]);
    galleryApiMocks.moveAssets.mockResolvedValue({ moved: [], failed: [{ id: a.id, detail: 'A file named a.png is already there.' }] });

    await moveFrom(target.querySelector('#ws-history-shell')!, a, 'Night Bus');

    expect(stripIds()).toEqual(['out/a.png', 'out/b.png']);
    expect(toasts.at(-1)).toMatchObject({ type: 'error' });
    expect(toasts.at(-1)?.message).toContain('A file named a.png is already there');
  });

  it('drops a file moved out of the picked project from the strip and the count', async () => {
    projectPick.set('Night Bus');
    const [a, b] = [inBus('a.png'), inBus('b.png')];
    galleryApiMocks.getGallery.mockImplementation(async (...args: unknown[]) => makeGalleryPage(args[3] === 'Night Bus' ? [a, b] : []));
    await mountWorkspace(makeContext({ history_assets: [a, b] }));
    galleryApiMocks.moveAssets.mockResolvedValue({ moved: [{ from: a.id, asset: top('a.png') }], failed: [] });

    await moveFrom(target.querySelector('#ws-history-shell')!, a, 'No project');

    expect(stripIds()).toEqual(['Night Bus/b.png']);
    expect(galleryStore.totalCount).toBe(1);
  });
});
