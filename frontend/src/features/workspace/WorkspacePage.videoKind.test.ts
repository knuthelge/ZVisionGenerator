// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { trackLengths } from '$lib/state/audioTrackLength.svelte';
import { draft } from '$lib/state/draft.svelte';
import { workspaceMemory } from '$lib/state/pageMemory';
import type { JobContext, WorkspaceContext } from '$lib/types';

import { makeContext } from '../../test-utils/workspace';
import { createHost, mountWorkspacePage, resetWorkspaceTest, settle, submitForm } from '../../test-utils/workspaceHarness';
import { extendApiMocks, layoutProbe, workspaceApiMocks } from '../../test-utils/workspaceMocks';
import { sceneSource } from '$lib/state/extend.fixtures';

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
  return { ...actual, getGallery: galleryApiMocks.getGallery, deleteAsset: galleryApiMocks.deleteAsset };
});

vi.mock('$lib/api/promptEnhance', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/promptEnhance')>();
  const { enhanceApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, enhancePrompt: enhanceApiMocks.enhancePrompt };
});

const trackApi = vi.hoisted(() => ({ inspectAudioTrack: vi.fn(), readFileLength: vi.fn() }));
vi.mock('$lib/api/audioTrack', () => ({ inspectAudioTrack: trackApi.inspectAudioTrack }));
vi.mock('$lib/state/audioLength', () => ({ readFileLength: trackApi.readFileLength }));

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

const SOURCE = '/clips/walk.mp4';

beforeEach(() => {
  resetWorkspaceTest();
  target = createHost();
  trackLengths.reset();
  trackApi.inspectAudioTrack.mockReset().mockRejectedValue(new Error('unreadable'));
  trackApi.readFileLength.mockReset().mockResolvedValue(null);
  layoutProbe.compact = false;
  workspaceApiMocks.submitGenerate.mockResolvedValue({ job_id: 'job-1' } as JobContext);
  extendApiMocks.inspectExtendSource.mockResolvedValue(sceneSource({ path: SOURCE }));
  window.location.hash = '#/workspace';
});

afterEach(async () => {
  vi.useRealTimers();
  if (app) { await unmount(app); app = null; }
  target.remove();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
  layoutProbe.compact = false;
});

/** Mount the Workspace on a machine that can extend (or not), open on a workflow. */
async function mountVideo(workflow: 'txt2vid' | 'extend' = 'txt2vid', options: { extend?: boolean } = {}): Promise<WorkspaceContext> {
  const context = makeContext();
  if (options.extend !== false) context.workflow_contract.values = [...context.workflow_contract.values, 'extend'];
  draft.reset();
  draft.patch({ workflow });
  app = await mountWorkspacePage(context, target);
  return context;
}

const kindControl = () => target.querySelector('[data-testid="kind-control"]');
const kindButton = (kind: 'new' | 'extend') => target.querySelector(`[data-testid="kind-control"] [data-value="${kind}"]`) as HTMLButtonElement;

async function pickKind(kind: 'new' | 'extend'): Promise<void> {
  kindButton(kind).click();
  await settle();
}

async function submit(): Promise<FormData> {
  return submitForm(target);
}

describe('WorkspacePage Video kinds', () => {
  it('offers New and Extend where the machine can extend, and the Video tab alone where it cannot', async () => {
    await mountVideo('txt2vid');
    expect(kindButton('new')).not.toBeNull();
    expect(kindButton('extend')).not.toBeNull();
    await unmount(app!);
    app = null;

    workspaceMemory.forget();
    await mountVideo('txt2vid', { extend: false });
    expect(kindControl()).toBeNull();
  });

  it('switches between the workflows of the two kinds, and Generate sends the one of the open kind', async () => {
    await mountVideo('txt2vid');
    draft.setExtendSourcePath(SOURCE);
    await settle();

    expect((await submit()).get('workflow')).toBe('txt2vid');

    await pickKind('extend');
    expect(draft.state.workflow).toBe('extend');
    const extended = await submit();
    expect(extended.get('workflow')).toBe('extend');
    expect(extended.get('source_video')).toBe(SOURCE);

    await pickKind('new');
    expect(draft.state.workflow).toBe('txt2vid');
  });

  it('keeps each kind\'s own inputs and text while the kind changes', async () => {
    await mountVideo('txt2vid');
    draft.setStartImage('video', { path: '/in/first.png' }, null);
    draft.patch({ frameCount: 49, scene: 'A street.', prompt: 'She walks.' });
    await settle();

    await pickKind('extend');
    expect(draft.state).toMatchObject({ scene: '', prompt: '' });
    draft.setExtendSourcePath(SOURCE);
    draft.patch({ extendFrames: 48, scene: 'A hall.', prompt: 'She runs.' });
    draft.setAudioTrack('extend', { path: '/audio/extend.wav' });
    await settle();

    await pickKind('new');
    expect(draft.state).toMatchObject({
      workflow: 'img2vid', firstFramePath: '/in/first.png', frameCount: 49, scene: 'A street.', prompt: 'She walks.',
      extendSourcePath: SOURCE, extendFrames: 48, extendAudioTrackPath: '/audio/extend.wav',
    });

    await pickKind('extend');
    expect(draft.state).toMatchObject({ workflow: 'extend', scene: 'A hall.', prompt: 'She runs.' });
  });

  it('opens Video on the kind it was left on', async () => {
    await mountVideo('txt2vid');
    await pickKind('extend');

    flushSync(() => draft.setTab('image'));
    await settle();
    flushSync(() => draft.setTab('video'));
    await settle();

    expect(draft.state.workflow).toBe('extend');
    expect(kindButton('extend').getAttribute('aria-pressed')).toBe('true');
  });

  it('opens a saved draft that was on Extend on the Extend kind', async () => {
    await mountVideo('extend');
    expect(kindButton('extend').getAttribute('aria-pressed')).toBe('true');
  });

  describe('on a phone', () => {
    const chip = (key: string) => target.querySelector(`[data-chip="${key}"]`) as HTMLButtonElement | null;
    const popoverRows = () => Array.from(target.querySelectorAll('[data-testid="setting-popover"] [role="radio"]')) as HTMLButtonElement[];

    // The popover slides in; jsdom has no animations.
    beforeEach(() => {
      Element.prototype.animate = vi.fn(() => ({ cancel: vi.fn(), finished: Promise.resolve(), onfinish: null })) as unknown as Element['animate'];
    });
    afterEach(() => {
      delete (Element.prototype as Partial<Element>).animate;
    });

    it('chooses the kind from the Kind chip, and calls the extension length Length', async () => {
      layoutProbe.compact = true;
      await mountVideo('txt2vid');

      chip('kind')!.click();
      await settle();
      expect(popoverRows()).toHaveLength(2);
      popoverRows()[1].click();
      await settle();

      expect(draft.state.workflow).toBe('extend');
      expect(chip('kind')!.getAttribute('aria-label')).toContain('Extend');
      expect(chip('extendSeconds')!.getAttribute('aria-label')).toContain('Length');
    });

    it('has no Kind chip where the machine cannot extend', async () => {
      layoutProbe.compact = true;
      await mountVideo('txt2vid', { extend: false });
      expect(chip('kind')).toBeNull();
    });
  });
});
