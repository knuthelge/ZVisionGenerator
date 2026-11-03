// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { trackLengths } from '$lib/state/audioTrackLength.svelte';
import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { memoryUploadStore } from '$lib/state/uploadStore';
import type { JobContext, WorkspaceContext } from '$lib/types';

import { makeContext, makeVideoDefaults } from '../../test-utils/workspace';
import { createHost, mountWorkspacePage, resetWorkspaceTest, settle, submitForm } from '../../test-utils/workspaceHarness';
import { requestJobSettings } from '$lib/state/pendingJobSettings.svelte';
import { extendApiMocks, workspaceApiMocks } from '../../test-utils/workspaceMocks';
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

// Every test starts from empty stores, reset API mocks, an empty server-side queue and a fresh host element.
beforeEach(() => {
  resetWorkspaceTest();
  target = createHost();
});

afterEach(async () => {
  vi.useRealTimers();
  if (app) { await unmount(app); app = null; }
  target.remove();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

/** Mount the Workspace page on the host element and let it settle. */
async function mountWorkspace(context: WorkspaceContext): Promise<void> {
  app = await mountWorkspacePage(context, target);
}


const TRACK_CONTROLS = ['audio_track', 'audio_track_path', 'audio_track_start'];

/** Two video models: LTX follows a track and can upscale one; H3 follows one but cannot upscale it; "plain" follows none. */
function trackContext(): WorkspaceContext {
  const context = makeContext({
    video_models: [{ id: 'ltx-8', label: 'ltx-8', type: 'video' }, { id: 'h3', label: 'h3', type: 'video' }, { id: 'plain', label: 'plain', type: 'video' }],
    current_video_model: 'ltx-8',
    video_model_defaults: {
      'ltx-8': makeVideoDefaults({ hidden_controls: [] }),
      h3: makeVideoDefaults({ hidden_controls: ['low_memory'], supports_audio_track_upscale: false }),
      plain: makeVideoDefaults({ hidden_controls: TRACK_CONTROLS }),
    },
  });
  context.workflow_contract.values = [...context.workflow_contract.values, 'extend'];
  return context;
}

beforeEach(() => {
  trackLengths.reset();
  trackApi.inspectAudioTrack.mockReset().mockRejectedValue(new Error('unreadable'));
  trackApi.readFileLength.mockReset().mockResolvedValue(null);
  workspaceApiMocks.submitGenerate.mockResolvedValue({ job_id: 'job-1' } as JobContext);
  window.location.hash = '#/workspace';
});

async function mountTab(workflow: 'txt2vid' | 'extend', model = 'ltx-8'): Promise<WorkspaceContext> {
  const context = trackContext();
  draft.reset();
  draft.patch({ workflow, model });
  app = await mountWorkspacePage(context, target);
  draft.update('model', model);
  await settle();
  return context;
}

const memo = (name = 'memo.m4a') => new File(['audio'], name, { type: 'audio/mp4' });
const section = () => target.querySelector('[data-section="audio-track"]') as HTMLElement | null;
const audioSwitch = () => target.querySelector('#ws-audio');
const trackLength = () => target.querySelector('[data-testid="ws-audio-track-detail"]')?.textContent;

async function pickTrack(file: File): Promise<void> {
  const input = target.querySelector('#ws-audio-track-file') as HTMLInputElement;
  Object.defineProperty(input, 'files', { configurable: true, value: [file] });
  input.dispatchEvent(new Event('change', { bubbles: true }));
  await settle();
}

async function submit(): Promise<FormData> {
  (target.querySelector('#ws-prompt') as HTMLTextAreaElement).value = 'a singer';
  (target.querySelector('#ws-prompt') as HTMLTextAreaElement).dispatchEvent(new Event('input', { bubbles: true }));
  await settle();
  return submitForm(target);
}

describe('WorkspacePage audio track', () => {
  describe('length of the track', () => {
    it('reads the length of an upload in the browser', async () => {
      trackApi.readFileLength.mockResolvedValue(3.6);
      await mountTab('txt2vid');

      await pickTrack(memo());

      expect(trackApi.inspectAudioTrack).not.toHaveBeenCalled();
      expect(trackLength()).toBe('0:03.6');
    });
  });

  it('sends a chosen path with Start at, shows no Audio switch, and sends no audio', async () => {
    await mountTab('txt2vid');
    draft.setAudioTrack('video', { path: '/audio/song.mp3' });
    draft.setAudioTrackStart('video', 90);
    await settle();

    expect(audioSwitch()).toBeNull();

    const submitted = await submit();
    expect(submitted.get('audio_track_path')).toBe('/audio/song.mp3');
    expect(submitted.get('audio_track_start')).toBe('90');
    expect(submitted.has('audio_track_file')).toBe(false);
    expect(submitted.has('audio')).toBe(false);
  });

  it('sends a file chosen from the device as audio_track_file, over a path', async () => {
    await mountTab('txt2vid');
    const file = memo();

    await pickTrack(file);

    expect(draft.state).toMatchObject({ audioTrackUpload: 'memo.m4a', audioTrackPath: null });
    const submitted = await submit();
    expect(submitted.get('audio_track_file')).toBe(file);
    expect(submitted.get('audio_track_start')).toBe('0');
    expect(submitted.has('audio_track_path')).toBe(false);
  });

  it('keeps a voice memo when the page is left and comes back', async () => {
    await mountTab('txt2vid');
    const file = memo();
    await pickTrack(file);
    await unmount(app!);
    app = null;

    await mountWorkspace(trackContext());
    const submitted = await submit();

    expect(submitted.get('audio_track_file')).toBe(file);
  });

  it('keeps a voice memo across a reload, read back from the upload store', async () => {
    const store = memoryUploadStore();
    heldUploads.useStore(store);
    await mountTab('txt2vid');
    const file = memo();
    await pickTrack(file);
    await heldUploads.persisted();
    await unmount(app!);
    app = null;

    // A reload empties what the page holds in memory; the draft and the store stay.
    heldUploads.useStore(store);
    expect(heldUploads.audioTrack).toBeNull();
    await mountWorkspace(trackContext());
    const submitted = await submit();

    expect(submitted.get('audio_track_file')).toEqual(file);
    expect(target.querySelector('#ws-generate-reason')).toBeNull();
  });

  it('blocks Generate with a reason when an uploaded track did not come back', async () => {
    await mountTab('txt2vid');
    draft.patch({ audioTrackUpload: 'memo.m4a' });
    await settle();

    expect((target.querySelector('#ws-submit') as HTMLButtonElement).disabled).toBe(true);
    expect(target.querySelector('#ws-generate-reason')?.textContent).toBeTruthy();
  });

  it('clears the track with its × and brings the Audio switch back', async () => {
    await mountTab('txt2vid');
    draft.setAudioTrack('video', { path: '/audio/song.mp3' });
    await settle();

    (target.querySelector('#ws-audio-track-clear') as HTMLButtonElement).click();
    await settle();

    expect(draft.state).toMatchObject({ audioTrackPath: null, audioTrackStart: 0 });
    expect(audioSwitch()).not.toBeNull();
  });

  describe('on a model without the capability', () => {
    it('hides the section, shows the Audio switch and sends neither the kept track nor its Start at', async () => {
      await mountTab('txt2vid');
      draft.setAudioTrack('video', { path: '/audio/song.mp3' });
      draft.setAudioTrackStart('video', 90);
      await settle();
      expect(audioSwitch()).toBeNull();

      draft.hydrateFromContext(trackContext(), 'plain', { keepSettings: true });
      await settle();

      expect(section()).toBeNull();
      expect(audioSwitch()).not.toBeNull();
      const submitted = await submit();
      expect(Array.from(submitted.keys()).filter((key) => key.startsWith('audio_track'))).toEqual([]);
      expect(submitted.has('audio')).toBe(true);
      // The track stays in the draft and comes back with a model that follows one.
      expect(draft.state).toMatchObject({ audioTrackPath: '/audio/song.mp3', audioTrackStart: 90 });
    });
  });

  describe('Upscale', () => {
    it('is hidden and not sent while a track is in effect on a model that cannot upscale one, and returns without', async () => {
      await mountTab('txt2vid', 'h3');
      draft.update('videoUpscaleEnabled', true);
      await settle();
      expect(target.querySelector('#ws-video-upscale')).not.toBeNull();

      draft.setAudioTrack('video', { path: '/audio/song.mp3' });
      await settle();
      expect(target.querySelector('#ws-video-upscale')).toBeNull();
      const submitted = await submit();
      expect(submitted.has('upscale')).toBe(false);
      expect(submitted.has('video_upscale_factor')).toBe(false);

      draft.setAudioTrack('video', null);
      await settle();
      expect(target.querySelector('#ws-video-upscale')).not.toBeNull();
    });
  });

  describe('in Extend', () => {
    async function mountExtend(): Promise<void> {
      extendApiMocks.inspectExtendSource.mockResolvedValue(sceneSource({ path: '/clips/walk.mp4', audio_track_path: '/audio/song.mp3', audio_track_end: 6.65 }));
      draft.setExtendSourcePath('/clips/walk.mp4');
      await mountTab('extend');
      draft.setExtendSourcePath('/clips/walk.mp4');
      await settle();
    }

    it('takes the section\'s place without an Images section, filled in from a source made from a track', async () => {
      await mountExtend();
      draft.applyExtendSource(sceneSource({ path: '/clips/walk.mp4', audio_track_path: '/audio/song.mp3', audio_track_end: 6.65 }), trackContext());
      await settle();

      expect(target.querySelector('[data-section="images"]')).toBeNull();
      expect(draft.state).toMatchObject({ extendAudioTrackPath: '/audio/song.mp3', extendAudioTrackStart: 6.65 });
    });

    it('sends the Extend track fields and not the Video track', async () => {
      await mountExtend();
      draft.setAudioTrack('video', { path: '/audio/video-only.mp3' });
      draft.applyExtendSource(sceneSource({ path: '/clips/walk.mp4', audio_track_path: '/audio/song.mp3', audio_track_end: 6.65 }), trackContext());
      await settle();

      const submitted = await submit();

      expect(submitted.get('audio_track_path')).toBe('/audio/song.mp3');
      expect(submitted.get('audio_track_start')).toBe('6.65');
      expect(submitted.has('audio')).toBe(false);
    });
  });

  describe('Load settings', () => {
    it('Load settings of an extension puts its track in Extend and leaves the Video track alone', async () => {
      await mountTab('txt2vid');
      draft.setAudioTrack('video', { path: '/audio/video.mp3' });
      await settle();

      requestJobSettings({ mode: 'video', workflow: 'extend', model: 'ltx-8', source_video: '/clips/walk.mp4', extend_frames: '48', scene: 's', prompt_source: 'inline', audio_track_path: '/audio/song.mp3', audio_track_start: '6.65' });
      await settle();

      expect(draft.state).toMatchObject({ workflow: 'extend', extendAudioTrackPath: '/audio/song.mp3', extendAudioTrackStart: 6.65, audioTrackPath: '/audio/video.mp3' });
    });
  });
});
