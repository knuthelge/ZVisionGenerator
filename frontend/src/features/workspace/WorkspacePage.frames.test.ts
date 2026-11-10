// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import { reactiveProps } from '../../test-utils/reactiveProps.svelte';
import { draft } from '$lib/state/draft.svelte';
import { EXTEND_SHORT_REASON } from '$lib/state/framePoint';
import { legacySource, sceneSource, sourceInfo } from '$lib/state/extend.fixtures';
import { galleryStore } from '$lib/state/gallery.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { jobStore } from '$lib/state/job.svelte';
import { hasPendingReference, requestReference } from '$lib/state/pendingReference.svelte';
import { requestPromptRun } from '$lib/state/pendingPromptRun';
import { toasts } from '$lib/state/toasts.svelte';
import { installFakeMedia, type FakeMedia } from '../../test-utils/media';
import { memoryUploadStore } from '$lib/state/uploadStore';
import type { EnhanceFrame, SettingsPreset, JobContext, PromptEnhancerContract, WorkspaceContext, GalleryAsset } from '$lib/types';

import WorkspacePage from './WorkspacePage.svelte';
import { chooseInBrowser, listingWith, placesFor } from '../../test-utils/pathBrowser';
import { lastEventSource } from '../../test-utils/eventSource';
import { makeAudioAsset, makeSpeechAsset } from '../../test-utils/audio';
import { EMPTY_AUDIO_MATRIX, makeAsset, makeContext, makeGalleryPage, makeImageDefaults } from '../../test-utils/workspace';
import { addFiles, answerConfirm, createHost, fakeDebounceTimers, field, holdInspection, mockClipboard, mountWorkspacePage, openEnhanceRun, resetWorkspaceTest, runEnhance, settle, type, waitDebounce } from '../../test-utils/workspaceHarness';
import { browseApiMocks, enhanceApiMocks, extendApiMocks, galleryApiMocks, layoutProbe, presetApiMocks, workspaceApiMocks } from '../../test-utils/workspaceMocks';

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

vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);

const frameApi = vi.hoisted(() => ({ grabFrame: vi.fn() }));
vi.mock('$lib/api/videoFrame', () => ({ grabFrame: frameApi.grabFrame }));

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
let media: FakeMedia;

// 6 s at 24 fps: 144 frames, the last is frame 143.
const FACTS = { fps: 24, frames: 144, start: 0, seconds: 6 };
const LAST = FACTS.frames - 1;
const BOX = { left: 100, top: 700, width: 400, height: 46, right: 500, bottom: 746, x: 100, y: 700, toJSON: () => ({}) };
const xOf = (frame: number): number => 100 + (frame / LAST) * 400;
const SOURCE = '/clips/walk.mp4';
const SAVED = { path: '/uploads/abc/walk_3.0s.png' };

beforeEach(() => {
  resetWorkspaceTest();
  target = createHost();
  media = installFakeMedia();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  browseApiMocks.browseInfo.mockReset().mockResolvedValue(FACTS as never);
  frameApi.grabFrame.mockReset().mockResolvedValue(SAVED);
  extendApiMocks.inspectExtendSource.mockResolvedValue(sceneSource());
});

afterEach(async () => {
  vi.useRealTimers();
  if (app) { await unmount(app); app = null; }
  target.remove();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

const clip = (overrides: Partial<GalleryAsset> = {}): GalleryAsset =>
  makeAsset({
    id: 'out/walk.mp4', url: '/media/out/walk.mp4', thumbnail_url: '/media/out/walk.mp4', filename: 'walk.mp4', media_type: 'video', workflow: 'txt2vid',
    file_path: SOURCE, scene: 'A woman on a rainy street.', prompt: 'She looks up.', extend: { available: true, reason: null }, ...overrides,
  });

async function mountWithClip(asset: GalleryAsset): Promise<void> {
  galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([asset]));
  const context = makeContext({ history_assets: [asset] });
  context.workflow_contract.values = [...context.workflow_contract.values, 'extend'];
  workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(context);
  draft.update('workflow', 'txt2vid');
  app = await mountWorkspacePage(context, target);
}

const viewer = () => document.querySelector<HTMLElement>('[data-testid="asset-viewer"]');
const timeline = () => viewer()!.querySelector<HTMLElement>('[data-testid="media-timeline"]')!;
const player = () => viewer()!.querySelector<HTMLVideoElement>('video')!;

/** Open the viewer on the clip, give its player its metadata and let it pause on *frame*. */
async function pausedOn(asset: GalleryAsset, frame: number): Promise<void> {
  (target.querySelector(`#ws-history-shell article[aria-label="${asset.filename}"] button[aria-label="View ${asset.filename}"]`) as HTMLButtonElement).click();
  await settle();
  media.load(player(), 6);
  flushSync();
  timeline().getBoundingClientRect = () => BOX;
  for (const type of ['pointerdown', 'pointerup']) {
    timeline().dispatchEvent(new PointerEvent(type, { clientX: xOf(frame), clientY: 720, pointerType: 'mouse', pointerId: 1, button: 0, bubbles: true }));
  }
  media.land(player());
  flushSync();
}

function press(key: string): void {
  document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
}

/** Open the paused frame's menu from its button and run one of its rows. */
async function choose(row: 'extend' | 'frame-first' | 'frame-last'): Promise<HTMLElement> {
  viewer()!.querySelector<HTMLButtonElement>('[data-testid="frame-menu"]')!.click();
  flushSync();
  const action = document.querySelector<HTMLElement>(`[role="menu"] [data-action="${row}"]`)!;
  action.click();
  await settle();
  return action;
}

describe('a paused video frame in the Workspace', () => {
  it('fills the First frame slot with the extracted frame, takes the clip\'s scene and empties the Action, then closes the viewer', async () => {
    const asset = clip();
    await mountWithClip(asset);
    draft.patch({ scene: 'old scene', prompt: 'old action' });
    await pausedOn(asset, 72);

    await choose('frame-first');

    expect(frameApi.grabFrame).toHaveBeenCalledWith(SOURCE, expect.closeTo(3.0208, 3));
    expect(draft.state).toMatchObject({ workflow: 'img2vid', firstFramePath: SAVED.path, scene: 'A woman on a rainy street.', prompt: '' });
    expect(viewer()).toBeNull();
    expect(toasts.map((toast) => toast.type)).toEqual(['success']);
  });

  it('leaves the text alone when the clip recorded no scene, and never writes a file name', async () => {
    const asset = clip({ scene: null, prompt: 'walk mp4' });
    await mountWithClip(asset);
    draft.patch({ scene: 'my scene', prompt: 'my action' });
    await pausedOn(asset, 72);

    await choose('frame-first');

    expect(draft.state).toMatchObject({ firstFramePath: SAVED.path, scene: 'my scene', prompt: 'my action' });
  });

  it('keeps a last frame that is already there and says so', async () => {
    const asset = clip();
    await mountWithClip(asset);
    draft.patch({ workflow: 'img2vid', endImagePath: '/uploads/old/other_6.0s.png' });
    await pausedOn(asset, 72);

    await choose('frame-first');

    expect(draft.state).toMatchObject({ firstFramePath: SAVED.path, endImagePath: '/uploads/old/other_6.0s.png' });
    expect(toasts.map((toast) => toast.type)).toEqual(['info']);
    expect(toasts[0].message).toContain('other_6.0s.png');
  });

  it('fills only the Last frame slot, touching no text and not the first frame', async () => {
    const asset = clip();
    await mountWithClip(asset);
    draft.patch({ workflow: 'img2vid', firstFramePath: '/pics/start.png', scene: 'my scene', prompt: 'my action' });
    await pausedOn(asset, 72);

    await choose('frame-last');

    expect(draft.state).toMatchObject({ workflow: 'img2vid', endImagePath: SAVED.path, firstFramePath: '/pics/start.png', scene: 'my scene', prompt: 'my action' });
    expect(toasts.map((toast) => toast.message)).toEqual([]);
  });

  it('says where a last frame came from when no first frame is set', async () => {
    const asset = clip();
    await mountWithClip(asset);
    await pausedOn(asset, 72);

    await choose('frame-last');

    expect(toasts.map((toast) => toast.type)).toEqual(['success']);
  });

  it('extends from the frame, with the handle there, from the frame menu', async () => {
    const asset = clip();
    await mountWithClip(asset);
    await pausedOn(asset, 72);

    await choose('extend');

    expect(draft.state).toMatchObject({ workflow: 'extend', extendSourcePath: SOURCE });
    expect(draft.state.extendFrom).toBeCloseTo(3.0208, 3);
  });

  it('extends the whole clip, with no handle, from the menu at the last frame', async () => {
    const asset = clip();
    await mountWithClip(asset);
    await pausedOn(asset, LAST);

    await choose('extend');

    expect(draft.state).toMatchObject({ workflow: 'extend', extendSourcePath: SOURCE, extendFrom: null });
  });

  it('refuses a frame too early to extend from: nothing arrives and the reason is raised', async () => {
    const asset = clip();
    await mountWithClip(asset);
    await pausedOn(asset, 10);

    const row = await choose('extend');

    expect(draft.state.workflow).toBe('txt2vid');
    expect(row.getAttribute('aria-disabled')).toBe('true');
    expect(row.textContent).toContain(EXTEND_SHORT_REASON);
  });

  it('shows the error and leaves the slot as it was when the frame can not be read', async () => {
    const asset = clip();
    await mountWithClip(asset);
    draft.patch({ workflow: 'img2vid', firstFramePath: '/pics/start.png' });
    frameApi.grabFrame.mockRejectedValue(new ApiError('failed', 500, 'ffmpeg could not read the frame'));
    await pausedOn(asset, 72);

    await choose('frame-first');

    expect(draft.state.firstFramePath).toBe('/pics/start.png');
    expect(toasts.map((toast) => [toast.message, toast.type])).toEqual([['ffmpeg could not read the frame', 'error']]);
    expect(viewer()).not.toBeNull();
  });

  it('still fills the slot when the viewer was closed while the frame was extracted', async () => {
    const asset = clip();
    await mountWithClip(asset);
    let finish: (value: typeof SAVED) => void = () => undefined;
    frameApi.grabFrame.mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    await pausedOn(asset, 72);

    await choose('frame-last');
    press('Escape');
    press('Escape');
    await settle();
    finish(SAVED);
    await settle();

    expect(draft.state.endImagePath).toBe(SAVED.path);
    expect(viewer()).toBeNull();
  });

  it('flashes the slot a frame arrived in', async () => {
    const asset = clip();
    await mountWithClip(asset);
    await pausedOn(asset, 72);

    await choose('frame-first');

    expect(target.querySelector('[aria-labelledby="ws-start-group"]')?.hasAttribute('data-arrived')).toBe(true);
  });

  describe('on the stage', () => {
    const stageTimeline = () => target.querySelector<HTMLElement>('.latest-region [data-testid="media-timeline"]')!;
    const stageMenuButton = () => target.querySelector<HTMLButtonElement>('.latest-region [data-testid="frame-menu"]');

    /** Let the stage's own player pause on *frame*. */
    async function stagePausedOn(asset: GalleryAsset, frame: number): Promise<void> {
      await mountWithClip(asset);
      const video = target.querySelector<HTMLVideoElement>('.latest-region video')!;
      media.load(video, 6);
      flushSync();
      stageTimeline().getBoundingClientRect = () => BOX;
      for (const type of ['pointerdown', 'pointerup']) {
        stageTimeline().dispatchEvent(new PointerEvent(type, { clientX: xOf(frame), clientY: 720, pointerType: 'mouse', pointerId: 1, button: 0, bubbles: true }));
      }
      media.land(video);
      flushSync();
    }

    async function chooseOnStage(row: 'extend' | 'frame-first' | 'frame-last'): Promise<void> {
      stageMenuButton()!.click();
      flushSync();
      document.querySelector<HTMLElement>(`[role="menu"] [data-action="${row}"]`)!.click();
      await settle();
    }

    it('steps the frame with the arrows while the menu has the focus, instead of the stage through its history', async () => {
      const asset = clip();
      const older = clip({ id: 'out/older.mp4', url: '/media/out/older.mp4', thumbnail_url: '/media/out/older.mp4', filename: 'older.mp4' });
      galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([asset, older]));
      const context = makeContext({ history_assets: [asset, older] });
      context.workflow_contract.values = [...context.workflow_contract.values, 'extend'];
      workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(context);
      draft.update('workflow', 'txt2vid');
      app = await mountWorkspacePage(context, target);
      const video = target.querySelector<HTMLVideoElement>('.latest-region video')!;
      media.load(video, 6);
      flushSync();
      stageTimeline().getBoundingClientRect = () => BOX;
      for (const type of ['pointerdown', 'pointerup']) {
        stageTimeline().dispatchEvent(new PointerEvent(type, { clientX: xOf(72), clientY: 720, pointerType: 'mouse', pointerId: 1, button: 0, bubbles: true }));
      }
      media.land(video);
      flushSync();
      const src = video.getAttribute('src');
      stageMenuButton()!.dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 0 }));
      await settle();
      const focused = document.activeElement as HTMLElement;
      expect(focused.closest('[role="menu"]')).not.toBeNull();
      const before = stageMenuButton()!.getAttribute('aria-label');

      const event = new KeyboardEvent('keydown', { key: 'ArrowRight', shiftKey: true, bubbles: true, cancelable: true });
      focused.dispatchEvent(event);
      media.land(video);
      flushSync();
      await settle();

      expect(event.defaultPrevented).toBe(true);
      expect(target.querySelector('.latest-region video')?.getAttribute('src')).toBe(src);
      expect(stageMenuButton()!.getAttribute('aria-label')).not.toBe(before);
      expect(document.querySelector('[role="menu"]')).not.toBeNull();
    });

    it('shows the menu button when the stage\'s video is paused mid-clip, and none at the start', async () => {
      const asset = clip();
      await mountWithClip(asset);
      expect(stageMenuButton()).toBeNull();

      await stagePausedOn(asset, 72);
      expect(stageMenuButton()?.textContent).toContain('3.0 s');
    });

    it('extends from the frame with the menu\'s Extend row, with the handle there', async () => {
      const asset = clip();
      await stagePausedOn(asset, 72);

      await chooseOnStage('extend');

      expect(draft.state).toMatchObject({ workflow: 'extend', extendSourcePath: SOURCE });
      expect(draft.state.extendFrom).toBeCloseTo(3.0208, 3);
    });

    it('fills the Last frame slot from the menu, as from the viewer', async () => {
      const asset = clip();
      draft.patch({ workflow: 'img2vid', firstFramePath: '/pics/start.png' });
      await stagePausedOn(asset, 72);

      await chooseOnStage('frame-last');

      expect(frameApi.grabFrame).toHaveBeenCalledWith(SOURCE, expect.closeTo(3.0208, 3));
      expect(draft.state).toMatchObject({ endImagePath: SAVED.path, firstFramePath: '/pics/start.png' });
    });
  });
});
