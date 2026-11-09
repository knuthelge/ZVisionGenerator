// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { jobStore } from '$lib/state/job.svelte';
import { mascotHome } from '$lib/state/mascotHome.svelte';
import { projectPick } from '$lib/state/projectPick.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import { router } from '$lib/state/router.svelte';
import { voiceWindow } from '$lib/state/voiceWindow.svelte';
import { lastEventSource } from '../test-utils/eventSource';
import type { AppConfig, WorkspaceContext } from '$lib/types';

const configApiMocks = vi.hoisted(() => ({
  getConfig: vi.fn<() => Promise<AppConfig>>(),
  updateConfig: vi.fn<(patch: Record<string, string | number | boolean | null>) => Promise<AppConfig>>(),
}));

const workspaceApiMocks = vi.hoisted(() => ({
  getWorkspaceContext: vi.fn<() => Promise<WorkspaceContext>>(),
  getWorkspaceCoreContext: vi.fn<() => Promise<WorkspaceContext>>(),
}));

const projectApiMocks = vi.hoisted(() => ({ listProjects: vi.fn(), createProject: vi.fn() }));

vi.mock('$lib/api/projects', () => projectApiMocks);
const voicesApiMocks = vi.hoisted(() => ({
  getDesignState: vi.fn(),
}));

vi.mock('$lib/api/voices', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/voices')>();
  return { ...actual, getDesignState: voicesApiMocks.getDesignState };
});

vi.mock('$lib/api/config', () => ({
  getConfig: configApiMocks.getConfig,
  updateConfig: configApiMocks.updateConfig,
}));

vi.mock('$lib/api/workspace', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/workspace')>();
  return {
    ...actual,
    getWorkspaceContext: workspaceApiMocks.getWorkspaceContext,
    getWorkspaceCoreContext: workspaceApiMocks.getWorkspaceCoreContext,
  };
});

import App from './App.svelte';

function makeConfig(startupView: AppConfig['ui']['startup_view'] = 'config'): AppConfig {
  return {
    output_dir: '/tmp/outputs',
    log_level: 'info',
    ui: {
      gallery_page_size: 12,
      startup_view: startupView,
      output_dir: '/tmp/outputs',
      default_models: { image: 'zit', video: 'ltx-8' },
      image_model_options: ['zit'],
      video_model_options: ['ltx-8'],
      image_size_labels: [{ value: 'm', label: 'Medium' }],
      model_cache_dir: '/tmp/models',
      loras_dir: '/tmp/loras',
      huggingface_token_configured: false,
      huggingface_token_source: null,
      default_image_size: 'm',
    },
    writable_config: {
      version: 1,
      semantics: {
        omitted: 'unchanged',
        null: 'clear for clearable fields',
        empty_string: 'normalized by field empty_string behavior before persistence',
      },
      fields: [],
    },
    models: {},
  };
}

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await Promise.resolve();
  await Promise.resolve();
  flushSync();
}

describe('App startup routing', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    router.replace('workspace');
    window.history.replaceState({}, '', '/');
    configApiMocks.getConfig.mockReset();
    configApiMocks.updateConfig.mockReset();
    workspaceApiMocks.getWorkspaceContext.mockReset();
    workspaceApiMocks.getWorkspaceCoreContext.mockReset();
    configApiMocks.getConfig.mockResolvedValue(makeConfig('config'));
    projectApiMocks.listProjects.mockReset().mockResolvedValue([{ name: 'Night Bus', path: '/out/Night Bus' }]);
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
    document.body.innerHTML = '';
  });

  it('uses config instead of workspace context to resolve startup_view', async () => {
    app = flushSync(() => mount(App, { target }));
    await settle();

    expect(configApiMocks.getConfig).toHaveBeenCalled();
    expect(workspaceApiMocks.getWorkspaceContext).not.toHaveBeenCalled();
    expect(workspaceApiMocks.getWorkspaceCoreContext).not.toHaveBeenCalled();
    expect(router.page).toBe('config');
  });

  it('reads the project list once at load', async () => {
    const { projectsStore } = await import('$lib/state/projects.svelte');
    app = flushSync(() => mount(App, { target }));
    await settle();

    expect(projectApiMocks.listProjects).toHaveBeenCalledOnce();
    expect(projectsStore.list.map((project) => project.name)).toEqual(['Night Bus']);
  });

  it('ignores unsupported page query parameters', async () => {
    window.history.replaceState({}, '', '/?page=gallery');
    window.dispatchEvent(new PopStateEvent('popstate'));

    expect(router.page).toBe('workspace');
  });

  it('puts a review marker in the tab title while a render waits, and takes it out after the decision', async () => {
    configApiMocks.getConfig.mockResolvedValue(makeConfig('gallery'));
    document.title = 'ZVision Generator';
    app = flushSync(() => mount(App, { target }));
    await settle();
    expect(document.title).toBe('ZVision Generator');

    jobStore.clearJob();
    jobStore.startJob({ job_id: 'job-title', workflow: 'txt2img', prompt: 'P', model: 'zit', runs: 1, created_at: '' });
    const source = lastEventSource();
    source.emit('job_paused', { type: 'job_paused', review: { iteration: 1 }, preview_version: 1 });
    await settle();
    expect(document.title).toBe('⏸ Review · ZVision Generator');

    source.emit('job_resumed', { type: 'job_resumed' });
    await settle();
    expect(document.title).toBe('ZVision Generator');
    jobStore.clearJob();
  });

  it('calls the mascot home to react when a job ends off the Workspace', async () => {
    configApiMocks.getConfig.mockResolvedValue(makeConfig('gallery'));
    app = flushSync(() => mount(App, { target }));
    await settle();

    jobStore.clearJob();
    jobStore.startJob({ job_id: 'job-away', workflow: 'txt2img', prompt: 'P', model: 'zit', runs: 1, created_at: '' });
    await settle();
    expect(mascotHome.placement.away).toBe(true);

    const source = lastEventSource();
    source.emit('job_completed', { type: 'job_completed', outputs: [] });
    await settle();
    expect(mascotHome.placement).toMatchObject({ away: false, mood: 'cheerful' });
    expect(toasts.map((toast) => toast.type)).toContain('success');
    jobStore.clearJob();
  });

  it('names the project a job saved into when it is not the picked one, on a page other than the Workspace', async () => {
    configApiMocks.getConfig.mockResolvedValue(makeConfig('gallery'));
    projectPick.set('Night Bus');
    app = flushSync(() => mount(App, { target }));
    await settle();

    jobStore.clearJob();
    jobStore.startJob({ job_id: 'job-elsewhere', workflow: 'txt2img', prompt: 'P', model: 'zit', runs: 1, created_at: '' });
    const asset = { id: 'Feature Reel/a.png', url: '/media/Feature Reel/a.png', thumbnail_url: '/thumbs/a.png', filename: 'a.png', media_type: 'image', project: 'Feature Reel' };
    lastEventSource().emit('job_completed', { type: 'job_completed', outputs: [asset] });
    await settle();

    const finished = toasts.filter((toast) => toast.message.includes('Feature Reel'));
    expect(finished).toHaveLength(1);
    expect(finished[0].type).toBe('success');
    projectPick.set(null);
    jobStore.clearJob();
  });

  describe('a finished voice design job', () => {
    const DESIGN_CONTEXT = { job_id: 'job-design', workflow: 'voice_design', media_type: 'audio', prompt: 'A warm woman', model: 'qwen3_tts', runs: 3, created_at: '' } as const;

    async function finishDesignJob(): Promise<void> {
      configApiMocks.getConfig.mockResolvedValue(makeConfig('gallery'));
      app = flushSync(() => mount(App, { target }));
      await settle();
      jobStore.clearJob();
      jobStore.startJob({ ...DESIGN_CONTEXT });
      await settle();
      lastEventSource().emit('job_completed', { type: 'job_completed', outputs: [] });
      await settle();
    }

    beforeEach(() => {
      toasts.splice(0, toasts.length);
      voiceWindow.open = false;
      voiceWindow.requested = false;
      voicesApiMocks.getDesignState.mockReset();
      voicesApiMocks.getDesignState.mockResolvedValue({ batches: [], unpicked: 3 });
    });

    afterEach(() => {
      jobStore.clearJob();
      toasts.splice(0, toasts.length);
      voiceWindow.open = false;
      voiceWindow.requested = false;
    });

    it('announces the new voices on any page with a notice that stays, not the generic one', async () => {
      await finishDesignJob();

      expect(toasts.map((toast) => toast.message)).toEqual(['Your new voices are ready']);
      expect(toasts[0]).toMatchObject({ type: 'success', timeout: 0, action: { label: 'Pick one' } });
      expect(mascotHome.placement).toMatchObject({ mood: 'cheerful' });
    });

    it('reads the kept candidates again, so the count on New voice is current', async () => {
      await finishDesignJob();

      expect(voicesApiMocks.getDesignState).toHaveBeenCalled();
    });

    it('opens the Workspace on New voice from Pick one', async () => {
      await finishDesignJob();
      expect(router.page).toBe('gallery');
      // The Workspace that opens has nothing to load in this test.
      workspaceApiMocks.getWorkspaceCoreContext.mockReturnValue(new Promise(() => undefined));
      workspaceApiMocks.getWorkspaceContext.mockReturnValue(new Promise(() => undefined));

      toasts[0].action!.run();

      expect(router.page).toBe('workspace');
      expect(voiceWindow.requested).toBe(true);
    });

    it('stays quiet while the Voices window is open on New voice, where the candidates arrive', async () => {
      voiceWindow.show('new');

      await finishDesignJob();

      expect(toasts).toEqual([]);
      expect(voicesApiMocks.getDesignState).toHaveBeenCalled();
    });

    it('announces them when the window is open on another tab', async () => {
      voiceWindow.show('yours');

      await finishDesignJob();

      expect(toasts.map((toast) => toast.message)).toEqual(['Your new voices are ready']);
    });

    it('keeps the failure toast for a design job that failed, and reads the candidates it kept', async () => {
      configApiMocks.getConfig.mockResolvedValue(makeConfig('gallery'));
      app = flushSync(() => mount(App, { target }));
      await settle();
      jobStore.clearJob();
      jobStore.startJob({ ...DESIGN_CONTEXT });
      await settle();

      lastEventSource().emit('job_failed', { type: 'job_failed', message: 'Out of memory.' });
      await settle();

      expect(toasts.map((toast) => toast.message)).toEqual(['Generation failed: Out of memory.']);
      expect(voicesApiMocks.getDesignState).toHaveBeenCalled();
    });
  });
});
