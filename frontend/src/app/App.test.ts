// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { jobStore } from '$lib/state/job.svelte';
import { mascotHome } from '$lib/state/mascotHome.svelte';
import { projectPick } from '$lib/state/projectPick.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import { router } from '$lib/state/router.svelte';
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
});
