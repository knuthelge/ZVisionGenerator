// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { router } from '$lib/state/router.svelte';
import type { AppConfig, GalleryPage, WorkspaceContext } from '$lib/types';

const configApiMocks = vi.hoisted(() => ({
  getConfig: vi.fn<() => Promise<AppConfig>>(),
  updateConfig: vi.fn<(patch: Record<string, string | number | boolean | null>) => Promise<AppConfig>>(),
}));

const workspaceApiMocks = vi.hoisted(() => ({
  getWorkspaceContext: vi.fn<() => Promise<WorkspaceContext>>(),
  getWorkspaceCoreContext: vi.fn<() => Promise<WorkspaceContext>>(),
  getHistory: vi.fn<(page?: number) => Promise<GalleryPage>>(),
}));

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
    getHistory: workspaceApiMocks.getHistory,
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
      huggingface_token_env_var: null,
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
    workspaceApiMocks.getHistory.mockReset();
    configApiMocks.getConfig.mockResolvedValue(makeConfig('config'));
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

  it('ignores unsupported page query parameters', async () => {
    window.history.replaceState({}, '', '/?page=gallery');
    window.dispatchEvent(new PopStateEvent('popstate'));

    expect(router.page).toBe('workspace');
  });
});
