// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ModelInventory } from '$lib/types';

const modelApiMocks = vi.hoisted(() => ({
  getModelInventory: vi.fn<() => Promise<ModelInventory>>(),
  convertCheckpoint: vi.fn(),
  importLoraLocal: vi.fn(),
  importLoraHF: vi.fn(),
}));

const promptFileApiMocks = vi.hoisted(() => ({
  openPathPicker: vi.fn(),
}));

vi.mock('$lib/api/models', () => ({
  getModelInventory: modelApiMocks.getModelInventory,
  convertCheckpoint: modelApiMocks.convertCheckpoint,
  importLoraLocal: modelApiMocks.importLoraLocal,
  importLoraHF: modelApiMocks.importLoraHF,
}));

vi.mock('$lib/api/promptFiles', () => ({
  openPathPicker: promptFileApiMocks.openPathPicker,
}));

vi.mock('$lib/state/toasts.svelte', () => ({
  addToast: vi.fn(),
}));

import ModelsPage from './ModelsPage.svelte';

function makeInventory(): ModelInventory {
  return {
    models_dir: '/models',
    loras_dir: '/loras',
    image_models: [],
    video_models: [],
    loras: [],
    huggingface_configured: false,
    huggingface_token_env_var: 'HF_TOKEN',
  };
}

async function settle(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

function browseButtonFor(target: HTMLElement, inputId: string): HTMLButtonElement {
  const input = target.querySelector(`#${inputId}`);
  const field = input?.closest('.flex.flex-col');
  const button = Array.from(field?.querySelectorAll('button') ?? []).find((candidate) => candidate.textContent?.trim() === 'Browse') as HTMLButtonElement | undefined;
  if (!button) throw new Error(`Browse button not found for ${inputId}`);
  return button;
}

describe('ModelsPage Browse buttons', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    modelApiMocks.getModelInventory.mockReset();
    modelApiMocks.getModelInventory.mockResolvedValue(makeInventory());
    promptFileApiMocks.openPathPicker.mockReset();
    promptFileApiMocks.openPathPicker.mockResolvedValue({ status: 'cancelled', path: null, message: null });
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.remove();
    document.body.innerHTML = '';
  });

  it('uses backend-supported picker purposes for local model files', async () => {
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    browseButtonFor(target, 'convert-input-path').dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();
    browseButtonFor(target, 'import-local-source-path').dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(promptFileApiMocks.openPathPicker).toHaveBeenNthCalledWith(1, {
      kind: 'existing_file',
      purpose: 'checkpoint_file',
      initial_path: null,
    });
    expect(promptFileApiMocks.openPathPicker).toHaveBeenNthCalledWith(2, {
      kind: 'existing_file',
      purpose: 'lora_file',
      initial_path: null,
    });
  });

  it('renders image model sizes from size_label', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      image_models: [{ name: 'zit', family: 'zimage', size_label: 'xl' }],
    });

    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    expect(target.textContent).toContain('zit');
    expect(target.textContent).toContain('zimage');
    expect(target.textContent).toContain('xl');
  });
});
