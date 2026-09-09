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
  await new Promise((resolve) => setTimeout(resolve, 0));
  flushSync();
}

function browseButtonFor(target: HTMLElement, inputId: string): HTMLButtonElement {
  const input = target.querySelector(`#${inputId}`);
  const field = input?.closest('.flex.flex-col');
  const button = Array.from(field?.querySelectorAll('button') ?? []).find((candidate) => candidate.textContent?.trim() === 'Browse') as HTMLButtonElement | undefined;
  if (!button) throw new Error(`Browse button not found for ${inputId}`);
  return button;
}

function pathAndForm(target: HTMLElement, inputId: string): { input: HTMLInputElement; form: HTMLFormElement } {
  const input = target.querySelector(`#${inputId}`) as HTMLInputElement;
  return { input, form: input.closest('form') as HTMLFormElement };
}

function typePath(input: HTMLInputElement, value: string): void {
  input.value = value;
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

describe('ModelsPage Browse buttons', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    modelApiMocks.getModelInventory.mockReset();
    modelApiMocks.convertCheckpoint.mockReset();
    modelApiMocks.importLoraLocal.mockReset();
    modelApiMocks.importLoraHF.mockReset();
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

  it('submits the first browse-selected checkpoint path in the conversion API payload', async () => {
    promptFileApiMocks.openPathPicker.mockResolvedValueOnce({
      status: 'selected', path: '/models/first-checkpoint.safetensors', message: null,
    });
    modelApiMocks.convertCheckpoint.mockResolvedValue({ tone: 'error', message: 'Keep values for assertion.' });

    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    browseButtonFor(target, 'convert-input-path').click();
    await settle();
    const modelType = target.querySelector('#convert-model-type') as HTMLSelectElement;
    modelType.value = 'zimage';
    modelType.dispatchEvent(new Event('change', { bubbles: true }));
    (target.querySelector('#convert-input-path') as HTMLInputElement).closest('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(modelApiMocks.convertCheckpoint).toHaveBeenCalledWith(expect.objectContaining({
      input_path: '/models/first-checkpoint.safetensors',
      model_type: 'zimage',
    }));
  });

  it('submits the first browse-selected local LoRA path in the import API payload', async () => {
    promptFileApiMocks.openPathPicker.mockResolvedValueOnce({
      status: 'selected', path: '/models/first-lora.safetensors', message: null,
    });
    modelApiMocks.importLoraLocal.mockResolvedValue({ tone: 'error', message: 'Keep values for assertion.' });

    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    browseButtonFor(target, 'import-local-source-path').click();
    await settle();
    (target.querySelector('#import-local-source-path') as HTMLInputElement).closest('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(modelApiMocks.importLoraLocal).toHaveBeenCalledWith(expect.objectContaining({
      source_path: '/models/first-lora.safetensors',
    }));
  });

  it('is expected to clear a successful checkpoint path and begin the next browse with no initial path', async () => {
    promptFileApiMocks.openPathPicker.mockResolvedValue({ status: 'cancelled', path: null, message: null });
    modelApiMocks.convertCheckpoint.mockResolvedValue({ tone: 'success', message: 'Converted.' });

    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const pathInput = target.querySelector('#convert-input-path') as HTMLInputElement;
    pathInput.value = '/models/to-convert.safetensors';
    pathInput.dispatchEvent(new Event('input', { bubbles: true }));
    pathInput.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();
    expect(pathInput.value).toBe('/models/to-convert.safetensors');
    expect(target.textContent).toContain('Path loaded from the host machine.');

    const modelType = target.querySelector('#convert-model-type') as HTMLSelectElement;
    modelType.value = 'zimage';
    modelType.dispatchEvent(new Event('change', { bubbles: true }));
    pathInput.closest('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(modelApiMocks.convertCheckpoint).toHaveBeenCalledWith(expect.objectContaining({ input_path: '/models/to-convert.safetensors' }));
    expect.soft(pathInput.value).toBe('');
    expect.soft(new FormData(pathInput.closest('form')!).get('input_path')).toBe('');
    expect.soft(target.textContent).not.toContain('Path loaded from the host machine.');

    browseButtonFor(target, 'convert-input-path').click();
    await settle();
    expect.soft(promptFileApiMocks.openPathPicker).toHaveBeenLastCalledWith(expect.objectContaining({
      purpose: 'checkpoint_file',
      initial_path: null,
    }));
  });

  it('preserves checkpoint retry state and surfaces an operation error result', async () => {
    modelApiMocks.convertCheckpoint.mockResolvedValue({ tone: 'error', message: 'Checkpoint is not a valid safetensors file.' });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const { input, form } = pathAndForm(target, 'convert-input-path');
    typePath(input, '/models/retry-checkpoint.safetensors');
    const modelType = target.querySelector('#convert-model-type') as HTMLSelectElement;
    modelType.value = 'zimage';
    modelType.dispatchEvent(new Event('change', { bubbles: true }));
    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(modelApiMocks.convertCheckpoint).toHaveBeenCalledWith(expect.objectContaining({
      input_path: '/models/retry-checkpoint.safetensors',
      model_type: 'zimage',
    }));
    expect(input.value).toBe('/models/retry-checkpoint.safetensors');
    expect(new FormData(form).get('input_path')).toBe('/models/retry-checkpoint.safetensors');
    expect(target.textContent).toContain('Checkpoint is not a valid safetensors file.');
  });

  it('preserves local LoRA retry state and surfaces a thrown operation error', async () => {
    modelApiMocks.importLoraLocal.mockRejectedValue(new Error('Local import request failed.'));
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const { input, form } = pathAndForm(target, 'import-local-source-path');
    typePath(input, '/loras/retry-lora.safetensors');
    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(modelApiMocks.importLoraLocal).toHaveBeenCalledWith(expect.objectContaining({
      source_path: '/loras/retry-lora.safetensors',
    }));
    expect(input.value).toBe('/loras/retry-lora.safetensors');
    expect(new FormData(form).get('source_path')).toBe('/loras/retry-lora.safetensors');
    expect(target.textContent).toContain('Local import request failed.');
  });

  it('clears a successful local LoRA path and resets its next picker origin', async () => {
    promptFileApiMocks.openPathPicker.mockResolvedValue({ status: 'cancelled', path: null, message: null });
    modelApiMocks.importLoraLocal.mockResolvedValue({ tone: 'success', message: 'Imported.' });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const { input, form } = pathAndForm(target, 'import-local-source-path');
    typePath(input, '/loras/import-me.safetensors');
    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(modelApiMocks.importLoraLocal).toHaveBeenCalledWith(expect.objectContaining({
      source_path: '/loras/import-me.safetensors',
    }));
    expect(input.value).toBe('');
    expect(new FormData(form).get('source_path')).toBe('');
    expect(target.textContent).not.toContain('Path loaded from the host machine.');

    browseButtonFor(target, 'import-local-source-path').click();
    await settle();
    expect(promptFileApiMocks.openPathPicker).toHaveBeenLastCalledWith(expect.objectContaining({
      purpose: 'lora_file',
      initial_path: null,
    }));
  });

  it('blocks an empty Model Type through native requestSubmit validation and submits after a type is selected', async () => {
    modelApiMocks.convertCheckpoint.mockResolvedValue({ tone: 'error', message: 'Keep values for assertion.' });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const checkpoint = pathAndForm(target, 'convert-input-path');
    const modelType = target.querySelector('#convert-model-type') as HTMLSelectElement;
    typePath(checkpoint.input, '/models/valid-checkpoint.safetensors');

    expect(modelType.required).toBe(true);
    expect(modelType.value).toBe('');
    expect(modelType.validity.valueMissing).toBe(true);
    expect(checkpoint.form.checkValidity()).toBe(false);

    checkpoint.form.requestSubmit();
    await settle();
    expect(modelApiMocks.convertCheckpoint).not.toHaveBeenCalled();

    modelType.value = 'zimage';
    modelType.dispatchEvent(new Event('change', { bubbles: true }));
    expect(modelType.validity.valid).toBe(true);
    expect(checkpoint.form.checkValidity()).toBe(true);

    checkpoint.form.requestSubmit();
    await settle();
    expect(modelApiMocks.convertCheckpoint).toHaveBeenCalledWith(expect.objectContaining({
      input_path: '/models/valid-checkpoint.safetensors',
      model_type: 'zimage',
    }));
  });

  it('uses native required validation for both local path controls', async () => {
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const checkpoint = pathAndForm(target, 'convert-input-path');
    const localLora = pathAndForm(target, 'import-local-source-path');
    expect(checkpoint.input.required).toBe(true);
    expect(localLora.input.required).toBe(true);
    expect(checkpoint.input.validity.valueMissing).toBe(true);
    expect(localLora.form.checkValidity()).toBe(false);

    localLora.form.requestSubmit();
    await settle();
    expect(modelApiMocks.importLoraLocal).not.toHaveBeenCalled();

    typePath(localLora.input, '/loras/valid.safetensors');
    expect(localLora.form.checkValidity()).toBe(true);
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

  it('keeps the Models information and operation grids responsive and overflow-safe', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      models_dir: `/models/${'long-directory-segment/'.repeat(16)}`,
      image_models: [{ name: 'a-very-long-model-name-that-must-not-widen-its-card', family: 'a-very-long-family', size_label: 'extra-long-size' }],
    });

    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const content = target.querySelector('.space-y-8') as HTMLElement;
    expect(content.className).toContain('min-w-0');
    // pb-6 is Tailwind's 1.5rem / 24 CSS px at the app's default root size.
    expect(content.className).toContain('pb-6');

    const responsiveGrids = Array.from(target.querySelectorAll('.grid')).filter((grid) =>
      grid.className.includes('md:grid-cols-2') && grid.className.includes('xl:grid-cols-3'),
    );
    expect(responsiveGrids).toHaveLength(2);
    for (const grid of responsiveGrids) {
      expect(grid.className).toContain('min-w-0');
      expect(grid.className).toContain('grid-cols-1');
    }

    const inventoryCards = Array.from(target.querySelectorAll('.admin-section.overflow-hidden'));
    expect(inventoryCards).toHaveLength(3);
    for (const card of inventoryCards) {
      expect(card.className).toContain('min-w-0');
      expect(card.className).toContain('overflow-hidden');
    }
    expect(Array.from(target.querySelectorAll('table')).every((table) => table.className.includes('table-fixed'))).toBe(true);

    for (const form of Array.from(target.querySelectorAll('form'))) {
      expect(form.className).toContain('admin-section');
      expect(form.className).toContain('min-w-0');
    }
  });
});
