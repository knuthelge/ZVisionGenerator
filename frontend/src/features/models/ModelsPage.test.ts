// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ModelInventory } from '$lib/types';

const modelApiMocks = vi.hoisted(() => ({
  getModelInventory: vi.fn<() => Promise<ModelInventory>>(),
  convertCheckpoint: vi.fn(),
  importLoraLocal: vi.fn(),
  importLoraHF: vi.fn(),
  deleteModel: vi.fn(),
  deleteLora: vi.fn(),
}));

const promptFileApiMocks = vi.hoisted(() => ({
  openPathPicker: vi.fn(),
}));

vi.mock('$lib/api/models', () => ({
  getModelInventory: modelApiMocks.getModelInventory,
  convertCheckpoint: modelApiMocks.convertCheckpoint,
  importLoraLocal: modelApiMocks.importLoraLocal,
  importLoraHF: modelApiMocks.importLoraHF,
  deleteModel: modelApiMocks.deleteModel,
  deleteLora: modelApiMocks.deleteLora,
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
    modelApiMocks.deleteModel.mockReset();
    modelApiMocks.deleteLora.mockReset();
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

  it('marks stored quants with their base model', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      image_models: [
        { name: 'atlas', family: 'flux2_klein', size_label: '9b', stored_quant: null },
        { name: 'atlas@q8', family: 'flux2_klein', size_label: '9b', stored_quant: { base_model: 'atlas', bits: 8 } },
      ],
    });

    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const markers = target.querySelectorAll('[data-testid="stored-quant"]');
    expect(markers).toHaveLength(1);
    expect(markers[0].closest('tr')?.querySelector('[data-testid="model-name"]')?.textContent).toBe('atlas@q8');
  });

  it('offers a quantized copy on conversion only where stored quants are supported', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({ ...makeInventory(), stored_quants_supported: false });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();
    expect(target.querySelector('#convert-quantize')).toBeNull();
    await unmount(app);

    modelApiMocks.getModelInventory.mockResolvedValue({ ...makeInventory(), stored_quants_supported: true });
    modelApiMocks.convertCheckpoint.mockResolvedValue({ tone: 'error', message: 'Keep values for assertion.' });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const { input, form } = pathAndForm(target, 'convert-input-path');
    typePath(input, '/models/checkpoint.safetensors');
    const modelType = target.querySelector('#convert-model-type') as HTMLSelectElement;
    modelType.value = 'zimage';
    modelType.dispatchEvent(new Event('change', { bubbles: true }));
    const quantize = target.querySelector('#convert-quantize') as HTMLSelectElement;
    quantize.value = '8';
    quantize.dispatchEvent(new Event('change', { bubbles: true }));
    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(modelApiMocks.convertCheckpoint).toHaveBeenCalledWith(expect.objectContaining({ quantize: '8' }));
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

  it('shows long model names and folders in full, in named sections and forms', async () => {
    const longName = 'a-very-long-model-name-that-must-not-widen-its-card';
    const modelsDir = `/models/${'long-directory-segment/'.repeat(16)}`;
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      models_dir: modelsDir,
      image_models: [{ name: longName, family: 'a-very-long-family', size_label: 'extra-long-size' }],
    });

    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const names = Array.from(target.querySelectorAll('[data-testid="model-name"]')).map((n) => n.textContent);
    expect(names).toContain(longName);
    const values = Array.from(target.querySelectorAll('dd')).map((dd) => dd.textContent ?? '');
    expect(values.some((value) => value.includes(modelsDir))).toBe(true);

    const headings = Array.from(target.querySelectorAll('h2')).map((h) => h.textContent?.trim());
    expect(headings).toEqual(expect.arrayContaining(['Folders and access', 'Image models', 'Video models', 'LoRAs']));
    expect(target.querySelectorAll('form')).toHaveLength(3);
  });

  it('shows one add form at a time and keeps what was typed when switching', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue(makeInventory());

    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const visibleForms = () => Array.from(target.querySelectorAll('form')).filter((form) => !form.hidden);
    const modes = target.querySelector('[role="group"][aria-label="What to add"]') as HTMLElement;
    expect(visibleForms()).toHaveLength(1);
    expect(visibleForms()[0].querySelector('#convert-input-path')).not.toBeNull();

    const name = target.querySelector('#convert-name') as HTMLInputElement;
    name.value = 'my-model';
    (modes.querySelector('[data-value="hf"]') as HTMLButtonElement).click();
    await settle();
    expect(visibleForms()).toHaveLength(1);
    expect(visibleForms()[0].querySelector('#import-hf-repo-id')).not.toBeNull();

    (modes.querySelector('[data-value="convert"]') as HTMLButtonElement).click();
    await settle();
    expect((target.querySelector('#convert-name') as HTMLInputElement).value).toBe('my-model');
  });

  it('marks download state on the name and shows one memory badge per model', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      image_models: [
        {
          name: 'zit',
          family: 'zimage',
          downloaded: true,
          memory_fit: { budget_gb: 10.7, by_quantize: { none: { status: 'too_large', required_gb: 20.8 }, '8': { status: 'fits', required_gb: 8.1 } } },
        },
        { name: 'klein4b', family: 'flux2_klein', downloaded: false, memory_fit: null },
      ],
      video_models: [{ name: 'ltx-4', family: 'ltx', supports_i2v: true, downloaded: true, memory_fit: null }],
    });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const names = Object.fromEntries(
      Array.from(target.querySelectorAll('[data-testid="model-name"]')).map((el) => [el.textContent, el.getAttribute('data-downloaded')]),
    );
    expect(names).toEqual({ zit: 'true', klein4b: 'false', 'ltx-4': 'true' });

    const fits = Array.from(target.querySelectorAll('[data-testid="model-memory-fit"]'));
    expect(fits.map((el) => el.querySelector('[data-status]')?.getAttribute('data-status'))).toEqual(['too_large']);
    expect(fits[0].querySelector('[role="tooltip"]')?.textContent).toContain('q8');

    const klein = Array.from(target.querySelectorAll('[data-testid="model-name"]')).find((el) => el.textContent === 'klein4b');
    const kleinTooltip = klein?.parentElement?.querySelector('[role="tooltip"]')?.textContent ?? '';
    expect(kleinTooltip.startsWith('klein4b')).toBe(true); // full name stays readable when the cell truncates
    expect(kleinTooltip).toContain('Not downloaded');
    // Only the memory badges are tab stops; the per-row name tooltips are hover-only.
    expect(klein?.parentElement?.hasAttribute('tabindex')).toBe(false);
    expect(fits[0].getAttribute('tabindex')).toBe('0');
    expect(target.querySelector('[data-testid="model-download-status"]')).toBeNull();
  });

  it('lists the stored quants a model delete removes', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      image_models: [
        { name: 'zit', family: 'zimage', downloaded: true, delete: { kind: 'huggingface', repo_id: 'org/zit', linked_by: [], stored_quants: ['zit@q8'] } },
        { name: 'atlas', family: 'flux2_klein', downloaded: true, delete: { kind: 'installed', repo_id: null, linked_by: [], stored_quants: [] } },
      ],
    });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();
    const buttons = target.querySelectorAll('[data-testid="delete-button"]');

    (buttons[0] as HTMLButtonElement).click();
    await settle();
    expect(document.querySelector('[data-testid="delete-stored-quants"]')?.textContent).toContain('zit@q8');

    (Array.from(document.querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Cancel') as HTMLButtonElement).click();
    await settle();
    (buttons[1] as HTMLButtonElement).click();
    await settle();
    expect(document.querySelector('[data-testid="delete-stored-quants"]')).toBeNull();
  });

  it('offers delete only for deletable models and warns about converted models linked to a download', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      image_models: [
        { name: 'artaix', family: 'flux2_klein', downloaded: true, delete: { kind: 'installed', repo_id: null, linked_by: [] } },
        { name: 'klein9b', family: 'flux2_klein', downloaded: true, delete: { kind: 'huggingface', repo_id: 'org/klein', linked_by: ['artaix'] } },
        { name: 'klein4b', family: 'flux2_klein', downloaded: false, delete: null },
      ],
      loras: [{ name: 'style', size_label: '10 MB' }],
    });
    modelApiMocks.deleteModel.mockResolvedValue({ tone: 'success', message: "Deleted the HuggingFace download of 'org/klein'.", detail: '' });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const labels = Array.from(target.querySelectorAll('[data-testid="delete-button"]')).map((el) => el.getAttribute('aria-label'));
    expect(labels).toEqual(['Delete artaix', 'Delete the Hugging Face download of klein9b', 'Delete style']);

    (target.querySelectorAll('[data-testid="delete-button"]')[1] as HTMLButtonElement).click();
    await settle();
    const dialog = document.querySelector('[role="alertdialog"]');
    expect(dialog?.textContent).toContain('org/klein');
    expect(document.querySelector('[data-testid="delete-linked-warning"]')?.textContent).toContain('artaix');
    expect(modelApiMocks.deleteModel).not.toHaveBeenCalled();

    const confirm = Array.from(document.querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Delete') as HTMLButtonElement;
    confirm.click();
    await settle();

    expect(modelApiMocks.deleteModel).toHaveBeenCalledWith('klein9b');
    expect(modelApiMocks.getModelInventory).toHaveBeenCalledTimes(2);
    expect(document.querySelector('[data-testid="delete-dialog"]')).toBeNull();
  });

  it('deletes a LoRA after confirmation and surfaces a refused delete', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({ ...makeInventory(), loras: [{ name: 'style', size_label: '10 MB' }] });
    modelApiMocks.deleteLora.mockRejectedValue(new Error('Wait for the running job to finish before deleting models or LoRAs.'));
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    (target.querySelector('[data-testid="delete-button"]') as HTMLButtonElement).click();
    await settle();
    expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain('style.safetensors');
    (Array.from(document.querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Delete') as HTMLButtonElement).click();
    await settle();

    expect(modelApiMocks.deleteLora).toHaveBeenCalledWith('style');
    expect(target.textContent).toContain('Wait for the running job to finish');
  });
});
