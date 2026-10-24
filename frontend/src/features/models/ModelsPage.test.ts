// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import { downloadStore } from '$lib/state/downloads.svelte';
import { addToast, dismissToast } from '$lib/state/toasts.svelte';
import type { DownloadItem, DownloadPlan, ModelInventory } from '$lib/types';

const modelApiMocks = vi.hoisted(() => ({
  getModelInventory: vi.fn<() => Promise<ModelInventory>>(),
  convertCheckpoint: vi.fn(),
  inspectLtxCheckpoint: vi.fn(),
  getConversionStatus: vi.fn(),
  importLoraLocal: vi.fn(),
  importLoraHF: vi.fn(),
  deleteModel: vi.fn(),
  deleteEnhancerModel: vi.fn(),
  deleteOtherDownload: vi.fn(),
  freeStrandedCacheFiles: vi.fn(),
  deleteLora: vi.fn(),
  getDownloadPlan: vi.fn(),
  startDownload: vi.fn(),
  getDownloads: vi.fn(),
  cancelDownload: vi.fn(),
}));

const browseApiMocks = vi.hoisted(() => ({
  listDirectory: vi.fn(),
  browsePlaces: vi.fn(),
  createFolder: vi.fn(),
}));

vi.mock('$lib/api/models', () => ({
  getModelInventory: modelApiMocks.getModelInventory,
  convertCheckpoint: modelApiMocks.convertCheckpoint,
  inspectLtxCheckpoint: modelApiMocks.inspectLtxCheckpoint,
  getConversionStatus: modelApiMocks.getConversionStatus,
  importLoraLocal: modelApiMocks.importLoraLocal,
  importLoraHF: modelApiMocks.importLoraHF,
  deleteModel: modelApiMocks.deleteModel,
  deleteEnhancerModel: modelApiMocks.deleteEnhancerModel,
  deleteOtherDownload: modelApiMocks.deleteOtherDownload,
  freeStrandedCacheFiles: modelApiMocks.freeStrandedCacheFiles,
  deleteLora: modelApiMocks.deleteLora,
  getDownloadPlan: modelApiMocks.getDownloadPlan,
  startDownload: modelApiMocks.startDownload,
  getDownloads: modelApiMocks.getDownloads,
  cancelDownload: modelApiMocks.cancelDownload,
}));

vi.mock('$lib/api/browse', () => browseApiMocks);

vi.mock('$lib/state/toasts.svelte', () => ({
  addToast: vi.fn(() => 'toast-1'),
  dismissToast: vi.fn(),
}));

import { DOWNLOADABLE_TOOLTIP, NOT_DOWNLOADED_FIRST_USE_TOOLTIP, PREPARED_INSTANT_TOOLTIP, PREPARED_SEEDVR2_TOOLTIP } from '$lib/components/molecules/ModelStatusBadges.svelte';
import ModelsPage from './ModelsPage.svelte';
import { chooseInBrowser, listingWith, placesFor } from '../../test-utils/pathBrowser';
import { pickOption } from '../../test-utils/select';
import { inventoryMemory } from '$lib/state/pageMemory';

beforeEach(() => inventoryMemory.forget());

function makeInventory(): ModelInventory {
  return {
    models_dir: '/models',
    loras_dir: '/loras',
    image_models: [],
    video_models: [],
    loras: [],
    huggingface_configured: false,
    huggingface_token_source: null,
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
  const button = field?.querySelector<HTMLButtonElement>('button[aria-label="Browse"]') ?? undefined;
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
    downloadStore.reset();
    modelApiMocks.getModelInventory.mockReset();
    modelApiMocks.convertCheckpoint.mockReset();
    modelApiMocks.inspectLtxCheckpoint.mockReset();
    modelApiMocks.getConversionStatus.mockReset();
    modelApiMocks.getConversionStatus.mockResolvedValue({ state: 'idle' });
    modelApiMocks.importLoraLocal.mockReset();
    modelApiMocks.importLoraHF.mockReset();
    modelApiMocks.deleteModel.mockReset();
    modelApiMocks.deleteEnhancerModel.mockReset();
    modelApiMocks.deleteOtherDownload.mockReset();
    modelApiMocks.freeStrandedCacheFiles.mockReset();
    modelApiMocks.deleteLora.mockReset();
    modelApiMocks.getDownloadPlan.mockReset();
    modelApiMocks.startDownload.mockReset();
    modelApiMocks.getDownloads.mockReset();
    modelApiMocks.getDownloads.mockResolvedValue({ items: [] });
    modelApiMocks.cancelDownload.mockReset();
    modelApiMocks.getModelInventory.mockResolvedValue(makeInventory());
    for (const mock of Object.values(browseApiMocks)) mock.mockReset();
    browseApiMocks.browsePlaces.mockImplementation(async (purpose: 'checkpoint_file' | 'lora_file') => placesFor(purpose));
    browseApiMocks.listDirectory.mockResolvedValue(listingWith('/models/first.safetensors'));
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.remove();
    document.body.innerHTML = '';
  });

  it('browses with the checkpoint and LoRA purposes for local model files', async () => {
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    browseButtonFor(target, 'convert-input-path').click();
    await settle();
    expect(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('checkpoint_file', '', expect.anything());
    document.querySelector<HTMLButtonElement>('[role="dialog"] button[aria-label="Close dialog"]')!.click();
    await settle();

    browseButtonFor(target, 'import-local-source-path').click();
    await settle();
    expect(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('lora_file', '', expect.anything());
  });

  it('submits the first browse-selected checkpoint path in the conversion API payload', async () => {
    browseApiMocks.listDirectory.mockResolvedValue(listingWith('/models/first-checkpoint.safetensors'));
    modelApiMocks.convertCheckpoint.mockResolvedValue({ tone: 'error', message: 'Keep values for assertion.' });

    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    browseButtonFor(target, 'convert-input-path').click();
    await settle();
    chooseInBrowser('first-checkpoint.safetensors');
    await settle();
    const modelType = target.querySelector('#convert-model-type') as HTMLButtonElement;
    pickOption(modelType, 'zimage');
    (target.querySelector('#convert-input-path') as HTMLInputElement).closest('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(modelApiMocks.convertCheckpoint).toHaveBeenCalledWith(expect.objectContaining({
      input_path: '/models/first-checkpoint.safetensors',
      model_type: 'zimage',
    }));
  });

  it('still imports a LoRA from Hugging Face with the form values', async () => {
    modelApiMocks.importLoraHF.mockResolvedValue({ tone: 'error', message: 'Keep values for assertion.' });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    typePath(target.querySelector('#import-hf-repo-id') as HTMLInputElement, 'owner/repository');
    typePath(target.querySelector('#import-hf-filename') as HTMLInputElement, 'style.safetensors');
    (target.querySelector('#import-hf-repo-id') as HTMLInputElement).closest('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(modelApiMocks.importLoraHF).toHaveBeenCalledWith(expect.objectContaining({ repo_id: 'owner/repository', filename: 'style.safetensors' }));
  });

  it('submits the first browse-selected local LoRA path in the import API payload', async () => {
    browseApiMocks.listDirectory.mockResolvedValue(listingWith('/models/first-lora.safetensors'));
    modelApiMocks.importLoraLocal.mockResolvedValue({ tone: 'error', message: 'Keep values for assertion.' });

    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    browseButtonFor(target, 'import-local-source-path').click();
    await settle();
    chooseInBrowser('first-lora.safetensors');
    await settle();
    (target.querySelector('#import-local-source-path') as HTMLInputElement).closest('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(modelApiMocks.importLoraLocal).toHaveBeenCalledWith(expect.objectContaining({
      source_path: '/models/first-lora.safetensors',
    }));
  });

  it('is expected to clear a successful checkpoint path and begin the next browse with no initial path', async () => {
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

    const modelType = target.querySelector('#convert-model-type') as HTMLButtonElement;
    pickOption(modelType, 'zimage');
    pathInput.closest('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(modelApiMocks.convertCheckpoint).toHaveBeenCalledWith(expect.objectContaining({ input_path: '/models/to-convert.safetensors' }));
    expect.soft(pathInput.value).toBe('');
    expect.soft(new FormData(pathInput.closest('form')!).get('input_path')).toBe('');
    expect.soft(target.textContent).not.toContain('Path loaded from the host machine.');
    // The model type is cleared too, and the submitted value follows what is shown.
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect((target.querySelector('#convert-model-type') as HTMLButtonElement).dataset.value).toBe('');
    expect((target.querySelector('select[name="model_type"]') as HTMLSelectElement).value).toBe('');

    browseButtonFor(target, 'convert-input-path').click();
    await settle();
    expect.soft(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('checkpoint_file', '', expect.anything());
  });

  it('preserves checkpoint retry state and surfaces an operation error result', async () => {
    modelApiMocks.convertCheckpoint.mockResolvedValue({ tone: 'error', message: 'Checkpoint is not a valid safetensors file.' });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const { input, form } = pathAndForm(target, 'convert-input-path');
    typePath(input, '/models/retry-checkpoint.safetensors');
    const modelType = target.querySelector('#convert-model-type') as HTMLButtonElement;
    pickOption(modelType, 'zimage');
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
    expect(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('lora_file', '', expect.anything());
  });

  it('blocks an empty Model Type through native requestSubmit validation and submits after a type is selected', async () => {
    modelApiMocks.convertCheckpoint.mockResolvedValue({ tone: 'error', message: 'Keep values for assertion.' });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const checkpoint = pathAndForm(target, 'convert-input-path');
    const modelType = target.querySelector('#convert-model-type') as HTMLButtonElement;
    typePath(checkpoint.input, '/models/valid-checkpoint.safetensors');

    const modelTypeField = target.querySelector('select[name="model_type"]') as HTMLSelectElement;
    expect(modelTypeField.required).toBe(true);
    expect(modelTypeField.value).toBe('');
    expect(modelTypeField.validity.valueMissing).toBe(true);
    expect(checkpoint.form.checkValidity()).toBe(false);

    checkpoint.form.requestSubmit();
    await settle();
    expect(modelApiMocks.convertCheckpoint).not.toHaveBeenCalled();

    pickOption(modelType, 'zimage');
    expect(modelTypeField.validity.valid).toBe(true);
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
        { download_needed: false, name: 'atlas', label: 'atlas', family: 'flux2_klein', size_label: '9b', stored_quant: null },
        { download_needed: false, name: 'atlas@q8', label: 'atlas@q8', family: 'flux2_klein', size_label: '9b', stored_quant: { base_model: 'atlas', bits: 8 } },
      ],
    });

    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const markers = target.querySelectorAll('[data-testid="stored-quant"]');
    expect(markers).toHaveLength(1);
    expect(markers[0].closest('tr')?.querySelector('[data-testid="model-name"]')?.textContent).toBe('atlas@q8');
  });

  it('marks converted LTX models with their stored level', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      video_models: [
        { download_needed: false, name: 'ltx-8', label: 'ltx-8', family: 'ltx', supports_i2v: true, converted_bits: null },
        { download_needed: false, name: 'my-finetune', label: 'my-finetune', family: 'ltx', supports_i2v: true, converted_bits: 8 },
      ],
    });

    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const markers = target.querySelectorAll('[data-testid="converted-level"]');
    expect(markers).toHaveLength(1);
    expect(markers[0].closest('tr')?.querySelector('[data-testid="model-name"]')?.textContent).toBe('my-finetune');
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
    const modelType = target.querySelector('#convert-model-type') as HTMLButtonElement;
    pickOption(modelType, 'zimage');
    const quantize = target.querySelector('#convert-quantize') as HTMLButtonElement;
    pickOption(quantize, '8');
    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(modelApiMocks.convertCheckpoint).toHaveBeenCalledWith(expect.objectContaining({ quantize: '8' }));
  });

  it('converts an LTX checkpoint at the inspected level, with Distill preselected for plain dev, as a background job', async () => {
    modelApiMocks.inspectLtxCheckpoint.mockResolvedValue({
      layout: 'full', fp8_weights: 440, distillation: 'dev', advice: 'Convert it with Distill.', recommend_distill: true, recommended_quantize: 8,
    });
    modelApiMocks.convertCheckpoint.mockResolvedValue({ tone: 'success', message: 'Converting.', detail: '', background: true });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const { input, form } = pathAndForm(target, 'convert-input-path');
    typePath(input, '/models/ltx-finetune.safetensors');
    const modelType = target.querySelector('#convert-model-type') as HTMLButtonElement;
    pickOption(modelType, 'ltx-2.3');
    await new Promise((resolve) => setTimeout(resolve, 450));
    await settle();

    expect(modelApiMocks.inspectLtxCheckpoint).toHaveBeenCalledWith('/models/ltx-finetune.safetensors');
    expect(target.querySelector('#convert-base-model')).toBeNull();
    expect((target.querySelector('#convert-ltx-quantize') as HTMLButtonElement).dataset.value).toBe('8');
    expect((target.querySelector('#convert-distill') as HTMLInputElement).checked).toBe(true);

    let answerStatus: (value: unknown) => void = () => {};
    modelApiMocks.getConversionStatus.mockImplementationOnce(() => new Promise((resolve) => { answerStatus = resolve; }));
    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    // Before the first status read, the progress line already names the model being converted.
    expect(target.textContent).toContain('Converting ltx-finetune: Starting');
    answerStatus({ state: 'running', name: 'ltx-finetune', stage: 'Writing the transformer', done: 3, total: 10 });
    await settle();

    expect(modelApiMocks.convertCheckpoint).toHaveBeenCalledWith(expect.objectContaining({ model_type: 'ltx-2.3', quantize: '8', distill: true }));
    expect(target.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe('3');
  });

  it('ignores an inspect answer that arrives after the path changed', async () => {
    let answerFirst: (value: unknown) => void = () => {};
    modelApiMocks.inspectLtxCheckpoint
      .mockImplementationOnce(() => new Promise((resolve) => { answerFirst = resolve; }))
      .mockRejectedValueOnce(new Error('not an LTX checkpoint'));
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const { input } = pathAndForm(target, 'convert-input-path');
    const modelType = target.querySelector('#convert-model-type') as HTMLButtonElement;
    pickOption(modelType, 'ltx-2.3');
    typePath(input, '/models/dev.safetensors');
    await new Promise((resolve) => setTimeout(resolve, 450));
    typePath(input, '/models/other.safetensors');
    answerFirst({ layout: 'full', fp8_weights: 0, distillation: 'dev', advice: 'Convert it with Distill.', recommend_distill: true, recommended_quantize: 4 });
    await new Promise((resolve) => setTimeout(resolve, 450));
    await settle();

    expect((target.querySelector('#convert-distill') as HTMLInputElement).checked).toBe(false);
    expect(target.textContent).not.toContain('Convert it with Distill.');
  });

  it('keeps reading the conversion status after a missed read', async () => {
    modelApiMocks.getConversionStatus
      .mockResolvedValueOnce({ state: 'running', name: 'ft', stage: 'Writing the transformer', done: 1, total: 10 })
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValue({ state: 'done', name: 'ft', message: 'Converted.' });
    vi.useFakeTimers();
    try {
      app = flushSync(() => mount(ModelsPage, { target }));
      await vi.advanceTimersByTimeAsync(10_000);
    } finally {
      vi.useRealTimers();
    }
    await settle();

    expect(modelApiMocks.getConversionStatus.mock.calls.length).toBeGreaterThanOrEqual(3);
    expect(target.textContent).toContain('Converted.');
  });

  it('shows the disk use of image and video models, and a dash when nothing is on disk', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      image_models: [
        { download_needed: false, name: 'zit', label: 'zit', family: 'zimage', size_label: 'xl', disk_size_label: '32.9 GB' },
        { download_needed: false, name: 'klein4b', label: 'klein4b', family: 'flux2_klein', size_label: '4b', disk_size_label: null },
      ],
      video_models: [{ download_needed: false, name: 'ltx-8', label: 'ltx-8', family: 'ltx', supports_i2v: true, disk_size_label: '112.3 GB' }],
    });

    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    expect(Array.from(target.querySelectorAll('[data-testid="disk-size"]')).map((el) => el.textContent)).toEqual(['32.9 GB', '112.3 GB']);
  });

  it('shows long model names and folders in full, in named sections and forms', async () => {
    const longName = 'a-very-long-model-name-that-must-not-widen-its-card';
    const modelsDir = `/models/${'long-directory-segment/'.repeat(16)}`;
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      models_dir: modelsDir,
      image_models: [{ download_needed: false, name: longName, label: longName, family: 'a-very-long-family', size_label: 'extra-long-size' }],
    });

    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const names = Array.from(target.querySelectorAll('[data-testid="model-name"]')).map((n) => n.textContent);
    expect(names).toContain(longName);
    const values = Array.from(target.querySelectorAll('dd')).map((dd) => dd.textContent ?? '');
    expect(values.some((value) => value.includes(modelsDir))).toBe(true);

    expect(target.querySelectorAll('form')).toHaveLength(3);
  });

  it('marks download state on the name and shows one memory badge per model', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      image_models: [
        {
          download_needed: false,
          name: 'zit',
          label: 'zit',
          family: 'zimage',
          downloaded: true,
          memory_fit: { budget_gb: 10.7, by_quantize: { none: { status: 'too_large', required_gb: 20.8 }, '8': { status: 'fits', required_gb: 8.1 } } },
        },
        { download_needed: true, name: 'klein4b', label: 'klein4b', family: 'flux2_klein', downloaded: false, memory_fit: null },
      ],
      video_models: [{ download_needed: false, name: 'ltx-4', label: 'ltx-4', family: 'ltx', supports_i2v: true, downloaded: true, memory_fit: null }],
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
        { download_needed: false, name: 'zit', label: 'zit', family: 'zimage', downloaded: true, delete: { kind: 'huggingface', repo_id: 'org/zit', linked_by: [], stored_quants: ['zit@q8'] } },
        { download_needed: false, name: 'atlas', label: 'atlas', family: 'flux2_klein', downloaded: true, delete: { kind: 'installed', repo_id: null, linked_by: [], stored_quants: [] } },
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

  it('deletes only the prepared copy of a model whose download another model still uses', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      video_models: [
        { download_needed: false, name: 'ltx-4', label: 'ltx-4', family: 'ltx', supports_i2v: true, downloaded: true, delete: { kind: 'copy', repo_id: 'org/ltx', linked_by: [], stored_quants: ['ltx-4@q4'], shared_with: ['ltx-8'] } },
      ],
    });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();
    const button = target.querySelector('[data-testid="delete-button"]') as HTMLButtonElement;
    expect(button.getAttribute('aria-label')).toBe('Delete the prepared copy of ltx-4');

    button.click();
    await settle();
    const copy = document.querySelector('[data-testid="delete-copy"]')?.textContent ?? '';
    expect(copy).toContain('ltx-4@q4');
    expect(copy).toContain('ltx-8');
    expect(document.querySelector('[data-testid="delete-stored-quants"]')).toBeNull();
  });

  it('offers delete only for deletable models and warns about converted models linked to a download', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      image_models: [
        { download_needed: false, name: 'artaix', label: 'artaix', family: 'flux2_klein', downloaded: true, delete: { kind: 'installed', repo_id: null, linked_by: [] } },
        { download_needed: false, name: 'klein9b', label: 'klein9b', family: 'flux2_klein', downloaded: true, delete: { kind: 'huggingface', repo_id: 'org/klein', linked_by: ['artaix'] } },
        { download_needed: true, name: 'klein4b', label: 'klein4b', family: 'flux2_klein', downloaded: false, delete: null },
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

  describe('SeedVR2 row', () => {
    const seedvr2Row = {
      name: 'seedvr2-7b@q8',
      label: 'seedvr2-7b@q8',
      family: 'seedvr2',
      size_label: '7b',
      source: 'seedvr2',
      stored_quant: null,
      memory_fit: null,
      download_size_label: '17.0 GB',
    };

    function tooltipFor(name: string): string {
      const label = Array.from(target.querySelectorAll('[data-testid="model-name"]')).find((el) => el.textContent === name);
      return label?.parentElement?.querySelector('[role="tooltip"]')?.textContent ?? '';
    }

    it('shows the disk use only while the stored copy exists, with a delete for it', async () => {
      modelApiMocks.getModelInventory.mockResolvedValue({
        ...makeInventory(),
        image_models: [
          { download_needed: false, ...seedvr2Row, downloaded: true, disk_size_label: '9.3 GB', delete: { kind: 'installed', repo_id: null, linked_by: [], stored_quants: [] } },
          { download_needed: false, name: 'zit', label: 'zit', family: 'zimage', downloaded: true, delete: null },
        ],
      });
      app = flushSync(() => mount(ModelsPage, { target }));
      await settle();

      const disk = target.querySelectorAll('[data-testid="disk-size"]');
      expect(disk).toHaveLength(1);
      expect(disk[0].textContent).toContain('9.3 GB');
      expect(Array.from(target.querySelectorAll('[data-testid="delete-button"]')).map((el) => el.getAttribute('aria-label'))).toEqual(['Delete seedvr2-7b@q8']);
      expect(target.querySelector('[data-testid="model-memory-fit"]')).toBeNull();
    });

    it('names the download size in the not-downloaded tooltip and leaves image rows alone', async () => {
      modelApiMocks.getModelInventory.mockResolvedValue({
        ...makeInventory(),
        image_models: [
          { download_needed: true, ...seedvr2Row, downloaded: false, disk_size_label: null, delete: null },
          { download_needed: true, name: 'klein4b', label: 'klein4b', family: 'flux2_klein', downloaded: false, delete: null },
        ],
      });
      app = flushSync(() => mount(ModelsPage, { target }));
      await settle();

      expect(tooltipFor('seedvr2-7b@q8')).toContain(seedvr2Row.download_size_label);
      expect(tooltipFor('seedvr2-7b@q8')).not.toContain(NOT_DOWNLOADED_FIRST_USE_TOOLTIP);
      expect(tooltipFor('klein4b')).toContain(DOWNLOADABLE_TOOLTIP);
      expect(target.querySelector('[data-testid="disk-size"]')).toBeNull();
    });

    it('confirms with the re-download size instead of the installed-model line and deletes by name', async () => {
      modelApiMocks.getModelInventory.mockResolvedValue({
        ...makeInventory(),
        image_models: [
          { download_needed: false, ...seedvr2Row, downloaded: true, disk_size_label: '9.3 GB', delete: { kind: 'installed', repo_id: null, linked_by: [], stored_quants: [] } },
          { download_needed: false, name: 'artaix', label: 'artaix', family: 'flux2_klein', downloaded: true, delete: { kind: 'installed', repo_id: null, linked_by: [] } },
        ],
      });
      modelApiMocks.deleteModel.mockResolvedValue({ tone: 'success', message: "Deleted model 'seedvr2-7b@q8'.", detail: '' });
      app = flushSync(() => mount(ModelsPage, { target }));
      await settle();
      const buttons = target.querySelectorAll('[data-testid="delete-button"]');

      (buttons[1] as HTMLButtonElement).click();
      await settle();
      expect(document.querySelector('[data-testid="delete-redownload"]')).toBeNull();
      const installedNotes = Array.from(document.querySelectorAll('[data-testid="delete-dialog"] p')).map((el) => el.textContent);
      expect(installedNotes.length).toBeGreaterThan(0);
      (Array.from(document.querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Cancel') as HTMLButtonElement).click();
      await settle();

      (buttons[0] as HTMLButtonElement).click();
      await settle();
      expect(document.querySelector('[data-testid="delete-redownload"]')?.textContent).toContain(seedvr2Row.download_size_label);
      const seedvr2Notes = Array.from(document.querySelectorAll('[data-testid="delete-dialog"] p')).map((el) => el.textContent);
      expect(seedvr2Notes.filter((note) => installedNotes.includes(note))).toEqual([]);
      (Array.from(document.querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Delete') as HTMLButtonElement).click();
      await settle();

      expect(modelApiMocks.deleteModel).toHaveBeenCalledWith('seedvr2-7b@q8');
    });
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

  it('lists enhancer tiers and deletes only a downloaded one, by tier', async () => {
    const tier = { revision: null, download_size_label: null, unavailable: null };
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      enhancer_models: [
        { download_needed: false, ...tier, tier: 'fast', repo: 'org/fast', downloaded: true, disk_size_label: '9.1 GB', deletable: true },
        { download_needed: false, ...tier, tier: 'smart', repo: 'org/smart', downloaded: false, disk_size_label: null, deletable: false, unavailable: 'Needs about 9 GB.' },
      ],
    });
    modelApiMocks.deleteEnhancerModel.mockResolvedValue({ status: 'ok', tone: 'success', message: "Deleted the HuggingFace download of 'org/fast'." });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const rows = Array.from(target.querySelectorAll<HTMLElement>('[data-testid="enhancer-row"]'));
    expect(rows.map((row) => row.dataset.tier)).toEqual(['fast', 'smart']);
    expect(rows[1].querySelector('[data-testid="delete-button"]')).toBeNull();
    expect(rows[1].querySelector('[data-testid="enhancer-unavailable"]')).not.toBeNull();

    (rows[0].querySelector('[data-testid="delete-button"]') as HTMLButtonElement).click();
    await settle();
    expect(document.querySelector('[data-testid="delete-enhancer"]')).not.toBeNull();
    (Array.from(document.querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Delete') as HTMLButtonElement).click();
    await settle();

    expect(modelApiMocks.deleteEnhancerModel).toHaveBeenCalledWith('fast');
    expect(modelApiMocks.getModelInventory).toHaveBeenCalledTimes(2);
  });

  it('lists other downloads and unused cache files, deleting each through its own route', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      other_downloads: { cache_dir: '/hub', repos: [{ repo_id: 'org/old', size_label: '16.1 GB' }], stranded: { count: 12, size_label: '60.1 GB' } },
    });
    const ok = { status: 'ok', tone: 'success', message: 'Done.' };
    modelApiMocks.deleteOtherDownload.mockResolvedValue(ok);
    modelApiMocks.freeStrandedCacheFiles.mockResolvedValue(ok);
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    const confirm = async (row: string) => {
      (target.querySelector(`[data-testid="${row}"] [data-testid="delete-button"]`) as HTMLButtonElement).click();
      await settle();
      (Array.from(document.querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Delete') as HTMLButtonElement).click();
      await settle();
    };
    await confirm('other-download-row');
    expect(modelApiMocks.deleteOtherDownload).toHaveBeenCalledWith('org/old');
    await confirm('stranded-row');
    expect(modelApiMocks.freeStrandedCacheFiles).toHaveBeenCalledOnce();
  });

  it('hides other downloads when the cache holds nothing unused', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({ ...makeInventory(), other_downloads: { cache_dir: '/hub', repos: [], stranded: null } });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    expect(target.querySelector('[data-testid="other-download-row"], [data-testid="stranded-row"]')).toBeNull();
  });

  it('hides the enhancer table when no tier is configured', async () => {
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();

    expect(target.querySelector('[data-testid="enhancer-row"]')).toBeNull();
  });
});

describe('ModelsPage downloads', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  const plan = (overrides: Partial<DownloadPlan> = {}): DownloadPlan => ({
    kind: 'model',
    name: 'klein9b',
    label: 'klein9b',
    parts: [{ repo: 'org/klein', files: 4, bytes: 34.7e9 }],
    total_bytes: 34.7e9,
    space: { path: '/hub', needed_bytes: 34.7e9, free_bytes: 512e9, enough: true },
    ...overrides,
  });
  const item = (overrides: Partial<DownloadItem> = {}): DownloadItem => ({ id: 'd1', kind: 'model', name: 'klein9b', label: 'klein9b', state: 'queued', message: null, ...overrides });
  const running = (overrides: Partial<DownloadItem> = {}): DownloadItem =>
    item({ state: 'running', repo: 'org/klein', part: 1, parts: 1, bytes_done: 12.4e9, bytes_total: 34.7e9, bytes_per_sec: 48e6, download_eta_secs: 460, shared: false, ...overrides });

  function inventoryWith(overrides: Partial<ModelInventory> = {}): ModelInventory {
    return {
      ...makeInventory(),
      image_models: [
        { name: 'klein9b', label: 'klein9b', family: 'flux2_klein', downloaded: false, download_needed: true, delete: null },
        { name: 'klein4b', label: 'klein4b', family: 'flux2_klein', downloaded: true, download_needed: false, delete: { kind: 'installed', repo_id: null, linked_by: [] } },
      ],
      ...overrides,
    };
  }

  /** Let pending promises settle and run the timers due within *ms* (a poll fires every second). */
  async function tickFor(ms = 0): Promise<void> {
    await vi.advanceTimersByTimeAsync(ms);
    flushSync();
  }

  async function mountPage(inventory: ModelInventory = inventoryWith()): Promise<void> {
    modelApiMocks.getModelInventory.mockResolvedValue(inventory);
    app = flushSync(() => mount(ModelsPage, { target }));
    await tickFor();
  }

  function downloadButton(name = 'klein9b'): HTMLButtonElement | null {
    return target.querySelector<HTMLButtonElement>(`[data-testid="download-button"][aria-label="Download ${name}"]`);
  }

  const confirmDialog = (): HTMLElement | null => document.querySelector('[role="alertdialog"]');
  const confirmButton = (): HTMLButtonElement => document.querySelector('[data-action="confirm"]') as HTMLButtonElement;

  async function startAndConfirm(): Promise<void> {
    downloadButton()!.click();
    await tickFor();
    confirmButton().click();
    await tickFor();
  }

  beforeEach(() => {
    vi.useFakeTimers();
    downloadStore.reset();
    for (const mock of Object.values(modelApiMocks)) mock.mockReset();
    modelApiMocks.getConversionStatus.mockResolvedValue({ state: 'idle' });
    modelApiMocks.getDownloads.mockResolvedValue({ items: [] });
    vi.mocked(addToast).mockClear();
    vi.mocked(dismissToast).mockClear();
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.remove();
    document.body.innerHTML = '';
    vi.useRealTimers();
  });

  it('offers Download only on rows that need files, and Delete on the rest', async () => {
    await mountPage();

    expect(downloadButton('klein9b')).not.toBeNull();
    expect(target.querySelectorAll('[data-testid="download-button"]')).toHaveLength(1);
    expect(Array.from(target.querySelectorAll('[data-testid="delete-button"]')).map((el) => el.getAttribute('aria-label'))).toEqual(['Delete klein4b']);
  });

  it('reads the plan, confirms with the plain label, then starts the download for the row', async () => {
    modelApiMocks.getDownloadPlan.mockResolvedValue(plan());
    modelApiMocks.startDownload.mockResolvedValue(item());
    await mountPage();

    downloadButton()!.click();
    await tickFor();
    expect(modelApiMocks.getDownloadPlan).toHaveBeenCalledWith('model', 'klein9b');
    expect(confirmDialog()).not.toBeNull();
    expect(confirmButton().textContent?.trim()).toBe('Download');
    expect(confirmButton().classList.contains('ui-btn-danger')).toBe(false);
    expect(modelApiMocks.startDownload).not.toHaveBeenCalled();

    confirmButton().click();
    await tickFor();
    expect(modelApiMocks.startDownload).toHaveBeenCalledWith('model', 'klein9b');
  });

  it('asks to Download anyway only when the free space is short', async () => {
    modelApiMocks.getDownloadPlan.mockResolvedValue(plan({ space: { path: '/hub', needed_bytes: 34.7e9, free_bytes: 21e9, enough: false } }));
    await mountPage();

    downloadButton()!.click();
    await tickFor();
    expect(confirmButton().textContent?.trim()).toBe('Download anyway');
  });

  it('starts nothing when the confirmation is declined', async () => {
    modelApiMocks.getDownloadPlan.mockResolvedValue(plan());
    await mountPage();

    downloadButton()!.click();
    await tickFor();
    (document.querySelector('[data-action="cancel"]') as HTMLButtonElement).click();
    await tickFor();

    expect(modelApiMocks.startDownload).not.toHaveBeenCalled();
  });

  it('skips the confirmation for an empty plan, tells so and reloads the inventory', async () => {
    modelApiMocks.getDownloadPlan.mockResolvedValue(plan({ parts: [], total_bytes: 0, space: null }));
    await mountPage();
    const reads = modelApiMocks.getModelInventory.mock.calls.length;

    downloadButton()!.click();
    await tickFor();

    expect(confirmDialog()).toBeNull();
    expect(modelApiMocks.startDownload).not.toHaveBeenCalled();
    expect(addToast).toHaveBeenCalledWith(expect.any(String), 'info');
    expect(modelApiMocks.getModelInventory.mock.calls.length).toBe(reads + 1);
  });

  it('downloads an enhancer tier by kind and tier name', async () => {
    const tier = { revision: null, download_size_label: null, unavailable: null, disk_size_label: null, deletable: false };
    modelApiMocks.getDownloadPlan.mockResolvedValue(plan({ kind: 'enhancer', name: 'smart', label: 'Smart enhancer model' }));
    modelApiMocks.startDownload.mockResolvedValue(item({ kind: 'enhancer', name: 'smart' }));
    await mountPage({ ...inventoryWith(), enhancer_models: [{ ...tier, tier: 'smart', repo: 'org/smart', downloaded: false, download_needed: true }] });

    target.querySelector<HTMLButtonElement>('[data-testid="enhancer-row"] [data-testid="download-button"]')!.click();
    await tickFor();
    confirmButton().click();
    await tickFor();

    expect(modelApiMocks.getDownloadPlan).toHaveBeenCalledWith('enhancer', 'smart');
    expect(modelApiMocks.startDownload).toHaveBeenCalledWith('enhancer', 'smart');
  });

  it('shows a queued item with Cancel in place of Download, and no progress bar', async () => {
    modelApiMocks.getDownloads.mockResolvedValue({ items: [item({ queue_position: 2 })] });
    await mountPage();

    expect(downloadButton()).toBeNull();
    expect(target.querySelector('[data-testid="download-cancel"]')).not.toBeNull();
    expect(target.querySelector('[data-testid="download-subrow"]')).not.toBeNull();
    expect(target.querySelector('[data-testid="download-progress"]')).toBeNull();
  });

  it('shows a running item with a progress bar and cancels it by id without a toast', async () => {
    modelApiMocks.getDownloads.mockResolvedValue({ items: [running()] });
    modelApiMocks.cancelDownload.mockResolvedValue(item({ state: 'cancelled' }));
    await mountPage();

    const bar = target.querySelector('[data-testid="download-progress"] [role="progressbar"]');
    expect(bar?.getAttribute('aria-valuenow')).toBe(String(12.4e9));
    expect(bar?.getAttribute('aria-valuemax')).toBe(String(34.7e9));

    target.querySelector<HTMLButtonElement>('[data-testid="download-cancel"]')!.click();
    await tickFor();

    expect(modelApiMocks.cancelDownload).toHaveBeenCalledWith('d1');
    expect(addToast).not.toHaveBeenCalled();
    expect(downloadButton()).not.toBeNull();
  });

  it('shows a running item of unknown size as an indeterminate bar', async () => {
    modelApiMocks.getDownloads.mockResolvedValue({ items: [running({ bytes_total: null, download_eta_secs: null })] });
    await mountPage();

    const bar = target.querySelector('[data-testid="download-progress"] [role="progressbar"]');
    expect(bar?.classList.contains('is-live')).toBe(true);
    expect(bar?.getAttribute('aria-valuenow')).toBe('0');
  });

  it('polls every second while a download is active and stops once it is done, reloading the inventory', async () => {
    modelApiMocks.getDownloads.mockResolvedValue({ items: [running()] });
    await mountPage();
    const reads = modelApiMocks.getModelInventory.mock.calls.length;
    const polled = modelApiMocks.getDownloads.mock.calls.length;

    await tickFor(1000);
    expect(modelApiMocks.getDownloads.mock.calls.length).toBe(polled + 1);

    modelApiMocks.getDownloads.mockResolvedValue({ items: [item({ state: 'done' })] });
    await tickFor(1000);
    expect(addToast).toHaveBeenCalledWith(expect.any(String), 'success');
    expect(modelApiMocks.getModelInventory.mock.calls.length).toBe(reads + 1);

    const settled = modelApiMocks.getDownloads.mock.calls.length;
    await tickFor(3000);
    expect(modelApiMocks.getDownloads.mock.calls.length).toBe(settled);
  });

  it('does not poll while nothing is active, nor announce what finished before the page opened', async () => {
    modelApiMocks.getDownloads.mockResolvedValue({ items: [item({ state: 'done' }), item({ id: 'd2', state: 'failed', message: 'Disk full' })] });
    await mountPage();
    const polled = modelApiMocks.getDownloads.mock.calls.length;

    await tickFor(3000);

    expect(modelApiMocks.getDownloads.mock.calls.length).toBe(polled);
    expect(addToast).not.toHaveBeenCalled();
  });

  it('starts polling after a download is queued and stops polling when the page is destroyed', async () => {
    modelApiMocks.getDownloadPlan.mockResolvedValue(plan());
    modelApiMocks.startDownload.mockResolvedValue(running());
    await mountPage();
    modelApiMocks.getDownloads.mockResolvedValue({ items: [running()] });

    await startAndConfirm();
    const polled = modelApiMocks.getDownloads.mock.calls.length;
    await tickFor(1000);
    expect(modelApiMocks.getDownloads.mock.calls.length).toBeGreaterThan(polled);

    await unmount(app!);
    app = null;
    const afterDestroy = modelApiMocks.getDownloads.mock.calls.length;
    await tickFor(3000);
    expect(modelApiMocks.getDownloads.mock.calls.length).toBe(afterDestroy);
  });

  it('raises an error toast with a Retry action when a download fails, and the toast survives the page', async () => {
    modelApiMocks.getDownloads.mockResolvedValue({ items: [running()] });
    await mountPage();

    modelApiMocks.getDownloads.mockResolvedValue({ items: [item({ state: 'failed', message: 'No space left on device' })] });
    await tickFor(1000);

    expect(addToast).toHaveBeenCalledWith('No space left on device', 'error', { action: { label: 'Retry', run: expect.any(Function) } });
    await unmount(app!);
    app = null;
    expect(dismissToast).not.toHaveBeenCalled();

    const request = vi.spyOn(downloadStore, 'request').mockResolvedValue('declined');
    vi.mocked(addToast).mock.calls.at(-1)?.[2]?.action?.run();
    expect(request).toHaveBeenCalledWith('model', 'klein9b', 'klein9b');
    request.mockRestore();
  });

  it('runs the Download flow again from the Retry action', async () => {
    modelApiMocks.getDownloads.mockResolvedValue({ items: [running()] });
    modelApiMocks.getDownloadPlan.mockResolvedValue(plan());
    await mountPage();
    modelApiMocks.getDownloads.mockResolvedValue({ items: [item({ state: 'failed', message: 'Network error' })] });
    await tickFor(1000);
    const options = vi.mocked(addToast).mock.calls.at(-1)?.[2];

    options?.action?.run();
    await tickFor();

    expect(modelApiMocks.getDownloadPlan).toHaveBeenCalledWith('model', 'klein9b');
  });

  it('shows a model by its name with the alias under it, and uses the name in the row controls', async () => {
    await mountPage(inventoryWith({
      image_models: [
        { name: 'klein9b', label: 'FLUX.2 Klein 9B', family: 'flux2_klein', downloaded: false, download_needed: true, delete: null },
        { name: 'plain', label: 'plain', family: 'flux2_klein', downloaded: true, download_needed: false, delete: { kind: 'installed', repo_id: null, linked_by: [] } },
      ],
    }));

    const [named, plain] = Array.from(target.querySelectorAll('tbody tr')).filter((row) => row.querySelector('[data-testid="model-name"]'));
    expect(named.querySelector('[data-testid="model-name"]')?.textContent).toBe('FLUX.2 Klein 9B');
    expect(named.querySelector('[data-testid="model-id"]')?.textContent).toBe('klein9b');
    expect(named.querySelector('[data-testid="download-button"]')?.getAttribute('aria-label')).toBe('Download FLUX.2 Klein 9B');
    expect(plain.querySelector('[data-testid="model-name"]')?.textContent).toBe('plain');
    expect(plain.querySelector('[data-testid="model-id"]')).toBeNull();
    expect(plain.querySelector('[data-testid="delete-button"]')?.getAttribute('aria-label')).toBe('Delete plain');
  });

  it('keeps a partly downloaded row downloaded and offers Download and Delete together', async () => {
    await mountPage(inventoryWith({
      image_models: [{ name: 'partial', label: 'partial', family: 'flux2_klein', downloaded: true, download_needed: true, delete: { kind: 'installed', repo_id: null, linked_by: [] } }],
    }));

    expect(target.querySelector('[data-testid="model-name"]')?.getAttribute('data-downloaded')).toBe('true');
    const download = target.querySelector('[data-testid="download-button"]');
    expect(download).not.toBeNull();
    expect(download?.getAttribute('aria-label')).not.toBe('Download partial');
    expect(target.querySelector('[data-testid="delete-button"]')).not.toBeNull();
    expect(download!.compareDocumentPosition(target.querySelector('[data-testid="delete-button"]')!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('offers Cancel only, without Delete, while a row is downloading', async () => {
    modelApiMocks.getDownloads.mockResolvedValue({ items: [running({ name: 'partial' })] });
    await mountPage(inventoryWith({
      image_models: [{ name: 'partial', label: 'partial', family: 'flux2_klein', downloaded: true, download_needed: true, delete: { kind: 'installed', repo_id: null, linked_by: [] } }],
    }));

    expect(target.querySelector('[data-testid="download-cancel"]')).not.toBeNull();
    expect(target.querySelector('[data-testid="download-button"]')).toBeNull();
    expect(target.querySelector('[data-testid="delete-button"]')).toBeNull();
  });

  it('offers Download and Delete for a deletable enhancer that needs files', async () => {
    const tier = { revision: null, download_size_label: null, unavailable: null, disk_size_label: '4.0 GB' };
    await mountPage({ ...inventoryWith(), enhancer_models: [{ ...tier, tier: 'fast', repo: 'org/fast', downloaded: true, download_needed: true, deletable: true }] });

    const row = target.querySelector('[data-testid="enhancer-row"]')!;
    expect(row.querySelector('[data-testid="download-button"]')).not.toBeNull();
    expect(row.querySelector('[data-testid="delete-button"]')).not.toBeNull();
  });

  describe('prepared on first use', () => {
    function nameTooltip(name: string): string {
      const label = Array.from(target.querySelectorAll('[data-testid="model-name"]')).find((el) => el.textContent === name);
      return label?.parentElement?.querySelector('[role="tooltip"]')?.textContent ?? '';
    }

    it('says the SeedVR2 copy and the Ideogram 4 Instant build are made on first use when nothing is left to fetch', async () => {
      await mountPage(inventoryWith({
        image_models: [
          { name: 'seedvr2-7b@q8', label: 'seedvr2-7b@q8', family: 'seedvr2', source: 'seedvr2', downloaded: false, download_needed: false, download_size_label: '17.0 GB', delete: null },
          { name: 'ideo', label: 'ideo', family: 'ideogram4', downloaded: false, download_needed: false, delete: null },
          { name: 'klein9b', label: 'klein9b', family: 'flux2_klein', downloaded: false, download_needed: true, delete: null },
        ],
      }));

      expect(nameTooltip('seedvr2-7b@q8')).toContain(PREPARED_SEEDVR2_TOOLTIP);
      expect(nameTooltip('ideo')).toContain(PREPARED_INSTANT_TOOLTIP);
      expect(nameTooltip('klein9b')).toContain(DOWNLOADABLE_TOOLTIP);
      expect(target.querySelectorAll('[data-testid="download-button"]')).toHaveLength(1);
    });
  });

  it('keeps the Retry toast of a failed request after the page is destroyed, and Retry asks the store again', async () => {
    modelApiMocks.getDownloadPlan.mockRejectedValueOnce(new ApiError('Bad gateway', 502, 'Hub unreachable'));
    await mountPage();

    downloadButton()!.click();
    await tickFor();
    await unmount(app!);
    app = null;
    expect(dismissToast).not.toHaveBeenCalled();

    const request = vi.spyOn(downloadStore, 'request').mockResolvedValue('declined');
    vi.mocked(addToast).mock.calls.at(-1)?.[2]?.action?.run();
    expect(request).toHaveBeenCalledWith('model', 'klein9b', 'klein9b');
    request.mockRestore();
  });

  it('keeps the focus inside a row when a poll replaces that row\'s control', async () => {
    modelApiMocks.getDownloads.mockResolvedValue({ items: [item({ queue_position: 1 })] });
    await mountPage();
    const cancel = target.querySelector<HTMLButtonElement>('[data-dl-key="model:klein9b"] [data-testid="download-cancel"]')!;
    cancel.focus();
    expect(document.activeElement).toBe(cancel);

    modelApiMocks.getDownloads.mockResolvedValue({ items: [item({ state: 'cancelled' })] });
    await tickFor(1000);

    expect(target.querySelector('[data-testid="download-cancel"]')).toBeNull();
    expect(document.activeElement?.closest('[data-dl-key="model:klein9b"]')).not.toBeNull();
    expect(document.activeElement?.getAttribute('data-testid')).toBe('download-button');
  });

  it('reads the lane once per second after a submit, however often the row was submitted', async () => {
    modelApiMocks.getDownloadPlan.mockResolvedValue(plan());
    modelApiMocks.startDownload.mockResolvedValue(running());
    await mountPage();
    modelApiMocks.getDownloads.mockResolvedValue({ items: [running()] });

    await startAndConfirm();
    await tickFor(1000);
    const polled = modelApiMocks.getDownloads.mock.calls.length;
    await tickFor(3000);

    expect(modelApiMocks.getDownloads.mock.calls.length).toBe(polled + 3);
  });

  it('offers Retry on a failed request only when repeating it could work', async () => {
    modelApiMocks.getDownloadPlan.mockRejectedValueOnce(new ApiError('Not found', 404, 'Unknown model')).mockRejectedValueOnce(new ApiError('Bad gateway', 502, 'Hub unreachable'));
    await mountPage();

    downloadButton()!.click();
    await tickFor();
    expect(addToast).toHaveBeenLastCalledWith('Unknown model', 'error');

    downloadButton()!.click();
    await tickFor();
    expect(addToast).toHaveBeenLastCalledWith('Hub unreachable', 'error', { action: { label: 'Retry', run: expect.any(Function) } });
  });
});

describe('ModelsPage memory', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    for (const mock of Object.values(modelApiMocks)) mock.mockReset();
    modelApiMocks.getDownloads.mockResolvedValue({ items: [] });
    modelApiMocks.getConversionStatus.mockResolvedValue({ state: 'idle' });
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
  });

  it('comes back with the inventory a previous visit loaded, before the refresh answers', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      image_models: [{ download_needed: false, name: 'atlas', label: 'atlas', family: 'flux2_klein', size_label: '9b', stored_quant: null }],
    });
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();
    expect(target.querySelector('[data-testid="model-name"]')?.textContent).toBe('atlas');
    await unmount(app!);
    app = null;

    modelApiMocks.getModelInventory.mockImplementation(() => new Promise<ModelInventory>(() => undefined));
    app = flushSync(() => mount(ModelsPage, { target }));
    await settle();
    expect(target.querySelector('[data-testid="model-name"]')?.textContent).toBe('atlas');
    expect(modelApiMocks.getModelInventory).toHaveBeenCalledTimes(2);
  });
});
