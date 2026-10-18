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
  deleteEnhancerModel: vi.fn(),
  deleteOtherDownload: vi.fn(),
  freeStrandedCacheFiles: vi.fn(),
  deleteLora: vi.fn(),
}));

const browseApiMocks = vi.hoisted(() => ({
  listDirectory: vi.fn(),
  browsePlaces: vi.fn(),
  createFolder: vi.fn(),
}));

vi.mock('$lib/api/models', () => ({
  getModelInventory: modelApiMocks.getModelInventory,
  convertCheckpoint: modelApiMocks.convertCheckpoint,
  importLoraLocal: modelApiMocks.importLoraLocal,
  importLoraHF: modelApiMocks.importLoraHF,
  deleteModel: modelApiMocks.deleteModel,
  deleteEnhancerModel: modelApiMocks.deleteEnhancerModel,
  deleteOtherDownload: modelApiMocks.deleteOtherDownload,
  freeStrandedCacheFiles: modelApiMocks.freeStrandedCacheFiles,
  deleteLora: modelApiMocks.deleteLora,
}));

vi.mock('$lib/api/browse', () => browseApiMocks);

vi.mock('$lib/state/toasts.svelte', () => ({
  addToast: vi.fn(),
}));

import { NOT_DOWNLOADED_TOOLTIP } from '$lib/components/molecules/ModelStatusBadges.svelte';
import ModelsPage from './ModelsPage.svelte';
import { chooseInBrowser, listingWith, placesFor } from '../../test-utils/pathBrowser';

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
    modelApiMocks.getModelInventory.mockReset();
    modelApiMocks.convertCheckpoint.mockReset();
    modelApiMocks.importLoraLocal.mockReset();
    modelApiMocks.importLoraHF.mockReset();
    modelApiMocks.deleteModel.mockReset();
    modelApiMocks.deleteEnhancerModel.mockReset();
    modelApiMocks.deleteOtherDownload.mockReset();
    modelApiMocks.freeStrandedCacheFiles.mockReset();
    modelApiMocks.deleteLora.mockReset();
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
    expect.soft(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('checkpoint_file', '', expect.anything());
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

  it('shows the disk use of image and video models, and a dash when nothing is on disk', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      image_models: [
        { name: 'zit', family: 'zimage', size_label: 'xl', disk_size_label: '32.9 GB' },
        { name: 'klein4b', family: 'flux2_klein', size_label: '4b', disk_size_label: null },
      ],
      video_models: [{ name: 'ltx-8', family: 'ltx', supports_i2v: true, disk_size_label: '112.3 GB' }],
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
      image_models: [{ name: longName, family: 'a-very-long-family', size_label: 'extra-long-size' }],
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

  it('deletes only the prepared copy of a model whose download another model still uses', async () => {
    modelApiMocks.getModelInventory.mockResolvedValue({
      ...makeInventory(),
      video_models: [
        { name: 'ltx-4', family: 'ltx', supports_i2v: true, downloaded: true, delete: { kind: 'copy', repo_id: 'org/ltx', linked_by: [], stored_quants: ['ltx-4@q4'], shared_with: ['ltx-8'] } },
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

  describe('SeedVR2 row', () => {
    const seedvr2Row = {
      name: 'seedvr2-7b@q8',
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
          { ...seedvr2Row, downloaded: true, disk_size_label: '9.3 GB', delete: { kind: 'installed', repo_id: null, linked_by: [], stored_quants: [] } },
          { name: 'zit', family: 'zimage', downloaded: true, delete: null },
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
          { ...seedvr2Row, downloaded: false, disk_size_label: null, delete: null },
          { name: 'klein4b', family: 'flux2_klein', downloaded: false, delete: null },
        ],
      });
      app = flushSync(() => mount(ModelsPage, { target }));
      await settle();

      expect(tooltipFor('seedvr2-7b@q8')).toContain(seedvr2Row.download_size_label);
      expect(tooltipFor('seedvr2-7b@q8')).not.toContain(NOT_DOWNLOADED_TOOLTIP);
      expect(tooltipFor('klein4b')).toContain(NOT_DOWNLOADED_TOOLTIP);
      expect(target.querySelector('[data-testid="disk-size"]')).toBeNull();
    });

    it('confirms with the re-download size instead of the installed-model line and deletes by name', async () => {
      modelApiMocks.getModelInventory.mockResolvedValue({
        ...makeInventory(),
        image_models: [
          { ...seedvr2Row, downloaded: true, disk_size_label: '9.3 GB', delete: { kind: 'installed', repo_id: null, linked_by: [], stored_quants: [] } },
          { name: 'artaix', family: 'flux2_klein', downloaded: true, delete: { kind: 'installed', repo_id: null, linked_by: [] } },
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
        { ...tier, tier: 'fast', repo: 'org/fast', downloaded: true, disk_size_label: '9.1 GB', deletable: true },
        { ...tier, tier: 'smart', repo: 'org/smart', downloaded: false, disk_size_label: null, deletable: false, unavailable: 'Needs about 9 GB.' },
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
