import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import type { AppConfig, WritableConfigField } from '$lib/types';

const configApiMocks = vi.hoisted(() => ({
  getConfig: vi.fn<() => Promise<AppConfig>>(),
  updateConfig: vi.fn<(patch: Record<string, string | number | boolean | null>) => Promise<AppConfig>>(),
}));

const toastMocks = vi.hoisted(() => ({
  addToast: vi.fn<(message: string, tone: string, options?: { action?: { label: string } }) => void>(),
}));

vi.mock('$lib/api/config', () => ({
  getConfig: configApiMocks.getConfig,
  updateConfig: configApiMocks.updateConfig,
}));

vi.mock('$lib/state/toasts.svelte', () => ({
  addToast: toastMocks.addToast,
  dismissToast: vi.fn(),
}));

import ConfigPage from './ConfigPage.svelte';
import { configMemory } from '$lib/state/pageMemory';
import { mountComponent, settle } from '../../test-utils/mount';

beforeEach(() => configMemory.forget());

function makeField(overrides: Partial<WritableConfigField> & Pick<WritableConfigField, 'key'>): WritableConfigField {
  const { key, ...rest } = overrides;
  return {
    key,
    type: 'string',
    clearable: true,
    empty_string: 'clear',
    omitted: 'unchanged',
    null: 'clear',
    value: '',
    persisted_value: '',
    persisted_value_shape: 'string or null',
    effective_value: '',
    effective_value_shape: 'string',
    default_source: 'backend default',
    validation_rules: ['Backend-owned writable field.'],
    owning_consumer: 'Config page',
    ...rest,
  };
}

function makeConfig(): AppConfig {
  return {
    output_dir: '/tmp/outputs',
    log_level: 'INFO',
    ui: {
      gallery_page_size: 12,
      output_dir: '/tmp/outputs',
      default_models: { image: 'zit', video: 'ltx-8' },
      image_model_options: ['zit', 'flux-dev'],
      video_model_options: ['ltx-8', 'wan-2.1'],
      image_size_labels: [
        { value: 'm', label: 'Medium' },
        { value: 'l', label: 'Large' },
      ],
      model_cache_dir: '/cache/models',
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
      fields: [
        makeField({
          key: 'ui.default_models.image',
          value: 'zit',
          persisted_value: 'zit',
          effective_value: 'zit',
          validation_rules: ['Must be one of image_model_options.'],
        }),
        makeField({
          key: 'ui.default_models.video',
          value: 'ltx-8',
          persisted_value: 'ltx-8',
          effective_value: 'ltx-8',
          validation_rules: ['Must be one of video_model_options.'],
        }),
        makeField({
          key: 'generation.default_size',
          value: 'm',
          persisted_value: 'm',
          effective_value: 'm',
          validation_rules: ['Must be valid for the effective ratio.'],
        }),
        makeField({
          key: 'ui.output_dir',
          value: '/tmp/outputs',
          persisted_value: '/tmp/outputs',
          effective_value: '/tmp/outputs',
          validation_rules: ['Empty string clears the override.'],
        }),
      ],
    },
    models: {},
  };
}

describe('ConfigPage', () => {
  let page: ReturnType<typeof mountComponent>;
  let target: HTMLElement;

  function mountPage(): void {
    page = mountComponent(ConfigPage);
    target = page.target;
  }

  beforeEach(() => {
    configApiMocks.getConfig.mockReset();
    configApiMocks.updateConfig.mockReset();
    toastMocks.addToast.mockReset();
  });

  it('names where a configured Hugging Face token comes from', async () => {
    const config = makeConfig();
    config.ui.huggingface_token_configured = true;
    config.ui.huggingface_token_source = 'hf auth login';
    configApiMocks.getConfig.mockResolvedValue(config);

    mountPage();
    await settle();

    const text = target.textContent ?? '';
    expect(text).toContain('Available');
    expect(text).toContain('hf auth login');
  });

  it('renders editable controls only for backend-declared writable schema fields', async () => {
    const config = makeConfig();
    config.writable_config.fields.push(
      makeField({
        key: 'ui.experimental_label',
        value: 'from-schema',
        persisted_value: 'from-schema',
        effective_value: 'from-schema',
      }),
    );
    configApiMocks.getConfig.mockResolvedValue(config);

    mountPage();
    await settle();

    const form = target.querySelector('form');
    expect(form).not.toBeNull();

    const namedControls = Array.from(form!.querySelectorAll('[name]'))
      .map((element) => element.getAttribute('name'))
      .filter((name): name is string => Boolean(name))
      .sort();

    expect(namedControls).toEqual([
      'generation.default_size',
      'ui.default_models.image',
      'ui.default_models.video',
      'ui.experimental_label',
      'ui.output_dir',
    ]);

    const schemaTextInput = form!.querySelector('input[name="ui.experimental_label"]') as HTMLInputElement | null;
    expect(schemaTextInput).not.toBeNull();
    expect(schemaTextInput!.value).toBe('from-schema');

    expect(form!.querySelector('[name="ui.model_cache_dir"]')).toBeNull();
    expect(form!.querySelector('[name="ui.loras_dir"]')).toBeNull();
  });

  it('submits backend snake_case schema keys, edits of backend-declared fields, and clears output_dir with null semantics', async () => {
    const config = makeConfig();
    config.writable_config.fields.push(makeField({ key: 'ui.experimental_label', value: 'from-schema', persisted_value: 'from-schema', effective_value: 'from-schema' }));
    configApiMocks.getConfig.mockResolvedValue(config);
    configApiMocks.updateConfig.mockResolvedValue(config);

    mountPage();
    await settle();

    const imageModelSelect = target.querySelector('select[name="ui.default_models.image"]') as HTMLSelectElement | null;
    expect(imageModelSelect).not.toBeNull();
    imageModelSelect!.value = 'flux-dev';
    imageModelSelect!.dispatchEvent(new Event('change', { bubbles: true }));

    const outputDirField = target.querySelector('#config-ui-output-dir') as HTMLInputElement | null;
    expect(outputDirField).not.toBeNull();
    outputDirField!.value = '';
    outputDirField!.dispatchEvent(new Event('input', { bubbles: true }));
    const schemaTextInput = target.querySelector('input[name="ui.experimental_label"]') as HTMLInputElement;
    schemaTextInput.value = 'edited-from-schema';
    schemaTextInput.dispatchEvent(new Event('input', { bubbles: true }));
    await settle();

    const form = target.querySelector('form') as HTMLFormElement | null;
    expect(form).not.toBeNull();
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(configApiMocks.updateConfig).toHaveBeenCalledTimes(1);
    expect(configApiMocks.updateConfig.mock.calls[0][0]).toEqual({
      'ui.default_models.image': 'flux-dev',
      'ui.default_models.video': 'ltx-8',
      'generation.default_size': 'm',
      'ui.output_dir': null,
      'ui.experimental_label': 'edited-from-schema',
    });
    expect(toastMocks.addToast).toHaveBeenCalledTimes(1);
    expect(toastMocks.addToast.mock.calls[0][1]).toBe('success');
  });

  it('shows the server reason for a refused save and offers Retry only for server faults', async () => {
    configApiMocks.getConfig.mockResolvedValue(makeConfig());
    configApiMocks.updateConfig
      .mockRejectedValueOnce(new ApiError('rejected', 422, 'Default video model must be one of the discovered video models.'))
      .mockRejectedValueOnce(new ApiError('crashed', 500, 'Could not write the config file.'));

    mountPage();
    await settle();
    const form = target.querySelector('form') as HTMLFormElement;

    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();
    expect(toastMocks.addToast.mock.calls[0]).toEqual([expect.any(String), 'error']);

    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();
    expect(toastMocks.addToast.mock.calls[1][1]).toBe('error');
    expect(toastMocks.addToast.mock.calls[1][2]?.action?.label).toBe('Retry');
  });

  it('marks settings that have their own value and resets one to the default', async () => {
    configApiMocks.getConfig.mockResolvedValue(makeConfig());
    configApiMocks.updateConfig.mockResolvedValue(makeConfig());

    mountPage();
    await settle();

    const reset = target.querySelector('button[aria-label="Reset Default image model to the default"]') as HTMLButtonElement;
    expect(reset.getAttribute('aria-hidden')).toBeNull();

    reset.click();
    await settle();
    const imageSelect = target.querySelector('select[name="ui.default_models.image"]') as HTMLSelectElement;
    expect(imageSelect.value).toBe('');
    expect(reset.getAttribute('aria-hidden')).toBe('true');

    (target.querySelector('form') as HTMLFormElement).dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();
    expect(configApiMocks.updateConfig.mock.calls[0][0]['ui.default_models.image']).toBeNull();
  });

  it('comes back with the config a previous visit loaded, before the refresh answers', async () => {
    configApiMocks.getConfig.mockResolvedValue(makeConfig());
    mountPage();
    await settle();
    expect(target.querySelector('#config-form')).not.toBeNull();
    await page.unmount();

    configApiMocks.getConfig.mockImplementation(() => new Promise<AppConfig>(() => undefined));
    mountPage();
    await settle();
    expect(target.querySelector('#config-form')).not.toBeNull();
    expect((target.querySelector('[data-testid="page-actions"]') as HTMLFieldSetElement).disabled).toBe(false);
    expect(configApiMocks.getConfig).toHaveBeenCalledTimes(2);
  });
});
