// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { AppConfig, WritableConfigField } from '$lib/types';

const configApiMocks = vi.hoisted(() => ({
  getConfig: vi.fn<() => Promise<AppConfig>>(),
  updateConfig: vi.fn<(patch: Record<string, string | number | boolean | null>) => Promise<AppConfig>>(),
}));

const toastMocks = vi.hoisted(() => ({
  addToast: vi.fn<(message: string, tone: string) => void>(),
}));

vi.mock('$lib/api/config', () => ({
  getConfig: configApiMocks.getConfig,
  updateConfig: configApiMocks.updateConfig,
}));

vi.mock('$lib/state/toasts.svelte', () => ({
  addToast: toastMocks.addToast,
}));

import ConfigPage from './ConfigPage.svelte';

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
      huggingface_token_env_var: 'HF_TOKEN',
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

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await Promise.resolve();
  await Promise.resolve();
  flushSync();
}

describe('ConfigPage', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
    configApiMocks.getConfig.mockReset();
    configApiMocks.updateConfig.mockReset();
    toastMocks.addToast.mockReset();
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
    document.body.innerHTML = '';
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

    app = flushSync(() => mount(ConfigPage, { target }));
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

  it('does not expose internal config schema terms as visible text', async () => {
    const config = makeConfig();
    configApiMocks.getConfig.mockResolvedValue(config);

    app = flushSync(() => mount(ConfigPage, { target }));
    await settle();

    const pageText = target.textContent ?? '';

    expect(pageText).not.toContain('ui.output_dir');
    expect(pageText).not.toContain('ui.default_models.image');
    expect(pageText).not.toContain('default_source');
    expect(pageText).not.toContain('writable_config');
    expect(pageText).not.toContain('backend default');
  });

  it('submits backend snake_case schema keys and clears output_dir with null semantics', async () => {
    const config = makeConfig();
    configApiMocks.getConfig.mockResolvedValue(config);
    configApiMocks.updateConfig.mockResolvedValue(config);

    app = flushSync(() => mount(ConfigPage, { target }));
    await settle();

    const imageModelSelect = target.querySelector('select[name="ui.default_models.image"]') as HTMLSelectElement | null;
    expect(imageModelSelect).not.toBeNull();
    imageModelSelect!.value = 'flux-dev';
    imageModelSelect!.dispatchEvent(new Event('change', { bubbles: true }));

    const outputDirField = target.querySelector('#config-ui-output-dir') as HTMLInputElement | null;
    expect(outputDirField).not.toBeNull();
    outputDirField!.value = '';
    outputDirField!.dispatchEvent(new Event('input', { bubbles: true }));
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
    });
    expect(toastMocks.addToast).toHaveBeenCalledTimes(1);
    expect(toastMocks.addToast.mock.calls[0][1]).toBe('success');
  });

  it('submits a manually typed output directory without requiring resolve or browse', async () => {
    const config = makeConfig();
    configApiMocks.getConfig.mockResolvedValue(config);
    configApiMocks.updateConfig.mockResolvedValue(config);

    app = flushSync(() => mount(ConfigPage, { target }));
    await settle();

    const outputDirInput = target.querySelector('#config-ui-output-dir') as HTMLInputElement | null;
    expect(outputDirInput).not.toBeNull();
    outputDirInput!.value = '/tmp/manual-output';
    outputDirInput!.dispatchEvent(new Event('input', { bubbles: true }));

    const form = target.querySelector('form') as HTMLFormElement | null;
    expect(form).not.toBeNull();
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(configApiMocks.updateConfig).toHaveBeenCalledTimes(1);
    expect(configApiMocks.updateConfig.mock.calls[0][0]).toMatchObject({
      'ui.output_dir': '/tmp/manual-output',
    });
  });

  it('submits edits for backend-declared writable fields without a local writable allowlist', async () => {
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
    configApiMocks.updateConfig.mockResolvedValue(config);

    app = flushSync(() => mount(ConfigPage, { target }));
    await settle();

    const schemaTextInput = target.querySelector('input[name="ui.experimental_label"]') as HTMLInputElement | null;
    expect(schemaTextInput).not.toBeNull();
    schemaTextInput!.value = 'edited-from-schema';
    schemaTextInput!.dispatchEvent(new Event('input', { bubbles: true }));

    const form = target.querySelector('form') as HTMLFormElement | null;
    expect(form).not.toBeNull();
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(configApiMocks.updateConfig).toHaveBeenCalledTimes(1);
    expect(configApiMocks.updateConfig.mock.calls[0][0]).toMatchObject({
      'ui.experimental_label': 'edited-from-schema',
    });
  });
});