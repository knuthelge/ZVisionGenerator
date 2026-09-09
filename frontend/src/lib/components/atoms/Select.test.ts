// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import PromptFileField from '$lib/components/molecules/PromptFileField.svelte';
import type { PromptFileContract } from '$lib/types';
import Select from './Select.svelte';

const promptFileContract: PromptFileContract = {
  accepted_extensions: ['.yaml'],
  browse_kind: 'existing_file',
  selection_required: false,
  trust_boundary: {
    scope: 'server_host_only',
    manual_entry: 'submitted_value_kept_until_backend_validation',
    picker: 'server_host_native_picker',
    read_write: 'existing_yaml_files_only',
  },
  help: {
    path: 'Path', editor: 'Editor', option_required: 'Required', option_optional: 'Optional',
    empty_options: 'Empty', stale_selection: 'Stale', loaded: 'Loaded', saved: 'Saved',
    ignored_negative_video: 'Ignored', ignored_negative_unsupported: 'Unsupported',
  },
};

describe('Select native required semantics', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
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

  it('is optional by default for existing consumers that do not opt into required', () => {
    const form = document.createElement('form');
    target.appendChild(form);
    app = flushSync(() => mount(Select, {
      target: form,
      props: {
        id: 'optional-select',
        name: 'optional_select',
        options: [{ value: 'choice', label: 'Choice' }],
      },
    }));

    const select = form.querySelector('#optional-select') as HTMLSelectElement;
    expect(select.required).toBe(false);
    expect(select.validity.valid).toBe(true);
    expect(form.checkValidity()).toBe(true);
  });

  it('forwards an opted-in required prop to the native select', () => {
    const form = document.createElement('form');
    target.appendChild(form);
    app = flushSync(() => mount(Select, {
      target: form,
      props: {
        id: 'required-select',
        name: 'required_select',
        required: true,
        options: [{ value: '', label: '-- Select --', disabled: true }, { value: 'choice', label: 'Choice' }],
      },
    }));

    const select = form.querySelector('#required-select') as HTMLSelectElement;
    expect(select.required).toBe(true);
    expect(select.validity.valueMissing).toBe(true);
    expect(form.checkValidity()).toBe(false);
  });

  it('keeps the existing optional prompt-option consumer optional', () => {
    const form = document.createElement('form');
    target.appendChild(form);
    app = flushSync(() => mount(PromptFileField, {
      target: form,
      props: {
        contract: promptFileContract,
        promptSource: 'inline',
        path: null,
        selectedOptionId: null,
        workflowMode: 'image',
        negativePromptSupported: true,
        onPathChange: () => undefined,
        onOptionChange: () => undefined,
      },
    }));

    const promptOption = form.querySelector('#ws-prompt-option') as HTMLSelectElement;
    expect(promptOption.required).toBe(false);
    expect(form.checkValidity()).toBe(true);
  });
});
