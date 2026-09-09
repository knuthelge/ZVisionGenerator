// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const promptFileApiMocks = vi.hoisted(() => ({
  openPathPicker: vi.fn(),
}));

vi.mock('$lib/api/promptFiles', () => ({
  openPathPicker: promptFileApiMocks.openPathPicker,
}));

import PathField from '$lib/components/molecules/PathField.svelte';

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await Promise.resolve();
  flushSync();
}

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function inputAndHidden(form: HTMLFormElement, id: string, name: string): {
  input: HTMLInputElement;
  hidden: HTMLInputElement;
} {
  return {
    input: form.querySelector(`#${id}`) as HTMLInputElement,
    hidden: form.querySelector(`input[type="hidden"][name="${name}"]`) as HTMLInputElement,
  };
}

describe('PathField behavior', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
    promptFileApiMocks.openPathPicker.mockReset();
    promptFileApiMocks.openPathPicker.mockResolvedValue({ status: 'cancelled', path: null, message: null });
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
    document.body.innerHTML = '';
  });

  it('submits the visible manually typed value without an explicit resolve action', async () => {
    let submittedPath: FormDataEntryValue | null = null;
    const form = document.createElement('form');
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      submittedPath = new FormData(form).get('path');
    });
    target.appendChild(form);

    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'manual-path',
        name: 'path',
        label: 'Path',
        value: '',
        onresolve: vi.fn(async (candidate: string) => candidate),
      },
    }));
    await settle();

    const input = form.querySelector('#manual-path') as HTMLInputElement | null;
    expect(input).not.toBeNull();
    input!.value = '/tmp/manual-value';
    input!.dispatchEvent(new Event('input', { bubbles: true }));
    await settle();

    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));

    expect(submittedPath).toBe('/tmp/manual-value');
  });

  it('keeps a first browse-selected path aligned between the visible input and named FormData', async () => {
    const form = document.createElement('form');
    target.appendChild(form);
    promptFileApiMocks.openPathPicker.mockResolvedValue({
      status: 'selected',
      path: '/models/first-selected.safetensors',
      message: null,
    });

    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'browse-path',
        name: 'path',
        label: 'Path',
        value: '',
        pickerPurpose: 'checkpoint_file',
        onresolve: vi.fn(async (candidate: string) => candidate),
      },
    }));
    await settle();

    const input = form.querySelector('#browse-path') as HTMLInputElement;
    const hidden = form.querySelector('input[type="hidden"][name="path"]') as HTMLInputElement;
    const browse = Array.from(form.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Browse') as HTMLButtonElement;
    browse.click();
    await settle();

    expect(input.value).toBe('/models/first-selected.safetensors');
    expect(hidden.value).toBe('/models/first-selected.safetensors');
    expect(new FormData(form).get('path')).toBe('/models/first-selected.safetensors');
    expect(promptFileApiMocks.openPathPicker).toHaveBeenCalledWith(expect.objectContaining({
      purpose: 'checkpoint_file',
      initial_path: null,
    }));
  });

  it('keeps explicit resolve as the normalization path', async () => {
    const onValueChange = vi.fn();
    const onResolve = vi.fn(async () => '/server/normalized.yaml');
    app = flushSync(() => mount(PathField, {
      target,
      props: {
        id: 'normalized-path',
        name: 'path',
        label: 'Path',
        value: '',
        onresolve: onResolve,
        onvaluechange: onValueChange,
      },
    }));
    await settle();

    const input = target.querySelector('#normalized-path') as HTMLInputElement | null;
    input!.value = '~/prompts.yaml';
    input!.dispatchEvent(new Event('input', { bubbles: true }));
    const enter = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true });
    input!.dispatchEvent(enter);
    await settle();

    expect(enter.defaultPrevented).toBe(true);
    expect(onResolve).toHaveBeenCalledWith('~/prompts.yaml');
    expect(input!.value).toBe('/server/normalized.yaml');
    expect(onValueChange).toHaveBeenLastCalledWith('/server/normalized.yaml');
  });

  it.each([
    ['cancelled', null, null, undefined],
    ['unsupported', null, 'The host does not support browsing.', 'The host does not support browsing.'],
    ['error', null, 'Only .safetensors files are accepted.', 'Only .safetensors files are accepted.'],
  ] as const)('preserves a manual value for the %s picker result', async (status, path, message, visibleMessage) => {
    const form = document.createElement('form');
    target.appendChild(form);
    promptFileApiMocks.openPathPicker.mockResolvedValue({ status, path, message });
    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'picker-outcome-path', name: 'path', label: 'Path', value: '',
        onresolve: vi.fn(async (candidate: string) => candidate),
      },
    }));
    await settle();

    const { input, hidden } = inputAndHidden(form, 'picker-outcome-path', 'path');
    input.value = '/models/preserve.safetensors';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    Array.from(form.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Browse')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(input.value).toBe('/models/preserve.safetensors');
    expect(hidden.value).toBe('/models/preserve.safetensors');
    expect(new FormData(form).get('path')).toBe('/models/preserve.safetensors');
    if (visibleMessage) expect(form.textContent).toContain(visibleMessage);
    else expect(form.textContent).not.toContain('Path picker failed.');
  });

  it('shows a field error and preserves the value when the picker request throws', async () => {
    const form = document.createElement('form');
    target.appendChild(form);
    promptFileApiMocks.openPathPicker.mockRejectedValue(new Error('Picker network failed.'));
    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'picker-network-path', name: 'path', label: 'Path', value: '',
        onresolve: vi.fn(async (candidate: string) => candidate),
      },
    }));
    await settle();

    const { input, hidden } = inputAndHidden(form, 'picker-network-path', 'path');
    input.value = '/models/retry.safetensors';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    Array.from(form.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Browse')!.click();
    await settle();

    expect(form.textContent).toContain('Picker network failed.');
    expect(input.value).toBe('/models/retry.safetensors');
    expect(hidden.value).toBe('/models/retry.safetensors');
    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(input.getAttribute('aria-describedby')).toBe('picker-network-path-feedback');
    const feedback = form.querySelector('#picker-network-path-feedback') as HTMLElement;
    expect(feedback.textContent).toBe('Picker network failed.');
    expect(feedback.getAttribute('role')).toBe('alert');
    expect(feedback.getAttribute('aria-live')).toBe('assertive');
    expect(feedback.getAttribute('aria-atomic')).toBe('true');
  });

  it('treats a backend picker error as an invalid field with assertive feedback', async () => {
    const form = document.createElement('form');
    target.appendChild(form);
    promptFileApiMocks.openPathPicker.mockResolvedValue({
      status: 'error', path: null, message: 'Only .safetensors files are accepted.',
    });
    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'picker-backend-error-path', name: 'path', label: 'Path', value: '',
        onresolve: vi.fn(async (candidate: string) => candidate),
      },
    }));
    await settle();

    const { input, hidden } = inputAndHidden(form, 'picker-backend-error-path', 'path');
    input.value = '/models/not-supported.bin';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    Array.from(form.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Browse')!.click();
    await settle();

    expect(input.value).toBe('/models/not-supported.bin');
    expect(hidden.value).toBe('/models/not-supported.bin');
    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(input.getAttribute('aria-describedby')).toBe('picker-backend-error-path-feedback');
    const feedback = form.querySelector('#picker-backend-error-path-feedback') as HTMLElement;
    expect(feedback.textContent).toBe('Only .safetensors files are accepted.');
    expect(feedback.getAttribute('role')).toBe('alert');
    expect(feedback.getAttribute('aria-live')).toBe('assertive');
    expect(feedback.getAttribute('aria-atomic')).toBe('true');
  });

  it('keeps one stable, polite feedback target while helper text becomes success or warning feedback', async () => {
    const form = document.createElement('form');
    target.appendChild(form);
    const onResolve = vi.fn(async (candidate: string) => `/resolved${candidate}`);
    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'accessible-path', name: 'path', label: 'Path', value: '',
        helper: 'Choose a model file from this computer.',
        onresolve: onResolve,
      },
    }));
    await settle();

    const { input } = inputAndHidden(form, 'accessible-path', 'path');
    const feedbackId = 'accessible-path-feedback';
    expect(input.getAttribute('aria-describedby')).toBe(feedbackId);
    expect(form.querySelectorAll(`#${feedbackId}`)).toHaveLength(1);
    const feedback = form.querySelector(`#${feedbackId}`) as HTMLElement;
    expect(feedback.textContent).toBe('Choose a model file from this computer.');
    expect(feedback.getAttribute('role')).toBe('status');
    expect(feedback.getAttribute('aria-live')).toBe('polite');
    expect(feedback.getAttribute('aria-atomic')).toBe('true');

    input.value = '/models/selected.safetensors';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    await settle();

    expect(form.querySelectorAll(`#${feedbackId}`)).toHaveLength(1);
    expect(feedback.textContent).toBe('Path loaded from the host machine.');
    expect(form.textContent?.match(/Path loaded from the host machine\./g)).toHaveLength(1);
    expect(feedback.getAttribute('role')).toBe('status');
    expect(feedback.getAttribute('aria-live')).toBe('polite');

    promptFileApiMocks.openPathPicker.mockResolvedValue({
      status: 'unsupported', path: null, message: 'Browse is unavailable on this host.',
    });
    Array.from(form.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Browse')!.click();
    await settle();

    expect(form.querySelectorAll(`#${feedbackId}`)).toHaveLength(1);
    expect(feedback.textContent).toBe('Browse is unavailable on this host.');
    expect(form.textContent?.match(/Browse is unavailable on this host\./g)).toHaveLength(1);
    expect(feedback.getAttribute('role')).toBe('status');
    expect(feedback.getAttribute('aria-live')).toBe('polite');
    expect(feedback.getAttribute('aria-atomic')).toBe('true');
  });

  it('exposes pending picker work through aria-busy before returning to the idle state', async () => {
    const form = document.createElement('form');
    target.appendChild(form);
    const picker = deferred<{ status: 'cancelled'; path: null; message: null }>();
    promptFileApiMocks.openPathPicker.mockReturnValue(picker.promise);
    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'busy-path', name: 'path', label: 'Path', value: '',
        onresolve: vi.fn(async (candidate: string) => candidate),
      },
    }));
    await settle();

    const { input } = inputAndHidden(form, 'busy-path', 'path');
    Array.from(form.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Browse')!.click();
    await Promise.resolve();
    flushSync();

    expect(input.getAttribute('aria-busy')).toBe('true');
    expect(form.querySelector('[aria-busy="true"]')).not.toBeNull();

    picker.resolve({ status: 'cancelled', path: null, message: null });
    await settle();
    expect(input.getAttribute('aria-busy')).toBeNull();
    expect(form.querySelector('[aria-busy="true"]')).toBeNull();
  });

  it('does not allow an obsolete async normalization result to repopulate a cleared path', async () => {
    const form = document.createElement('form');
    target.appendChild(form);
    const resolution = deferred<string>();
    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'stale-resolution-path', name: 'path', label: 'Path', value: '',
        onresolve: vi.fn(() => resolution.promise),
      },
    }));
    await settle();

    const { input, hidden } = inputAndHidden(form, 'stale-resolution-path', 'path');
    input.value = '/models/slow.safetensors';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    await settle();

    const clear = Array.from(form.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Clear')!;
    // A parent-controlled reset may occur while a resolution is pending. Dispatching
    // directly exercises the stale-result guard even though the disabled button blocks
    // this action for a user during the pending state.
    clear.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    resolution.resolve('/models/slow-normalized.safetensors');
    await settle();

    expect(input.value).toBe('');
    expect(hidden.value).toBe('');
    expect(new FormData(form).get('path')).toBe('');
    expect(form.textContent).not.toContain('Path loaded from the host machine.');
  });

  it('forwards required semantics to the visible native path input', async () => {
    const form = document.createElement('form');
    target.appendChild(form);
    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'required-path', name: 'path', label: 'Path', value: '', required: true,
        onresolve: vi.fn(async (candidate: string) => candidate),
      },
    }));
    await settle();

    const { input } = inputAndHidden(form, 'required-path', 'path');
    expect(input.required).toBe(true);
    expect(form.checkValidity()).toBe(false);
    input.value = '/models/valid.safetensors';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    expect(form.checkValidity()).toBe(true);
  });

  it('records native required invalid feedback without submitting and clears it after typing, a selection, or clear', async () => {
    const form = document.createElement('form');
    target.appendChild(form);
    const submit = vi.fn((event: SubmitEvent) => event.preventDefault());
    const invalid = vi.fn();
    form.addEventListener('submit', submit);
    promptFileApiMocks.openPathPicker.mockResolvedValue({
      status: 'selected', path: '/models/from-picker.safetensors', message: null,
    });
    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'required-feedback-path', name: 'path', label: 'Path', value: '', required: true,
        onresolve: vi.fn(async (candidate: string) => candidate),
      },
    }));
    await settle();

    const { input } = inputAndHidden(form, 'required-feedback-path', 'path');
    input.addEventListener('invalid', invalid);
    const nativeMessage = input.validationMessage;
    expect(nativeMessage).not.toBe('');

    form.requestSubmit();
    await settle();

    expect(invalid).toHaveBeenCalledTimes(1);
    expect(submit).not.toHaveBeenCalled();
    expect(input.getAttribute('aria-invalid')).toBe('true');
    const feedback = form.querySelector('#required-feedback-path-feedback') as HTMLElement;
    expect(feedback.textContent).toBe(nativeMessage);
    expect(feedback.getAttribute('role')).toBe('alert');
    expect(feedback.getAttribute('aria-live')).toBe('assertive');
    expect(feedback.getAttribute('aria-atomic')).toBe('true');

    input.value = '/models/typed.safetensors';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    await settle();
    expect(input.getAttribute('aria-invalid')).toBeNull();
    expect(form.querySelector('#required-feedback-path-feedback')).toBeNull();

    Array.from(form.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Browse')!.click();
    await settle();
    expect(input.value).toBe('/models/from-picker.safetensors');
    expect(input.getAttribute('aria-invalid')).toBeNull();

    Array.from(form.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Clear')!.click();
    await settle();
    expect(input.value).toBe('');
    expect(input.getAttribute('aria-invalid')).toBeNull();
    expect(form.querySelector('#required-feedback-path-feedback')).toBeNull();
  });
});
