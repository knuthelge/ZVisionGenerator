// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const browseApiMocks = vi.hoisted(() => ({
  listDirectory: vi.fn(),
  browsePlaces: vi.fn(),
  createFolder: vi.fn(),
}));

vi.mock('$lib/api/browse', () => browseApiMocks);

import PathField from '$lib/components/molecules/PathField.svelte';
import { chooseInBrowser, listingWith, openBrowser, placesFor } from '../test-utils/pathBrowser';

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

function button(root: ParentNode, label: string): HTMLButtonElement {
  return Array.from(root.querySelectorAll<HTMLButtonElement>('button')).find((item) => item.getAttribute('aria-label') === label || item.textContent?.trim() === label)!;
}

/** Open Browse and choose *name* in the browser. */
async function browseAndChoose(root: ParentNode, name: string): Promise<void> {
  button(root, 'Browse').click();
  await settle();
  chooseInBrowser(name);
  await settle();
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
    for (const mock of Object.values(browseApiMocks)) mock.mockReset();
    browseApiMocks.browsePlaces.mockResolvedValue(placesFor('checkpoint_file'));
    browseApiMocks.listDirectory.mockResolvedValue(listingWith('/models/first-selected.safetensors'));
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
        purpose: 'checkpoint_file',
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
    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'browse-path',
        name: 'path',
        label: 'Path',
        purpose: 'checkpoint_file',
        value: '',
        onresolve: vi.fn(async (candidate: string) => candidate),
      },
    }));
    await settle();

    const input = form.querySelector('#browse-path') as HTMLInputElement;
    const hidden = form.querySelector('input[type="hidden"][name="path"]') as HTMLInputElement;
    await browseAndChoose(form, 'first-selected.safetensors');

    expect(input.value).toBe('/models/first-selected.safetensors');
    expect(hidden.value).toBe('/models/first-selected.safetensors');
    expect(new FormData(form).get('path')).toBe('/models/first-selected.safetensors');
    expect(browseApiMocks.listDirectory).toHaveBeenCalledWith('checkpoint_file', '', expect.anything());
    expect(openBrowser()).toBeNull();
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
        purpose: 'checkpoint_file',
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

  it('opens the browser at the typed value and keeps it when the browser closes without a choice', async () => {
    const form = document.createElement('form');
    target.appendChild(form);
    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'cancel-path', name: 'path', label: 'Path', purpose: 'checkpoint_file', value: '',
        onresolve: vi.fn(async (candidate: string) => candidate),
      },
    }));
    await settle();

    const { input, hidden } = inputAndHidden(form, 'cancel-path', 'path');
    input.value = '/models/preserve.safetensors';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    button(form, 'Browse').click();
    await settle();

    expect(browseApiMocks.listDirectory).toHaveBeenCalledWith('checkpoint_file', '/models/preserve.safetensors', expect.anything());
    button(openBrowser()!, 'Cancel').click();
    await settle();

    expect(openBrowser()).toBeNull();
    expect(input.value).toBe('/models/preserve.safetensors');
    expect(hidden.value).toBe('/models/preserve.safetensors');
  });

  it('opens the browser at a newly typed path rather than the last resolved one', async () => {
    app = flushSync(() => mount(PathField, {
      target,
      props: {
        id: 'retyped-path', name: 'path', label: 'Path', purpose: 'checkpoint_file', value: '/data/models/a.safetensors',
        onresolve: vi.fn(async (candidate: string) => candidate),
      },
    }));
    await settle();

    const input = target.querySelector('#retyped-path') as HTMLInputElement;
    input.value = '/mnt/other/';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    button(target, 'Browse').click();
    await settle();

    expect(browseApiMocks.listDirectory).toHaveBeenCalledWith('checkpoint_file', '/mnt/other/', expect.anything());
  });

  it('shows a folder that cannot be listed inside the browser and keeps the value', async () => {
    const form = document.createElement('form');
    target.appendChild(form);
    browseApiMocks.listDirectory.mockRejectedValue(new Error('Permission denied: /root.'));
    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'denied-path', name: 'path', label: 'Path', purpose: 'checkpoint_file', value: '',
        onresolve: vi.fn(async (candidate: string) => candidate),
      },
    }));
    await settle();

    const { input } = inputAndHidden(form, 'denied-path', 'path');
    input.value = '/root/x.safetensors';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    button(form, 'Browse').click();
    await settle();

    expect(openBrowser()?.querySelector('[role="alert"]')?.textContent).toBe('Permission denied: /root.');
    expect(input.value).toBe('/root/x.safetensors');
  });

  it('keeps one stable, polite feedback target while helper text becomes success or warning feedback', async () => {
    const form = document.createElement('form');
    target.appendChild(form);
    const onResolve = vi.fn(async (candidate: string) => `/resolved${candidate}`);
    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'accessible-path', name: 'path', label: 'Path', purpose: 'checkpoint_file', value: '',
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
  });

  it('does not allow an obsolete async normalization result to repopulate a cleared path', async () => {
    const form = document.createElement('form');
    target.appendChild(form);
    const resolution = deferred<string>();
    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'stale-resolution-path', name: 'path', label: 'Path', purpose: 'checkpoint_file', value: '',
        onresolve: vi.fn(() => resolution.promise),
      },
    }));
    await settle();

    const { input, hidden } = inputAndHidden(form, 'stale-resolution-path', 'path');
    input.value = '/models/slow.safetensors';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    await settle();

    const clear = button(form, 'Clear');
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
        id: 'required-path', name: 'path', label: 'Path', purpose: 'checkpoint_file', value: '', required: true,
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
    browseApiMocks.listDirectory.mockResolvedValue(listingWith('/models/from-picker.safetensors'));
    app = flushSync(() => mount(PathField, {
      target: form,
      props: {
        id: 'required-feedback-path', name: 'path', label: 'Path', purpose: 'checkpoint_file', value: '', required: true,
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

    await browseAndChoose(form, 'from-picker.safetensors');
    expect(input.value).toBe('/models/from-picker.safetensors');
    expect(input.getAttribute('aria-invalid')).toBeNull();

    button(form, 'Clear').click();
    await settle();
    expect(input.value).toBe('');
    expect(input.getAttribute('aria-invalid')).toBeNull();
    expect(form.querySelector('#required-feedback-path-feedback')).toBeNull();
  });
});
