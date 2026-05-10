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
    input!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();

    expect(onResolve).toHaveBeenCalledWith('~/prompts.yaml');
    expect(input!.value).toBe('/server/normalized.yaml');
    expect(onValueChange).toHaveBeenLastCalledWith('/server/normalized.yaml');
  });
});
