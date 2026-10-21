// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { DownloadRequestResult } from '$lib/state/downloads.svelte';
import type { ModelOption } from '$lib/types';
import FirstRunDialog from './FirstRunDialog.svelte';

const models: ModelOption[] = [
  { id: 'klein9b', label: 'FLUX.2 Klein 9B', type: 'image', download_size_label: '35 GB' },
  { id: 'zit', label: 'Z-Image Turbo', type: 'image', download_size_label: '33 GB' },
  { id: 'krea2', label: 'Krea 2 Turbo', type: 'image', download_size_label: null },
];

describe('FirstRunDialog', () => {
  let app: Record<string, unknown> | null = null;
  const ondownload = vi.fn<(model: ModelOption) => Promise<DownloadRequestResult>>();
  const onskip = vi.fn();
  const onclose = vi.fn();

  beforeEach(() => {
    ondownload.mockReset();
    ondownload.mockResolvedValue('queued');
    onskip.mockReset();
    onclose.mockReset();
    // jsdom has no Web Animations API, which the mascot's blend calls.
    Element.prototype.animate = vi.fn(() => ({ cancel: vi.fn(), finished: Promise.resolve(), onfinish: null })) as unknown as Element['animate'];
  });

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    document.body.innerHTML = '';
    delete (Element.prototype as Partial<Element>).animate;
  });

  function open(props: { recommended?: string | null; models?: ModelOption[] } = {}): void {
    app = flushSync(() => mount(FirstRunDialog, {
      target: document.body,
      props: { models: props.models ?? models, recommended: 'recommended' in props ? props.recommended : 'klein9b', ondownload, onskip, onclose },
    }));
  }

  const radio = (id: string) => document.querySelector(`[data-testid="first-run-model-${id}"]`) as HTMLInputElement;
  const action = (name: string) => document.querySelector(`[data-action="${name}"]`) as HTMLButtonElement;
  const settle = async () => { await Promise.resolve(); await Promise.resolve(); flushSync(); };

  it('is a labelled dialog with one radio group holding every model', () => {
    open();

    const dialog = document.querySelector('[role="dialog"]') as HTMLElement;
    expect(document.getElementById(dialog.getAttribute('aria-labelledby')!)?.textContent).toBeTruthy();
    expect(document.getElementById(dialog.getAttribute('aria-describedby')!)?.textContent).toBeTruthy();
    const radios = Array.from(document.querySelectorAll<HTMLInputElement>('input[type="radio"]'));
    expect(radios.map((input) => input.value)).toEqual(['klein9b', 'zit', 'krea2']);
    expect(new Set(radios.map((input) => input.name)).size).toBe(1);
  });

  it('preselects the recommended model and names it on the Download button', () => {
    open();

    expect(radio('klein9b').checked).toBe(true);
    expect(action('first-run-download').textContent).toContain('Download FLUX.2 Klein 9B');
  });

  it('preselects the first model when none is recommended', () => {
    open({ recommended: null });

    expect(radio('klein9b').checked).toBe(true);
    expect(document.querySelector('[data-testid="first-run-recommended"]')).toBeNull();
  });

  it('shows names and sizes but never an alias, and leaves out an unknown size', () => {
    open();

    const text = document.body.textContent ?? '';
    expect(text).toContain('≈35 GB');
    expect(text).toContain('≈33 GB');
    expect(text).not.toContain('klein9b');
    expect(text).not.toContain('krea2');
    expect(document.querySelectorAll('[data-testid="first-run-size"]')).toHaveLength(1);
  });

  it('downloads the model that was chosen', async () => {
    open();

    radio('zit').click();
    flushSync();
    expect(action('first-run-download').textContent).toContain('Download Z-Image Turbo');
    action('first-run-download').click();
    await settle();

    expect(ondownload).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ id: 'zit' }));
  });

  it('does nothing on Enter in the group', () => {
    open();
    const event = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true });

    radio('klein9b').dispatchEvent(event);

    expect(event.defaultPrevented).toBe(true);
    expect(ondownload).not.toHaveBeenCalled();
  });

  it('locks the choice and the buttons while the download is prepared, keeping Download focusable', async () => {
    let finish: (result: DownloadRequestResult) => void = () => undefined;
    ondownload.mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    open();

    action('first-run-download').click();
    flushSync();

    expect((document.querySelector('[data-testid="first-run-models"]') as HTMLFieldSetElement).disabled).toBe(true);
    expect(action('first-run-download').getAttribute('aria-disabled')).toBe('true');
    expect(action('first-run-download').disabled).toBe(false);
    expect(action('first-run-skip').getAttribute('aria-disabled')).toBe('true');
    action('first-run-download').click();
    action('first-run-skip').click();
    expect(ondownload).toHaveBeenCalledOnce();
    expect(onskip).not.toHaveBeenCalled();

    finish('declined');
    await settle();

    expect((document.querySelector('[data-testid="first-run-models"]') as HTMLFieldSetElement).disabled).toBe(false);
    expect(action('first-run-download').getAttribute('aria-disabled')).toBeNull();
  });

  it('hands Skip to the caller', () => {
    open();

    action('first-run-skip').click();

    expect(onskip).toHaveBeenCalledOnce();
  });

  it('reports the backdrop as a close', () => {
    open();

    (document.querySelector('button[aria-label="Close dialog"]') as HTMLButtonElement).click();

    expect(onclose).toHaveBeenCalledOnce();
  });
});
