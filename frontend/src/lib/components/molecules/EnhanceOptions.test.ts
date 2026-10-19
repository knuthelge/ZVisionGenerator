// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { EnhanceAxis, EnhanceSettings } from '$lib/types';

import EnhanceOptions from './EnhanceOptions.svelte';

const axes: EnhanceAxis[] = [
  { key: 'style', label: 'Style', multi: false, video_only: false, audio_only: false, options: [{ slug: 'keep', label: 'Keep', description: '', group: null }, { slug: 'photo', label: 'Photo', description: '', group: null }], default: ['keep'] },
  { key: 'details', label: 'Details', multi: true, video_only: false, audio_only: false, options: [{ slug: 'lighting', label: 'Lighting', description: '', group: null }, { slug: 'composition', label: 'Composition', description: '', group: null }], default: [] },
];

describe('EnhanceOptions', () => {
  let app: Record<string, unknown> | null = null;
  const target = document.createElement('div');

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.innerHTML = '';
  });

  function mountOptions(settings: EnhanceSettings, onchange = vi.fn(), extra: Record<string, unknown> = {}): typeof onchange {
    document.body.appendChild(target);
    app = flushSync(() => mount(EnhanceOptions, { target, props: { axes, settings, mode: 'image', onchange, ...extra } }));
    return onchange;
  }

  const BASE: EnhanceSettings = { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'same', motion: [], sound: [], note: '' };
  const videoAxes: EnhanceAxis[] = [
    ...axes,
    { key: 'motion', label: 'Motion', multi: true, video_only: true, audio_only: false, options: [{ slug: 'action', label: 'Action', description: '', group: null }], default: [] },
    { key: 'sound', label: 'Sound', multi: true, video_only: true, audio_only: true, options: [{ slug: 'ambient', label: 'Ambience', description: '', group: null }], default: [] },
  ];

  it.each([
    ['image', false, false, false],
    ['video', false, true, false],
    ['video', true, true, true],
  ])('mode %s with audio %s shows motion %s and sound %s', (mode, audio, motion, sound) => {
    mountOptions(BASE, vi.fn(), { axes: videoAxes, mode, audio });
    expect(target.querySelector('[data-toggle-all="motion"]') !== null).toBe(motion);
    expect(target.querySelector('[data-toggle-all="sound"]') !== null).toBe(sound);
  });

  it('reports a typed direction', () => {
    const onchange = mountOptions({ ...BASE, note: 'at' }, vi.fn(), { idPrefix: 't' });
    const input = target.querySelector('#t-note') as HTMLInputElement;
    expect(input.value).toBe('at');
    input.value = 'at night';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    expect(onchange).toHaveBeenCalledWith(expect.objectContaining({ note: 'at night' }));
  });

  it('offers select-all only on multi-pick axes and selects every option', () => {
    const onchange = mountOptions({ style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: ['lighting'], length: 'same', motion: [], sound: [], note: '' });
    expect(target.querySelector('[data-toggle-all="style"]')).toBeNull();
    const toggle = target.querySelector('[data-toggle-all="details"]') as HTMLButtonElement;
    expect(toggle.dataset.allSelected).toBe('false');
    toggle.click();
    expect(onchange).toHaveBeenCalledWith(expect.objectContaining({ details: ['lighting', 'composition'] }));
  });

  it('clears every option once all are selected', () => {
    const onchange = mountOptions({ style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: ['lighting', 'composition'], length: 'same', motion: [], sound: [], note: '' });
    const toggle = target.querySelector('[data-toggle-all="details"]') as HTMLButtonElement;
    expect(toggle.dataset.allSelected).toBe('true');
    toggle.click();
    expect(onchange).toHaveBeenCalledWith(expect.objectContaining({ details: [] }));
  });
});
