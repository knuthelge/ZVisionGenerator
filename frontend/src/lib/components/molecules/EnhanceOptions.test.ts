// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { EnhanceAxis, EnhanceSettings } from '$lib/types';

import EnhanceOptions from './EnhanceOptions.svelte';

const axes: EnhanceAxis[] = [
  { key: 'style', label: 'Style', multi: false, video_only: false, options: [{ slug: 'keep', label: 'Keep' }, { slug: 'photo', label: 'Photo' }], default: ['keep'] },
  { key: 'details', label: 'Details', multi: true, video_only: false, options: [{ slug: 'lighting', label: 'Lighting' }, { slug: 'composition', label: 'Composition' }], default: [] },
];

describe('EnhanceOptions', () => {
  let app: Record<string, unknown> | null = null;
  const target = document.createElement('div');

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.innerHTML = '';
  });

  function mountOptions(settings: EnhanceSettings, onchange = vi.fn()): typeof onchange {
    document.body.appendChild(target);
    app = flushSync(() => mount(EnhanceOptions, { target, props: { axes, settings, mode: 'image', onchange } }));
    return onchange;
  }

  it('offers select-all only on multi-pick axes and selects every option', () => {
    const onchange = mountOptions({ style: 'keep', details: ['lighting'], length: 'same', motion: [] });
    expect(target.querySelector('[data-toggle-all="style"]')).toBeNull();
    const toggle = target.querySelector('[data-toggle-all="details"]') as HTMLButtonElement;
    expect(toggle.dataset.allSelected).toBe('false');
    toggle.click();
    expect(onchange).toHaveBeenCalledWith(expect.objectContaining({ details: ['lighting', 'composition'] }));
  });

  it('clears every option once all are selected', () => {
    const onchange = mountOptions({ style: 'keep', details: ['lighting', 'composition'], length: 'same', motion: [] });
    const toggle = target.querySelector('[data-toggle-all="details"]') as HTMLButtonElement;
    expect(toggle.dataset.allSelected).toBe('true');
    toggle.click();
    expect(onchange).toHaveBeenCalledWith(expect.objectContaining({ details: [] }));
  });
});
