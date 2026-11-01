// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { EnhanceAxis, EnhanceSettings, EntryEnhance } from '$lib/types';

import EnhanceChip from './EnhanceChip.svelte';

const AXES: EnhanceAxis[] = [
  { key: 'style', label: 'Style', multi: false, video_only: false, audio_only: false, options: [{ slug: 'keep', label: 'Keep', description: '', group: null }, { slug: 'cinematic', label: 'Cinematic', description: '', group: null }], default: ['keep'] },
];
// The backend sends `parts: null` in its defaults.
const DEFAULTS = { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'same', motion: [], sound: [], note: '', parts: null } as unknown as EnhanceSettings;

describe('EnhanceChip Rewrite choice', () => {
  let app: Record<string, unknown> | null = null;
  const target = document.createElement('div');

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    document.body.innerHTML = '';
    target.innerHTML = '';
  });

  function mountChip(enhance: EntryEnhance): ReturnType<typeof vi.fn> {
    const onchange = vi.fn();
    document.body.appendChild(target);
    app = flushSync(() => mount(EnhanceChip, { target, props: { enhance, axes: AXES, defaults: DEFAULTS, idPrefix: 'enh-e1', onchange } }));
    flushSync(() => (target.querySelector('button.enhance-button') as HTMLButtonElement).click());
    return onchange;
  }

  const rewrite = (): HTMLButtonElement[] => Array.from(document.querySelectorAll('[aria-labelledby="enh-e1-rewrite-label"] [role="radio"]'));
  const apply = (): void => flushSync(() => (Array.from(document.querySelectorAll('.enhance-actions button')).find((b) => b.textContent === 'Apply') as HTMLButtonElement).click());

  it('shows the entry value and applies it unchanged', () => {
    const onchange = mountChip({ parts: 'action' });
    expect(rewrite().map((chip) => [chip.textContent?.trim(), chip.getAttribute('aria-checked')])).toEqual([['Scene and action', 'false'], ['Action only', 'true']]);
    apply();
    expect(onchange).toHaveBeenCalledWith(expect.objectContaining({ parts: 'action' }));
  });

  it('omits Rewrite when none was chosen', () => {
    const onchange = mountChip({ style: 'cinematic' });
    expect(rewrite()[0].getAttribute('aria-checked')).toBe('true');
    apply();
    expect(onchange.mock.calls[0][0]).not.toHaveProperty('parts');
  });

  it('applies a chosen Rewrite', () => {
    const onchange = mountChip({ style: 'cinematic' });
    flushSync(() => rewrite()[1].click());
    apply();
    expect(onchange.mock.calls[0][0]).toMatchObject({ style: 'cinematic', parts: 'action' });
  });

});

describe('EnhanceChip Framing and Color', () => {
  let app: Record<string, unknown> | null = null;
  const target = document.createElement('div');
  const OPTION = (slug: string, label: string) => ({ slug, label, description: '', group: null });
  const single = (key: EnhanceAxis['key'], label: string, options: EnhanceAxis['options']): EnhanceAxis => ({ key, label, multi: false, video_only: false, audio_only: false, options, default: ['keep'] });
  const FRAMING_AXES: EnhanceAxis[] = [
    ...AXES,
    single('shot', 'Shot', [OPTION('keep', 'Keep'), OPTION('close-up', 'Close-up')]),
    single('angle', 'Angle', [OPTION('keep', 'Keep'), OPTION('low', 'Low angle')]),
    single('palette', 'Palette', [OPTION('keep', 'Keep'), OPTION('earth', 'Earth tones'), OPTION('custom', 'Custom')]),
  ];
  const OWN = ['#' + '1B3A4B'];

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    document.body.innerHTML = '';
    target.innerHTML = '';
  });

  function mountChip(enhance: EntryEnhance): ReturnType<typeof vi.fn> {
    const onchange = vi.fn();
    document.body.appendChild(target);
    app = flushSync(() => mount(EnhanceChip, { target, props: { enhance, axes: FRAMING_AXES, defaults: DEFAULTS, idPrefix: 'enh-e1', onchange } }));
    flushSync(() => (target.querySelector('button.enhance-button') as HTMLButtonElement).click());
    return onchange;
  }

  const palette = (): HTMLButtonElement => document.querySelector('#enh-e1-palette') as HTMLButtonElement;
  const apply = (): void => flushSync(() => (Array.from(document.querySelectorAll('.enhance-actions button')).find((b) => b.textContent === 'Apply') as HTMLButtonElement).click());

  it('offers the Framing and Color sections', () => {
    mountChip({ style: 'cinematic' });
    expect(document.querySelector('section[aria-label="Framing"]')).not.toBeNull();
    expect(document.querySelector('section[aria-label="Color"]')).not.toBeNull();
    expect(document.querySelector('#enh-e1-shot')).not.toBeNull();
    expect(document.querySelector('#enh-e1-angle')).not.toBeNull();
    expect(palette()).not.toBeNull();
  });

  it('shows an entry with its own colours and applies them unchanged', () => {
    const onchange = mountChip({ palette: 'custom', colors: OWN });
    expect(palette().dataset.value).toBe('custom');
    expect(document.querySelectorAll('[data-color-dot]')).toHaveLength(1);
    apply();
    expect(onchange).toHaveBeenCalledWith(expect.objectContaining({ palette: 'custom', colors: OWN }));
  });

  it('reports no colours once a preset palette replaces the Custom ones', () => {
    const onchange = mountChip({ palette: 'custom', colors: OWN });
    flushSync(() => palette().click());
    flushSync(() => (document.querySelector('[role="option"][data-slug="earth"]') as HTMLButtonElement).click());
    expect(palette().dataset.value).toBe('earth');
    expect(document.querySelector('[data-color-dot]')).toBeNull();
    apply();
    expect(onchange).toHaveBeenCalledWith(expect.objectContaining({ palette: 'earth', colors: [] }));
  });
});
