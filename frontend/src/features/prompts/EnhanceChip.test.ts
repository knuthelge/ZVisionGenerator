// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { EnhanceAxis, EnhanceSettings, EntryEnhance } from '$lib/types';

import EnhanceChip from './EnhanceChip.svelte';

const AXES: EnhanceAxis[] = [
  { key: 'style', label: 'Style', multi: false, video_only: false, audio_only: false, options: [{ slug: 'keep', label: 'Keep' }, { slug: 'cinematic', label: 'Cinematic' }], default: ['keep'] },
];
// The backend sends `parts: null` in its defaults.
const DEFAULTS = { style: 'keep', mood: 'keep', details: [], length: 'same', motion: [], sound: [], note: '', parts: null } as unknown as EnhanceSettings;

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
