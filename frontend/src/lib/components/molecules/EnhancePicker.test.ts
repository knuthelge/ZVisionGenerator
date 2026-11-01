// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, tick, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { EnhanceAxis } from '$lib/types';

import EnhancePicker, { groupOptions } from './EnhancePicker.svelte';

const STYLE: EnhanceAxis = {
  key: 'style',
  label: 'Style',
  multi: false,
  video_only: false,
  audio_only: false,
  options: [
    { slug: 'keep', label: 'Keep', description: "Don't impose one", group: null },
    { slug: 'photo', label: 'Natural photo', description: 'A real photograph', group: 'Photo' },
    { slug: 'candid', label: 'Candid', description: 'A phone snapshot', group: 'Photo' },
    { slug: 'anime', label: 'Anime', description: 'Line work, cel shading', group: 'Art' },
  ],
  default: ['keep'],
};
const LENGTH: EnhanceAxis = {
  key: 'length',
  label: 'Length',
  multi: false,
  video_only: false,
  audio_only: false,
  options: [
    { slug: 'shorter', label: 'Shorter', description: '', group: null },
    { slug: 'same', label: 'Same', description: '', group: null },
  ],
  default: ['same'],
};

const EARTH = ['#6B4226', '#C8643B', '#6B7032', '#C2A878', '#F3E5C0'];
const PALETTE: EnhanceAxis = {
  key: 'palette',
  label: 'Palette',
  multi: false,
  video_only: false,
  audio_only: false,
  options: [
    { slug: 'keep', label: 'Keep', description: "Don't impose one", group: null },
    { slug: 'earth', label: 'Earth tones', description: 'Browns, terracotta, olive', group: null, swatches: EARTH },
    { slug: 'custom', label: 'Custom', description: 'Pick your own colours', group: null },
  ],
  default: ['keep'],
};

describe('EnhancePicker', () => {
  let app: Record<string, unknown> | null = null;
  const target = document.createElement('div');

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.innerHTML = '';
    document.querySelectorAll('[role="listbox"]').forEach((node) => node.remove());
  });

  function mountPicker(axis: EnhanceAxis, value: string, onchange = vi.fn(), extra: Record<string, unknown> = {}): typeof onchange {
    document.body.appendChild(target);
    app = flushSync(() => mount(EnhancePicker, { target, props: { axis, value, id: 'p', onchange, ...extra } }));
    return onchange;
  }

  const trigger = () => target.querySelector('#p') as HTMLButtonElement;
  const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
  const options = () => Array.from(document.querySelectorAll('[role="listbox"] [role="option"]')) as HTMLButtonElement[];

  it('groups options in their listed order, an ungrouped Keep first', () => {
    expect(groupOptions(STYLE.options).map((group) => [group.name, group.options.map((option) => option.slug)])).toEqual([
      [null, ['keep']],
      ['Photo', ['photo', 'candid']],
      ['Art', ['anime']],
    ]);
  });

  it('shows the chosen option and opens a described, grouped menu with the choice selected', async () => {
    mountPicker(STYLE, 'candid');
    expect(trigger().textContent).toContain('Candid');
    expect(trigger().getAttribute('aria-expanded')).toBe('false');

    trigger().click();
    await settle();
    expect(trigger().getAttribute('aria-expanded')).toBe('true');
    expect(options().map((option) => option.dataset.slug)).toEqual(['keep', 'photo', 'candid', 'anime']);
    expect(options().find((option) => option.getAttribute('aria-selected') === 'true')?.dataset.slug).toBe('candid');
    expect(document.activeElement).toBe(options()[2]);
  });

  it('reports a pick and closes', async () => {
    const onchange = mountPicker(STYLE, 'keep');
    trigger().click();
    await settle();
    options()[3].click();
    await tick();
    expect(onchange).toHaveBeenCalledWith('anime');
    expect(options()).toEqual([]);
  });

  it('moves through the options with the arrow keys', async () => {
    mountPicker(STYLE, 'keep');
    trigger().click();
    await settle();
    const listbox = document.querySelector('[role="listbox"]') as HTMLElement;
    listbox.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
    expect(document.activeElement).toBe(options()[1]);
    listbox.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true }));
    listbox.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true }));
    expect(document.activeElement).toBe(options()[3]);
  });

  it('clears a set option back to Keep, and has no clear button when it is Keep', () => {
    const onchange = mountPicker(STYLE, 'photo');
    const clear = target.querySelector('[aria-label="Style: back to Keep"]') as HTMLButtonElement;
    clear.click();
    expect(onchange).toHaveBeenCalledWith('keep');
  });

  it('has no clear button at Keep or on an axis without Keep', async () => {
    mountPicker(STYLE, 'keep');
    expect(target.querySelector('[aria-label="Style: back to Keep"]')).toBeNull();
    await unmount(app); app = null; target.innerHTML = '';
    mountPicker(LENGTH, 'shorter');
    expect(target.querySelector('[aria-label="Length: back to Keep"]')).toBeNull();
  });

  const swatchHexes = (root: ParentNode) => Array.from(root.querySelectorAll<HTMLElement>('[data-swatch]')).map((node) => node.dataset.swatch);

  it('shows a swatch strip on palette options that have colours, and none on Keep or Custom', async () => {
    mountPicker(PALETTE, 'keep');
    trigger().click();
    await settle();
    const bySlug = (slug: string) => options().find((option) => option.dataset.slug === slug) as HTMLElement;
    expect(swatchHexes(bySlug('earth'))).toEqual(EARTH);
    expect(swatchHexes(bySlug('keep'))).toEqual([]);
    expect(swatchHexes(bySlug('custom'))).toEqual([]);
  });

  it('hides every swatch strip from screen readers', async () => {
    mountPicker(PALETTE, 'earth');
    trigger().click();
    await settle();
    const strips = Array.from(document.querySelectorAll('.picker-swatches'));
    expect(strips.length).toBe(2);
    for (const strip of strips) expect(strip.getAttribute('aria-hidden')).toBe('true');
  });

  it('shows the chosen option\'s swatches on the closed trigger, or the ones passed in', async () => {
    mountPicker(PALETTE, 'earth');
    expect(swatchHexes(trigger())).toEqual(EARTH);
    await unmount(app); app = null; target.innerHTML = '';
    mountPicker(PALETTE, 'earth', vi.fn(), { swatches: ['#111111'] });
    expect(swatchHexes(trigger())).toEqual(['#111111']);
    await unmount(app); app = null; target.innerHTML = '';
    mountPicker(PALETTE, 'custom');
    expect(swatchHexes(trigger())).toEqual([]);
  });
});
