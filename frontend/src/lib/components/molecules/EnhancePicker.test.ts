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

describe('EnhancePicker', () => {
  let app: Record<string, unknown> | null = null;
  const target = document.createElement('div');

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.innerHTML = '';
    document.querySelectorAll('[role="listbox"]').forEach((node) => node.remove());
  });

  function mountPicker(axis: EnhanceAxis, value: string, onchange = vi.fn()): typeof onchange {
    document.body.appendChild(target);
    app = flushSync(() => mount(EnhancePicker, { target, props: { axis, value, id: 'p', onchange } }));
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
    expect(Array.from(document.querySelectorAll('[role="listbox"] .picker-group')).map((node) => node.textContent)).toEqual(['Photo', 'Art']);
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
});
