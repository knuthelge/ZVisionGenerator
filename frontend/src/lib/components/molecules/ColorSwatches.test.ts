// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { SvelteMap } from 'svelte/reactivity';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import ColorSwatches from './ColorSwatches.svelte';

describe('ColorSwatches', () => {
  const colors = ['#112233', '#445566', '#778899'];
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
    vi.restoreAllMocks();
  });

  function unmountApp(): void {
    if (app) unmount(app);
    app = null;
    target.innerHTML = '';
  }

  function render(props: { colors?: string[]; max?: number; disabled?: boolean } = {}) {
    const onchange = vi.fn();
    app = mount(ColorSwatches, {
      target,
      props: { colors, max: 5, onchange, ...props },
    }) as Record<string, unknown>;
    flushSync();
    return onchange;
  }

  const addInput = () => target.querySelector<HTMLInputElement>('[data-color-add] input')!;

  it('renders one dot per colour inside a named group', () => {
    render();
    expect(target.querySelector('[role="group"]')?.getAttribute('aria-label')).toBe('Custom colors');
    expect(target.querySelectorAll('[data-color-dot]')).toHaveLength(3);
  });

  it('replaces a colour on an input event from its dot', () => {
    const onchange = render();
    const input = target.querySelector<HTMLInputElement>('[data-color-dot="1"] input')!;
    input.value = '#abcdef';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    expect(onchange).toHaveBeenCalledWith(['#112233', '#abcdef', '#778899']);
  });

  it('removes a colour from its remove button', () => {
    const onchange = render();
    target.querySelector<HTMLButtonElement>('[data-color-remove="0"]')!.click();
    expect(onchange).toHaveBeenCalledWith(['#445566', '#778899']);
  });

  it('appends the colour chosen in the add input', () => {
    const onchange = render();
    const input = addInput();
    input.value = '#fedcba';
    input.dispatchEvent(new Event('change', { bubbles: true }));
    expect(onchange).toHaveBeenCalledWith([...colors, '#fedcba']);
  });

  it('ignores a colour the row already holds', () => {
    const onchange = render();
    const input = addInput();
    input.value = '#445566';
    input.dispatchEvent(new Event('change', { bubbles: true }));
    expect(onchange).not.toHaveBeenCalled();
  });

  it('hides the add control at the maximum', () => {
    render({ max: 3 });
    expect(target.querySelector('[data-color-add]')).toBeNull();
  });

  it('rests the add input on a reset value, so the next pick always differs, even black', () => {
    const onchange = render();
    const input = addInput();
    const rest = input.value;
    expect(rest).not.toBe('#000000');
    input.value = '#000000';
    input.dispatchEvent(new Event('change', { bubbles: true }));
    expect(onchange).toHaveBeenCalledWith([...colors, '#000000']);
    expect(input.value).toBe(rest);
  });

  it('shows a hint only while there are no colours', () => {
    render({ colors: [] });
    expect(target.querySelector('[data-color-hint]')).not.toBeNull();
    unmountApp();
    render();
    expect(target.querySelector('[data-color-hint]')).toBeNull();
  });

  it('disables every input and button when disabled', () => {
    render({ disabled: true });
    const controls = target.querySelectorAll<HTMLInputElement | HTMLButtonElement>('input, button');
    expect(controls.length).toBe(7);
    for (const control of controls) expect(control.disabled).toBe(true);
  });

  it('keeps the same input element across a live edit so an open picker stays linked', () => {
    // A SvelteMap read inside a prop getter makes the props reactive without a .svelte.ts file.
    const state = new SvelteMap<string, string[]>([['colors', [...colors]]]);
    const onchange = vi.fn((next: string[]) => state.set('colors', next));
    app = mount(ColorSwatches, {
      target,
      props: {
        get colors() {
          return state.get('colors')!;
        },
        max: 5,
        onchange,
      },
    }) as Record<string, unknown>;
    flushSync();

    const input = target.querySelector<HTMLInputElement>('[data-color-dot="1"] input')!;
    input.value = '#abcdef';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();

    expect(state.get('colors')).toEqual(['#112233', '#abcdef', '#778899']);
    expect(target.querySelector('[data-color-dot="1"] input')).toBe(input);
    expect(input.value).toBe('#abcdef');
    expect(target.querySelectorAll('[data-color-dot]')).toHaveLength(3);
  });

  it('labels every control for assistive tech', () => {
    render();
    expect(target.querySelector('[data-color-dot="0"] input')?.getAttribute('aria-label')).toBe('Color 1');
    expect(target.querySelector('[data-color-remove="2"]')?.getAttribute('aria-label')).toBe('Remove color 3');
    expect(addInput().getAttribute('aria-label')).toBe('Add a color');
  });
});
