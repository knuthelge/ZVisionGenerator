// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import TextBoxHeader from './TextBoxHeader.svelte';

describe('TextBoxHeader', () => {
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
  });

  const ids = { mineId: 'a-tab', enhancedId: 'b-tab', mineControls: 'a', enhancedControls: 'b' };
  const props = (overrides: Record<string, unknown> = {}) => ({ view: 'mine', label: 'Style', ...ids, ...overrides });
  const show = (overrides: Record<string, unknown> = {}) => {
    app = flushSync(() => mount(TextBoxHeader, { target, props: props(overrides) }));
  };
  const tab = (id: string) => target.querySelector(`#${id}`) as HTMLButtonElement;
  const control = (label: string) => target.querySelector(`[aria-label="${label}"]`) as HTMLButtonElement | null;
  const description = () => document.getElementById(tab('b-tab').getAttribute('aria-describedby') ?? '')?.textContent ?? '';

  it('names the two tabs in a tablist, with your tab selected first', () => {
    show();

    expect(Array.from(target.querySelectorAll('[role="tab"]')).map((node) => node.textContent?.trim())).toEqual(['Style', 'Enhanced']);
    expect(tab('a-tab').getAttribute('aria-selected')).toBe('true');
    expect(tab('b-tab').tabIndex).toBe(-1);
  });

  it('puts ✕ and the stop ring after the tablist, never inside a tab, and never both', () => {
    show({ view: 'enhanced', used: true });
    const scratch = control('Scratch the enhanced text')!;
    expect(scratch.closest('[role="tablist"]')).toBeNull();
    expect(scratch.parentElement?.querySelector('[role="tablist"]')).not.toBeNull();
    expect(control('Stop enhancing')).toBeNull();
  });

  it('shows the stop ring in place of ✕ while Enhance runs on the open Enhanced tab, and calls onstop', () => {
    const onstop = vi.fn();
    show({ view: 'enhanced', used: true, running: true, onstop });

    expect(control('Scratch the enhanced text')).toBeNull();
    control('Stop enhancing')!.click();
    expect(onstop).toHaveBeenCalledTimes(1);
  });

  it('scratches: calls onscratch, shows your tab and focuses it', async () => {
    const onscratch = vi.fn();
    show({ view: 'enhanced', used: true, onscratch });

    control('Scratch the enhanced text')!.click();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(onscratch).toHaveBeenCalledTimes(1);
    expect(tab('a-tab').getAttribute('aria-selected')).toBe('true');
    expect(document.activeElement).toBe(tab('a-tab'));
  });

  it('moves between the two tabs with the arrow keys, and the focus follows', async () => {
    show();
    tab('a-tab').focus();

    tab('a-tab').dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    await Promise.resolve();
    flushSync();

    expect(tab('b-tab').getAttribute('aria-selected')).toBe('true');
    expect(document.activeElement).toBe(tab('b-tab'));
    tab('b-tab').dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    await Promise.resolve();
    flushSync();
    expect(tab('a-tab').getAttribute('aria-selected')).toBe('true');
  });

  it('describes the Enhanced tab by what Generate sends, and carries no mark on your tab', () => {
    show({ used: true, stale: true });

    expect(description()).toBe('sent by Generate, out of date');
    expect(tab('b-tab').querySelector('[data-mark="warn"]')).not.toBeNull();
    expect(tab('a-tab').querySelector('[data-mark]')).toBeNull();
    expect(tab('b-tab').querySelectorAll('[data-mark]')).toHaveLength(1);
  });

  it('shows the stepper last, in the Enhanced view only, after the controls given for the right group', () => {
    show({ view: 'mine', stepper: { index: 2, count: 3 } });
    expect(target.querySelector('[role="group"]')).toBeNull();

    app = null;
    target.innerHTML = '';
    show({ view: 'enhanced', stepper: { index: 2, count: 3 } });
    expect(target.querySelector('[role="group"][aria-label="Enhanced versions"] .rewrite-count')?.textContent).toBe('2 of 3');
  });

  it('is a heading with no tabs, marks or ✕ for a box with one view, and shows its stepper', () => {
    show({ heading: 'Prompt', used: true, stepper: { index: 1, count: 2 }, stepperLabel: 'Lyrics versions' });

    expect(target.querySelectorAll('[role="tab"]')).toHaveLength(0);
    expect(target.textContent).toContain('Prompt');
    expect(target.querySelector('[data-mark]')).toBeNull();
    expect(control('Scratch the enhanced text')).toBeNull();
    expect(target.querySelector('[role="group"][aria-label="Lyrics versions"]')).not.toBeNull();
  });
});
