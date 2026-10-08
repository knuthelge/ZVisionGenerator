// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import Button from './Button.svelte';
import Chip from './Chip.svelte';

describe('Button', () => {
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

  it('is a plain button by default and runs its click handler', () => {
    const onclick = vi.fn();
    app = flushSync(() => mount(Button, { target, props: { onclick } }));

    const button = target.querySelector('button') as HTMLButtonElement;
    expect(button.type).toBe('button');
    button.click();
    expect(onclick).toHaveBeenCalledOnce();
  });

  it('is disabled and marked busy while loading, so it cannot be pressed twice', () => {
    const onclick = vi.fn();
    app = flushSync(() => mount(Button, { target, props: { variant: 'primary', loading: true, onclick } }));

    const button = target.querySelector('button') as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    expect(button.getAttribute('aria-busy')).toBe('true');
    button.click();
    expect(onclick).not.toHaveBeenCalled();
  });

  it('passes through attributes such as aria-label and data attributes', () => {
    app = flushSync(() => mount(Button, { target, props: { icon: true, 'aria-label': 'Delete model', 'data-action': 'delete' } }));

    const button = target.querySelector('button') as HTMLButtonElement;
    expect(button.getAttribute('aria-label')).toBe('Delete model');
    expect(button.dataset.action).toBe('delete');
  });
});

describe('Chip', () => {
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

  it('is a toggle button when it has a pressed state', () => {
    const onclick = vi.fn();
    app = flushSync(() => mount(Chip, { target, props: { pressed: true, onclick } }));

    const chip = target.querySelector('button') as HTMLButtonElement;
    expect(chip.getAttribute('aria-pressed')).toBe('true');
    chip.click();
    expect(onclick).toHaveBeenCalledOnce();
  });

  it('offers a named remove button when it can be removed', () => {
    const onremove = vi.fn();
    app = flushSync(() => mount(Chip, { target, props: { onremove, removeLabel: 'Remove film-grain' } }));

    const remove = target.querySelector('button[aria-label="Remove film-grain"]') as HTMLButtonElement;
    expect(remove).not.toBeNull();
    remove.click();
    expect(onremove).toHaveBeenCalledOnce();
  });
});
