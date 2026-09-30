// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import Button from './Button.svelte';

describe('Button variants', () => {
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

  it('leaves font, weight, transitions, and disabled styling of primary to surface-button-primary', () => {
    app = flushSync(() => mount(Button, { target, props: { variant: 'primary', loading: true } }));

    const button = target.querySelector('button') as HTMLButtonElement;
    expect(button.classList.contains('surface-button-primary')).toBe(true);
    expect(button.disabled).toBe(true);
    for (const utility of ['font-display', 'font-medium', 'transition-colors', 'disabled:opacity-50']) {
      expect(button.classList.contains(utility)).toBe(false);
    }
  });

  it('keeps utility font, weight, transitions, and dimmed disabled state on other variants', () => {
    app = flushSync(() => mount(Button, { target, props: { variant: 'secondary' } }));

    const button = target.querySelector('button') as HTMLButtonElement;
    for (const utility of ['font-display', 'font-medium', 'transition-colors', 'disabled:opacity-50']) {
      expect(button.classList.contains(utility)).toBe(true);
    }
  });
});
