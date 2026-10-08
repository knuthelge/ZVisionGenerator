// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import Segmented from './Segmented.svelte';

describe('Segmented', () => {
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

  const options = [
    { value: 'inline', label: 'Inline' },
    { value: 'file', label: 'Prompt file' },
    { value: 'off', label: 'Off', disabled: true },
  ];

  it('marks the chosen option as pressed inside a named group', () => {
    app = flushSync(() => mount(Segmented, { target, props: { options, value: 'file', label: 'Prompt source', onchange: vi.fn() } }));

    const group = target.querySelector('[role="group"]') as HTMLElement;
    expect(group.getAttribute('aria-label')).toBe('Prompt source');
    const pressed = Array.from(group.querySelectorAll('button')).map((b) => b.getAttribute('aria-pressed'));
    expect(pressed).toEqual(['false', 'true', 'false']);
  });

  it('reports a new choice, and ignores the current or a disabled one', () => {
    const onchange = vi.fn();
    app = flushSync(() => mount(Segmented, { target, props: { options, value: 'inline', label: 'Prompt source', onchange } }));

    const button = (value: string) => target.querySelector(`button[data-value="${value}"]`) as HTMLButtonElement;
    button('inline').click();
    button('off').click();
    expect(onchange).not.toHaveBeenCalled();
    button('file').click();
    expect(onchange).toHaveBeenCalledExactlyOnceWith('file');
  });
});
