// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import NavDrawer from './NavDrawer.svelte';

describe('NavDrawer', () => {
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

  it('renders nothing while closed', () => {
    app = flushSync(() => mount(NavDrawer, { target, props: { open: false, currentPage: 'workspace', onnavigate: vi.fn(), onclose: vi.fn() } }));
    expect(document.body.querySelector('[data-testid="nav-drawer"]')).toBeNull();
  });

  it('lists the five pages, marks the current one, navigates and closes', () => {
    const onnavigate = vi.fn();
    const onclose = vi.fn();
    app = flushSync(() => mount(NavDrawer, { target, props: { open: true, currentPage: 'workspace', onnavigate, onclose } }));

    const items = Array.from(document.body.querySelectorAll<HTMLButtonElement>('nav button'));
    expect(items.map((item) => item.textContent?.trim())).toEqual(['Workspace', 'Prompts', 'Gallery', 'Models', 'Config']);
    expect(items[0].getAttribute('aria-current')).toBe('page');

    items[2].click();
    expect(onclose).toHaveBeenCalledOnce();
    expect(onnavigate).toHaveBeenCalledWith('gallery');
  });

  it('closes on Escape and on the scrim, without navigating', () => {
    const onnavigate = vi.fn();
    const onclose = vi.fn();
    app = flushSync(() => mount(NavDrawer, { target, props: { open: true, currentPage: 'gallery', onnavigate, onclose } }));

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    expect(onclose).toHaveBeenCalledTimes(1);
    (document.body.querySelector('button[aria-label="Close menu"]') as HTMLButtonElement).click();
    expect(onclose).toHaveBeenCalledTimes(2);
    expect(onnavigate).not.toHaveBeenCalled();
  });
});
