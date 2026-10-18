// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { router } from '$lib/state/router.svelte';
import TopStrip from './TopStrip.svelte';

describe('TopStrip', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
    draft.reset();
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
  });

  it('shows the menu, the workflow, the job state and Gallery on the Workspace', () => {
    const onmenu = vi.fn();
    app = flushSync(() => mount(TopStrip, { target, props: { currentPage: 'workspace', workflows: ['txt2img', 'img2img', 'img2vid', 'txt2vid', 'extend'], onmenu } }));

    (target.querySelector('button[aria-label="Menu"]') as HTMLButtonElement).click();
    expect(onmenu).toHaveBeenCalledOnce();

    const select = target.querySelector('select[aria-label="Workflow"]') as HTMLSelectElement;
    expect(Array.from(select.options).map((option) => option.value)).toEqual(['txt2img', 'img2img', 'img2vid', 'txt2vid', 'extend']);
    select.value = 'img2vid';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    expect(draft.state.workflow).toBe('img2vid');

    expect(target.querySelector('[data-testid="strip-status"]')?.textContent).toContain('Ready');
    expect(target.querySelector('button[aria-label="Gallery"]')).not.toBeNull();
    expect(target.querySelector('button[aria-label="Back to the Workspace"]')).toBeNull();
  });

  it('keeps the menu and adds an arrow back with the page name on other pages', () => {
    const navigate = vi.spyOn(router, 'navigate').mockImplementation(() => {});
    app = flushSync(() => mount(TopStrip, { target, props: { currentPage: 'gallery', onmenu: vi.fn() } }));

    expect(target.querySelector('button[aria-label="Menu"]')).not.toBeNull();
    expect(target.querySelector('h1')?.textContent).toBe('Gallery');
    expect(target.querySelector('select')).toBeNull();
    (target.querySelector('button[aria-label="Back to the Workspace"]') as HTMLButtonElement).click();
    expect(navigate).toHaveBeenCalledWith('workspace');
    navigate.mockRestore();
  });

  it('lists four workflows until the contract is known', () => {
    app = flushSync(() => mount(TopStrip, { target, props: { currentPage: 'workspace', workflows: null, onmenu: vi.fn() } }));
    const select = target.querySelector('select[aria-label="Workflow"]') as HTMLSelectElement;
    expect(select.options).toHaveLength(4);
  });
});
