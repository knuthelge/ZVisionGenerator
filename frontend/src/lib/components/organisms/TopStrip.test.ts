// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { router } from '$lib/state/router.svelte';
import TopStrip from './TopStrip.svelte';
import { optionValues, pickOption } from '../../../test-utils/select';

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

    const select = target.querySelector('[role="combobox"][aria-label="Workflow"]') as HTMLButtonElement;
    expect(optionValues(select)).toEqual(['txt2img', 'img2img', 'img2vid', 'txt2vid', 'extend']);
    pickOption(select, 'img2vid');
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
    expect(target.querySelector('[role="combobox"]')).toBeNull();
    (target.querySelector('button[aria-label="Back to the Workspace"]') as HTMLButtonElement).click();
    expect(navigate).toHaveBeenCalledWith('workspace');
    navigate.mockRestore();
  });

  it.each(['workspace', 'gallery'] as const)('opens the job log from the strip on %s', (currentPage) => {
    const onjoblog = vi.fn();
    app = flushSync(() => mount(TopStrip, { target, props: { currentPage, onmenu: vi.fn(), onjoblog } }));
    (target.querySelector('button[aria-label="Job log"]') as HTMLButtonElement).click();
    expect(onjoblog).toHaveBeenCalledOnce();
  });

  it('lists four workflows until the contract is known', () => {
    app = flushSync(() => mount(TopStrip, { target, props: { currentPage: 'workspace', workflows: null, onmenu: vi.fn() } }));
    const select = target.querySelector('[role="combobox"][aria-label="Workflow"]') as HTMLButtonElement;
    expect(optionValues(select)).toHaveLength(4);
  });
});
