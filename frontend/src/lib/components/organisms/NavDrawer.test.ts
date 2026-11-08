// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/state/layoutMode.svelte', () => ({ layout: { compact: true, mode: 'compact', coarse: true } }));
vi.mock('$lib/api/projects', () => ({
  listProjects: vi.fn().mockResolvedValue([{ name: 'Night Bus', path: '/out/Night Bus' }]),
  createProject: vi.fn(),
}));
vi.mock('$lib/api/gallery', () => ({ getGallery: vi.fn().mockResolvedValue({ assets: [], page: 1, total_pages: 1, total_count: 0 }) }));

import { projectPick } from '$lib/state/projectPick.svelte';
import { settle } from '../../../test-utils/mount';
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

  it('names the app Z-Vision for assistive technology, with only decorative drawings beside the name', () => {
    app = flushSync(() => mount(NavDrawer, { target, props: { open: true, currentPage: 'workspace', onnavigate: vi.fn(), onclose: vi.fn() } }));

    const brand = document.body.querySelector('h1') as HTMLElement;
    expect(brand.textContent?.trim()).toBe('Z-Vision');
    const drawings = Array.from(brand.querySelectorAll('svg, img'));
    expect(drawings.length).toBeGreaterThan(0);
    expect(drawings.every((drawing) => drawing.getAttribute('aria-hidden') === 'true' || drawing.getAttribute('alt') === '')).toBe(true);
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

  describe('project row', () => {
    const projectRow = () => document.body.querySelector<HTMLButtonElement>('.drawer-project')!;

    beforeEach(() => {
      localStorage.clear();
      projectPick.set(null);
    });

    it('sits between the brand and the pages, naming the pick', () => {
      projectPick.set('Night Bus');
      app = flushSync(() => mount(NavDrawer, { target, props: { open: true, currentPage: 'workspace', onnavigate: vi.fn(), onclose: vi.fn() } }));

      const drawer = document.body.querySelector('[role="dialog"]')!;
      expect(Array.from(drawer.children).map((child) => child.tagName)).toEqual(['H1', 'BUTTON', 'NAV']);
      expect(projectRow().getAttribute('aria-label')).toBe('Project: Night Bus');
    });

    it('opens the list as a sheet, and a pick changes the project and closes the drawer', async () => {
      const onclose = vi.fn();
      app = flushSync(() => mount(NavDrawer, { target, props: { open: true, currentPage: 'workspace', onnavigate: vi.fn(), onclose } }));

      projectRow().click();
      await settle();
      expect(document.body.querySelector('[data-testid="action-sheet"]')).not.toBeNull();
      expect(onclose).not.toHaveBeenCalled();

      document.body.querySelector<HTMLElement>('[data-action="project:Night Bus"]')!.click();
      await settle();

      expect(projectPick.current).toBe('Night Bus');
      expect(onclose).toHaveBeenCalledOnce();
    });

    it('leaves the drawer open when the sheet closes without a pick', async () => {
      const onclose = vi.fn();
      app = flushSync(() => mount(NavDrawer, { target, props: { open: true, currentPage: 'workspace', onnavigate: vi.fn(), onclose } }));

      projectRow().click();
      await settle();
      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
      await settle();

      expect(projectPick.current).toBeNull();
      expect(onclose).not.toHaveBeenCalled();
      expect(document.body.querySelector('[data-testid="nav-drawer"]')).not.toBeNull();
    });
  });
});
