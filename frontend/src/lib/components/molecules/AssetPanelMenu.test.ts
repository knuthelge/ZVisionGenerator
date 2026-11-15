import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const layoutMock = vi.hoisted(() => ({ compact: false, short: false, mode: 'wide', coarse: false }));

vi.mock('$lib/state/layoutMode.svelte', () => ({ layout: layoutMock }));

import { toasts } from '$lib/state/toasts.svelte';
import type { GalleryAsset } from '$lib/types';
import { mountComponent, settle } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';
import AssetPanelMenu from './AssetPanelMenu.svelte';
import { assetPanel } from './assetMenu';

const image: GalleryAsset = {
  id: 'out/koi.png', url: '/media/out/koi.png', thumbnail_url: '/media/out/koi-thumb.png', filename: 'web_2026-11-11_09-12-33.png', created_at: '2026-11-11T09:12:33Z', project: null,
  workflow: 'txt2img', prompt: 'koi', model: 'zit', media_type: 'image', file_path: '/outputs/out/koi.png', has_reusable_config: true, reuse_workspace_url: '#/workspace', width: 1024, height: 1024,
  upscale: {
    factors: [
      { factor: 2, width: 2048, height: 2048, allowed: true, reason: null, warning: null },
      { factor: 4, width: 4096, height: 4096, allowed: true, reason: null, warning: null },
    ],
    seedvr2: null,
  },
};

const handlers = () => ({ onreuse: vi.fn(), onreference: vi.fn(), onedit: vi.fn(), onupscale: vi.fn(), ondelete: vi.fn(), onstar: vi.fn() });

describe('AssetPanelMenu', () => {
  const menu = () => document.querySelector<HTMLElement>('[role="menu"]');
  const item = (id: string) => document.querySelector<HTMLElement>(`[data-action="${id}"]`);
  const keydown = (key: string) => flushSync(() => menu()?.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true })));
  const order = () => Array.from(menu()?.querySelectorAll<HTMLElement>('[role^="menuitem"]') ?? []).map((el) => el.dataset.action);

  beforeEach(() => {
    Object.assign(layoutMock, { compact: false, short: false, mode: 'wide', coarse: false });
    toasts.splice(0, toasts.length);
  });

  /** Mount the menu on a panel built as a surface builds it: from its handlers, the Use as state, Move and View. */
  function mountPanel({ handlers: on = handlers(), useAs, openMove = vi.fn(), onview, ...rest }: Record<string, unknown> = {}) {
    const props = reactiveProps({
      open: true, anchor: null as HTMLElement | null, asset: image, panel: assetPanel(image, on as ReturnType<typeof handlers>, { useAs: useAs as never, openMove: openMove as () => void, onview: onview as (() => void) | undefined }), onclose: vi.fn(), ...rest,
    });
    mountComponent(AssetPanelMenu, props);
    flushSync();
    return props;
  }

  describe('as a popover', () => {
    it('is a menu named after the file, with each section a named group', () => {
      mountPanel();

      expect(menu()?.getAttribute('aria-label')).toBe(image.filename);
      const groups = Array.from(menu()!.querySelectorAll('[role="group"]')).map((group) => group.getAttribute('aria-label'));
      expect(groups).toHaveLength(3);
      expect(groups.every((name) => Boolean(name))).toBe(true);
      expect(new Set(groups).size).toBe(3);
      expect(item('star')?.getAttribute('role')).toBe('menuitemcheckbox');
      expect(item('star')?.getAttribute('aria-checked')).toBe('false');
      // Use as and Upscale choices carry a complete accessible name of their own, one per choice.
      const names = ['reference-image', 'reference-reference', 'upscale-creative-2', 'upscale-creative-4'].map((id) => item(id)?.getAttribute('aria-label'));
      expect(names.every((name) => Boolean(name))).toBe(true);
      expect(new Set(names).size).toBe(names.length);
    });

    it('lists the actions in reading order: head, Make, choices, then the file row ending in Delete', () => {
      mountPanel({ onview: vi.fn() });

      expect(order()).toEqual([
        'view', 'star', 'reuse', 'edit',
        'reference-image', 'reference-reference', 'reference-video', 'reference-last-frame',
        'upscale-creative-2', 'upscale-creative-4',
        'download', 'move', 'copy-prompt', 'delete',
      ]);
    });

    it('shows an enabled choice\'s note as its second line, and a greyed one\'s reason instead', () => {
      const clip = { ...image, media_type: 'video' as const, extend: { available: true, reason: null } };
      const on = { ...handlers(), onextend: vi.fn() };
      const panel = assetPanel(clip, on);
      mountPanel({ asset: clip, panel });
      const note = panel.make.find((entry) => entry.id === 'extend')!.description;

      expect(note).toBeTruthy();
      expect(item('extend')?.textContent).toContain(note);

      const blocked = assetPanel({ ...clip, extend: { available: false, reason: 'blocked' } }, on);
      document.body.innerHTML = '';
      mountPanel({ asset: clip, panel: blocked });
      expect(item('extend')?.textContent).toContain('blocked');
      expect(item('extend')?.getAttribute('title')).toBe('blocked');
    });

    it('gives an image\'s Reuse its own row, with Edit and Adjust paired under it, and a video\'s Reuse its pair', () => {
      const on = { ...handlers(), onadjust: vi.fn() };
      mountPanel({ handlers: on });
      const row = (id: string) => item(id)?.dataset.row;

      expect(row('reuse')).not.toBe(row('edit'));
      expect(row('edit')).toBe(row('adjust'));
      expect(order().slice(1, 5)).toEqual(['reuse', 'edit', 'adjust', 'reference-image']);

      document.body.innerHTML = '';
      const clip = { ...image, media_type: 'video' as const, extend: { available: true, reason: null } };
      mountPanel({ asset: clip, panel: assetPanel(clip, { ...on, onextend: vi.fn() }) });
      expect(row('reuse')).toBe(row('extend'));
      expect(item('adjust')).toBeNull();
    });

    it('opens on Reuse settings', () => {
      mountPanel({ onview: vi.fn() });

      expect(document.activeElement).toBe(item('reuse'));
    });

    it('moves Up and Down through every item in reading order, wrapping at the ends', () => {
      mountPanel({ onview: vi.fn() });
      const visited: (string | undefined)[] = [];
      for (let step = 0; step < 4; step += 1) {
        keydown('ArrowDown');
        visited.push((document.activeElement as HTMLElement).dataset.action);
      }
      expect(visited).toEqual(['edit', 'reference-image', 'reference-reference', 'reference-video']);

      keydown('ArrowUp');
      expect((document.activeElement as HTMLElement).dataset.action).toBe('reference-reference');
      keydown('End');
      expect((document.activeElement as HTMLElement).dataset.action).toBe('delete');
      keydown('ArrowDown');
      expect((document.activeElement as HTMLElement).dataset.action).toBe('view');
      keydown('Home');
      expect((document.activeElement as HTMLElement).dataset.action).toBe('view');
    });

    it('moves Left and Right within a row only, and stops at its ends', () => {
      mountPanel();

      keydown('ArrowRight');
      expect((document.activeElement as HTMLElement).dataset.action).toBe('edit');
      keydown('ArrowRight');
      expect((document.activeElement as HTMLElement).dataset.action).toBe('edit');
      keydown('ArrowLeft');
      expect((document.activeElement as HTMLElement).dataset.action).toBe('reuse');
      keydown('ArrowLeft');
      expect((document.activeElement as HTMLElement).dataset.action).toBe('reuse');

      item('reference-last-frame')!.focus();
      keydown('ArrowLeft');
      expect((document.activeElement as HTMLElement).dataset.action).toBe('reference-video');
      item('download')!.focus();
      keydown('ArrowRight');
      expect((document.activeElement as HTMLElement).dataset.action).toBe('move');
    });

    it('closes before it runs an item, and stays open for the star', () => {
      const on = handlers();
      const props = mountPanel({ handlers: on });

      flushSync(() => item('star')!.click());
      expect(on.onstar).toHaveBeenCalledWith(image, true);
      expect(props.onclose).not.toHaveBeenCalled();

      flushSync(() => item('delete')!.click());
      expect(props.onclose).toHaveBeenCalledTimes(1);
      expect(on.ondelete).toHaveBeenCalledWith(image);
    });

    it('closes and opens the viewer from View in the head', () => {
      const onview = vi.fn();
      const props = mountPanel({ onview });

      flushSync(() => item('view')!.click());

      expect(props.onclose).toHaveBeenCalledTimes(1);
      expect(onview).toHaveBeenCalledOnce();
    });

    it('keeps a greyed choice reachable, with its reason as its hint, and explains it instead of running it', () => {
      const on = handlers();
      mountPanel({ handlers: on, useAs: { image: { disabled: true, hint: 'Ideogram 4 can\'t use a start image' } } });
      const choice = item('reference-image')!;

      expect(choice.getAttribute('aria-disabled')).toBe('true');
      expect(choice.textContent).toContain('Ideogram 4 can\'t use a start image');
      choice.focus();
      expect(document.activeElement).toBe(choice);
      flushSync(() => choice.click());

      expect(on.onreference).not.toHaveBeenCalled();
      expect(menu()).not.toBeNull();
      expect(toasts.map((toast) => toast.message)).toEqual(['Ideogram 4 can\'t use a start image']);
    });

    it('shows only one section\'s choices, with no head and no file row, for a popover opened from the viewer\'s bar', () => {
      mountPanel({ only: 'upscale' });

      expect(menu()?.getAttribute('aria-label')).toBeTruthy();
      expect(menu()?.getAttribute('aria-label')).not.toBe(image.filename);
      expect(order()).toEqual(['upscale-creative-2', 'upscale-creative-4']);
    });
  });

  describe('as a sheet', () => {
    beforeEach(() => Object.assign(layoutMock, { compact: true, mode: 'compact', coarse: true }));

    it('draws the file and its controls in the sheet\'s head, and the grid and file row in its body', async () => {
      mountPanel({ onview: vi.fn() });
      await settle();

      const head = document.querySelector('[data-testid="sheet"] .sheet-head') as HTMLElement;
      expect(head.querySelector('[data-action="view"]')).not.toBeNull();
      expect(head.querySelector('[data-action="star"]')).not.toBeNull();
      expect(head.querySelector('button[aria-label="Close"]')).not.toBeNull();
      expect(head.hidden).toBe(false);
      expect(document.querySelectorAll('[data-action="star"]')).toHaveLength(1);
      expect(document.querySelector('[data-testid="sheet"] [data-testid="file-panel"] [data-action="reuse"]')).not.toBeNull();
      expect(document.querySelector('[data-testid="file-panel"] [data-action="download"]')).not.toBeNull();
    });

    it('focuses the panel itself for a finger, not the first item', async () => {
      mountPanel();
      await settle();

      expect(document.activeElement).toBe(menu());
    });

    it('keeps the sheet\'s own title and Close for a list of choices held sideways, in the ordinary grid', async () => {
      Object.assign(layoutMock, { short: true });
      mountPanel({ only: 'upscale' });
      await settle();

      const head = document.querySelector('[data-testid="sheet"] .sheet-head') as HTMLElement;
      expect(head).not.toBeNull();
      expect(head.querySelector('button[aria-label="Close"]')).not.toBeNull();
      // The panel draws no Close of its own: the sheet's is the one.
      expect(document.querySelector('[data-testid="file-panel"] button[aria-label="Close"]')).toBeNull();
    });

    it('holds the panel\'s own head, with Close, in its grid when held sideways, and has no sheet head row', async () => {
      Object.assign(layoutMock, { short: true });
      const props = mountPanel({ onview: vi.fn() });
      await settle();

      expect(document.querySelector('[data-testid="sheet"] .sheet-head')).toBeNull();
      const inGrid = document.querySelector('[data-testid="file-panel"] .fp-head-cell') as HTMLElement;
      expect(inGrid.querySelector('[data-action="star"]')).not.toBeNull();
      expect(inGrid.querySelector('button[aria-label="Close"]')).not.toBeNull();
      expect(document.querySelectorAll('[data-action="star"]')).toHaveLength(1);
      flushSync(() => (inGrid.querySelector('button[aria-label="Close"]') as HTMLElement).click());
      expect(props.onclose).toHaveBeenCalledOnce();
    });
  });
});
