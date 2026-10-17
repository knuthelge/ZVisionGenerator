// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryAsset } from '$lib/types';
import HistoryStrip from './HistoryStrip.svelte';

class MockIntersectionObserver {
  static instances: MockIntersectionObserver[] = [];

  observe = vi.fn();
  disconnect = vi.fn();

  constructor(
    private readonly callback: IntersectionObserverCallback,
    readonly options?: IntersectionObserverInit
  ) {
    MockIntersectionObserver.instances.push(this);
  }

  trigger(isIntersecting: boolean): void {
    this.callback([{ isIntersecting } as IntersectionObserverEntry], this as unknown as IntersectionObserver);
  }
}

function makeAsset(id: string): GalleryAsset {
  return {
    id,
    url: `/media/${id}`,
    thumbnail_url: `/media/${id}`,
    filename: id,
    created_at: '2026-04-24T12:00:00Z',
    workflow: 'txt2img',
    prompt: 'A calm shoreline at dusk',
    model: 'zit',
    reuse_workspace_url: '#/workspace',
    has_reusable_config: true,
    media_type: 'image',
  };
}

const noop = (): void => {};
const handlers = { onpreview: noop, onreuse: noop, onreference: noop, onextend: noop, onupscale: noop, ondelete: noop };

let target: HTMLDivElement;
let app: unknown = null;

beforeEach(() => {
  MockIntersectionObserver.instances = [];
  globalThis.IntersectionObserver = MockIntersectionObserver as unknown as typeof IntersectionObserver;
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(async () => {
  if (app) { await unmount(app); app = null; }
  target.remove();
});

function render(props: Record<string, unknown>): void {
  app = flushSync(() => mount(HistoryStrip, { target, props: { assets: [makeAsset('a.png')], ...handlers, ...props } }));
}

describe('HistoryStrip infinite scroll', () => {
  it('loads more when the end of the strip comes into view', () => {
    const onloadmore = vi.fn();
    render({ hasMore: true, totalCount: 40, onloadmore });

    const observer = MockIntersectionObserver.instances.at(-1);
    expect(observer?.options?.root).toBe(target.querySelector('#ws-history-scroll'));
    observer?.trigger(false);
    expect(onloadmore).not.toHaveBeenCalled();
    observer?.trigger(true);
    expect(onloadmore).toHaveBeenCalledTimes(1);
  });

  it('counts every history asset, not only the loaded ones', () => {
    render({ hasMore: true, totalCount: 40, onloadmore: noop });
    expect(target.querySelector('#ws-history-toggle')?.textContent).toContain('40');
  });

  it('stops watching once everything is loaded or while a page is loading', () => {
    render({ hasMore: false, totalCount: 1, onloadmore: noop });
    expect(MockIntersectionObserver.instances).toHaveLength(0);
    unmount(app);

    app = flushSync(() => mount(HistoryStrip, { target, props: { assets: [makeAsset('a.png')], ...handlers, hasMore: true, loadingMore: true } }));
    expect(MockIntersectionObserver.instances).toHaveLength(0);
  });
});
