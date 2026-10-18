import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

type Listener = (event: { matches: boolean }) => void;

/** A controllable matchMedia: each query has a `matches` flag and change listeners the test can fire. */
function installMatchMedia(initial: Record<string, boolean>): { set: (query: string, matches: boolean) => void } {
  const lists = new Map<string, { matches: boolean; listeners: Set<Listener> }>();
  const entry = (query: string) => {
    let list = lists.get(query);
    if (!list) {
      list = { matches: initial[query] ?? false, listeners: new Set() };
      lists.set(query, list);
    }
    return list;
  };
  vi.stubGlobal('matchMedia', (query: string) => {
    const list = entry(query);
    return {
      get matches() { return list.matches; },
      media: query,
      addEventListener: (_type: string, listener: Listener) => { list.listeners.add(listener); },
      removeEventListener: (_type: string, listener: Listener) => { list.listeners.delete(listener); },
    };
  });
  return {
    set(query, matches) {
      const list = entry(query);
      list.matches = matches;
      list.listeners.forEach((listener) => listener({ matches }));
    },
  };
}

describe('layoutModeFor', () => {
  it('picks compact over tall, then tall, then wide', async () => {
    const { layoutModeFor } = await import('./layoutMode.svelte');
    expect(layoutModeFor({ compact: true, tall: true })).toBe('compact');
    expect(layoutModeFor({ compact: false, tall: true })).toBe('tall');
    expect(layoutModeFor({ compact: false, tall: false })).toBe('wide');
  });
});

describe('layout store', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('is wide with a fine pointer when no query matches', async () => {
    installMatchMedia({});
    const { layout } = await import('./layoutMode.svelte');
    expect(layout.mode).toBe('wide');
    expect(layout.compact).toBe(false);
    expect(layout.coarse).toBe(false);
  });

  it('follows the queries as they change', async () => {
    const media = installMatchMedia({});
    const { layout, COMPACT_QUERY, TALL_QUERY, COARSE_QUERY } = await import('./layoutMode.svelte');

    media.set(TALL_QUERY, true);
    expect(layout.mode).toBe('tall');

    media.set(COMPACT_QUERY, true);
    expect(layout.mode).toBe('compact');
    expect(layout.compact).toBe(true);

    media.set(COMPACT_QUERY, false);
    media.set(TALL_QUERY, false);
    expect(layout.mode).toBe('wide');

    media.set(COARSE_QUERY, true);
    expect(layout.coarse).toBe(true);
  });

  it('stays wide without matchMedia, as in a test or server environment', async () => {
    vi.stubGlobal('matchMedia', undefined);
    const { layout } = await import('./layoutMode.svelte');
    expect(layout.mode).toBe('wide');
    expect(layout.coarse).toBe(false);
  });
});
