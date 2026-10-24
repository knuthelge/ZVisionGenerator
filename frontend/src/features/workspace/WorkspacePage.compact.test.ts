// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { historyStore } from '$lib/state/history.svelte';
import { jobStore } from '$lib/state/job.svelte';
import type { JobSnapshot, WorkspaceContext } from '$lib/types';

vi.mock('$lib/state/layoutMode.svelte', () => ({
  layout: { compact: true, mode: 'compact', coarse: true },
}));

const workspaceApiMocks = vi.hoisted(() => ({
  getWorkspaceCoreContext: vi.fn<() => Promise<WorkspaceContext>>(),
  parseUrlPrefill: vi.fn<() => Record<string, string>>(() => ({})),
  listJobs: vi.fn(async () => ({ active_job: null as JobSnapshot | null, queued_jobs: [] as JobSnapshot[] })),
}));

vi.mock('$lib/api/workspace', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/workspace')>();
  return { ...actual, ...workspaceApiMocks };
});

import WorkspacePage from './WorkspacePage.svelte';

let target: HTMLDivElement;
let app: Record<string, unknown> | null = null;

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await Promise.resolve();
  flushSync();
}

describe('WorkspacePage on a phone', () => {
  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    draft.reset();
    historyStore.seedHistory([]);
    jobStore.clearJob();
    workspaceApiMocks.getWorkspaceCoreContext.mockReset();
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

  it('draws the settings strip and the composer before the context answers, as placeholders', async () => {
    draft.update('workflow', 'txt2vid');
    workspaceApiMocks.getWorkspaceCoreContext.mockImplementation(() => new Promise<WorkspaceContext>(() => undefined));

    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();

    const strip = target.querySelector('[data-testid="settings-strip"]') as HTMLElement | null;
    expect(strip).not.toBeNull();
    expect(strip!.getAttribute('aria-busy')).toBe('true');
    const chips = Array.from(strip!.querySelectorAll<HTMLButtonElement>('[data-chip]'));
    expect(chips.map((chip) => chip.dataset.chip)).toEqual(['preset', 'seconds', 'steps', 'guidance', 'seed', 'size', 'audio', 'loras', 'model']);
    expect(chips.every((chip) => chip.disabled)).toBe(true);
    expect((strip!.querySelector('[aria-label="All settings"]') as HTMLButtonElement).disabled).toBe(true);

    expect(target.querySelector('[data-testid="composer"]')).not.toBeNull();
    expect((target.querySelector('#ws-submit-compact') as HTMLButtonElement).disabled).toBe(true);
    expect((target.querySelector('.composer-summary') as HTMLButtonElement).disabled).toBe(true);
  });
});
