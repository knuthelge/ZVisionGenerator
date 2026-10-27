// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { galleryStore } from '$lib/state/gallery.svelte';
import { jobStore } from '$lib/state/job.svelte';
import { workspaceMemory } from '$lib/state/pageMemory';
import type { JobSnapshot, RecentPrompt, WorkflowMode, WorkspaceContext } from '$lib/types';
import { makeContext } from '../../test-utils/workspace';

vi.mock('$lib/state/layoutMode.svelte', () => ({
  layout: { compact: true, mode: 'compact', coarse: true },
}));

const workspaceApiMocks = vi.hoisted(() => ({
  getWorkspaceCoreContext: vi.fn<() => Promise<WorkspaceContext>>(),
  parseUrlPrefill: vi.fn<() => Record<string, string>>(() => ({})),
  listJobs: vi.fn(async () => ({ active_job: null as JobSnapshot | null, queued_jobs: [] as JobSnapshot[] })),
}));

const recentPromptsMock = vi.hoisted(() => ({
  getRecentPrompts: vi.fn<(query: string, mode: WorkflowMode) => Promise<{ entries: RecentPrompt[] }>>(),
  clearRecentPrompts: vi.fn(),
}));

vi.mock('$lib/api/recentPrompts', () => recentPromptsMock);

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
    workspaceMemory.forget();
    galleryStore.reset();
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

  it('opens the recent prompts as a sheet, with the filter left unfocused so the keyboard stays down', async () => {
    recentPromptsMock.getRecentPrompts.mockResolvedValue({ entries: [{ format: 'text', prompt: 'A red fox', scene: '', negative_prompt: '', mode: 'image', used_at: Date.now() / 1000 - 60 }] });
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(makeContext());
    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();

    (target.querySelector('#ws-recent-prompts') as HTMLButtonElement).click();
    await settle();

    const sheet = document.querySelector('[data-testid="sheet"] [role="dialog"][aria-label="Recent prompts"]');
    expect(sheet).not.toBeNull();
    expect(sheet!.querySelectorAll('[role="option"]')).toHaveLength(1);
    expect(document.activeElement).not.toBe(sheet!.querySelector('input'));

    (sheet!.querySelector('[role="option"]') as HTMLElement).click();
    await settle();
    expect(draft.state.prompt).toBe('A red fox');
  });
});
