// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { JobSnapshot } from '$lib/types';

import QueuePanel from './QueuePanel.svelte';

function makeJob(overrides: Partial<JobSnapshot> = {}): JobSnapshot {
  return {
    id: 'job-1',
    job_id: 'job-1',
    job_type: 'Text to Image',
    workflow: 'txt2img',
    prompt: 'a fox',
    model: 'zit',
    runs: 1,
    created_at: '2026-10-04T12:00:00Z',
    status: 'queued',
    event_count: 1,
    paused: false,
    settings: { prompt: 'a fox' },
    ...overrides,
  };
}

describe('QueuePanel', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.remove();
  });

  function mountPanel(jobs: JobSnapshot[]): void {
    app = flushSync(() => mount(QueuePanel, { target, props: { jobs, onremove: vi.fn(), onload: vi.fn(), onclear: vi.fn() } }));
  }

  function loadButtons(): HTMLButtonElement[] {
    return Array.from(target.querySelectorAll<HTMLButtonElement>('.queue-action'));
  }

  it('offers Load settings for jobs submitted from the form', () => {
    mountPanel([makeJob()]);
    expect(loadButtons()).toHaveLength(1);
  });

  it('labels a queued upscale and offers no settings to load', () => {
    mountPanel([makeJob({ workflow: 'upscale', job_type: 'Upscale', prompt: '', settings: {}, meta: '2× → 1664×2432' })]);

    expect(loadButtons()).toHaveLength(0);
    expect(target.querySelector('.queue-prompt')?.textContent).toBe('No prompt recorded');
    expect(target.querySelector('.queue-meta')?.textContent).toContain('upscale');
    expect(target.querySelector('.queue-meta')?.textContent).toContain('2× → 1664×2432');
  });
});
