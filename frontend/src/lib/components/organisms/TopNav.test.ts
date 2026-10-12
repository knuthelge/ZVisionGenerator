// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { jobStore } from '$lib/state/job.svelte';

import TopNav from './TopNav.svelte';

describe('TopNav', () => {
  let target: HTMLDivElement;
  let component: Record<string, unknown> | null = null;

  beforeEach(() => {
    sessionStorage.clear();
    jobStore.clearJob();
    target = document.createElement('div');
    document.body.appendChild(target);
    component = mount(TopNav, { target, props: { currentPage: 'gallery' } });
    flushSync();
  });

  afterEach(() => {
    if (component) unmount(component);
    component = null;
    jobStore.clearJob();
    document.body.innerHTML = '';
  });

  const workspaceTab = () => Array.from(target.querySelectorAll('.main-tabs .nav-tab')).find((tab) => tab.textContent?.includes('Workspace'))!;
  const source = () => (globalThis.EventSource as unknown as { lastInstance: { emit: (type: string, data: unknown) => void } }).lastInstance;

  it('badges the Workspace while a render waits for review, on any page, and clears it after the decision', () => {
    jobStore.startJob({ job_id: 'job-nav', workflow: 'txt2img', prompt: 'P', model: 'zit', runs: 1, created_at: '' });
    flushSync();
    expect(workspaceTab().querySelector('.ui-badge')).toBeNull();

    source().emit('job_paused', { type: 'job_paused', review: { iteration: 1 }, preview_version: 1 });
    flushSync();
    expect(workspaceTab().querySelector('.ui-badge')?.textContent).toBe('Review');
    expect(target.querySelectorAll('.ui-badge')).toHaveLength(1);

    source().emit('job_resumed', { type: 'job_resumed' });
    flushSync();
    expect(workspaceTab().querySelector('.ui-badge')).toBeNull();
  });

  it('badges the Workspace for a video review, which has no preview version', () => {
    jobStore.startJob({ job_id: 'job-nav-video', workflow: 'txt2vid', prompt: 'P', model: 'ltx', runs: 1, created_at: '' });
    source().emit('job_paused', { type: 'job_paused', mode: 'video', review: { iteration: 1 } });
    flushSync();
    expect(workspaceTab().querySelector('.ui-badge')?.textContent).toBe('Review');

    source().emit('job_resumed', { type: 'job_resumed' });
    flushSync();
    expect(workspaceTab().querySelector('.ui-badge')).toBeNull();
  });

  it('does not badge a plain pause', () => {
    jobStore.startJob({ job_id: 'job-nav', workflow: 'txt2img', prompt: 'P', model: 'zit', runs: 1, created_at: '' });
    source().emit('job_paused', { type: 'job_paused' });
    flushSync();
    expect(target.querySelector('.ui-badge')).toBeNull();
  });
});
