// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { jobStore } from '$lib/state/job.svelte';
import type { Workflow } from '$lib/types';

import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';
import TopNav from './TopNav.svelte';

const BASE: Workflow[] = ['txt2img', 'img2img', 'txt2vid', 'img2vid'];

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

describe('TopNav workflow tabs', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.remove();
    document.body.innerHTML = '';
  });

  function mountNav(props: Record<string, unknown>): void {
    app = flushSync(() => mount(TopNav, { target, props: { currentPage: 'workspace', ...props } }));
  }

  function workflowTabs(): string[] {
    return Array.from(target.querySelectorAll('nav[aria-label="Workflow"] button')).map((tab) => tab.textContent?.trim() ?? '');
  }

  it('lists the four workflows until the contract is known, then adds Extend Video after Text to Video when it lists extend', () => {
    const props = reactiveProps<{ currentPage: string; workflows: Workflow[] | null }>({ currentPage: 'workspace', workflows: null });
    app = flushSync(() => mount(TopNav, { target, props }));
    expect(workflowTabs()).toEqual(['Text to Image', 'Image to Image', 'Image to Video', 'Text to Video']);

    flushSync(() => { props.workflows = [...BASE, 'extend']; });
    expect(workflowTabs()).toEqual(['Text to Image', 'Image to Image', 'Image to Video', 'Text to Video', 'Extend Video']);
  });

  it('has no Extend Video tab when the contract does not list extend', () => {
    mountNav({ workflows: BASE });
    expect(workflowTabs()).not.toContain('Extend Video');
    expect(workflowTabs()).toHaveLength(4);
  });

  it('switches the draft to the extend workflow and marks the tab pressed', () => {
    mountNav({ workflows: [...BASE, 'extend'] });
    const tab = Array.from(target.querySelectorAll<HTMLButtonElement>('nav[aria-label="Workflow"] button')).find((button) => button.textContent?.trim() === 'Extend Video')!;
    expect(tab.getAttribute('aria-pressed')).toBe('false');

    flushSync(() => tab.click());
    expect(draft.state.workflow).toBe('extend');
    expect(tab.getAttribute('aria-pressed')).toBe('true');
  });

  it('shows no workflow tabs outside the Workspace', () => {
    mountNav({ currentPage: 'gallery', workflows: [...BASE, 'extend'] });
    expect(target.querySelector('nav[aria-label="Workflow"]')).toBeNull();
  });
});
