// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { JobLogEntry, JobLogPage } from '$lib/types';

import { reactiveProps } from '../../test-utils/reactiveProps.svelte';
import JobLogDrawer from './JobLogDrawer.svelte';

async function settle(): Promise<void> {
  for (let i = 0; i < 4; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
    flushSync();
  }
}

function entry(overrides: Partial<JobLogEntry> = {}): JobLogEntry {
  return {
    job_id: 'job-1',
    job_type: 'generate',
    workflow: 'txt2img',
    model: 'zit',
    prompt: 'a red fox\nsecond line',
    settings: {},
    status: 'completed',
    created_at: Date.now() / 1000 - 600,
    started_at: Date.now() / 1000 - 590,
    completed_at: Date.now() / 1000 - 540,
    duration_secs: 50,
    outputs: [],
    error: null,
    failed_runs: 0,
    ...overrides
  };
}

function pageOf(entries: JobLogEntry[], overrides: Partial<JobLogPage> = {}): JobLogPage {
  return { entries, page: 1, total_pages: 1, total_count: entries.length, page_size: 50, ...overrides };
}

function okResponse(body: unknown) {
  return { ok: true, status: 200, statusText: 'OK', json: async () => body, text: async () => JSON.stringify(body) };
}

describe('JobLogDrawer', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
    fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.remove();
    document.body.innerHTML = '';
    vi.unstubAllGlobals();
  });

  async function openDrawer(): Promise<{ open: boolean }> {
    const props = reactiveProps<{ open: boolean }>({ open: false });
    app = flushSync(() => mount(JobLogDrawer, { target, props }));
    flushSync(() => { props.open = true; });
    await settle();
    return props;
  }

  const requestedUrls = (): URL[] => fetchMock.mock.calls.map((call) => new URL(String(call[0]), 'http://localhost'));
  const rows = (): HTMLElement[] => Array.from(document.querySelectorAll<HTMLElement>('[data-testid="job-log-row"]'));
  const button = (name: string): HTMLButtonElement => Array.from(document.querySelectorAll<HTMLButtonElement>('button')).find((b) => b.textContent?.trim() === name)!;

  it('fetches the first page of every outcome on open and lists the entries', async () => {
    fetchMock.mockResolvedValue(okResponse(pageOf([entry(), entry({ job_id: 'job-2', status: 'failed', error: 'boom' })])));
    await openDrawer();

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = requestedUrls()[0];
    expect(url.pathname).toBe('/api/jobs/log');
    expect(url.searchParams.get('page')).toBe('1');
    expect(url.searchParams.get('status')).toBe('all');
    expect(rows().map((row) => row.dataset.tone)).toEqual(['completed', 'failed']);
    expect(rows()[0].textContent).toContain('a red fox');
    expect(rows()[0].textContent).not.toContain('second line');
  });

  it('does not fetch while closed', async () => {
    app = flushSync(() => mount(JobLogDrawer, { target, props: { open: false } }));
    await settle();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(document.querySelector('[role="dialog"]')).toBeNull();
  });

  it('refetches page 1 with the chosen status when the filter changes', async () => {
    fetchMock.mockResolvedValue(okResponse(pageOf([entry()])));
    await openDrawer();

    button('Failed').click();
    await settle();
    let url = requestedUrls().at(-1)!;
    expect(url.searchParams.get('status')).toBe('failed');
    expect(url.searchParams.get('page')).toBe('1');

    button('Completed').click();
    await settle();
    url = requestedUrls().at(-1)!;
    expect(url.searchParams.get('status')).toBe('completed');
  });

  it('pages with Older and Newer', async () => {
    fetchMock.mockImplementation(async (input: string) => {
      const page = Number(new URL(input, 'http://localhost').searchParams.get('page'));
      return okResponse(pageOf([entry({ job_id: `job-p${page}` })], { page, total_pages: 3, total_count: 120 }));
    });
    await openDrawer();
    expect(document.body.textContent).toContain('Page 1 of 3');
    expect(button('Newer').disabled).toBe(true);

    button('Older').click();
    await settle();
    expect(requestedUrls().at(-1)!.searchParams.get('page')).toBe('2');
    expect(document.body.textContent).toContain('Page 2 of 3');

    button('Newer').click();
    await settle();
    expect(requestedUrls().at(-1)!.searchParams.get('page')).toBe('1');
  });

  it('shows an empty state when nothing is logged', async () => {
    fetchMock.mockResolvedValue(okResponse(pageOf([])));
    await openDrawer();
    expect(rows()).toHaveLength(0);
    expect(document.querySelector('.ui-empty')).not.toBeNull();
  });

  it('expands a failed row to its error text', async () => {
    fetchMock.mockResolvedValue(okResponse(pageOf([entry({ status: 'failed', error: 'CUDA out of memory' })])));
    await openDrawer();

    expect(document.querySelector('[data-testid="job-log-detail"]')).toBeNull();
    const toggle = rows()[0].querySelector('button[aria-expanded]') as HTMLButtonElement;
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    toggle.click();
    flushSync();
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(document.querySelector('[data-testid="job-log-detail"]')?.textContent).toContain('CUDA out of memory');
  });

  it('marks a job with failed runs as partial and shows how many failed', async () => {
    fetchMock.mockResolvedValue(okResponse(pageOf([entry({ status: 'cancelled', failed_runs: 2 })])));
    await openDrawer();

    expect(rows()[0].dataset.tone).toBe('partial');
    (rows()[0].querySelector('button[aria-expanded]') as HTMLButtonElement).click();
    flushSync();
    expect(document.querySelector('[data-testid="job-log-detail"]')?.textContent).toContain('2 runs failed');
  });

  it('offers no expander on a row without error detail', async () => {
    fetchMock.mockResolvedValue(okResponse(pageOf([entry()])));
    await openDrawer();
    expect(rows()[0].querySelector('button[aria-expanded]')).toBeNull();
  });

  it('reports a failed load and retries it', async () => {
    fetchMock.mockResolvedValueOnce({ ok: false, status: 503, statusText: 'Unavailable', text: async () => '{"detail":"The job log cannot be read."}' });
    await openDrawer();
    expect(document.querySelector('[role="alert"]')?.textContent).toContain('The job log cannot be read.');

    fetchMock.mockResolvedValueOnce(okResponse(pageOf([entry()])));
    button('Retry').click();
    await settle();
    expect(rows()).toHaveLength(1);
    expect(document.querySelector('[role="alert"]')).toBeNull();
  });

  it('offers no retry when the server rejects the request', async () => {
    fetchMock.mockResolvedValueOnce({ ok: false, status: 422, statusText: 'Unprocessable', text: async () => '{"detail":"Bad page."}' });
    await openDrawer();
    expect(document.querySelector('[role="alert"]')?.textContent).toContain('Bad page.');
    expect(Array.from(document.querySelectorAll('button')).some((b) => b.textContent?.trim() === 'Retry')).toBe(false);
  });

  it('ignores a slow earlier response that arrives after a later one', async () => {
    let releaseFirst: (value: unknown) => void = () => {};
    fetchMock.mockImplementationOnce(() => new Promise((resolve) => { releaseFirst = resolve; }));
    fetchMock.mockResolvedValueOnce(okResponse(pageOf([entry({ job_id: 'failed-1', status: 'failed', error: 'boom' })])));
    const props = reactiveProps<{ open: boolean }>({ open: false });
    app = flushSync(() => mount(JobLogDrawer, { target, props }));
    flushSync(() => { props.open = true; });
    await settle();

    button('Failed').click();
    await settle();
    expect(rows().map((row) => row.dataset.tone)).toEqual(['failed']);

    releaseFirst(okResponse(pageOf([entry({ job_id: 'stale-1' }), entry({ job_id: 'stale-2' })])));
    await settle();
    expect(rows().map((row) => row.dataset.tone)).toEqual(['failed']);
  });

  it('starts from page 1 of every outcome each time it opens', async () => {
    fetchMock.mockResolvedValue(okResponse(pageOf([entry()])));
    const props = await openDrawer();
    button('Failed').click();
    await settle();

    flushSync(() => { props.open = false; });
    await settle();
    flushSync(() => { props.open = true; });
    await settle();
    const url = requestedUrls().at(-1)!;
    expect(url.searchParams.get('status')).toBe('all');
    expect(url.searchParams.get('page')).toBe('1');
  });
});
