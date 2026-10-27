import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { JobLogEntry, JobLogPage } from '$lib/types';

import { toasts } from '$lib/state/toasts.svelte';
import { mountComponent, settle } from '../../test-utils/mount';
import { reactiveProps } from '../../test-utils/reactiveProps.svelte';
import JobLogDrawer from './JobLogDrawer.svelte';

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
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => vi.unstubAllGlobals());

  async function openDrawer(): Promise<{ open: boolean }> {
    const props = reactiveProps<{ open: boolean }>({ open: false });
    mountComponent(JobLogDrawer, props);
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
    mountComponent(JobLogDrawer, { open: false });
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
    expect(document.querySelector('[data-testid="job-log-list"]')?.textContent?.trim()).not.toBe('');
  });

  it('expands a failed row to its error text, and offers no expander on a row without error detail', async () => {
    fetchMock.mockResolvedValue(okResponse(pageOf([entry({ status: 'failed', error: 'CUDA out of memory' }), entry({ job_id: 'job-ok' })])));
    await openDrawer();
    expect(rows()[1].querySelector('button[aria-expanded]')).toBeNull();

    expect(document.querySelector('[data-testid="job-log-detail"]')).toBeNull();
    const toggle = rows()[0].querySelector('button[aria-expanded]') as HTMLButtonElement;
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    toggle.click();
    flushSync();
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(document.querySelector('[data-testid="job-log-detail"]')?.textContent).toContain('CUDA out of memory');
  });

  it('shows how many runs of a job failed', async () => {
    fetchMock.mockResolvedValue(okResponse(pageOf([entry({ status: 'cancelled', failed_runs: 2 })])));
    await openDrawer();

    (rows()[0].querySelector('button[aria-expanded]') as HTMLButtonElement).click();
    flushSync();
    expect(document.querySelector('[data-testid="job-log-detail"]')?.textContent).toContain('2 runs failed');
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

  it('ignores a slow earlier response that arrives after a later one', async () => {
    let releaseFirst: (value: unknown) => void = () => {};
    fetchMock.mockImplementationOnce(() => new Promise((resolve) => { releaseFirst = resolve; }));
    fetchMock.mockResolvedValueOnce(okResponse(pageOf([entry({ job_id: 'failed-1', status: 'failed', error: 'boom' })])));
    const props = reactiveProps<{ open: boolean }>({ open: false });
    mountComponent(JobLogDrawer, props);
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

  describe('row actions', () => {
    const notFound = { ok: false, status: 404, statusText: 'Not Found', text: async () => '{"detail":"No such job."}' };
    const rowButton = (row: HTMLElement, name: string): HTMLButtonElement | undefined =>
      Array.from(row.querySelectorAll<HTMLButtonElement>('button')).find((b) => b.textContent?.trim() === name);
    const methodCalls = (method: string) => fetchMock.mock.calls.filter((call) => (call[1] as RequestInit | undefined)?.method === method);

    beforeEach(() => {
      toasts.splice(0, toasts.length);
    });

    afterEach(() => {
      Object.defineProperty(navigator, 'clipboard', { value: undefined, configurable: true });
    });

    it('hands the entry to the owner on Load settings and hides it for an entry without settings', async () => {
      const withSettings = entry({ job_id: 'job-s', settings: { model: 'zit', steps: '9' } });
      fetchMock.mockResolvedValue(okResponse(pageOf([withSettings, entry({ job_id: 'job-up', workflow: 'upscale', settings: {} })])));
      const onloadsettings = vi.fn();
      const props = reactiveProps<{ open: boolean; onloadsettings: (e: JobLogEntry) => void }>({ open: false, onloadsettings });
      mountComponent(JobLogDrawer, props);
      flushSync(() => { props.open = true; });
      await settle();

      expect(rowButton(rows()[1], 'Load settings')).toBeUndefined();
      rowButton(rows()[0], 'Load settings')!.click();
      expect(onloadsettings).toHaveBeenCalledOnce();
      expect(onloadsettings.mock.calls[0][0]).toMatchObject({ job_id: 'job-s' });
    });

    it('deletes one entry without asking and refetches the page', async () => {
      fetchMock.mockImplementation(async (_url: string, init?: RequestInit) => (init?.method === 'DELETE' ? okResponse({ deleted: 'job-2' }) : okResponse(pageOf([entry({ job_id: 'job-2' })], { page: 2, total_pages: 2, total_count: 51 }))));
      await openDrawer();
      button('Older').click();
      await settle();
      const before = fetchMock.mock.calls.length;

      rowButton(rows()[0], 'Delete')!.click();
      await settle();

      expect(document.querySelector('[role="alertdialog"]')).toBeNull();
      const deletes = methodCalls('DELETE');
      expect(deletes).toHaveLength(1);
      expect(String(deletes[0][0])).toBe('/api/jobs/log/job-2');
      const refetch = new URL(String(fetchMock.mock.calls[before + 1][0]), 'http://localhost');
      expect(refetch.pathname).toBe('/api/jobs/log');
      expect(refetch.searchParams.get('page')).toBe('2');
    });

    it('reports a failed delete, and refetches when the entry was already gone', async () => {
      fetchMock.mockImplementation(async (_url: string, init?: RequestInit) => (init?.method === 'DELETE' ? notFound : okResponse(pageOf([entry()]))));
      await openDrawer();
      const before = fetchMock.mock.calls.length;

      rowButton(rows()[0], 'Delete')!.click();
      await settle();

      expect(toasts.map((toast) => [toast.type, toast.message])).toContainEqual(['error', 'No such job.']);
      expect(fetchMock.mock.calls.length).toBe(before + 2);
    });

    it('copies the error of a failed row and says so once', async () => {
      const writeText = vi.fn().mockResolvedValue(undefined);
      vi.stubGlobal('isSecureContext', true);
      Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
      fetchMock.mockResolvedValue(okResponse(pageOf([entry({ status: 'failed', error: 'CUDA out of memory' }), entry({ job_id: 'job-ok' })])));
      await openDrawer();

      expect(rowButton(rows()[1], 'Copy error')).toBeUndefined();
      rowButton(rows()[0], 'Copy error')!.click();
      await settle();

      expect(writeText).toHaveBeenCalledWith('CUDA out of memory');
      expect(toasts.map((toast) => toast.type)).toEqual(['success']);
    });

    it('selects the error instead when the clipboard is unavailable', async () => {
      vi.stubGlobal('isSecureContext', false);
      fetchMock.mockResolvedValue(okResponse(pageOf([entry({ status: 'failed', error: 'CUDA out of memory' })])));
      await openDrawer();

      rowButton(rows()[0], 'Copy error')!.click();
      await settle();

      expect(window.getSelection()?.toString()).toBe('CUDA out of memory');
      expect(toasts.map((toast) => toast.type)).toEqual(['info']);
    });

    const clearButton = (): HTMLButtonElement => button('Clear log');
    const dialogButton = (action: string): HTMLButtonElement => document.querySelector(`[role="alertdialog"] [data-action="${action}"]`)!;

    it('clears nothing when the confirmation is cancelled', async () => {
      fetchMock.mockResolvedValue(okResponse(pageOf([entry(), entry({ job_id: 'job-2' })])));
      await openDrawer();

      clearButton().click();
      await settle();
      expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain('Clear the whole job log?');
      expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain('2');
      dialogButton('cancel').click();
      await settle();

      expect(methodCalls('DELETE')).toHaveLength(0);
    });

    it('clears the log after confirmation and shows the first page again', async () => {
      let cleared = false;
      fetchMock.mockImplementation(async (_url: string, init?: RequestInit) => {
        if (init?.method === 'DELETE') { cleared = true; return okResponse({ deleted_count: 2 }); }
        return okResponse(cleared ? pageOf([]) : pageOf([entry(), entry({ job_id: 'job-2' })]));
      });
      await openDrawer();

      clearButton().click();
      await settle();
      dialogButton('confirm').click();
      await settle();

      const deletes = methodCalls('DELETE');
      expect(deletes).toHaveLength(1);
      expect(String(deletes[0][0])).toBe('/api/jobs/log');
      expect(rows()).toHaveLength(0);
      expect(requestedUrls().at(-1)!.searchParams.get('page')).toBe('1');
      expect(toasts.map((toast) => toast.type)).toEqual(['success']);
    });

    it('enables Clear log only on the All filter and when there are entries', async () => {
      fetchMock.mockResolvedValue(okResponse(pageOf([entry()])));
      await openDrawer();
      expect(clearButton().disabled).toBe(false);

      button('Failed').click();
      await settle();
      expect(clearButton().disabled).toBe(true);

      fetchMock.mockResolvedValue(okResponse(pageOf([])));
      button('All').click();
      await settle();
      expect(clearButton().disabled).toBe(true);
    });
  });
});
