import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import type { DownloadItem, DownloadPlan } from '$lib/types';

const api = vi.hoisted(() => ({
  getDownloadPlan: vi.fn(),
  startDownload: vi.fn(),
  getDownloads: vi.fn(),
  cancelDownload: vi.fn(),
}));
const confirm = vi.hoisted(() => ({ requestConfirm: vi.fn() }));
const toasts = vi.hoisted(() => ({ addToast: vi.fn(() => 'toast-1') }));

vi.mock('$lib/api/models', () => api);
vi.mock('$lib/components/molecules/confirm.svelte', () => confirm);
vi.mock('./toasts.svelte', () => toasts);

import { downloadStore } from './downloads.svelte';

const plan = (overrides: Partial<DownloadPlan> = {}): DownloadPlan => ({
  kind: 'model',
  name: 'klein9b',
  label: 'klein9b',
  parts: [{ repo: 'org/klein', files: 4, bytes: 34.7e9 }],
  total_bytes: 34.7e9,
  space: { path: '/hub', needed_bytes: 34.7e9, free_bytes: 512e9, enough: true },
  ...overrides,
});
const item = (overrides: Partial<DownloadItem> = {}): DownloadItem => ({ id: 'd1', kind: 'model', name: 'klein9b', label: 'klein9b', state: 'queued', message: null, ...overrides });
const running = (overrides: Partial<DownloadItem> = {}): DownloadItem => item({ state: 'running', bytes_done: 1e9, bytes_total: 2e9, ...overrides });

async function advance(ms = 0): Promise<void> {
  await vi.advanceTimersByTimeAsync(ms);
}

describe('downloadStore', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    downloadStore.reset();
    for (const mock of [...Object.values(api), confirm.requestConfirm, toasts.addToast]) mock.mockReset();
    toasts.addToast.mockReturnValue('toast-1');
    api.getDownloads.mockResolvedValue({ items: [] });
  });

  afterEach(() => {
    downloadStore.reset();
    vi.useRealTimers();
  });

  describe('startSync', () => {
    it('reads the lane once per tick however many pages sync, and stops when nothing is active', async () => {
      api.getDownloads.mockResolvedValue({ items: [running()] });
      const stopA = downloadStore.startSync();
      const stopB = downloadStore.startSync();
      await advance();
      expect(api.getDownloads).toHaveBeenCalledTimes(1);

      await advance(3000);
      expect(api.getDownloads).toHaveBeenCalledTimes(4);

      api.getDownloads.mockResolvedValue({ items: [item({ state: 'done' })] });
      await advance(1000);
      const settled = api.getDownloads.mock.calls.length;
      await advance(5000);
      expect(api.getDownloads).toHaveBeenCalledTimes(settled);
      stopA();
      stopB();
    });

    it('keeps polling until the last page stops syncing', async () => {
      api.getDownloads.mockResolvedValue({ items: [running()] });
      const stopA = downloadStore.startSync();
      const stopB = downloadStore.startSync();
      await advance();

      stopA();
      const before = api.getDownloads.mock.calls.length;
      await advance(2000);
      expect(api.getDownloads.mock.calls.length).toBe(before + 2);

      stopB();
      const after = api.getDownloads.mock.calls.length;
      await advance(5000);
      expect(api.getDownloads).toHaveBeenCalledTimes(after);
    });

    it('reads again on window focus while syncing, and not after the last cleanup', async () => {
      const stop = downloadStore.startSync();
      await advance();
      window.dispatchEvent(new Event('focus'));
      await advance();
      expect(api.getDownloads).toHaveBeenCalledTimes(2);

      stop();
      window.dispatchEvent(new Event('focus'));
      await advance();
      expect(api.getDownloads).toHaveBeenCalledTimes(2);
    });

    it('does not poll while nothing is active', async () => {
      api.getDownloads.mockResolvedValue({ items: [item({ state: 'done' })] });
      const stop = downloadStore.startSync();
      await advance(5000);
      expect(api.getDownloads).toHaveBeenCalledTimes(1);
      stop();
    });
  });

  describe('transitions', () => {
    it('toasts and notifies once when a running download finishes', async () => {
      const onDownloaded = vi.fn();
      downloadStore.onDownloaded(onDownloaded);
      api.getDownloads.mockResolvedValue({ items: [running({ label: 'Klein 9B' })] });
      const stop = downloadStore.startSync();
      await advance();
      expect(downloadStore.active('model', 'klein9b')?.state).toBe('running');

      api.getDownloads.mockResolvedValue({ items: [item({ state: 'done', label: 'Klein 9B' })] });
      await advance(1000);
      await advance(3000);

      expect(onDownloaded).toHaveBeenCalledTimes(1);
      expect(toasts.addToast).toHaveBeenCalledWith('Klein 9B downloaded', 'success');
      expect(downloadStore.active('model', 'klein9b')).toBeNull();
      stop();
    });

    it('announces nothing for downloads already final on the first read', async () => {
      const onDownloaded = vi.fn();
      downloadStore.onDownloaded(onDownloaded);
      api.getDownloads.mockResolvedValue({ items: [item({ state: 'done' }), item({ id: 'd2', state: 'failed', message: 'Disk full' })] });
      const stop = downloadStore.startSync();
      await advance(3000);

      expect(onDownloaded).not.toHaveBeenCalled();
      expect(toasts.addToast).not.toHaveBeenCalled();
      stop();
    });

    it('raises an error toast with a Retry that requests the download again, and the toast outlasts the page', async () => {
      api.getDownloads.mockResolvedValue({ items: [running()] });
      api.getDownloadPlan.mockResolvedValue(plan());
      const stop = downloadStore.startSync();
      await advance();
      api.getDownloads.mockResolvedValue({ items: [item({ state: 'failed', message: 'No space left on device', label: 'klein9b' })] });
      await advance(1000);
      stop();

      expect(toasts.addToast).toHaveBeenCalledWith('No space left on device', 'error', { action: { label: 'Retry', run: expect.any(Function) } });
      const options = (toasts.addToast.mock.calls.at(-1) as unknown as [string, string, { action: { run: () => void } }])[2];
      options.action.run();
      await advance();
      expect(api.getDownloadPlan).toHaveBeenCalledWith('model', 'klein9b');
    });

    it('stops notifying a callback once it is unsubscribed', async () => {
      const onDownloaded = vi.fn();
      const unsubscribe = downloadStore.onDownloaded(onDownloaded);
      unsubscribe();
      api.getDownloadPlan.mockResolvedValue(plan({ parts: [] }));

      await downloadStore.request('model', 'klein9b', 'klein9b');

      expect(onDownloaded).not.toHaveBeenCalled();
    });
  });

  describe('request', () => {
    it('answers present for an empty plan without starting, tells so and notifies', async () => {
      const onDownloaded = vi.fn();
      downloadStore.onDownloaded(onDownloaded);
      api.getDownloadPlan.mockResolvedValue(plan({ parts: [], total_bytes: 0, space: null, label: 'Klein 9B' }));

      await expect(downloadStore.request('model', 'klein9b', 'fallback')).resolves.toBe('present');

      expect(api.startDownload).not.toHaveBeenCalled();
      expect(confirm.requestConfirm).not.toHaveBeenCalled();
      expect(toasts.addToast).toHaveBeenCalledWith('Klein 9B is already downloaded.', 'info');
      expect(onDownloaded).toHaveBeenCalledTimes(1);
    });

    it('starts nothing when the confirmation is declined', async () => {
      api.getDownloadPlan.mockResolvedValue(plan());
      confirm.requestConfirm.mockResolvedValue(false);

      await expect(downloadStore.request('model', 'klein9b', 'klein9b')).resolves.toBe('declined');

      expect(api.startDownload).not.toHaveBeenCalled();
      expect(downloadStore.busy('model', 'klein9b')).toBe(false);
    });

    it('confirms with the plan label, then queues the download and exposes it as active', async () => {
      api.getDownloadPlan.mockResolvedValue(plan({ label: 'Klein 9B' }));
      confirm.requestConfirm.mockResolvedValue(true);
      api.startDownload.mockResolvedValue(item());
      api.getDownloads.mockResolvedValue({ items: [item()] });

      await expect(downloadStore.request('model', 'klein9b', 'fallback')).resolves.toBe('queued');
      expect(downloadStore.active('model', 'klein9b')?.id).toBe('d1');
      await advance();

      expect(confirm.requestConfirm).toHaveBeenCalledWith(expect.objectContaining({ question: 'Download Klein 9B?', confirmLabel: 'Download', danger: false }));
      expect(api.startDownload).toHaveBeenCalledWith('model', 'klein9b');
      expect(downloadStore.active('model', 'klein9b')?.id).toBe('d1');
      expect(downloadStore.active('enhancer', 'klein9b')).toBeNull();
    });

    it('marks the row busy while the plan is read', async () => {
      let resolvePlan: (value: DownloadPlan) => void = () => undefined;
      api.getDownloadPlan.mockReturnValue(new Promise((resolve) => { resolvePlan = resolve; }));
      confirm.requestConfirm.mockResolvedValue(false);

      const pending = downloadStore.request('model', 'klein9b', 'klein9b');
      expect(downloadStore.busy('model', 'klein9b')).toBe(true);
      await expect(downloadStore.request('model', 'klein9b', 'klein9b')).resolves.toBe('declined');
      expect(api.getDownloadPlan).toHaveBeenCalledTimes(1);

      resolvePlan(plan());
      await pending;
      expect(downloadStore.busy('model', 'klein9b')).toBe(false);
    });

    it('raises an error toast with Retry for a server fault, naming the row with the fallback label', async () => {
      api.getDownloadPlan.mockRejectedValue(new ApiError('Unavailable', 503, 'Hub unreachable'));

      await expect(downloadStore.request('model', 'klein9b', 'klein9b')).resolves.toBe('failed');

      expect(toasts.addToast).toHaveBeenCalledWith('Hub unreachable', 'error', { action: { label: 'Retry', run: expect.any(Function) } });
    });

    it('uses the fallback label when the failure carries no detail', async () => {
      api.getDownloadPlan.mockRejectedValue(new ApiError('Unavailable', 503, ''));

      await downloadStore.request('enhancer', 'smart', 'the Smart enhancer model');

      expect(toasts.addToast).toHaveBeenCalledWith('Could not download the Smart enhancer model.', 'error', { action: { label: 'Retry', run: expect.any(Function) } });
    });

    it('offers no Retry when the server rejected the input', async () => {
      api.getDownloadPlan.mockRejectedValue(new ApiError('Not found', 404, 'Unknown model'));

      await downloadStore.request('model', 'nope', 'nope');

      expect(toasts.addToast).toHaveBeenCalledWith('Unknown model', 'error');
    });
  });

  describe('cancel', () => {
    it('cancels by id and replaces the item with the answer', async () => {
      api.getDownloads.mockResolvedValue({ items: [running()] });
      api.cancelDownload.mockResolvedValue(item({ state: 'cancelled' }));
      const stop = downloadStore.startSync();
      await advance();

      await downloadStore.cancel(downloadStore.active('model', 'klein9b')!);

      expect(api.cancelDownload).toHaveBeenCalledWith('d1');
      expect(downloadStore.active('model', 'klein9b')).toBeNull();
      expect(toasts.addToast).not.toHaveBeenCalled();
      stop();
    });

    it('raises an error toast when the cancel fails', async () => {
      api.cancelDownload.mockRejectedValue(new ApiError('Conflict', 409, 'Already finished'));

      await downloadStore.cancel(item());

      expect(toasts.addToast).toHaveBeenCalledWith('Already finished', 'error');
      expect(downloadStore.busy('model', 'klein9b')).toBe(false);
    });
  });
});
