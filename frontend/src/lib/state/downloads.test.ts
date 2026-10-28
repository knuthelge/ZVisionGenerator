import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import type { DownloadItem, DownloadPlan } from '$lib/types';

const api = vi.hoisted(() => ({
  getDownloadPlan: vi.fn(),
  startDownload: vi.fn(),
  getDownloads: vi.fn(),
  cancelDownload: vi.fn(),
  importLoraHF: vi.fn(),
}));
const discover = vi.hoisted(() => ({ startDiscoverDownload: vi.fn() }));
const confirm = vi.hoisted(() => ({ requestConfirm: vi.fn() }));
const toasts = vi.hoisted(() => ({ addToast: vi.fn(() => 'toast-1') }));

vi.mock('$lib/api/models', () => api);
vi.mock('$lib/api/discover', () => discover);
vi.mock('$lib/components/molecules/confirm.svelte', () => confirm);
vi.mock('./toasts.svelte', () => toasts);

import { downloadStore } from './downloads.svelte';
import { chipDownload } from './loraDownloads';

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
    for (const mock of [...Object.values(api), discover.startDiscoverDownload, confirm.requestConfirm, toasts.addToast]) mock.mockReset();
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

  describe('LoRA downloads', () => {
    const lora = (overrides: Partial<DownloadItem> = {}): DownloadItem => item({ id: 'l1', kind: 'lora', name: 'pixel_artist', label: 'LoRA pixel_artist', curated_id: 'a-pixel', ...overrides });

    it('takes an item the server answered with into the lane at once and reads the lane again', async () => {
      api.getDownloads.mockResolvedValue({ items: [lora({ state: 'running' })] });

      downloadStore.adopt(lora());
      expect(downloadStore.active('lora', 'pixel_artist')).not.toBeNull();
      await advance();

      expect(api.getDownloads).toHaveBeenCalled();
    });

    it('names every LoRA the lane knows, so a chip waiting on a file is kept', async () => {
      api.getDownloads.mockResolvedValue({ items: [lora(), item()] });
      const stop = downloadStore.startSync();
      await advance();

      expect(downloadStore.loraNames()).toEqual(['pixel_artist']);
      stop();
    });

    it('starts a failed LoRA download again from its curated entry, and from its repository file otherwise', async () => {
      discover.startDiscoverDownload.mockResolvedValue({ name: 'pixel_artist', status: 'queued', renamed: false, item: lora({ id: 'l2' }), trigger: null, trigger_source: null });

      expect(await downloadStore.retryLora(lora({ state: 'failed' }))).toBe(true);
      expect(discover.startDiscoverDownload).toHaveBeenLastCalledWith({ curated_id: 'a-pixel' });
      expect(downloadStore.active('lora', 'pixel_artist')?.id).toBe('l2');

      await downloadStore.retryLora(lora({ state: 'failed', curated_id: null, source_repo: 'o/r', source_file: 'f.safetensors' }));
      expect(discover.startDiscoverDownload).toHaveBeenLastCalledWith({ repo: 'o/r', file: 'f.safetensors', name: 'pixel_artist' });
    });

    it('starts a failed Import again through the Import route with the same file and name, never through Discover', async () => {
      api.importLoraHF.mockResolvedValue({ tone: 'success', message: 'ok', name: 'pixel_artist', outcome: 'queued', download: lora({ id: 'l4', anonymous: false }) });
      const imported = lora({ state: 'failed', anonymous: false, curated_id: null, source_repo: 'o/private', source_file: 'f.safetensors' });

      expect(await downloadStore.retryLora(imported)).toBe(true);

      expect(api.importLoraHF).toHaveBeenCalledWith({ repo_id: 'o/private', filename: 'f.safetensors', name: 'pixel_artist' });
      expect(discover.startDiscoverDownload).not.toHaveBeenCalled();
      expect(downloadStore.active('lora', 'pixel_artist')?.id).toBe('l4');
    });

    it('starts a failed anonymous Discover download again through Discover, never through Import', async () => {
      discover.startDiscoverDownload.mockResolvedValue({ name: 'pixel_artist', status: 'queued', renamed: false, item: lora({ id: 'l5' }), trigger: null, trigger_source: null });

      await downloadStore.retryLora(lora({ state: 'failed', anonymous: true, curated_id: null, source_repo: 'o/r', source_file: 'f.safetensors' }));

      expect(discover.startDiscoverDownload).toHaveBeenCalledWith({ repo: 'o/r', file: 'f.safetensors', name: 'pixel_artist' });
      expect(api.importLoraHF).not.toHaveBeenCalled();
    });

    it('cannot retry an Import that does not say which file it was', async () => {
      expect(await downloadStore.retryLora(lora({ anonymous: false, curated_id: null }))).toBe(false);
      expect(api.importLoraHF).not.toHaveBeenCalled();
    });

    it('forgets the finished downloads of a LoRA whose file was deleted, so none speaks for it any more, and keeps a running one', async () => {
      api.getDownloads.mockResolvedValue({ items: [lora({ state: 'done' }), lora({ id: 'l9', name: 'other', state: 'done' })] });
      const stop = downloadStore.startSync();
      await advance();
      expect(chipDownload('pixel_artist', [], downloadStore.items)).toMatchObject({ state: 'downloading', percent: 100 });
      downloadStore.markUnlisted('l1');
      api.getDownloads.mockResolvedValue({ items: [lora({ id: 'l9', name: 'other', state: 'done' })] });

      downloadStore.forgetLora('pixel_artist');
      await advance();

      expect(downloadStore.loraNames()).toEqual(['other']);
      expect(chipDownload('pixel_artist', [], downloadStore.items)).toEqual({ state: 'ready' });
      expect(downloadStore.unlisted.has('l1')).toBe(false);

      api.getDownloads.mockResolvedValue({ items: [lora({ id: 'l10', state: 'running' }), lora({ id: 'l9', name: 'other', state: 'done' })] });
      await downloadStore.refresh();
      downloadStore.forgetLora('pixel_artist');
      expect(downloadStore.active('lora', 'pixel_artist')?.id).toBe('l10');
      stop();
    });

    it('keeps a failed LoRA download the server stopped listing, until a newer item for that LoRA arrives', async () => {
      api.getDownloads.mockResolvedValueOnce({ items: [lora({ id: 'l1', state: 'failed' }), lora({ id: 'l2', name: 'other', state: 'done' })] });
      await downloadStore.refresh();
      api.getDownloads.mockResolvedValueOnce({ items: [] });
      await downloadStore.refresh();

      expect(downloadStore.items.map((item) => item.id)).toEqual(['l1']);
      expect(chipDownload('pixel_artist', [], downloadStore.items).state).toBe('failed');

      api.getDownloads.mockResolvedValueOnce({ items: [lora({ id: 'l3', state: 'running' })] });
      await downloadStore.refresh();
      expect(downloadStore.items.map((item) => item.id)).toEqual(['l3']);
    });

    it('says so when there is nothing to start it from, and when the request fails', async () => {
      expect(await downloadStore.retryLora(lora({ curated_id: null }))).toBe(false);
      expect(toasts.addToast).toHaveBeenCalledWith('Could not retry LoRA pixel_artist. Download it again from Discover.', 'error');

      discover.startDiscoverDownload.mockRejectedValue(new ApiError('Bad', 409, 'The name is taken.'));
      expect(await downloadStore.retryLora(lora())).toBe(false);
      expect(toasts.addToast).toHaveBeenLastCalledWith('The name is taken.', 'error');
      expect(downloadStore.busy('lora', 'pixel_artist')).toBe(false);
    });

    it('raises the failed toast with a Retry that starts the LoRA download again', async () => {
      discover.startDiscoverDownload.mockResolvedValue({ name: 'pixel_artist', status: 'queued', renamed: false, item: lora({ id: 'l3' }), trigger: null, trigger_source: null });
      api.getDownloads.mockResolvedValueOnce({ items: [lora({ state: 'running' })] }).mockResolvedValue({ items: [lora({ state: 'failed', message: 'The connection dropped.' })] });
      const stop = downloadStore.startSync();
      await advance();
      await advance(1500);

      const call = toasts.addToast.mock.calls.find((args) => (args as unknown[])[0] === 'The connection dropped.') as unknown as [string, string, { action: { run: () => void } }];
      call[2].action.run();
      await advance();

      expect(discover.startDiscoverDownload).toHaveBeenCalledWith({ curated_id: 'a-pixel' });
      stop();
    });

    describe('a failure the Workspace shows in its own error slot', () => {
      const failedAfterRunning = async (items: DownloadItem[]): Promise<void> => {
        api.getDownloads.mockResolvedValueOnce({ items: items.map((one) => ({ ...one, state: 'running' as const })) }).mockResolvedValue({ items });
        const stop = downloadStore.startSync();
        await advance();
        await advance(1500);
        stop();
      };

      it('never holds back the failure of a model download', async () => {
        downloadStore.showFailuresInline(['klein9b']);

        await failedAfterRunning([item({ state: 'failed', message: 'No space left.' })]);

        expect(toasts.addToast.mock.calls.map((args) => (args as unknown[])[0])).toContain('No space left.');
      });
    });
  });
});
