import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./client', () => ({ api: { get: vi.fn(), post: vi.fn(), delete: vi.fn() } }));

import { api } from './client';
import { cancelDownload, deleteBackgroundRemoval, getDownloadPlan, getDownloads, startDownload } from './models';

beforeEach(() => {
  vi.clearAllMocks();
});

describe('download API', () => {
  it('requests the download plan for a row', async () => {
    vi.mocked(api.get).mockResolvedValue({ parts: [] });
    await getDownloadPlan('model', 'flux dev');
    expect(api.get).toHaveBeenCalledWith('/api/models/download/plan?kind=model&name=flux%20dev');
  });

  it('starts a download with the kind and name', async () => {
    vi.mocked(api.post).mockResolvedValue({ id: 'd1' });
    await expect(startDownload('enhancer', 'fast')).resolves.toEqual({ id: 'd1' });
    expect(api.post).toHaveBeenCalledWith('/api/models/download', { kind: 'enhancer', name: 'fast' });
  });

  it('plans and starts the background removal download as kind matte', async () => {
    vi.mocked(api.get).mockResolvedValue({ parts: [] });
    await getDownloadPlan('matte', 'background-removal');
    expect(api.get).toHaveBeenCalledWith('/api/models/download/plan?kind=matte&name=background-removal');
    vi.mocked(api.post).mockResolvedValue({ id: 'd2' });
    await startDownload('matte', 'background-removal');
    expect(api.post).toHaveBeenCalledWith('/api/models/download', { kind: 'matte', name: 'background-removal' });
  });

  it('deletes the background removal models through their own route', async () => {
    vi.mocked(api.delete).mockResolvedValue({ status: 'ok' });
    await deleteBackgroundRemoval();
    expect(api.delete).toHaveBeenCalledWith('/api/background-removal');
  });

  it('lists the downloads lane', async () => {
    vi.mocked(api.get).mockResolvedValue({ items: [] });
    await expect(getDownloads()).resolves.toEqual({ items: [] });
    expect(api.get).toHaveBeenCalledWith('/api/models/downloads');
  });

  it('cancels a download by id', async () => {
    vi.mocked(api.post).mockResolvedValue({ id: 'd 1' });
    await cancelDownload('d 1');
    expect(vi.mocked(api.post).mock.calls[0][0]).toBe('/api/models/downloads/d%201/cancel');
  });
});
