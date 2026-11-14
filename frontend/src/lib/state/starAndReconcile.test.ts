import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';

const api = vi.hoisted(() => ({ setStarred: vi.fn<(id: string, starred: boolean) => Promise<{ id: string; starred: boolean }>>() }));
const toast = vi.hoisted(() => ({ addToast: vi.fn() }));
vi.mock('$lib/api/gallery', () => ({ setStarred: api.setStarred, getGallery: vi.fn() }));
vi.mock('$lib/state/toasts.svelte', () => toast);

import { galleryStore } from './gallery.svelte';
import { jobStore } from './job.svelte';
import { starAndReconcile } from './starAndReconcile';

beforeEach(() => {
  api.setStarred.mockReset();
  toast.addToast.mockReset();
  vi.restoreAllMocks();
});

describe('starAndReconcile', () => {
  it('saves the star, then updates the Gallery lists and the job outputs', async () => {
    api.setStarred.mockResolvedValue({ id: 'a.png', starred: true });
    const gallery = vi.spyOn(galleryStore, 'setStarred');
    const job = vi.spyOn(jobStore, 'setStarred');

    expect(await starAndReconcile({ id: 'a.png' }, true)).toBe(true);

    expect(api.setStarred).toHaveBeenCalledWith('a.png', true);
    expect(gallery).toHaveBeenCalledWith('a.png', true);
    expect(job).toHaveBeenCalledWith('a.png', true);
    expect(toast.addToast).not.toHaveBeenCalled();
  });

  it('changes neither store and says why when the request fails', async () => {
    api.setStarred.mockRejectedValue(new ApiError('POST /api/gallery/star → 500', 500, 'The star could not be saved.'));
    const gallery = vi.spyOn(galleryStore, 'setStarred');
    const job = vi.spyOn(jobStore, 'setStarred');

    expect(await starAndReconcile({ id: 'a.png' }, false)).toBe(false);

    expect(gallery).not.toHaveBeenCalled();
    expect(job).not.toHaveBeenCalled();
    expect(toast.addToast).toHaveBeenCalledWith('The star could not be saved.', 'error');
  });

  it('uses a generic message when the failure carries none', async () => {
    api.setStarred.mockRejectedValue('offline');

    await starAndReconcile({ id: 'a.png' }, true);

    expect(toast.addToast).toHaveBeenCalledWith(expect.stringMatching(/\S/), 'error');
  });

  it('sends one request for a file while its first is pending, and returns false for the second', async () => {
    let settle!: (value: { id: string; starred: boolean }) => void;
    api.setStarred.mockReturnValue(new Promise((resolve) => (settle = resolve)));
    const gallery = vi.spyOn(galleryStore, 'setStarred');

    const first = starAndReconcile({ id: 'a.png' }, true);
    expect(await starAndReconcile({ id: 'a.png' }, true)).toBe(false);
    expect(api.setStarred).toHaveBeenCalledTimes(1);
    expect(toast.addToast).not.toHaveBeenCalled();

    settle({ id: 'a.png', starred: true });
    expect(await first).toBe(true);
    expect(gallery).toHaveBeenCalledTimes(1);
  });

  it('frees the file once a call has settled, whether it succeeded or failed', async () => {
    api.setStarred
      .mockResolvedValueOnce({ id: 'a.png', starred: true })
      .mockResolvedValueOnce({ id: 'a.png', starred: false })
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce({ id: 'a.png', starred: true });

    expect(await starAndReconcile({ id: 'a.png' }, true)).toBe(true);
    expect(await starAndReconcile({ id: 'a.png' }, false)).toBe(true);
    expect(await starAndReconcile({ id: 'a.png' }, true)).toBe(false);
    expect(await starAndReconcile({ id: 'a.png' }, true)).toBe(true);

    expect(api.setStarred).toHaveBeenCalledTimes(4);
  });

  it('does not block a different file while one is pending', async () => {
    let settle!: (value: { id: string; starred: boolean }) => void;
    api.setStarred.mockReturnValueOnce(new Promise((resolve) => (settle = resolve)));
    api.setStarred.mockResolvedValueOnce({ id: 'b.png', starred: true });

    const first = starAndReconcile({ id: 'a.png' }, true);
    expect(await starAndReconcile({ id: 'b.png' }, true)).toBe(true);

    settle({ id: 'a.png', starred: true });
    await first;
  });
});
