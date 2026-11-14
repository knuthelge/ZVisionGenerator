import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import type { GalleryAsset, MoveResult } from '$lib/types';

const api = vi.hoisted(() => ({ moveAssets: vi.fn<(ids: string[], project: string | null) => Promise<MoveResult>>() }));
const toast = vi.hoisted(() => ({ addToast: vi.fn() }));
const inspection = vi.hoisted(() => ({ inspect: vi.fn() }));
vi.mock('$lib/api/gallery', () => ({ moveAssets: api.moveAssets, getGallery: vi.fn() }));
vi.mock('$lib/state/toasts.svelte', () => toast);
vi.mock('$lib/state/extendInspection.svelte', () => ({ extendInspection: inspection }));

import { draft } from './draft.svelte';
import { galleryStore } from './gallery.svelte';
import { jobStore } from './job.svelte';
import { moveAndReconcile } from './moveAndReconcile';

function asset(id: string, project: string | null, filePath: string): GalleryAsset {
  return { id, project, file_path: filePath, media_type: 'image' } as GalleryAsset;
}

const after = asset('Bus/a.png', 'Bus', '/o/Bus/a.png');

beforeEach(() => {
  api.moveAssets.mockReset();
  toast.addToast.mockReset();
  inspection.inspect.mockReset();
  localStorage.clear();
  draft.reset();
  jobStore.clearJob();
});

describe('moveAndReconcile', () => {
  it('sends the IDs and the project, and returns the server\'s answer', async () => {
    const answer = { moved: [{ from: 'a.png', from_path: '/o/a.png', asset: after }], failed: [{ id: 'c.png', detail: 'The file is missing.' }] };
    api.moveAssets.mockResolvedValue(answer);

    const result = await moveAndReconcile(['a.png', 'b.png', 'c.png'], 'Bus', null);

    expect(api.moveAssets).toHaveBeenCalledWith(['a.png', 'b.png', 'c.png'], 'Bus');
    expect(result).toBe(answer);
  });

  it('runs moves started together one after another, in the order asked, and both land', async () => {
    const log: string[] = [];
    let releaseFirst: (result: MoveResult) => void = () => {};
    api.moveAssets
      .mockImplementationOnce(() => { log.push('first sent'); return new Promise((resolve) => { releaseFirst = resolve; }); })
      .mockImplementationOnce(async () => { log.push('second sent'); return { moved: [{ from: 'b.png', from_path: '/o/b.png', asset: asset('Zoo/b.png', 'Zoo', '/o/Zoo/b.png') }], failed: [] }; });
    const applyMoves = vi.spyOn(galleryStore, 'applyMoves').mockImplementation((moves) => { log.push(`applied ${moves[0].asset.id}`); });

    const first = moveAndReconcile(['a.png'], 'Bus', null);
    const second = moveAndReconcile(['b.png'], 'Zoo', null);
    await Promise.resolve();
    await Promise.resolve();
    expect(log).toEqual(['first sent']);

    releaseFirst({ moved: [{ from: 'a.png', from_path: '/o/a.png', asset: after }], failed: [] });
    const results = await Promise.all([first, second]);

    expect(log).toEqual(['first sent', 'applied Bus/a.png', 'second sent', 'applied Zoo/b.png']);
    expect(results.map((result) => result?.moved.length)).toEqual([1, 1]);
    applyMoves.mockRestore();
  });

  it('runs the next move after one whose request failed', async () => {
    api.moveAssets
      .mockRejectedValueOnce(new Error('down'))
      .mockResolvedValueOnce({ moved: [{ from: 'a.png', from_path: '/o/a.png', asset: after }], failed: [] });

    const first = moveAndReconcile(['a.png'], 'Bus', null);
    const second = moveAndReconcile(['a.png'], 'Bus', null);

    expect(await first).toBeNull();
    expect((await second)?.moved).toHaveLength(1);
  });

  it('points the slots at the new path, takes the move into the lists and the job, and reports the outcome', async () => {
    api.moveAssets.mockResolvedValue({ moved: [{ from: 'a.png', from_path: '/o/a.png', asset: after }], failed: [] });
    draft.setStartImage('image', { path: '/o/a.png' }, null);
    const applyMoves = vi.spyOn(galleryStore, 'applyMoves');
    const replaceOutputs = vi.spyOn(jobStore, 'replaceOutputs');

    await moveAndReconcile(['a.png'], 'Bus', null);

    expect(draft.state.referenceImagePath).toBe('/o/Bus/a.png');
    expect(applyMoves).toHaveBeenCalledWith([{ from: 'a.png', from_path: '/o/a.png', asset: after }]);
    expect(replaceOutputs).toHaveBeenCalledWith([{ from: 'a.png', from_path: '/o/a.png', asset: after }]);
    expect(toast.addToast).toHaveBeenCalledWith(expect.any(String), 'success');
    expect(inspection.inspect).not.toHaveBeenCalled();
  });

  it('retargets a slot from the path the server reports, for a file the page never loaded', async () => {
    const ids = Array.from({ length: 30 }, (_, index) => `f${index}.png`);
    api.moveAssets.mockResolvedValue({
      moved: ids.map((id) => ({ from: id, from_path: `/o/${id}`, asset: asset(`Bus/${id}`, 'Bus', `/o/Bus/${id}`) })),
      failed: [],
    });
    draft.setStartImage('image', { path: '/o/f27.png' }, null);

    await moveAndReconcile(ids, 'Bus', null);

    expect(draft.state.referenceImagePath).toBe('/o/Bus/f27.png');
  });

  it('inspects the extend source again when it moved', async () => {
    api.moveAssets.mockResolvedValue({ moved: [{ from: 'a.mp4', from_path: '/o/a.mp4', asset: asset('Bus/a.mp4', 'Bus', '/o/Bus/a.mp4') }], failed: [] });
    draft.setExtendSourcePath('/o/a.mp4');

    await moveAndReconcile(['a.mp4'], 'Bus', null);

    expect(draft.state.extendSourcePath).toBe('/o/Bus/a.mp4');
    expect(inspection.inspect).toHaveBeenCalledOnce();
  });

  it('leaves the slots alone for a file that stayed, and for a file that failed', async () => {
    api.moveAssets.mockResolvedValue({ moved: [], failed: [{ id: 'a.png', detail: 'Could not move the file: Permission denied' }] });
    draft.setStartImage('image', { path: '/o/a.png' }, null);

    await moveAndReconcile(['a.png'], 'Bus', null);

    expect(draft.state.referenceImagePath).toBe('/o/a.png');
    expect(toast.addToast).toHaveBeenCalledWith(expect.any(String), 'error');
  });

  it('says so and changes nothing when the request fails', async () => {
    api.moveAssets.mockRejectedValue(new Error('down'));
    draft.setStartImage('image', { path: '/o/a.png' }, null);

    expect(await moveAndReconcile(['a.png'], 'Bus', null)).toBeNull();

    expect(draft.state.referenceImagePath).toBe('/o/a.png');
    expect(toast.addToast).toHaveBeenCalledWith(expect.any(String), 'error');
  });

  it('shows the server\'s reason when it refuses the whole request', async () => {
    api.moveAssets.mockRejectedValue(new ApiError('POST /api/gallery/move → 409', 409, 'A job is using that file.'));

    expect(await moveAndReconcile(['a.png'], 'Bus', null)).toBeNull();

    expect(toast.addToast).toHaveBeenCalledWith('A job is using that file.', 'error');
  });
});
