import { beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { memoryUploadStore } from '$lib/state/uploadStore';
import type { GalleryAsset } from '$lib/types';

const probe = vi.hoisted(() => ({ readSize: vi.fn(), readCutout: vi.fn() }));
vi.mock('./editSource', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./editSource')>();
  return { ...actual, readSize: probe.readSize, readCutout: probe.readCutout };
});
vi.mock('$lib/api/browse', async () => (await import('../../../test-utils/workspaceMocks')).browseApiMocks);

import { chooseSource, clearSource, followDraft } from './editActions';
import { sourceOfAsset, sourceOfFile, sourceOfPath } from './editSource';
import { imageEdit } from './editStore.svelte';

const asset = (overrides: Partial<GalleryAsset> = {}): GalleryAsset => ({
  id: 'out/a.png', url: '/media/out/a.png', thumbnail_url: '/thumbs/a.png', filename: 'a.png', file_path: '/out/a.png', media_type: 'image', width: 800, height: 600, ...overrides,
} as GalleryAsset);

const later = async (): Promise<void> => {
  await Promise.resolve();
  await Promise.resolve();
};

describe('choosing the image to edit', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    heldUploads.useStore(memoryUploadStore());
    imageEdit.setSource(null);
    probe.readSize.mockReset().mockResolvedValue(null);
    probe.readCutout.mockReset().mockResolvedValue(false);
    URL.createObjectURL = vi.fn(() => 'blob:edit');
    URL.revokeObjectURL = vi.fn();
  });

  it('puts a path in the draft and shows the image', () => {
    chooseSource(sourceOfPath('/out/a.png'));
    expect(draft.state.editSourcePath).toBe('/out/a.png');
    expect(imageEdit.source?.path).toBe('/out/a.png');
  });

  it('holds an uploaded file with its marker', () => {
    const file = new File(['x'], 'photo.png', { type: 'image/png' });
    chooseSource(sourceOfFile(file));
    expect(draft.state).toMatchObject({ editSourcePath: null, editSourceUpload: 'photo.png' });
    expect(heldUploads.editSource).toBe(file);
  });

  it('clears the held mask, its marker, the share and the extension with any new source', () => {
    for (const next of [sourceOfPath('/out/b.png'), sourceOfFile(new File(['x'], 'photo.png')), sourceOfAsset(asset({ file_path: '/out/c.png' }))]) {
      draft.setEditSource({ path: '/out/a.png' });
      draft.commitEditPaint({ mask: new File(['png'], 'mask.png'), share: 0.3, extension: { side: 'top', amount: 256 }, amount: 256 });
      chooseSource(next);
      expect(draft.state).toMatchObject({ editMaskUpload: null, editPaintedShare: 0, editExtendSide: null, editRepaint: false });
      expect(heldUploads.editMask).toBeNull();
    }
  });

  it('learns whether the image is a cut-out in the background, even though the painter may never open', async () => {
    probe.readCutout.mockResolvedValue(true);
    chooseSource(sourceOfPath('/out/a.png'));
    expect(imageEdit.source?.cutout).toBeNull();
    await later();
    expect(probe.readCutout).toHaveBeenCalledOnce();
    expect(imageEdit.source?.cutout).toBe(true);
  });

  it('takes a cut-out the gallery marks as one without looking, and looks at any other gallery image', async () => {
    chooseSource(sourceOfAsset(asset({ transparent: true })));
    await later();
    expect(imageEdit.source?.cutout).toBe(true);
    expect(probe.readCutout).not.toHaveBeenCalled();

    chooseSource(sourceOfAsset(asset({ file_path: '/out/b.png' })));
    await later();
    expect(probe.readCutout).toHaveBeenCalledOnce();
    expect(imageEdit.source?.cutout).toBe(false);
  });

  it('leaves the cut-out unknown when the image cannot be read', async () => {
    probe.readCutout.mockResolvedValue(null);
    chooseSource(sourceOfPath('/out/a.png'));
    await later();
    expect(imageEdit.source?.cutout).toBeNull();
  });

  it('drops the answer about an image that is no longer the one shown', async () => {
    let answer: (cutout: boolean) => void = () => undefined;
    probe.readCutout.mockReturnValueOnce(new Promise((resolve) => { answer = resolve; }));
    chooseSource(sourceOfPath('/out/a.png'));
    probe.readCutout.mockResolvedValue(false);
    chooseSource(sourceOfPath('/out/b.png'));
    answer(true);
    await later();
    expect(imageEdit.source?.path).toBe('/out/b.png');
    expect(imageEdit.source?.cutout).toBe(false);
  });

  it('reads the size of a source that does not tell it', async () => {
    probe.readSize.mockResolvedValue({ width: 640, height: 480 });
    chooseSource(sourceOfPath('/out/a.png'));
    await later();
    expect([imageEdit.source?.width, imageEdit.source?.height]).toEqual([640, 480]);
  });

  it('reads the size of a gallery image too, whatever size the gallery recorded', async () => {
    probe.readSize.mockResolvedValue({ width: 600, height: 800 });
    chooseSource(sourceOfAsset(asset()));
    expect(imageEdit.source).toMatchObject({ width: null, height: null });
    await later();
    expect(probe.readSize).toHaveBeenCalledOnce();
    expect([imageEdit.source?.width, imageEdit.source?.height]).toEqual([600, 800]);
  });

  it('forgets the image and everything done to it on Clear', () => {
    chooseSource(sourceOfPath('/out/a.png'));
    draft.commitEditPaint({ mask: new File(['png'], 'mask.png'), share: 0.3, extension: null, amount: 256 });
    clearSource();
    expect(imageEdit.source).toBeNull();
    expect(draft.state).toMatchObject({ editSourcePath: null, editMaskUpload: null });
    expect(heldUploads.editMask).toBeNull();
  });
});

describe('following the draft', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    heldUploads.useStore(memoryUploadStore());
    imageEdit.setSource(null);
    probe.readSize.mockReset().mockResolvedValue(null);
    probe.readCutout.mockReset().mockResolvedValue(false);
    URL.createObjectURL = vi.fn(() => 'blob:edit');
    URL.revokeObjectURL = vi.fn();
  });

  it('shows the draft\'s path after a reload, and its upload once the held file is back', () => {
    draft.patch({ editSourcePath: '/out/a.png' });
    followDraft();
    expect(imageEdit.source?.path).toBe('/out/a.png');

    const file = new File(['x'], 'photo.png');
    draft.patch({ editSourcePath: null, editSourceUpload: 'photo.png' });
    followDraft();
    expect(imageEdit.source).toBeNull();
    heldUploads.setEditSource(file);
    followDraft();
    expect(imageEdit.source?.file).toBe(file);
  });

  it('shows no image when the draft has none, after a Reset or a delete', () => {
    chooseSource(sourceOfPath('/out/a.png'));
    draft.setEditSource(null);
    followDraft();
    expect(imageEdit.source).toBeNull();
  });

  it('shows the file at its new path after a move, and keeps the paint and the history', () => {
    chooseSource(sourceOfPath('/out/a.png'));
    imageEdit.addStroke({ erase: false, size: 100, points: [10, 10] });
    draft.commitEditPaint({ mask: new File(['png'], 'mask.png'), share: 0.3, extension: { side: 'top', amount: 256 }, amount: 256 });

    draft.retargetPaths(new Map([['/out/a.png', '/out/Bus/a.png']]));
    followDraft();

    expect(imageEdit.source?.path).toBe('/out/Bus/a.png');
    expect(draft.state).toMatchObject({ editMaskUpload: 'mask.png', editExtendSide: 'top' });
    expect(heldUploads.editMask).not.toBeNull();
    expect(imageEdit.canUndo).toBe(true);
  });

  it('does nothing while the shown image is the draft\'s', () => {
    chooseSource(sourceOfPath('/out/a.png'));
    const shown = imageEdit.source;
    followDraft();
    expect(imageEdit.source).toBe(shown);
  });
});
