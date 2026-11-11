import { beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { memoryUploadStore } from '$lib/state/uploadStore';
import { makeAsset } from '../../../test-utils/workspace';

const probe = vi.hoisted(() => ({ readPicture: vi.fn() }));
vi.mock('./editSource', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./editSource')>();
  return { ...actual, readPicture: probe.readPicture };
});
vi.mock('$lib/api/browse', async () => (await import('../../../test-utils/workspaceMocks')).browseApiMocks);

import { chooseSource, clearSource, followDraft } from './editActions';
import { sourceOfAsset, sourceOfFile, sourceOfPath } from './editSource';
import { imageEdit } from './editStore.svelte';

beforeEach(() => {
  localStorage.clear();
  draft.reset();
  heldUploads.useStore(memoryUploadStore());
  imageEdit.setSource(null);
  probe.readPicture.mockReset().mockResolvedValue({ width: 640, height: 480, cutout: false });
  URL.createObjectURL = vi.fn(() => 'blob:edit');
  URL.revokeObjectURL = vi.fn();
});

const later = async (): Promise<void> => {
  await Promise.resolve();
  await Promise.resolve();
};

describe('choosing the image to edit', () => {
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

  it('learns the size and whether the image is a cut-out from one look, in the background, even though the painter may never open', async () => {
    probe.readPicture.mockResolvedValue({ width: 640, height: 480, cutout: true });
    chooseSource(sourceOfPath('/out/a.png'));
    expect(imageEdit.source).toMatchObject({ width: null, height: null, cutout: null });
    await later();
    expect(probe.readPicture).toHaveBeenCalledOnce();
    expect(imageEdit.source).toMatchObject({ width: 640, height: 480, cutout: true });
  });

  it('takes a cut-out the gallery marks as one at once, and reads the size of a gallery image whatever size the gallery recorded', async () => {
    probe.readPicture.mockResolvedValue({ width: 600, height: 800, cutout: true });
    chooseSource(sourceOfAsset(makeAsset({ file_path: '/out/a.png', transparent: true })));
    expect(imageEdit.source).toMatchObject({ width: null, height: null, cutout: true });
    await later();
    expect([imageEdit.source?.width, imageEdit.source?.height]).toEqual([600, 800]);
  });

  it('leaves the size and the cut-out unknown when the image cannot be read', async () => {
    probe.readPicture.mockResolvedValue(null);
    chooseSource(sourceOfPath('/out/a.png'));
    await later();
    expect(imageEdit.source).toMatchObject({ width: null, cutout: null });
  });

  it('drops the answer about an image that is no longer the one shown', async () => {
    let answer: (picture: { width: number; height: number; cutout: boolean }) => void = () => undefined;
    probe.readPicture.mockReturnValueOnce(new Promise((resolve) => { answer = resolve; }));
    chooseSource(sourceOfPath('/out/a.png'));
    probe.readPicture.mockResolvedValue({ width: 640, height: 480, cutout: false });
    chooseSource(sourceOfPath('/out/b.png'));
    answer({ width: 10, height: 10, cutout: true });
    await later();
    expect(imageEdit.source?.path).toBe('/out/b.png');
    expect(imageEdit.source).toMatchObject({ width: 640, cutout: false });
  });

  it('forgets the image and everything done to it on Clear', () => {
    chooseSource(sourceOfPath('/out/a.png'));
    draft.commitEdit({ mode: 'paint', mask: new File(['png'], 'mask.png'), share: 0.3, extension: null, amount: 256 });
    clearSource();
    expect(imageEdit.source).toBeNull();
    expect(draft.state).toMatchObject({ editSourcePath: null, editMaskUpload: null });
    expect(heldUploads.editMask).toBeNull();
  });
});

describe('following the draft', () => {
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
    draft.commitEdit({ mode: 'paint', mask: new File(['png'], 'mask.png'), share: 0.3, extension: { side: 'top', amount: 256 }, amount: 256 });

    draft.retargetPaths(new Map([['/out/a.png', '/out/Bus/a.png']]));
    followDraft();

    expect(imageEdit.source?.path).toBe('/out/Bus/a.png');
    expect(draft.state).toMatchObject({ editMaskUpload: 'mask.png', editMode: 'paint' });
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
