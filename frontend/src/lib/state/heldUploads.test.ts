import { beforeEach, describe, expect, it, vi } from 'vitest';
import { heldUploads } from './heldUploads.svelte';
import { memoryUploadStore, type UploadStore } from './uploadStore';

const file = (name: string): File => new File(['x'], name, { type: 'image/png' });

describe('heldUploads', () => {
  let store: UploadStore;

  beforeEach(() => {
    store = memoryUploadStore();
    heldUploads.useStore(store);
  });

  it('holds the start image of each tab apart', () => {
    const start = file('start.png');
    const first = file('first.png');
    heldUploads.setStart('image', start);
    heldUploads.setStart('video', first);
    expect([heldUploads.start('image'), heldUploads.start('video')]).toEqual([start, first]);
  });

  it('clearTab forgets the start image and references of Image, or the frames of Video', () => {
    heldUploads.setStart('image', file('start.png'));
    heldUploads.setReferences([file('ref.png')]);
    heldUploads.setStart('video', file('first.png'));
    heldUploads.setLastFrame(file('last.png'));

    heldUploads.clearTab('video');
    expect([heldUploads.firstFrame, heldUploads.lastFrame]).toEqual([null, null]);
    expect(heldUploads.startImage).not.toBeNull();
    expect(heldUploads.references).toHaveLength(1);

    heldUploads.clearTab('image');
    expect(heldUploads.startImage).toBeNull();
    expect(heldUploads.references).toEqual([]);
  });

  it('clear forgets every upload', () => {
    heldUploads.setStart('image', file('start.png'));
    heldUploads.setStart('video', file('first.png'));
    heldUploads.setLastFrame(file('last.png'));
    heldUploads.setReferences([file('ref.png')]);

    heldUploads.clear();

    expect([heldUploads.startImage, heldUploads.firstFrame, heldUploads.lastFrame, heldUploads.references]).toEqual([null, null, null, []]);
  });

  it("each write persists its tab's record", async () => {
    const start = file('start.png');
    const ref = file('ref.png');
    const first = file('first.png');
    const last = file('last.png');

    heldUploads.setStart('image', start);
    heldUploads.setReferences([ref, null]);
    heldUploads.setStart('video', first);
    heldUploads.setLastFrame(last);
    await heldUploads.persisted();

    expect(await store.get('image')).toEqual({ start, end: null, references: [ref, null] });
    expect(await store.get('video')).toEqual({ start: first, end: last, references: [] });

    heldUploads.setStart('image', null);
    heldUploads.setReferences([]);
    await heldUploads.persisted();
    expect(await store.get('image')).toBeNull();
    expect(await store.get('video')).not.toBeNull();
  });

  it('restore brings back both tabs after a reload', async () => {
    const start = file('start.png');
    const ref = file('ref.png');
    const first = file('first.png');
    const last = file('last.png');
    heldUploads.setStart('image', start);
    heldUploads.setReferences([null, ref]);
    heldUploads.setStart('video', first);
    heldUploads.setLastFrame(last);
    await heldUploads.persisted();

    heldUploads.useStore(store);
    expect(heldUploads.restored).toBe(false);
    expect(heldUploads.startImage).toBeNull();

    await heldUploads.restore();

    expect(heldUploads.restored).toBe(true);
    expect([heldUploads.startImage, heldUploads.firstFrame, heldUploads.lastFrame]).toEqual([start, first, last]);
    expect(heldUploads.references).toEqual([null, ref]);
  });

  it('a write before the restore finishes wins', async () => {
    const stored = file('stored.png');
    const storedRef = file('stored-ref.png');
    const storedLast = file('stored-last.png');
    await store.put('image', { start: stored, end: null, references: [storedRef] });
    await store.put('video', { start: file('stored-first.png'), end: storedLast, references: [] });
    heldUploads.useStore(store);

    const restoring = heldUploads.restore();
    const written = file('written.png');
    heldUploads.setStart('image', written);
    heldUploads.setLastFrame(null);
    await restoring;
    await heldUploads.persisted();

    expect(heldUploads.startImage).toBe(written);
    expect(heldUploads.lastFrame).toBeNull();
    expect(heldUploads.references).toEqual([storedRef]);
    expect(heldUploads.firstFrame?.name).toBe('stored-first.png');
    expect(await store.get('image')).toEqual({ start: written, end: null, references: [storedRef] });
  });

  it("a write before the Workspace ever restores keeps the stored fields it didn't change", async () => {
    const storedLast = file('stored-last.png');
    const storedStart = file('stored-start.png');
    await store.put('video', { start: file('stored-first.png'), end: storedLast, references: [] });
    await store.put('image', { start: storedStart, end: null, references: [] });
    heldUploads.useStore(store);

    // The Gallery page drops the first frame; nothing has read the store back yet.
    heldUploads.setStart('video', null);
    await heldUploads.persisted();

    expect(await store.get('video')).toEqual({ start: null, end: storedLast, references: [] });
    expect(await store.get('image')).toEqual({ start: storedStart, end: null, references: [] });
    expect(heldUploads.lastFrame).toBe(storedLast);
    expect(heldUploads.restored).toBe(true);
  });

  it('restore is shared between callers', async () => {
    const get = vi.spyOn(store, 'get');
    await Promise.all([heldUploads.restore(), heldUploads.restore()]);
    await heldUploads.restore();
    expect(get).toHaveBeenCalledTimes(2);
  });

  it('clear deletes both records, clearTab only its own', async () => {
    heldUploads.setStart('image', file('start.png'));
    heldUploads.setStart('video', file('first.png'));
    await heldUploads.persisted();

    heldUploads.clearTab('video');
    await heldUploads.persisted();
    expect(await store.get('video')).toBeNull();
    expect(await store.get('image')).not.toBeNull();

    heldUploads.setStart('video', file('first.png'));
    heldUploads.clear();
    await heldUploads.persisted();
    expect(await store.get('image')).toBeNull();
    expect(await store.get('video')).toBeNull();
  });

  it('restore settles when the store fails', async () => {
    const failing: UploadStore = {
      get: () => Promise.reject(new Error('unreadable')),
      put: () => Promise.reject(new Error('unwritable')),
      delete: () => Promise.reject(new Error('unwritable')),
    };
    heldUploads.useStore(failing);

    await heldUploads.restore();
    expect(heldUploads.restored).toBe(true);

    const start = file('start.png');
    heldUploads.setStart('image', start);
    await heldUploads.persisted();
    expect(heldUploads.startImage).toBe(start);
  });

  it('forgetting a held file removes it from the store', async () => {
    await heldUploads.restore();
    const remove = vi.spyOn(store, 'delete');

    heldUploads.setStart('image', file('start.png'));
    await heldUploads.persisted();
    expect(await store.get('image')).not.toBeNull();

    heldUploads.setStart('image', null);
    await heldUploads.persisted();
    expect(remove).toHaveBeenCalledTimes(1);
    expect(await store.get('image')).toBeNull();
  });
});
