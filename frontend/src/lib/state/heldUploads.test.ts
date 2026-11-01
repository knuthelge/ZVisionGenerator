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

    expect(await store.get('image')).toEqual({ start, end: null, references: [ref, null], track: null });
    expect(await store.get('video')).toEqual({ start: first, end: last, references: [], track: null });

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
    await store.put('image', { start: stored, end: null, references: [storedRef], track: null });
    await store.put('video', { start: file('stored-first.png'), end: storedLast, references: [], track: null });
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
    expect(await store.get('image')).toEqual({ start: written, end: null, references: [storedRef], track: null });
  });

  it("a write before the Workspace ever restores keeps the stored fields it didn't change", async () => {
    const storedLast = file('stored-last.png');
    const storedStart = file('stored-start.png');
    await store.put('video', { start: file('stored-first.png'), end: storedLast, references: [], track: null });
    await store.put('image', { start: storedStart, end: null, references: [], track: null });
    heldUploads.useStore(store);

    // The Gallery page drops the first frame; nothing has read the store back yet.
    heldUploads.setStart('video', null);
    await heldUploads.persisted();

    expect(await store.get('video')).toEqual({ start: null, end: storedLast, references: [], track: null });
    expect(await store.get('image')).toEqual({ start: storedStart, end: null, references: [], track: null });
    expect(heldUploads.lastFrame).toBe(storedLast);
    expect(heldUploads.restored).toBe(true);
  });

  it('restore is shared between callers', async () => {
    const get = vi.spyOn(store, 'get');
    await Promise.all([heldUploads.restore(), heldUploads.restore()]);
    await heldUploads.restore();
    // One read per tab: Image, Video and Extend Video.
    expect(get).toHaveBeenCalledTimes(3);
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

  describe('the audio track', () => {
    const memo = (name: string): File => new File(['x'], name, { type: 'audio/mp4' });

    it('is held per tab, beside the Video tab\'s frames', () => {
      const video = memo('video.m4a');
      const extend = memo('extend.m4a');
      heldUploads.setTrack('video', video);
      heldUploads.setTrack('extend', extend);

      expect([heldUploads.audioTrack, heldUploads.extendAudioTrack]).toEqual([video, extend]);
    });

    it('persists in its own tab record, so a voice memo survives a visit to the Gallery', async () => {
      const video = memo('video.m4a');
      const extend = memo('extend.m4a');
      const first = file('first.png');
      heldUploads.setStart('video', first);
      heldUploads.setTrack('video', video);
      heldUploads.setTrack('extend', extend);
      await heldUploads.persisted();

      expect(await store.get('video')).toEqual({ start: first, end: null, references: [], track: video });
      expect(await store.get('extend')).toEqual({ start: null, end: null, references: [], track: extend });

      heldUploads.useStore(store);
      expect(heldUploads.audioTrack).toBeNull();
      await heldUploads.restore();

      expect([heldUploads.audioTrack, heldUploads.extendAudioTrack, heldUploads.firstFrame]).toEqual([video, extend, first]);
    });

    it('is forgotten with the Video tab and by clear, and kept when another tab is cleared', async () => {
      heldUploads.setStart('video', file('first.png'));
      heldUploads.setTrack('video', memo('video.m4a'));
      heldUploads.setTrack('extend', memo('extend.m4a'));

      heldUploads.clearTab('image');
      expect([heldUploads.audioTrack?.name, heldUploads.extendAudioTrack?.name]).toEqual(['video.m4a', 'extend.m4a']);

      heldUploads.clearTab('video');
      expect([heldUploads.firstFrame, heldUploads.audioTrack, heldUploads.extendAudioTrack?.name]).toEqual([null, null, 'extend.m4a']);

      heldUploads.clearTab('extend');
      expect(heldUploads.extendAudioTrack).toBeNull();

      heldUploads.setTrack('video', memo('video.m4a'));
      heldUploads.setTrack('extend', memo('extend.m4a'));
      heldUploads.clear();
      await heldUploads.persisted();
      expect([heldUploads.audioTrack, heldUploads.extendAudioTrack, await store.get('video'), await store.get('extend')]).toEqual([null, null, null, null]);
    });

  });
});
