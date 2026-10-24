import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import type { ExtendSourceInfo } from '$lib/types';

import { draft } from './draft.svelte';
import { legacySource, sceneSource } from './extend.fixtures';
import { extendInspection } from './extendInspection.svelte';

const apiMocks = vi.hoisted(() => ({ inspectExtendSource: vi.fn<(body: Record<string, unknown>) => Promise<ExtendSourceInfo>>() }));
vi.mock('$lib/api/extend', () => ({ inspectExtendSource: apiMocks.inspectExtendSource }));

const SOURCE = '/clips/walk.mp4';

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

describe('extendInspection', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    extendInspection.reset();
    apiMocks.inspectExtendSource.mockReset();
    draft.patch({ workflow: 'extend', model: 'ltx-8', lowMemory: true });
    draft.setExtendSourcePath(SOURCE);
  });

  it('sends the source, the length, the model, the low-memory setting and the steps, and hands the answer to the draft', async () => {
    apiMocks.inspectExtendSource.mockResolvedValue(sceneSource());
    draft.update('extendFrames', 64);
    draft.update('steps', 4);
    await extendInspection.inspect(null);

    expect(apiMocks.inspectExtendSource).toHaveBeenCalledWith({ path: SOURCE, extend_frames: 64, model: 'ltx-8', low_memory: true, steps: 4, lora: null });
    expect(draft.extendSource?.filename).toBe('walk.mp4');
    expect(extendInspection.statusFor(SOURCE)).toBe('ready');
    expect(draft.state.scene).toBe('A woman on a rainy street.');
  });

  it('sends the selected LoRAs, and counts a change of them as a change to inspect again', async () => {
    apiMocks.inspectExtendSource.mockResolvedValue(sceneSource());
    draft.update('loraString', 'detail:0.8');
    await extendInspection.inspect(null);

    expect(apiMocks.inspectExtendSource).toHaveBeenCalledWith(expect.objectContaining({ lora: 'detail:0.8' }));
    expect(extendInspection.isCurrent()).toBe(true);
    draft.update('loraString', 'detail:0.5');
    expect(extendInspection.isCurrent()).toBe(false);
  });

  it('is checking while the server thinks, and reports the status only for the path it was made for', async () => {
    const answer = deferred<ExtendSourceInfo>();
    apiMocks.inspectExtendSource.mockReturnValue(answer.promise);
    const done = extendInspection.inspect(null);
    expect(extendInspection.statusFor(SOURCE)).toBe('checking');
    expect(extendInspection.statusFor('/clips/other.mp4')).toBe('idle');
    expect(extendInspection.statusFor(null)).toBe('idle');
    answer.resolve(sceneSource());
    await done;
    expect(extendInspection.statusFor(SOURCE)).toBe('ready');
  });

  it('replaces the text with the new source\'s when asked, and leaves edits alone when re-inspecting', async () => {
    apiMocks.inspectExtendSource.mockResolvedValue(sceneSource());
    await extendInspection.inspect(null);
    draft.patch({ scene: 'My scene.', prompt: 'My action.' });

    apiMocks.inspectExtendSource.mockResolvedValue(legacySource());
    await extendInspection.inspect(null);
    expect(draft.state).toMatchObject({ scene: 'My scene.', prompt: 'My action.' });

    await extendInspection.inspect(null, { replaceText: true });
    expect(draft.state).toMatchObject({ scene: '', prompt: 'A woman stands in the rain and says "Hello".' });
  });

  it('records a missing file as not found and forgets the facts', async () => {
    apiMocks.inspectExtendSource.mockResolvedValueOnce(sceneSource());
    await extendInspection.inspect(null);
    apiMocks.inspectExtendSource.mockRejectedValue(new ApiError('gone', 404, 'Source video not found: /clips/walk.mp4'));
    await extendInspection.inspect(null);
    expect(extendInspection.statusFor(SOURCE)).toBe('not_found');
    expect(draft.extendSource).toBeNull();
    expect(draft.state.extendSourcePath).toBe(SOURCE);
  });

  it('records any other failure with the server\'s explanation', async () => {
    apiMocks.inspectExtendSource.mockRejectedValue(new ApiError('bad', 422, 'Source video must be .mp4, .mov, .m4v, .mkv or .webm.'));
    await extendInspection.inspect(null);
    expect(extendInspection.statusFor(SOURCE)).toBe('failed');
    expect(extendInspection.failureFor(SOURCE)).toBe('Source video must be .mp4, .mov, .m4v, .mkv or .webm.');

    apiMocks.inspectExtendSource.mockRejectedValue(new TypeError('Failed to fetch'));
    await extendInspection.inspect(null);
    expect(extendInspection.failureFor(SOURCE)).toBe('Failed to fetch');
  });

  it('ignores an answer that a newer inspection or another source has overtaken', async () => {
    const first = deferred<ExtendSourceInfo>();
    apiMocks.inspectExtendSource.mockReturnValueOnce(first.promise);
    const slow = extendInspection.inspect(null);

    apiMocks.inspectExtendSource.mockResolvedValueOnce(sceneSource({ filename: 'second.mp4' }));
    await extendInspection.inspect(null);
    first.resolve(sceneSource({ filename: 'first.mp4' }));
    await slow;
    expect(draft.extendSource?.filename).toBe('second.mp4');

    const moved = deferred<ExtendSourceInfo>();
    apiMocks.inspectExtendSource.mockReturnValueOnce(moved.promise);
    const pending = extendInspection.inspect(null);
    draft.setExtendSourcePath('/clips/other.mp4');
    moved.resolve(sceneSource({ filename: 'stale.mp4' }));
    await pending;
    expect(draft.extendSource).toBeNull();
  });

  it('forgets the source when there is none to inspect', async () => {
    apiMocks.inspectExtendSource.mockResolvedValue(sceneSource());
    await extendInspection.inspect(null);
    draft.setExtendSourcePath(null);
    await extendInspection.inspect(null);
    expect(apiMocks.inspectExtendSource).toHaveBeenCalledTimes(1);
    expect(draft.extendSource).toBeNull();
    expect(extendInspection.statusFor(null)).toBe('idle');
  });

  describe('knowing what was asked', () => {
    it('is current after an inspection, and stale once the length, model, low-memory setting or source changes', async () => {
      apiMocks.inspectExtendSource.mockResolvedValue(sceneSource({ extend_frames: null }));
      expect(extendInspection.isCurrent()).toBe(false);
      draft.update('extendFrames', 48);
      await extendInspection.inspect(null);
      expect(extendInspection.isCurrent()).toBe(true);
      expect(extendInspection.requestedPath()).toBe(SOURCE);

      draft.update('extendFrames', 72);
      expect(extendInspection.isCurrent()).toBe(false);
      draft.update('extendFrames', 48);
      expect(extendInspection.isCurrent()).toBe(true);
      draft.update('lowMemory', false);
      expect(extendInspection.isCurrent()).toBe(false);
      draft.update('lowMemory', true);
      draft.update('steps', 4);
      expect(extendInspection.isCurrent()).toBe(false);
      draft.update('steps', 0);
      expect(extendInspection.isCurrent()).toBe(true);
      draft.update('model', 'other');
      expect(extendInspection.isCurrent()).toBe(false);
    });

    it('does not count the default length the answer fills in as a change worth asking about again', async () => {
      apiMocks.inspectExtendSource.mockResolvedValue(sceneSource({ extend_frames: 48 }));
      expect(draft.state.extendFrames).toBeNull();
      await extendInspection.inspect(null);
      expect(draft.state.extendFrames).toBe(48);
      expect(extendInspection.isCurrent()).toBe(true);
    });

    it('forgets what it asked after a reset', async () => {
      apiMocks.inspectExtendSource.mockResolvedValue(sceneSource());
      await extendInspection.inspect(null);
      extendInspection.reset();
      expect(extendInspection.isCurrent()).toBe(false);
      expect(extendInspection.requestedPath()).toBeNull();
      expect(extendInspection.statusFor(SOURCE)).toBe('idle');
    });
  });
});
