import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { WorkspaceContext } from '$lib/types';

const draftMock = vi.hoisted(() => ({
  state: {} as Record<string, string | string[] | null>,
  setStartImage: vi.fn(),
  setLastFrame: vi.fn(),
  setAudioTrack: vi.fn(),
  setVoice: vi.fn(),
  setExtendSourcePath: vi.fn(),
  setEditSource: vi.fn(),
}));
const inspectionMock = vi.hoisted(() => ({ inspect: vi.fn() }));
vi.mock('$lib/state/draft.svelte', () => ({ draft: draftMock }));
vi.mock('$lib/state/extendInspection.svelte', () => ({ extendInspection: inspectionMock }));

import { emptySlotsHolding } from './assetRemoval';

const context = {} as WorkspaceContext;

const emptyState = { referenceImagePath: null, firstFramePath: null, endImagePath: null, referenceImagePaths: [] as string[], audioTrackPath: null, extendAudioTrackPath: null, extendSourcePath: null, voicePath: null, editSourcePath: null };

beforeEach(() => {
  draftMock.state = { ...emptyState };
  for (const fn of [draftMock.setStartImage, draftMock.setLastFrame, draftMock.setAudioTrack, draftMock.setVoice, draftMock.setExtendSourcePath, draftMock.setEditSource, inspectionMock.inspect]) fn.mockReset();
});

describe('emptySlotsHolding', () => {
  it('empties the image to edit, with its paint, when its file is gone, and leaves another', () => {
    draftMock.state = { ...emptyState, editSourcePath: '/o/e.png' };
    emptySlotsHolding(['/o/other.png'], context);
    expect(draftMock.setEditSource).not.toHaveBeenCalled();
    emptySlotsHolding(['/o/e.png'], context);
    expect(draftMock.setEditSource).toHaveBeenCalledWith(null);
  });

  it('empties the start image, first frame and last frame that hold a gone path', () => {
    draftMock.state = { ...emptyState, referenceImagePath: '/o/a.png', firstFramePath: '/o/b.png', endImagePath: '/o/c.png' };
    expect(emptySlotsHolding(['/o/a.png', '/o/b.png', '/o/c.png'], context)).toBe(false);
    expect(draftMock.setStartImage).toHaveBeenCalledWith('image', null, context);
    expect(draftMock.setStartImage).toHaveBeenCalledWith('video', null, context);
    expect(draftMock.setLastFrame).toHaveBeenCalledWith(null);
  });

  it('leaves slots that hold other files', () => {
    draftMock.state = { ...emptyState, referenceImagePath: '/o/a.png', endImagePath: '/o/c.png', audioTrackPath: '/o/t.wav', extendAudioTrackPath: '/o/u.wav', extendSourcePath: '/o/v.mp4', voicePath: '/o/w.wav' };
    emptySlotsHolding(['/o/zzz.png'], null);
    expect(draftMock.setStartImage).not.toHaveBeenCalled();
    expect(draftMock.setLastFrame).not.toHaveBeenCalled();
    expect(draftMock.setAudioTrack).not.toHaveBeenCalled();
    expect(draftMock.setVoice).not.toHaveBeenCalled();
    expect(draftMock.setExtendSourcePath).not.toHaveBeenCalled();
    expect(inspectionMock.inspect).not.toHaveBeenCalled();
  });

  it('empties the Video and Extend audio tracks and the voice that hold a gone path', () => {
    draftMock.state = { ...emptyState, audioTrackPath: '/o/t.wav', extendAudioTrackPath: '/o/u.wav', voicePath: '/o/w.wav' };
    expect(emptySlotsHolding(['/o/t.wav', '/o/u.wav', '/o/w.wav'], context)).toBe(false);
    expect(draftMock.setAudioTrack).toHaveBeenCalledWith('video', null);
    expect(draftMock.setAudioTrack).toHaveBeenCalledWith('extend', null);
    expect(draftMock.setVoice).toHaveBeenCalledWith(null);
  });

  it('empties only the audio track that holds the path', () => {
    draftMock.state = { ...emptyState, audioTrackPath: '/o/t.wav', extendAudioTrackPath: '/o/u.wav' };
    emptySlotsHolding(['/o/u.wav'], context);
    expect(draftMock.setAudioTrack).toHaveBeenCalledTimes(1);
    expect(draftMock.setAudioTrack).toHaveBeenCalledWith('extend', null);
  });

  it('empties the extend source and forgets its inspection', () => {
    draftMock.state = { ...emptyState, extendSourcePath: '/o/v.mp4' };
    emptySlotsHolding(['/o/v.mp4'], context);
    expect(draftMock.setExtendSourcePath).toHaveBeenCalledWith(null);
    expect(inspectionMock.inspect).toHaveBeenCalledWith(context);
  });

  it('reports a reference row that holds a gone path and leaves the row', () => {
    draftMock.state = { ...emptyState, referenceImagePaths: ['/o/x.png', '/o/y.png'] };
    expect(emptySlotsHolding(['/o/y.png'], context)).toBe(true);
    expect(draftMock.setStartImage).not.toHaveBeenCalled();
  });

  it('ignores missing paths', () => {
    expect(emptySlotsHolding([undefined, null, ''], context)).toBe(false);
    expect(draftMock.setStartImage).not.toHaveBeenCalled();
  });
});
