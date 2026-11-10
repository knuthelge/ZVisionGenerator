import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { FrameTarget } from '$lib/state/assetActions';
import { EXTEND_SHORT_REASON, type FramePoint } from '$lib/state/framePoint';
import { videoFacts } from '$lib/state/videoFacts.svelte';
import type { GalleryAsset } from '$lib/types';

import type { ActionMenuEntry, ActionMenuItem } from './ActionMenu.svelte';
import { assetMenuEntries } from './assetMenu';
import { FRAME_BUSY_LABEL, frameMenuEntries } from './frameMenu';

const mocks = vi.hoisted(() => ({ busyFor: vi.fn(), browseInfo: vi.fn() }));
vi.mock('$lib/state/frameGrab.svelte', () => ({ busyFor: mocks.busyFor }));
vi.mock('$lib/api/browse', () => ({ browseInfo: mocks.browseInfo }));

const clip = (overrides: Partial<GalleryAsset> = {}): GalleryAsset => ({
  id: 'out/c.mp4', url: '/media/out/c.mp4', thumbnail_url: '', filename: 'c.mp4', created_at: '2026-04-30T12:00:00Z', project: null, workflow: 'txt2vid', prompt: 'A clip', model: 'ltx',
  media_type: 'video', file_path: '/outputs/out/c.mp4', reuse_workspace_url: '', extend: { available: true, reason: null }, ...overrides,
});
const point = (overrides: Partial<FramePoint> = {}): FramePoint => ({ time: 3.02, seconds: 3, tooShort: false, atEnd: false, ...overrides });
const handlers = () => ({ onextend: vi.fn(), onuseframe: vi.fn() });

beforeEach(() => {
  videoFacts.reset();
  mocks.busyFor.mockReset().mockReturnValue(null);
  mocks.browseInfo.mockReset();
});

describe('frameMenuEntries', () => {
  it('builds Extend, First frame and Last frame with their notes', async () => {
    mocks.browseInfo.mockResolvedValue({ fps: 24, frames: 144, start: 0 });
    videoFacts.ensure('/outputs/out/c.mp4');
    await vi.waitFor(() => expect(videoFacts.lengthOf('/outputs/out/c.mp4')).toBe(6));

    const rows = frameMenuEntries(clip(), point(), handlers());

    expect(rows.map((row) => [row.id, row.label])).toEqual([['extend', 'Extend from 3.0 s'], ['frame-first', 'First frame'], ['frame-last', 'Last frame']]);
    expect(rows.map((row) => row.description)).toEqual(['Keeps 3.0 s of 6.0 s, drops what came after', 'Start a new clip on this frame', 'End a new clip on this frame']);
    expect(rows.every((row) => row.hint === undefined && row.disabled === false && row.busy !== true)).toBe(true);
  });

  it('leaves the length out of Extend\'s note until the video\'s facts are known', () => {
    const [extend] = frameMenuEntries(clip(), point(), handlers());

    expect(extend.description).toBe('Keeps 3.0 s, drops what came after');
  });

  it('reads the end as Extend the clip', () => {
    const [extend] = frameMenuEntries(clip(), point({ atEnd: true }), handlers());

    expect(extend).toMatchObject({ label: 'Extend the clip', description: 'Continues from the end of the clip' });
  });

  it('keeps the rows it can not use, disabled, with their reasons', () => {
    const useAs = { video: { disabled: true, hint: 'LTX can\'t start from an image' }, last: { disabled: true } };
    const rows = frameMenuEntries(clip(), point({ tooShort: true }), handlers(), useAs);

    expect(rows.map((row) => row.hint)).toEqual([EXTEND_SHORT_REASON, 'LTX can\'t start from an image', 'Last frame is not available']);
    expect(rows.map((row) => row.disabled)).toEqual([true, true, true]);
    expect(rows.every((row) => row.description === undefined)).toBe(true);
  });

  it('reads an Extend that is not available as Extend with its reason, disabled', () => {
    const [extend] = frameMenuEntries(clip({ extend: { available: false, reason: 'Gone' } }), point(), handlers());

    expect(extend).toMatchObject({ label: 'Extend', hint: 'Gone', disabled: true });
  });

  it('disables every frame row while a frame is extracted, and marks only the running one busy', () => {
    mocks.busyFor.mockReturnValue('last' satisfies FrameTarget);

    const rows = frameMenuEntries(clip(), point(), handlers());

    expect(rows.map((row) => [row.id, row.disabled, row.busy ?? false])).toEqual([['extend', false, false], ['frame-first', true, false], ['frame-last', true, true]]);
    expect(rows[2].label).toBe(FRAME_BUSY_LABEL);
    expect(rows[1].label).toBe('First frame');
    expect(mocks.busyFor).toHaveBeenCalledWith('out/c.mp4');
  });

  it('leaves out the rows whose handler is absent or whose file is not on the host', () => {
    expect(frameMenuEntries(clip(), point(), { onextend: vi.fn() }).map((row) => row.id)).toEqual(['extend']);
    expect(frameMenuEntries(clip(), point(), { onuseframe: vi.fn() }).map((row) => row.id)).toEqual(['frame-first', 'frame-last']);
    expect(frameMenuEntries(clip({ file_path: undefined }), point(), handlers()).map((row) => row.id)).toEqual(['extend']);
    expect(frameMenuEntries(clip({ extend: undefined }), point(), handlers()).map((row) => row.id)).toEqual(['frame-first', 'frame-last']);
  });

  it('runs the handlers with the frame', () => {
    const on = handlers();
    const frame = point();

    frameMenuEntries(clip(), frame, on).forEach((row) => row.onselect?.());

    expect(on.onextend).toHaveBeenCalledWith(expect.objectContaining({ id: 'out/c.mp4' }), frame);
    expect(on.onuseframe.mock.calls.map((call) => call[2])).toEqual(['video', 'last']);
  });
});

describe('the time chip\'s menu and the stage ⋯ menu (which lists Use as before Extend)', () => {
  const frameRows = (entries: readonly ActionMenuEntry[]): ActionMenuItem[] =>
    entries.filter((entry): entry is ActionMenuItem => entry.kind === 'item' && ['extend', 'frame-first', 'frame-last'].includes(entry.id));
  const view = (rows: readonly ActionMenuItem[]) => [...rows].sort((a, b) => a.id.localeCompare(b.id)).map(({ id, label, icon, description, hint, disabled, busy }) => ({ id, label, icon, description, hint, disabled, busy: busy ?? false }));

  it.each([
    ['mid-clip', point(), {}, null],
    ['at the end of the clip', point({ atEnd: true }), {}, null],
    ['too early to extend', point({ tooShort: true, seconds: 0.4 }), {}, null],
    ['with a target the model can not use', point(), { last: { disabled: true, hint: 'No last frame' } }, null],
    ['while a frame is extracted', point(), {}, 'video'],
    ['at the end while a frame is extracted', point({ atEnd: true }), {}, 'last'],
  ])('give the same rows %s', (_name, frame, useAs, busy) => {
    mocks.busyFor.mockReturnValue(busy);
    const on = handlers();

    const chip = frameMenuEntries(clip(), frame, on, useAs);
    const stage = frameRows(assetMenuEntries(clip(), on, false, useAs, undefined, frame));

    expect(chip).toHaveLength(3);
    expect(view(stage)).toEqual(view(chip));
  });
});
