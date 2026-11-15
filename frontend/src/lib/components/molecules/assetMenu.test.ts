import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { AssetUpscaleFactor, GalleryAsset } from '$lib/types';

import { makeAudioAsset, makeSpeechAsset } from '../../../test-utils/audio';
import { EXTEND_SHORT_REASON, type FramePoint } from '$lib/state/framePoint';
import { assetKindLine, assetPanel, referenceItems, upscaleItems, voiceItems, type AssetPanel, type PanelItem } from './assetMenu';

const mocks = vi.hoisted(() => ({ copyPrompt: vi.fn() }));
vi.mock('$lib/state/copyPrompt', async (original) => ({ ...(await original<typeof import('$lib/state/copyPrompt')>()), copyPrompt: mocks.copyPrompt }));

beforeEach(() => mocks.copyPrompt.mockReset());

function option(factor: 2 | 4, overrides: Partial<AssetUpscaleFactor> = {}): AssetUpscaleFactor {
  return { factor, width: 832 * factor, height: 1216 * factor, allowed: true, reason: null, warning: null, ...overrides };
}

function makeAsset(upscale: GalleryAsset['upscale']): GalleryAsset {
  return {
    id: 'out/a.png', url: '/media/out/a.png', thumbnail_url: '/media/out/a.png', filename: 'a.png', created_at: '2026-10-03T12:00:00Z', project: null,
    workflow: 'txt2img', prompt: 'a fox', model: 'zit', media_type: 'image', reuse_workspace_url: '', upscale,
  };
}

const seedvr2 = (options: AssetUpscaleFactor[], overrides = {}) => ({
  options, model: 'SeedVR2 7B q8', downloaded: true, download_size_label: null, ...overrides,
});

const ids = (items: readonly PanelItem[]) => items.map((item) => item.id);
const sectionIds = (panel: AssetPanel) => panel.sections.map((section) => section.id);
const itemIn = (panel: AssetPanel, id: string): PanelItem | undefined =>
  [...panel.make, ...panel.sections.flatMap((section) => section.items), ...panel.file, ...[panel.head.view, panel.head.star].flatMap((item) => (item ? [item] : []))].find((item) => item.id === id);

describe('upscaleItems', () => {
  it('lists the plain choices, then the creative ones, each with its output size in its hint', () => {
    const asset = makeAsset({ factors: [option(2), option(4)], seedvr2: seedvr2([option(2), option(4)]) });
    const items = upscaleItems(asset, vi.fn());

    expect(ids(items)).toEqual(['upscale-seedvr2-2', 'upscale-seedvr2-4', 'upscale-creative-2', 'upscale-creative-4']);
    expect(new Set(items.map((item) => item.label)).size).toBe(4);
    // The two factors give two sizes, and a plain and a creative choice of one factor the same size.
    expect(items[0].hint).toBe(items[2].hint);
    expect(items[1].hint).toBe(items[3].hint);
    expect(items[0].hint).not.toBe(items[1].hint);
    expect(items.some((item) => item.span)).toBe(false);
  });

  it('gives a file with one choice that choice alone, spanning its row', () => {
    const onupscale = vi.fn();
    const video = { ...makeAsset({ factors: [], seedvr2: null, ltx: seedvr2([option(2)], { model: 'LTX-2.3 4-bit' }) }), media_type: 'video' as const };

    const items = upscaleItems(video, onupscale);

    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({ id: 'upscale-ltx-2', span: true });
    expect(items[0].hint).toBeTruthy();
    items[0].onselect?.();
    expect(onupscale).toHaveBeenCalledWith(video, 2, 'ltx');
  });

  it('marks a warned plain choice on its name and puts the warning in its tooltip, without greying it', () => {
    const warning = 'a warning from the server';
    const [plain, warned] = upscaleItems(makeAsset({ factors: [], seedvr2: seedvr2([option(2), option(4, { warning, warning_code: 'memory' })]) }), vi.fn());

    expect(plain).toMatchObject({ warn: undefined, title: undefined });
    expect(warned).toMatchObject({ id: 'upscale-seedvr2-4', title: warning });
    expect(warned.warn).toBeTruthy();
    expect(warned.disabled).not.toBe(true);
  });

  it('marks a one-frame-batch warning differently from a memory warning, by the warning code', () => {
    const warn = (warning_code: 'memory' | 'one_frame_batch') => {
      const [item] = upscaleItems(makeAsset({ factors: [], seedvr2: seedvr2([option(2, { warning: 'w', warning_code }), option(4)]) }), vi.fn());
      return item.warn;
    };

    expect(warn('one_frame_batch')).toBeDefined();
    expect(warn('one_frame_batch')).not.toBe(warn('memory'));
  });

  it('puts the first-use download in the hint of a plain choice while the model is not on disk, and nowhere once it is', () => {
    const pending = seedvr2([option(2), option(4)], { downloaded: false, download_size_label: '17.0 GB' });
    const [first, second, creative] = upscaleItems(makeAsset({ factors: [option(2)], seedvr2: pending }), vi.fn());

    // The size of the download is in the hint only while the model is not on disk.
    expect(first.hint).toContain('17.0 GB');
    expect(second.hint).toContain('17.0 GB');
    expect(creative.hint).not.toContain('17.0 GB');
    const [downloaded] = upscaleItems(makeAsset({ factors: [], seedvr2: seedvr2([option(2), option(4)]) }), vi.fn());
    expect(downloaded.hint).toBe(creative.hint);
  });

  it('offers only the creative choices where SeedVR2 is unavailable, and none where nothing is offered', () => {
    expect(ids(upscaleItems(makeAsset({ factors: [option(2), option(4)], seedvr2: null }), vi.fn()))).toEqual(['upscale-creative-2', 'upscale-creative-4']);
    expect(upscaleItems(makeAsset(null), vi.fn())).toEqual([]);
  });

  it('greys a creative choice over a limit and gives its reason as the hint and the tooltip', () => {
    const blocked = option(4, { allowed: false, reason: 'Over the limit.' });
    const [open, over] = upscaleItems(makeAsset({ factors: [option(2), blocked], seedvr2: null }), vi.fn());

    expect(open).toMatchObject({ disabled: false });
    expect(over).toMatchObject({ id: 'upscale-creative-4', disabled: true, hint: 'Over the limit.', title: 'Over the limit.' });
  });

  it('reports each choice with its method', () => {
    const onupscale = vi.fn();
    const asset = makeAsset({ factors: [option(2), option(4)], seedvr2: seedvr2([option(2), option(4)]) });
    for (const item of upscaleItems(asset, onupscale)) item.onselect?.();

    expect(onupscale.mock.calls.map(([, factor, method]) => [factor, method])).toEqual([[2, 'seedvr2'], [4, 'seedvr2'], [2, 'creative'], [4, 'creative']]);
  });
});

describe('referenceItems', () => {
  const asset = makeAsset(undefined);

  it('lists the four targets, each with a hint of what it makes', () => {
    const items = referenceItems(asset, vi.fn(), { reference: { disabled: false, hint: 'as image 2' } });

    expect(ids(items)).toEqual(['reference-image', 'reference-reference', 'reference-video', 'reference-last-frame']);
    expect(items.every((item) => Boolean(item.hint))).toBe(true);
    // A reference adds the slot the state names.
    expect(items[1].hint).toContain('as image 2');
    expect(items[0].hint).not.toContain('as image 2');
    expect(items.every((item) => item.disabled === false)).toBe(true);
  });

  it('greys a target the model can not use, with its reason as the hint and the tooltip', () => {
    const items = referenceItems(asset, vi.fn(), { last: { disabled: true, hint: 'LTX has no last frame' } });

    expect(items[3]).toMatchObject({ disabled: true, hint: 'LTX has no last frame', title: 'LTX has no last frame' });
    expect(items[0]).toMatchObject({ disabled: false, title: undefined });
  });

  it('passes a greyed choice\'s reason into its accessible name as it is, with its capitals', () => {
    const reason = 'Ideogram 4 can\'t use a start image';
    const items = referenceItems(asset, vi.fn(), { image: { disabled: true, hint: reason } });

    expect(items[0].name).toContain(reason);
  });

  it('reports the chosen target', () => {
    const onreference = vi.fn();
    for (const item of referenceItems(asset, onreference)) item.onselect?.();
    expect(onreference.mock.calls.map(([, target]) => target)).toEqual(['image', 'reference', 'video', 'last']);
  });
});

describe('the audio choices', () => {
  it('offers Voice and Saved voice for a speech take only', () => {
    expect(ids(voiceItems(makeSpeechAsset(), vi.fn()))).toEqual(['use-voice', 'use-save-voice']);
    expect(voiceItems(makeAudioAsset(), vi.fn())).toEqual([]);
    expect(voiceItems(makeAudioAsset({ kind: 'sound' }), vi.fn())).toEqual([]);
  });

  it('greys both for a take longer than the model takes, with the reason as hint and tooltip', () => {
    const items = voiceItems(makeSpeechAsset({ duration: 44 }), vi.fn(), { voiceMaxSeconds: 30 });

    for (const item of items) expect(item).toMatchObject({ disabled: true, hint: 'Use a take of 30 s or less as a voice', title: 'Use a take of 30 s or less as a voice' });
  });

  it('greys both where no speech model can follow a voice, with that reason', () => {
    const none = { disabled: true, hint: 'No speech model here can use a voice' };
    const items = voiceItems(makeSpeechAsset(), vi.fn(), { voice: none, save_voice: none });

    for (const item of items) expect(item).toMatchObject({ disabled: true, hint: none.hint, title: none.hint });
  });

  it('reports the chosen target', () => {
    const onuseaudio = vi.fn();
    for (const item of voiceItems(makeSpeechAsset(), onuseaudio)) item.onselect?.();
    expect(onuseaudio.mock.calls.map(([, target]) => target)).toEqual(['voice', 'save_voice']);
  });
});

describe('assetPanel', () => {
  const everything = () => ({ onreuse: vi.fn(), onreference: vi.fn(), onextend: vi.fn(), onedit: vi.fn(), onuseframe: vi.fn(), onupscale: vi.fn(), onuseaudio: vi.fn(), ondelete: vi.fn(), onstar: vi.fn() });
  const image = { ...makeAsset({ factors: [option(2), option(4)], seedvr2: seedvr2([option(2), option(4)]) }), file_path: '/outputs/out/a.png', has_reusable_config: true };
  const video = (overrides: Partial<GalleryAsset> = {}): GalleryAsset => ({
    ...makeAsset({ factors: [], seedvr2: seedvr2([option(2)]) }), id: 'out/c.mp4', url: '/media/out/c.mp4', filename: 'c.mp4', media_type: 'video', file_path: '/outputs/out/c.mp4',
    extend: { available: true, reason: null }, ...overrides,
  });
  const point = (overrides: Partial<FramePoint> = {}): FramePoint => ({ time: 3.02, seconds: 3, tooShort: false, atEnd: false, ...overrides });

  it('puts an image in the one order: Reuse then Edit, Use as, Upscale, then the file row ending in Delete', () => {
    const panel = assetPanel(image, everything(), { openMove: vi.fn() });

    expect(ids(panel.make)).toEqual(['reuse', 'edit']);
    expect(sectionIds(panel)).toEqual(['use-as', 'upscale']);
    expect(ids(panel.sections[0].items)).toEqual(['reference-image', 'reference-reference', 'reference-video', 'reference-last-frame']);
    expect(ids(panel.file)).toEqual(['download', 'move', 'copy-prompt', 'delete']);
    expect(panel.make[0]).toMatchObject({ primary: true, disabled: false });
    expect(panel.make[0].hint).toBeTruthy();
  });

  it('gives a video Reuse with Extend and one Upscale 2× that spans its row, and no Use as', () => {
    const panel = assetPanel(video(), everything());

    expect(ids(panel.make)).toEqual(['reuse', 'extend']);
    expect(panel.make[1].description).toBeTruthy();
    expect(sectionIds(panel)).toEqual(['upscale']);
    expect(panel.sections[0].items[0]).toMatchObject({ span: true });
  });

  it('offers Audio track for every take, and Use as only for a speech take', () => {
    const handlers = everything();
    for (const take of [makeAudioAsset(), makeAudioAsset({ kind: 'sound' }), makeSpeechAsset()]) {
      expect(ids(assetPanel(take, handlers).make)).toEqual(['reuse', 'use-audio-track']);
    }
    expect(sectionIds(assetPanel(makeAudioAsset(), handlers))).toEqual([]);
    expect(sectionIds(assetPanel(makeAudioAsset({ kind: 'sound' }), handlers))).toEqual([]);
    const speech = assetPanel(makeSpeechAsset(), handlers);
    expect(sectionIds(speech)).toEqual(['use-as']);
    expect(ids(speech.sections[0].items)).toEqual(['use-voice', 'use-save-voice']);
  });

  it('greys Audio track with the reason as its hint where the Video model can not follow one', () => {
    const useAs = { audio_track: { disabled: true, hint: 'LTX 2.3 can\'t follow an audio track' } };
    const track = itemIn(assetPanel(makeAudioAsset(), everything(), { useAs }), 'use-audio-track');

    expect(track).toMatchObject({ disabled: true, hint: 'LTX 2.3 can\'t follow an audio track', title: 'LTX 2.3 can\'t follow an audio track' });
    const open = itemIn(assetPanel(makeAudioAsset(), everything()), 'use-audio-track');
    expect(open).toMatchObject({ disabled: false });
    expect(open?.hint).toBeTruthy();
    expect(open?.hint).not.toBe(track?.hint);
  });

  it('leaves out what has no handler and has no Select anywhere', () => {
    const panel = assetPanel(image, {});

    expect(panel.make).toEqual([]);
    expect(panel.sections).toEqual([]);
    expect(ids(panel.file)).toEqual(['download', 'copy-prompt']);
    expect(panel.head).toEqual({ view: null, star: null });
    expect(itemIn(assetPanel(image, everything(), { onview: vi.fn() }), 'select')).toBeUndefined();
  });

  it('greys Reuse with its reason for a file that recorded no settings', () => {
    const reuse = assetPanel({ ...image, has_reusable_config: false }, everything()).make[0];

    expect(reuse).toMatchObject({ id: 'reuse', disabled: true });
    expect(reuse.title).toBeTruthy();
    expect(reuse.hint).toBe(reuse.title);
  });

  it('offers Move only when the owner can open the list, and greys it while the file is being deleted', () => {
    const openMove = vi.fn();
    expect(ids(assetPanel(image, { ondelete: vi.fn() }).file)).toEqual(['download', 'copy-prompt', 'delete']);
    const move = itemIn(assetPanel(image, {}, { openMove }), 'move');
    move?.onselect?.();
    expect(openMove).toHaveBeenCalledOnce();
    expect(itemIn(assetPanel(image, {}, { openMove, deleting: true }), 'move')).toMatchObject({ disabled: true });
    expect(itemIn(assetPanel(image, { ondelete: vi.fn() }, { deleting: true }), 'delete')).toMatchObject({ label: 'Deleting…', disabled: true });
  });

  it('copies the prompt, and greys Copy prompt with its reason for a file with no recorded text', () => {
    const copy = itemIn(assetPanel(image, {}), 'copy-prompt');
    copy?.onselect?.();
    expect(mocks.copyPrompt).toHaveBeenCalledWith('a fox');

    const blank = itemIn(assetPanel({ ...image, prompt: '' }, {}), 'copy-prompt');
    expect(blank?.disabled).toBe(true);
    expect(blank?.title).toBeTruthy();
    const take = itemIn(assetPanel(makeAudioAsset({ lyrics: 'la la' }), {}), 'copy-prompt');
    expect(take?.disabled).toBe(false);
  });

  it('keeps Delete a plain action that hands the asset to ondelete, which asks first', () => {
    const handlers = everything();
    itemIn(assetPanel(image, handlers), 'delete')?.onselect?.();

    expect(handlers.ondelete).toHaveBeenCalledWith(image);
    expect(itemIn(assetPanel(image, handlers), 'delete')).toMatchObject({ danger: true });
  });

  it('puts View in the head when asked, and the star as a checkbox that flips with the file\'s state', () => {
    const onview = vi.fn();
    const onstar = vi.fn();
    const panel = assetPanel(image, { onstar }, { onview });
    panel.head.view?.onselect?.();
    expect(onview).toHaveBeenCalledOnce();
    expect(panel.head.star).toMatchObject({ label: 'Star', checked: false });
    panel.head.star?.onselect?.();
    expect(onstar).toHaveBeenCalledWith(image, true);

    const starred = { ...image, starred: true };
    expect(assetPanel(starred, { onstar }).head.star).toMatchObject({ label: 'Unstar', checked: true });
    expect(assetPanel(image, {}).head.star).toBeNull();
  });

  describe('for a paused video frame', () => {
    it('reads Extend from the frame and adds the frame\'s own First frame and Last frame section', () => {
      const panel = assetPanel(video(), everything(), { point: point() });

      expect(panel.make[1].id).toBe('extend');
      expect(panel.make[1].label).toContain('3.0');
      // The note says what a cut keeps, and differs from the whole-clip and the end-of-clip notes.
      const whole = assetPanel(video(), everything()).make[1].description;
      const atEnd = assetPanel(video(), everything(), { point: point({ atEnd: true }) }).make[1].description;
      expect(panel.make[1].description).toBeTruthy();
      expect(panel.make[1].description).not.toBe(whole);
      expect(atEnd).toBe(whole);
      expect(sectionIds(panel)).toEqual(['frames', 'upscale']);
      expect(panel.sections[0].label).toContain('3.0');
      expect(ids(panel.sections[0].items)).toEqual(['frame-first', 'frame-last']);
    });

    it('has no frame section without a frame or a handler, and plain Extend', () => {
      expect(sectionIds(assetPanel(video(), everything()))).toEqual(['upscale']);
      expect(sectionIds(assetPanel(video(), { onextend: vi.fn() }, { point: point() }))).toEqual([]);
      expect(itemIn(assetPanel(video(), { onextend: vi.fn() }, { point: point() }), 'extend')).toMatchObject({ label: 'Extend from 3.0 s' });
    });

    it('runs Extend and the frame choices with the frame, and Extend without one when there is none', () => {
      const on = everything();
      const frame = point();
      const panel = assetPanel(video(), on, { point: frame });
      for (const id of ['extend', 'frame-first', 'frame-last']) itemIn(panel, id)?.onselect?.();

      expect(on.onextend).toHaveBeenCalledWith(expect.objectContaining({ id: 'out/c.mp4' }), frame);
      expect(on.onuseframe).toHaveBeenNthCalledWith(1, expect.anything(), frame, 'video');
      expect(on.onuseframe).toHaveBeenNthCalledWith(2, expect.anything(), frame, 'last');
      itemIn(assetPanel(video(), on), 'extend')?.onselect?.();
      expect(on.onextend.mock.calls.at(-1)).toHaveLength(1);
    });

    it('greys Extend on a frame too early, with the reason as its hint, and leaves the frame choices open', () => {
      const panel = assetPanel(video(), everything(), { point: point({ tooShort: true, seconds: 0.4 }) });

      expect(itemIn(panel, 'extend')).toMatchObject({ disabled: true, hint: EXTEND_SHORT_REASON });
      expect(itemIn(panel, 'frame-first')?.disabled).toBe(false);
      expect(itemIn(panel, 'frame-first')?.description).toBeTruthy();
      expect(itemIn(panel, 'extend')?.title).toBe(EXTEND_SHORT_REASON);
      // A greyed choice gives its reason, not its note.
      expect(itemIn(panel, 'extend')?.description).toBeUndefined();
    });

    it('keeps Extend greyed with its reason when the clip can not be extended, and never reads from', () => {
      const panel = assetPanel(video({ extend: { available: false, reason: 'Upscaled videos can\'t be extended' } }), everything(), { point: point() });

      expect(itemIn(panel, 'extend')).toMatchObject({ disabled: true, hint: 'Upscaled videos can\'t be extended' });
      // It never reads "from" a frame while the clip can't be extended: the label stays as it is without a frame.
      expect(itemIn(panel, 'extend')?.label).toBe(itemIn(assetPanel(video({ extend: { available: false, reason: 'x' } }), everything()), 'extend')?.label);
    });

    it('greys a frame target the model can not use, with its reason', () => {
      const panel = assetPanel(video(), everything(), { point: point(), useAs: { last: { disabled: true, hint: 'No last frame' } } });

      expect(itemIn(panel, 'frame-last')).toMatchObject({ disabled: true, hint: 'No last frame' });
    });
  });
});

describe('assetKindLine', () => {
  it('carries the size of an image or video and the length of a clip or take', () => {
    const image = assetKindLine({ ...makeAsset(undefined), width: 1216, height: 832 });
    expect(image).toContain('1216');
    expect(image).toContain('832');
    const clip = assetKindLine({ ...makeAsset(undefined), media_type: 'video', width: 768, height: 448, duration: 5 });
    expect(clip).toContain('768');
    expect(clip).toContain('5.0');
    expect(assetKindLine(makeSpeechAsset({ duration: 6 }))).toContain('0:06');
  });
});

describe('assetPanel adjust', () => {
  const picture = (overrides: Partial<GalleryAsset> = {}): GalleryAsset => ({ ...makeAsset(undefined), file_path: '/outputs/out/a.png', ...overrides });
  const ids = (items: readonly PanelItem[]) => items.map((item) => item.id);

  it('offers Adjust after Edit for an image with a file, and runs it on the asset', () => {
    const onadjust = vi.fn();
    const asset = picture();
    const panel = assetPanel(asset, { onedit: vi.fn(), onadjust });
    expect(ids(panel.make)).toEqual(['edit', 'adjust']);
    const item = panel.make[1];
    expect(item.icon).toBe('sliders');
    item.onselect?.();
    expect(onadjust).toHaveBeenCalledWith(asset);
  });

  it('offers Adjust without Edit, and not without a handler', () => {
    expect(ids(assetPanel(picture(), { onadjust: vi.fn() }).make)).toContain('adjust');
    expect(ids(assetPanel(picture(), { onedit: vi.fn() }).make)).not.toContain('adjust');
  });

  it('offers no Adjust for a video, an audio take, or an image without a file', () => {
    const handlers = { onedit: vi.fn(), onadjust: vi.fn() };
    expect(ids(assetPanel(picture({ media_type: 'video', file_path: '/outputs/a.mp4' }), handlers).make)).not.toContain('adjust');
    expect(ids(assetPanel(makeAudioAsset(), handlers).make)).not.toContain('adjust');
    expect(ids(assetPanel(picture({ file_path: undefined }), handlers).make)).not.toContain('adjust');
  });
});
