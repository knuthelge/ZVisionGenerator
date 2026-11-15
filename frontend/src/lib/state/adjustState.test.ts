import { describe, expect, it } from 'vitest';

import { makeAsset } from '../../test-utils/workspace';
import { NEUTRAL_LIGHT, MIX_NAMES } from './adjustMath';
import { adjustBody, canAdjust, canReplace, draftChanged, neutralDraft, withAdjustedCopy } from './adjustState';

describe('neutralDraft', () => {
  it('changes nothing and gives each call its own object', () => {
    const draft = neutralDraft();
    expect(draftChanged(draft)).toBe(false);
    expect(draft).toEqual({ light: { exposure: 0, contrast: 0, highlights: 0, shadows: 0, saturation: 0, vibrance: 0, warmth: 0, mix_red: 0, mix_orange: 0, mix_yellow: 0, mix_green: 0, mix_aqua: 0, mix_blue: 0, mix_purple: 0, mix_magenta: 0 }, geometry: { rotate: 0, flip: false, crop: null }, aspect: { preset: 'free', ratio: null } });
    draft.light.exposure = 5;
    expect(neutralDraft().light.exposure).toBe(0);
  });
});

describe('draftChanged', () => {
  it.each([
    ['a slider', (d: ReturnType<typeof neutralDraft>) => { d.light.warmth = -1; }],
    ['a rotation', (d: ReturnType<typeof neutralDraft>) => { d.geometry.rotate = 90; }],
    ['a flip', (d: ReturnType<typeof neutralDraft>) => { d.geometry.flip = true; }],
    ['a crop', (d: ReturnType<typeof neutralDraft>) => { d.geometry.crop = { x: 0, y: 0, width: 10, height: 10 }; }],
  ])('is true for %s', (_name, change) => {
    const draft = neutralDraft();
    change(draft);
    expect(draftChanged(draft)).toBe(true);
  });

  it.each(MIX_NAMES)('is true for the %s band of the colour mixer', (name) => {
    const draft = neutralDraft();
    draft.light[`mix_${name}`] = 1;
    expect(draftChanged(draft)).toBe(true);
  });

  it('does not count the shape chip', () => {
    const draft = neutralDraft();
    draft.aspect = { preset: '1:1', ratio: 1 };
    expect(draftChanged(draft)).toBe(false);
  });

  it('does not count a crop that covers the whole frame once the frame is known', () => {
    const draft = neutralDraft();
    draft.geometry.crop = { x: 0, y: 0, width: 200, height: 100 };
    expect(draftChanged(draft, { width: 200, height: 100 })).toBe(false);
    expect(draftChanged(draft, { width: 200, height: 101 })).toBe(true);
  });
});

describe('adjustBody', () => {
  const asset = makeAsset({ id: 'Zoo/a.png' });

  it('names the file, the save and every setting, neutral ones included', () => {
    const draft = neutralDraft();
    draft.light = { ...NEUTRAL_LIGHT, exposure: 10, highlights: -20, shadows: 15, vibrance: 20, warmth: 5, mix_orange: -40, mix_magenta: 25 };
    draft.geometry = { rotate: 90, flip: true, crop: { x: 0, y: 128, width: 1024, height: 1024 } };
    expect(adjustBody(asset, draft, 'copy', { width: 1500, height: 2000 })).toEqual({
      asset_id: 'Zoo/a.png',
      save: 'copy',
      rotate: 90,
      flip: true,
      crop: { x: 0, y: 128, width: 1024, height: 1024 },
      exposure: 10,
      contrast: 0,
      highlights: -20,
      shadows: 15,
      saturation: 0,
      vibrance: 20,
      warmth: 5,
      mix_red: 0,
      mix_orange: -40,
      mix_yellow: 0,
      mix_green: 0,
      mix_aqua: 0,
      mix_blue: 0,
      mix_purple: 0,
      mix_magenta: 25,
    });
  });

  it('sends the eight colour-mixer fields, zero when untouched', () => {
    const body = adjustBody(asset, neutralDraft(), 'copy', { width: 10, height: 10 });
    for (const name of MIX_NAMES) expect(body[`mix_${name}`]).toBe(0);
  });

  it('sends a crop that covers the whole frame as null, and carries the save mode', () => {
    const draft = neutralDraft();
    draft.geometry.crop = { x: 0, y: 0, width: 640, height: 480 };
    const body = adjustBody(asset, draft, 'replace', { width: 640, height: 480 });
    expect(body.crop).toBeNull();
    expect(body.save).toBe('replace');
  });

  it('keeps a crop of the whole upright size when a quarter turn makes the frame another shape', () => {
    const draft = neutralDraft();
    draft.geometry = { rotate: 90, flip: false, crop: { x: 0, y: 0, width: 640, height: 480 } };
    expect(adjustBody(asset, draft, 'copy', { width: 480, height: 640 }).crop).toEqual({ x: 0, y: 0, width: 640, height: 480 });
  });
});

describe('Reset all', () => {
  it('is a fresh neutral draft: every colour-mixer band back to 0', () => {
    const draft = neutralDraft();
    draft.light = { ...draft.light, mix_blue: 40, mix_red: -20 };
    expect(draftChanged(draft)).toBe(true);
    expect(draftChanged(neutralDraft())).toBe(false);
    expect(neutralDraft().light).toEqual(NEUTRAL_LIGHT);
  });
});

describe('canAdjust and canReplace', () => {
  it('adjusts an image whose file the server knows', () => {
    expect(canAdjust(makeAsset({ file_path: '/out/a.png' }))).toBe(true);
    expect(canAdjust(makeAsset({ file_path: undefined }))).toBe(false);
    expect(canAdjust(makeAsset({ file_path: '' }))).toBe(false);
    expect(canAdjust(makeAsset({ media_type: 'video', file_path: '/out/a.mp4' }))).toBe(false);
    expect(canAdjust(makeAsset({ media_type: 'audio', file_path: '/out/a.wav' }))).toBe(false);
  });

  it('replaces only a PNG, whatever the case of its extension', () => {
    expect(canReplace(makeAsset({ filename: 'a.png' }))).toBe(true);
    expect(canReplace(makeAsset({ filename: 'A.PNG' }))).toBe(true);
    expect(canReplace(makeAsset({ filename: 'a.jpg' }))).toBe(false);
    expect(canReplace(makeAsset({ filename: 'a.png.jpg' }))).toBe(false);
  });
});

describe('withAdjustedCopy', () => {
  const [a, b, c] = ['a', 'b', 'c'].map((name) => makeAsset({ id: `${name}.png`, filename: `${name}.png` }));
  const copy = makeAsset({ id: 'b_adjusted.png', filename: 'b_adjusted.png' });

  it('returns the list itself with nothing pinned', () => {
    const list = [a, b, c];
    expect(withAdjustedCopy(list, null)).toBe(list);
  });

  it('returns the list itself when it already holds the copy', () => {
    const list = [a, copy, b];
    expect(withAdjustedCopy(list, { asset: copy, besideId: 'b.png' })).toBe(list);
  });

  it('puts the copy right after its source', () => {
    const list = [a, b, c];
    const result = withAdjustedCopy(list, { asset: copy, besideId: 'b.png' });
    expect(result.map((item) => item.id)).toEqual(['a.png', 'b.png', 'b_adjusted.png', 'c.png']);
    expect(list).toHaveLength(3);
  });

  it('puts the copy first when there is no source to sit beside or the list does not hold it', () => {
    expect(withAdjustedCopy([a, b], { asset: copy, besideId: null }).map((item) => item.id)).toEqual(['b_adjusted.png', 'a.png', 'b.png']);
    expect(withAdjustedCopy([a, c], { asset: copy, besideId: 'b.png' })[0]).toBe(copy);
  });
});
