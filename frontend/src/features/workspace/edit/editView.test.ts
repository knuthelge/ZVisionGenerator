import { describe, expect, it } from 'vitest';

import { editPlaceholder, editReason, modeSummary, sizeLine, wholePictureEdit, type EditFacts } from './editView';

const facts = (overrides: Partial<EditFacts> = {}): EditFacts => ({
  mode: 'paint', hasImage: true, lost: null, painted: 0, extension: null, repaint: false, width: 832, height: 1216, cutout: false, ...overrides,
});
const extending = { mode: 'extend' as const, extension: { side: 'right' as const, amount: 256 } };

describe('why Generate is off in Edit', () => {
  it('asks for an image first', () => {
    expect(editReason(facts({ hasImage: false }))).toBe('Choose an image to edit');
  });

  it('lets Paint with nothing painted through on an opaque picture: the prompt changes the whole picture', () => {
    expect(editReason(facts())).toBeNull();
    expect(editReason(facts({ painted: 0.1 }))).toBeNull();
  });

  it('needs a side in Extend', () => {
    expect(editReason(facts({ mode: 'extend' }))).toBe('Choose a side to extend');
    expect(editReason(facts(extending))).toBeNull();
  });

  it('says why an image that cannot be used is no image to edit', () => {
    expect(editReason(facts({ lost: 'Choose this image again.' }))).toBe('Choose this image again.');
  });

  it('asks a reused edit for its paint again, so it never becomes a whole-picture edit by itself', () => {
    expect(editReason(facts({ repaint: true }))).toBe('Paint the area again');
    expect(editReason(facts({ repaint: true, painted: 0.1 }))).toBeNull();
  });

  it('has no paint to ask for in Extend', () => {
    expect(editReason(facts({ ...extending, repaint: true }))).toBeNull();
  });

  it('refuses an empty Paint on a cut-out, and lets it through on an opaque image or while that is unknown', () => {
    expect(editReason(facts({ cutout: true }))).toBe('Paint the area to change');
    expect(editReason(facts({ cutout: false }))).toBeNull();
    expect(editReason(facts({ cutout: null }))).toBeNull();
    expect(editReason(facts({ cutout: true, painted: 0.1 }))).toBeNull();
  });
});

describe('what Edit says about itself', () => {
  it('hints at what should be there in the marked area, and at what should change when nothing is marked', () => {
    expect(editPlaceholder(facts())).toBe('What should change?');
    expect(editPlaceholder(facts({ painted: 0.2 }))).toBe('What should be here?');
    expect(editPlaceholder(facts(extending))).toBe('What should be here?');
  });

  it('words the mode line for each state: the mode and what it holds', () => {
    expect(modeSummary(facts({ painted: 0.06 }))).toEqual({ mode: 'paint', label: 'Paint', detail: '6% of the picture', tone: 'normal' });
    expect(modeSummary(facts({ painted: 0.002 })).detail).toBe('under 1% of the picture');
    expect(modeSummary(facts())).toMatchObject({ mode: 'paint', detail: 'Whole picture, from the prompt', spoken: 'Nothing painted, so the prompt changes the whole picture', tone: 'normal' });
    expect(modeSummary(facts({ cutout: true })).detail).toBe('Nothing painted yet');
    expect(modeSummary(facts({ repaint: true }))).toMatchObject({ detail: 'Paint the area again', tone: 'warning' });
    expect(modeSummary(facts(extending))).toEqual({ mode: 'extend', label: 'Extend', detail: 'Right, 256 px · 1088 × 1216', tone: 'normal' });
    expect(modeSummary(facts({ ...extending, width: null, height: null })).detail).toBe('Right, 256 px');
    expect(modeSummary(facts({ mode: 'extend' }))).toMatchObject({ mode: 'extend', detail: 'No side chosen yet', tone: 'muted' });
  });

  it('tells a whole-picture edit from one with a region', () => {
    expect(wholePictureEdit(facts())).toBe(true);
    expect(wholePictureEdit(facts({ painted: 0.1 }))).toBe(false);
    expect(wholePictureEdit(facts({ repaint: true }))).toBe(false);
    expect(wholePictureEdit(facts(extending))).toBe(false);
    expect(wholePictureEdit(facts({ cutout: true }))).toBe(false);
  });

  it('gives the painter its size line', () => {
    expect(sizeLine(832, 1216, null)).toBe('832 × 1216');
    expect(sizeLine(832, 1216, { side: 'bottom', amount: 256 })).toBe('832 × 1216 → 832 × 1472');
  });
});
