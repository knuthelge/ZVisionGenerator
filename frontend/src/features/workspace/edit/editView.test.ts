import { describe, expect, it } from 'vitest';

import { cardSummary, editPlaceholder, editReason, extendSummary, paintSummary, sizeLine, type EditFacts } from './editView';

const facts = (overrides: Partial<EditFacts> = {}): EditFacts => ({
  hasImage: true, lost: null, painted: 0, extension: null, repaint: false, width: 832, height: 1216, cutout: false, wholePicture: false, ...overrides,
});

describe('why Generate is off in Edit', () => {
  it('asks for an image first', () => {
    expect(editReason(facts({ hasImage: false, wholePicture: true }))).toBe('Choose an image to edit');
  });

  it('holds Generate back until Z-Image and Krea 2 have a region, and Klein needs none', () => {
    expect(editReason(facts())).not.toBeNull();
    expect(editReason(facts({ painted: 0.1 }))).toBeNull();
    expect(editReason(facts({ wholePicture: true }))).toBeNull();
  });

  it('counts paint or an extension as the area to change', () => {
    expect(editReason(facts({ painted: 0.1 }))).toBeNull();
    expect(editReason(facts({ extension: { side: 'top', amount: 256 } }))).toBeNull();
  });

  it('says why an image that cannot be used is no image to edit', () => {
    expect(editReason(facts({ lost: 'Choose this image again.', wholePicture: true }))).toBe('Choose this image again.');
  });

  it('asks a reused edit for its paint again, on Klein too, so it never becomes a whole-picture edit by itself', () => {
    expect(editReason(facts({ repaint: true }))).toBe('Paint the area again');
    expect(editReason(facts({ repaint: true, wholePicture: true }))).toBe('Paint the area again');
    expect(editReason(facts({ repaint: true, painted: 0.1 }))).toBeNull();
  });

  it('keeps asking for the paint behind an extension a reuse restored, until it is painted', () => {
    expect(editReason(facts({ repaint: true, extension: { side: 'top', amount: 256 } }))).toBe('Paint the area again');
  });

  it('refuses Klein with nothing painted on a cut-out, and lets it through on an opaque image or while that is unknown', () => {
    expect(editReason(facts({ wholePicture: true, cutout: true }))).toBe('Paint the area to change');
    expect(editReason(facts({ wholePicture: true, cutout: false }))).toBeNull();
    expect(editReason(facts({ wholePicture: true, cutout: null }))).toBeNull();
    expect(editReason(facts({ wholePicture: true, cutout: true, painted: 0.1 }))).toBeNull();
  });
});

describe('what Edit says about itself', () => {
  it('hints at what should be there with paint, and at what should change without', () => {
    expect(editPlaceholder(facts())).toBe('What should change?');
    expect(editPlaceholder(facts({ painted: 0.2 }))).toBe('What should be here?');
    expect(editPlaceholder(facts({ extension: { side: 'left', amount: 128 } }))).toBe('What should be here?');
  });

  it('words the painted share, and says nothing when there is none', () => {
    expect(paintSummary(facts())).toBeNull();
    expect(paintSummary(facts({ painted: 0.18 }))).toBe('Painted area: 18% of the image');
    expect(paintSummary(facts({ painted: 0.002 }))).toBe('Painted area: under 1% of the image');
  });

  it('words the extension with the new size', () => {
    expect(extendSummary(facts())).toBeNull();
    expect(extendSummary(facts({ extension: { side: 'top', amount: 256 } }))).toBe('Extends top by 256 px, 832 × 1472');
    expect(extendSummary(facts({ extension: { side: 'left', amount: 128 }, width: null, height: null }))).toBe('Extends left by 128 px');
  });

  it('gives the phone card one short line', () => {
    expect(cardSummary(facts())).toBe('832 × 1216');
    expect(cardSummary(facts({ width: null, height: null }))).toBe('No paint yet');
    expect(cardSummary(facts({ painted: 0.18 }))).toBe('18% painted');
    expect(cardSummary(facts({ painted: 0.18, extension: { side: 'top', amount: 256 } }))).toBe('18% painted · extends top by 256 px');
    expect(cardSummary(facts({ extension: { side: 'left', amount: 128 } }))).toBe('Extends left by 128 px');
  });

  it('gives the painter its size line', () => {
    expect(sizeLine(832, 1216, null)).toBe('832 × 1216');
    expect(sizeLine(832, 1216, { side: 'bottom', amount: 256 })).toBe('832 × 1216 → 832 × 1472');
  });
});
