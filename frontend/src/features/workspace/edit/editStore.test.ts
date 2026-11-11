import { beforeEach, describe, expect, it } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { sameOps, type Stroke } from './editOps';
import { imageEdit, painterLayer } from './editStore.svelte';
import type { EditSource } from './editSource';
import { makeEditContext } from '../../../test-utils/workspace';

let nextId = 100;
const source = (overrides: Partial<EditSource> = {}): EditSource => ({
  id: nextId++, name: 'scene.png', path: '/out/scene.png', file: null, url: '/media/scene.png', thumbUrl: '/thumbs/scene.png', width: 800, height: 800, cutout: null, ...overrides,
});
const dab = (x = 400, y = 400, size = 200, erase = false): Stroke => ({ erase, size, points: [x, y] });

describe('the image to edit and its paint', () => {
  beforeEach(() => {
    localStorage.clear();
    // A new image starts the history over (`draft.editEpoch`).
    draft.reset();
    imageEdit.setSource(null);
    imageEdit.setBrush(64);
  });

  it('starts with nothing', () => {
    expect([imageEdit.source, imageEdit.strokes.length, imageEdit.extension, imageEdit.canUndo]).toEqual([null, 0, null, false]);
  });

  it('keeps the strokes it is given', () => {
    imageEdit.setSource(source());
    imageEdit.addStroke(dab());
    expect(imageEdit.strokes).toHaveLength(1);
    expect(imageEdit.canUndo).toBe(true);
  });

  it('undoes the last stroke', () => {
    imageEdit.setSource(source());
    imageEdit.addStroke(dab());
    imageEdit.undo();
    expect(imageEdit.strokes).toHaveLength(0);
    expect(imageEdit.canUndo).toBe(false);
  });

  it('clears the paint as one step that Undo brings back, and clears nothing when nothing is painted', () => {
    imageEdit.setSource(source());
    imageEdit.clearPaint();
    expect(imageEdit.canUndo).toBe(false);
    imageEdit.addStroke(dab());
    imageEdit.clearPaint();
    expect(imageEdit.strokes).toHaveLength(0);
    imageEdit.undo();
    expect(imageEdit.strokes).toHaveLength(1);
  });

  it('starts an extension on the side chosen, with the amount set, and removes it when the side is chosen again', () => {
    imageEdit.setSource(source());
    imageEdit.toggleSide('top');
    expect(imageEdit.extension).toEqual({ side: 'top', amount: 256 });
    imageEdit.toggleSide('top');
    expect(imageEdit.extension).toBeNull();
  });

  it('extends one side at a time', () => {
    imageEdit.setSource(source());
    imageEdit.toggleSide('top');
    imageEdit.toggleSide('right');
    expect(imageEdit.extension).toEqual({ side: 'right', amount: 256 });
  });

  it('changes the amount in steps of 16 within 128 to 512, as one step to undo', () => {
    imageEdit.setSource(source());
    imageEdit.toggleSide('top');
    imageEdit.setExtendAmount(300);
    imageEdit.setExtendAmount(9999);
    expect(imageEdit.extension?.amount).toBe(512);
    imageEdit.setExtendAmount(1);
    expect(imageEdit.extension?.amount).toBe(128);
    imageEdit.undo();
    expect(imageEdit.extension).toBeNull();
  });

  it('keeps the amount for the next side when there is no extension', () => {
    imageEdit.setSource(source());
    imageEdit.setExtendAmount(400);
    imageEdit.toggleSide('left');
    expect(imageEdit.extension).toEqual({ side: 'left', amount: 400 });
  });

  it('keeps the brush between 24 and 256 px', () => {
    imageEdit.setBrush(1);
    expect(imageEdit.brush).toBe(24);
    imageEdit.setBrush(1000);
    expect(imageEdit.brush).toBe(256);
  });

  it('forgets paint and extension when the draft starts the image over (a new image, a Reset, a Reuse)', () => {
    imageEdit.setSource(source());
    imageEdit.addStroke(dab());
    imageEdit.toggleSide('top');
    draft.setEditSource({ path: '/out/other.png' });
    expect([imageEdit.strokes.length, imageEdit.extension, imageEdit.canUndo]).toEqual([0, null, false]);
    imageEdit.addStroke(dab());
    const context = makeEditContext();
    draft.setEditing(true, context);
    draft.resetSelections(context);
    expect([imageEdit.strokes.length, imageEdit.canUndo]).toEqual([0, false]);
  });

  it('keeps its history through a Reset of New, which leaves the image to edit alone', () => {
    const context = makeEditContext();
    draft.setEditSource({ path: '/out/scene.png' });
    imageEdit.setSource(source());
    imageEdit.addStroke(dab());
    draft.resetSelections(context);
    expect(imageEdit.canUndo).toBe(true);
  });

  it('keeps its history when the same image moves: showing it at its new path is not a new image', () => {
    imageEdit.setSource(source());
    imageEdit.addStroke(dab());
    draft.setEditSource({ path: '/out/scene.png' });
    imageEdit.addStroke(dab());
    draft.retargetPaths(new Map([['/out/scene.png', '/out/shots/scene.png']]));
    imageEdit.setSource(source({ path: '/out/shots/scene.png' }));
    expect(imageEdit.canUndo).toBe(true);
    expect(imageEdit.strokes.length).toBeGreaterThan(0);
  });

  it('opens on the extension the draft holds, and the history has the last word', () => {
    draft.commitEditPaint({ mask: null, share: 0, extension: { side: 'left', amount: 320 }, amount: 320 });
    expect(imageEdit.extension).toEqual({ side: 'left', amount: 320 });
    expect(imageEdit.extendAmount).toBe(320);
    imageEdit.toggleSide('left');
    expect(imageEdit.extension).toBeNull();
    imageEdit.undo();
    expect(imageEdit.extension).toEqual({ side: 'left', amount: 320 });
  });

  it('undoes an extension kept with Done once the painter is opened again', () => {
    imageEdit.setSource(source());
    imageEdit.toggleSide('top');
    draft.commitEditPaint({ mask: null, share: 0, extension: imageEdit.extension, amount: 256 });
    expect(imageEdit.extension).toEqual({ side: 'top', amount: 256 });
    imageEdit.undo();
    expect(imageEdit.extension).toBeNull();
  });

  it('sits its history on the paint the draft holds when it has none yet, and shows that paint until a Clear removes it', () => {
    const held = new File(['png'], 'mask.png', { type: 'image/png' });
    imageEdit.setSource(source());
    imageEdit.begin(held);
    expect([imageEdit.base, imageEdit.baseShown]).toEqual([held, true]);
    imageEdit.clearPaint();
    expect(imageEdit.baseShown).toBe(false);
    imageEdit.undo();
    expect(imageEdit.baseShown).toBe(true);
  });

  it('leaves the paint under the history alone once a history exists', () => {
    const held = new File(['png'], 'mask.png', { type: 'image/png' });
    imageEdit.setSource(source());
    imageEdit.addStroke(dab());
    imageEdit.begin(held);
    expect(imageEdit.base).toBeNull();
  });

  it('comes back to how it was when the painter opened (Cancel)', () => {
    imageEdit.setSource(source());
    imageEdit.addStroke(dab(200, 200, 100));
    const opened = imageEdit.snapshot();
    const strokes = imageEdit.strokes.length;
    imageEdit.addStroke(dab(600, 600, 100));
    imageEdit.toggleSide('bottom');
    expect(sameOps(imageEdit.ops, opened)).toBe(false);
    imageEdit.restore(opened);
    expect(sameOps(imageEdit.ops, opened)).toBe(true);
    expect(imageEdit.strokes).toHaveLength(strokes);
    expect(imageEdit.extension).toBeNull();
  });

  it('keeps the paint when the painter closes with Done: nothing but Clear, a new image or Undo removes it', () => {
    imageEdit.setSource(source());
    imageEdit.addStroke(dab());
    painterLayer.show(null);
    painterLayer.hide();
    expect(imageEdit.strokes.length).toBeGreaterThan(0);
  });

  it('learns the size of an image later, and drops an answer about an image that is gone', () => {
    const first = source({ width: null, height: null });
    imageEdit.setSource(first);
    imageEdit.learn(first.id, { width: 640, height: 480 });
    expect([imageEdit.source?.width, imageEdit.source?.height]).toEqual([640, 480]);
    const second = source({ width: null, height: null });
    imageEdit.setSource(second);
    imageEdit.learn(first.id, { width: 1, height: 1 });
    expect(imageEdit.source?.width).toBeNull();
    imageEdit.learn(second.id, { cutout: true });
    expect(imageEdit.source?.cutout).toBe(true);
  });

  it('opens and closes the painter layer', () => {
    expect(painterLayer.open).toBe(false);
    painterLayer.show(null);
    expect(painterLayer.open).toBe(true);
    painterLayer.hide();
    expect(painterLayer.open).toBe(false);
  });
});
