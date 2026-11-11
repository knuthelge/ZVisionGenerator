import { describe, expect, it } from 'vitest';

import {
  BRUSH_DEFAULT, BRUSH_MAX, BRUSH_MIN, EXTEND_MAX, EXTEND_MIN, clampBrush, clampExtend, contentRect, extendedSize, extensionOf, extensionRect,
  pushOp, sameOps, strokesOf, undoOp, type EditOp, type Stroke,
} from './editOps';

const stroke = (size = BRUSH_DEFAULT, erase = false): EditOp => ({ kind: 'stroke', erase, size, points: [10, 10, 20, 20] });

describe('the history of the paint', () => {
  it('keeps the strokes after the last Clear, and nothing before it', () => {
    const ops: EditOp[] = [stroke(24), stroke(32), { kind: 'clear' }, stroke(48)];
    expect(strokesOf(ops).map((s: Stroke) => s.size)).toEqual([48]);
  });

  it('brings the paint back when Clear is undone', () => {
    const ops: EditOp[] = [stroke(24), stroke(32), { kind: 'clear' }];
    expect(strokesOf(ops)).toHaveLength(0);
    expect(strokesOf(undoOp(ops)).map((s) => s.size)).toEqual([24, 32]);
  });

  it('reads the extension from the last extend op, and none once it was removed', () => {
    const ops: EditOp[] = [{ kind: 'extend', side: 'top', amount: 256 }, stroke(), { kind: 'extend', side: 'left', amount: 128 }];
    expect(extensionOf(ops)).toEqual({ side: 'left', amount: 128 });
    expect(extensionOf([...ops, { kind: 'extend', side: null, amount: 128 }])).toBeNull();
    expect(extensionOf([stroke()])).toBeNull();
  });

  it('merges a change of amount into the step before it, so dragging the size is one step to undo', () => {
    let ops: EditOp[] = [{ kind: 'extend', side: 'top', amount: 256 }];
    for (const amount of [272, 288, 304]) ops = pushOp(ops, { kind: 'extend', side: 'top', amount }, true);
    expect(ops).toHaveLength(1);
    expect(extensionOf(ops)?.amount).toBe(304);
  });

  it('does not merge a new side or a removal into the one before it', () => {
    let ops: EditOp[] = [{ kind: 'extend', side: 'top', amount: 256 }];
    ops = pushOp(ops, { kind: 'extend', side: 'left', amount: 256 }, true);
    ops = pushOp(ops, { kind: 'extend', side: null, amount: 256 }, true);
    expect(ops).toHaveLength(3);
  });

  it('undoes one op at a time and tells a changed history from the one it started from', () => {
    const start: EditOp[] = [stroke()];
    const later = pushOp(start, stroke());
    expect(sameOps(start, later)).toBe(false);
    expect(sameOps(start, undoOp(later))).toBe(true);
    expect(undoOp([])).toEqual([]);
  });
});

describe('sizes', () => {
  it('keeps the brush within 24 to 256 px on steps of 4', () => {
    expect([clampBrush(0), clampBrush(1000), clampBrush(65)]).toEqual([BRUSH_MIN, BRUSH_MAX, 64]);
  });

  it('keeps an extension within 128 to 512 px on steps of 16', () => {
    expect([clampExtend(0), clampExtend(9999), clampExtend(250)]).toEqual([EXTEND_MIN, EXTEND_MAX, 256]);
  });

  it('adds the extension to one dimension only', () => {
    expect(extendedSize(832, 1216, { side: 'top', amount: 256 })).toEqual({ width: 832, height: 1472 });
    expect(extendedSize(832, 1216, { side: 'right', amount: 128 })).toEqual({ width: 960, height: 1216 });
    expect(extendedSize(832, 1216, null)).toEqual({ width: 832, height: 1216 });
  });

  it('puts the canvas and the new area where the extension reaches', () => {
    expect(contentRect(832, 1216, { side: 'top', amount: 256 })).toEqual({ x: 0, y: -256, w: 832, h: 1472 });
    expect(contentRect(832, 1216, { side: 'left', amount: 128 })).toEqual({ x: -128, y: 0, w: 960, h: 1216 });
    expect(extensionRect(832, 1216, { side: 'bottom', amount: 256 })).toEqual({ x: 0, y: 1216, w: 832, h: 256 });
    expect(extensionRect(832, 1216, { side: 'right', amount: 128 })).toEqual({ x: 832, y: 0, w: 128, h: 1216 });
  });
});
