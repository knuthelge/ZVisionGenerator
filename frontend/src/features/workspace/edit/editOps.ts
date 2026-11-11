/**
 * The painter's history as plain data. Every change to the paint or the extension is one op, so Undo is a pop and the
 * mask is always the replay of the ops. Nothing here touches a canvas.
 */

import { EXTEND_SIDES } from '$lib/state/editSlots';
import type { ExtendSide } from '$lib/types';

export type { ExtendSide };
export { EXTEND_SIDES };

/** The brush in image pixels. */
export const BRUSH_MIN = 24;
export const BRUSH_MAX = 256;
export const BRUSH_STEP = 4;
export const BRUSH_DEFAULT = 64;

/** How far an extension reaches, in image pixels; sizes stay aligned to 16 px. */
export const EXTEND_MIN = 128;
export const EXTEND_MAX = 512;
export const EXTEND_STEP = 16;

/** One brush or eraser stroke: its size in image pixels and its points as flat `x, y` pairs in image pixels. */
export interface Stroke {
  erase: boolean;
  size: number;
  points: number[];
}

export type EditOp =
  | ({ kind: 'stroke' } & Stroke)
  | { kind: 'clear' }
  | { kind: 'extend'; side: ExtendSide | null; amount: number };

/** One side of the picture pushed out by some pixels. */
export interface Extension {
  side: ExtendSide;
  amount: number;
}

/** A rectangle in image pixels; the picture itself starts at 0, 0, and an extension reaches into negative numbers. */
export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

const snap = (value: number, min: number, max: number, step: number): number => Math.min(max, Math.max(min, Math.round(value / step) * step));

/** The brush size within its range, on its step. */
export const clampBrush = (size: number): number => snap(size, BRUSH_MIN, BRUSH_MAX, BRUSH_STEP);

/** The extension within its range, on its step of 16. */
export const clampExtend = (amount: number): number => snap(amount, EXTEND_MIN, EXTEND_MAX, EXTEND_STEP);

/** The strokes that make up the paint: those after the last Clear. */
export function strokesOf(ops: readonly EditOp[]): Stroke[] {
  const strokes: Stroke[] = [];
  for (const op of ops) {
    if (op.kind === 'clear') strokes.length = 0;
    else if (op.kind === 'stroke') strokes.push(op);
  }
  return strokes;
}

/** The extension in effect: the last extend op, or none when it was removed; with no extend op at all, `before` (the extension the painter opened with). */
export function extensionOf(ops: readonly EditOp[], before: Extension | null = null): Extension | null {
  for (let index = ops.length - 1; index >= 0; index -= 1) {
    const op = ops[index];
    if (op.kind === 'extend') return op.side === null ? null : { side: op.side, amount: op.amount };
  }
  return before;
}

/**
 * Add an op. With `merge`, an extend op that only changes the amount of the extension in effect replaces the one before it,
 * so dragging the size is one step to undo, not one per tick.
 */
export function pushOp(ops: readonly EditOp[], op: EditOp, merge = false): EditOp[] {
  const last = ops[ops.length - 1];
  if (merge && op.kind === 'extend' && last?.kind === 'extend' && last.side !== null && last.side === op.side) {
    return [...ops.slice(0, -1), op];
  }
  return [...ops, op];
}

/** The ops without the last one. */
export function undoOp(ops: readonly EditOp[]): EditOp[] {
  return ops.slice(0, -1);
}

/** Whether two histories are the same one: ops are only ever added and removed, never changed, so identity is enough. */
export function sameOps(a: readonly EditOp[], b: readonly EditOp[]): boolean {
  return a.length === b.length && a.every((op, index) => op === b[index]);
}

/** The picture's size with its extension. */
export function extendedSize(width: number, height: number, extension: Extension | null): { width: number; height: number } {
  if (!extension) return { width, height };
  const horizontal = extension.side === 'left' || extension.side === 'right';
  return { width: width + (horizontal ? extension.amount : 0), height: height + (horizontal ? 0 : extension.amount) };
}

/** The whole canvas in image pixels: the picture and the new area. */
export function contentRect(width: number, height: number, extension: Extension | null): Rect {
  const size = extendedSize(width, height, extension);
  return {
    x: extension?.side === 'left' ? -extension.amount : 0,
    y: extension?.side === 'top' ? -extension.amount : 0,
    w: size.width,
    h: size.height,
  };
}

/** The new area an extension adds, in image pixels. */
export function extensionRect(width: number, height: number, extension: Extension): Rect {
  const { side, amount } = extension;
  if (side === 'top') return { x: 0, y: -amount, w: width, h: amount };
  if (side === 'bottom') return { x: 0, y: height, w: width, h: amount };
  if (side === 'left') return { x: -amount, y: 0, w: amount, h: height };
  return { x: width, y: 0, w: amount, h: height };
}
