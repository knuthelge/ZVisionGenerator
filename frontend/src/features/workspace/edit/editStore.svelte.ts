import { draft } from '$lib/state/draft.svelte';
import {
  BRUSH_DEFAULT, clampBrush, clampExtend, extensionOf, pushOp, strokesOf, undoOp,
  type EditOp, type ExtendSide, type Extension, type Stroke,
} from './editOps';
import type { EditSource } from './editSource';

const NO_OPS: readonly EditOp[] = [];

/** What the painter did since the image to edit started over: its history, the paint it started from and the amount an extension starts with. */
interface Session {
  /** The `draft.editEpoch` it belongs to: a session of an earlier count is empty. */
  epoch: number;
  ops: readonly EditOp[];
  /** The PNG of the paint the history sits on (a mask held before the history began, such as one restored after a reload); null for none. */
  base: File | null;
  /** The amount a newly chosen side starts with; null until the user changes it. */
  amount: number | null;
  /** The extension the history sits on, taken from the draft when the first op was added; undefined until then. Undo goes back to it, not to what the draft holds now. */
  before?: Extension | null;
}

/**
 * The image to edit as the painter and the page see it, and the painter's working state.
 *
 * The draft owns what Generate sends: the image, the committed mask with its share, the extension. This store holds what the
 * painter needs on top of that and that is not saved: the strokes and the undo history, the brush, what is known about the image
 * (its size, whether it is a cut-out) and where its picture is drawn from. The history belongs to the draft's `editEpoch`:
 * a new image, a Reset or a Reuse empties it, a move does not.
 */
class ImageEditStore {
  #source = $state.raw<EditSource | null>(null);
  #session = $state.raw<Session>({ epoch: 0, ops: NO_OPS, base: null, amount: null });
  #brush = $state(BRUSH_DEFAULT);
  readonly #current = $derived<Session>(
    this.#session.epoch === draft.editEpoch ? this.#session : { epoch: draft.editEpoch, ops: NO_OPS, base: null, amount: null },
  );
  readonly #strokes = $derived(strokesOf(this.#current.ops));
  /** The extension the draft holds, which the painter opens with. */
  readonly #committed = $derived<Extension | null>(
    draft.state.editExtendSide ? { side: draft.state.editExtendSide, amount: draft.state.editExtendAmount } : null,
  );

  get source(): EditSource | null { return this.#source; }
  get ops(): readonly EditOp[] { return this.#current.ops; }
  get strokes(): Stroke[] { return this.#strokes; }
  /** The extension in effect: the history's last word, or the one it sits on (the draft's until the history has a first op). */
  get extension(): Extension | null { return extensionOf(this.#current.ops, this.#current.before === undefined ? this.#committed : this.#current.before); }
  /** The paint the history sits on, or null. It is gone from the picture once the history clears the paint. */
  get base(): File | null { return this.#current.base; }
  /** Whether the paint under the history still shows: there is one, and no Clear in the history has removed it. */
  get baseShown(): boolean { return this.#current.base !== null && !this.#current.ops.some((op) => op.kind === 'clear'); }
  /** Brush size in image pixels; kept between visits to the painter. */
  get brush(): number { return this.#brush; }
  /** The extension a newly chosen side starts with. */
  get extendAmount(): number { return this.#current.amount ?? draft.state.editExtendAmount; }
  get canUndo(): boolean { return this.#current.ops.length > 0; }

  /** Show another image. What the painter did is not touched: it follows the draft's `editEpoch`. */
  setSource(source: EditSource | null): void {
    this.#release(this.#source);
    this.#source = source;
  }

  /** Fill in what is learnt about the source later (its size, whether it is a cut-out). A late answer about an earlier image is dropped. */
  learn(id: number, facts: Partial<Pick<EditSource, 'width' | 'height' | 'cutout'>>): void {
    if (this.#source?.id !== id) return;
    const known = Object.entries(facts).filter(([key, value]) => value !== null && this.#source?.[key as keyof EditSource] !== value);
    if (known.length > 0) this.#source = { ...this.#source, ...Object.fromEntries(known) };
  }

  /** The painter opens: with no history yet, the paint the draft holds is what the history sits on. */
  begin(held: File | null): void {
    if (this.#current.ops.length > 0 || this.#current.base === held) return;
    this.#session = { ...this.#current, base: held };
  }

  setBrush(size: number): void {
    this.#brush = clampBrush(size);
  }

  addStroke(stroke: Stroke): void {
    this.#push({ kind: 'stroke', ...stroke });
  }

  /** Remove all paint, the paint under the history included. It is one step in the history, so Undo brings it back. */
  clearPaint(): void {
    if (this.#strokes.length > 0 || this.baseShown) this.#push({ kind: 'clear' });
  }

  /** Extend one side by the chosen amount; choosing the side that is already extended removes the extension. */
  toggleSide(side: ExtendSide): void {
    const now = this.extension;
    this.#push({ kind: 'extend', side: now?.side === side ? null : side, amount: now?.side === side ? now.amount : this.extendAmount });
  }

  /** Remove the extension. */
  clearExtension(): void {
    if (this.extension) this.#push({ kind: 'extend', side: null, amount: this.extendAmount });
  }

  /** Change how far the extension reaches. Without an extension it sets the amount the next side starts with. */
  setExtendAmount(amount: number): void {
    const next = clampExtend(amount);
    this.#session = { ...this.#current, amount: next };
    const now = this.extension;
    if (now) this.#push({ kind: 'extend', side: now.side, amount: next }, true);
  }

  undo(): void {
    this.#session = { ...this.#current, ops: undoOp(this.#current.ops) };
  }

  /** The history as it is now, to come back to with `restore` (Cancel). */
  snapshot(): readonly EditOp[] {
    return this.#current.ops;
  }

  restore(ops: readonly EditOp[]): void {
    this.#session = { ...this.#current, ops: [...ops] };
  }

  #push(op: EditOp, merge = false): void {
    const current = this.#current;
    this.#session = { ...current, before: current.before === undefined ? this.#committed : current.before, ops: pushOp(current.ops, op, merge) };
  }

  #release(source: EditSource | null): void {
    if (source?.file && source.url.startsWith('blob:')) URL.revokeObjectURL(source.url);
  }
}

export const imageEdit = new ImageEditStore();

/**
 * Whether the painter is open, and the control that opened it, which gets the focus back when it closes. Kept apart from the
 * image store because the sections that open it sit in different panes (Settings, the phone's card).
 */
class PainterLayer {
  #open = $state(false);
  #opener: HTMLElement | null = null;

  get open(): boolean { return this.#open; }

  show(opener: HTMLElement | null): void {
    this.#opener = opener;
    this.#open = true;
  }

  hide(): void {
    if (!this.#open) return;
    this.#open = false;
    const opener = this.#opener;
    this.#opener = null;
    queueMicrotask(() => { if (opener?.isConnected) opener.focus(); });
  }
}

export const painterLayer = new PainterLayer();
