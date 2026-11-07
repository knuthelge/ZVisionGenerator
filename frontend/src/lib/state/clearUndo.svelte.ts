import type { DraftState } from '$lib/types';
import { draft } from './draft.svelte';

/** What a Clear took: the fields as they were and as Clear left them, and whatever the box shows besides (its open tab). */
export interface TakenFields<V> {
  before: Partial<DraftState>;
  after: Partial<DraftState>;
  view: V;
}

/** The fields of *state* that *cleared* sets, as they are now. */
function pick(state: DraftState, cleared: Partial<DraftState>): Partial<DraftState> {
  const fields: Record<string, unknown> = {};
  for (const key of Object.keys(cleared) as (keyof DraftState)[]) fields[key] = state[key];
  return fields as Partial<DraftState>;
}

/** Remember the fields *cleared* is about to set, with the *view* that Undo brings back. */
export function takeFields<V>(state: DraftState, cleared: Partial<DraftState>, view: V): TakenFields<V> {
  return { before: pick(state, cleared), after: cleared, view };
}

/** Whether the fields still hold what Clear left: nothing was typed, run or stepped since (the draft's values are plain JSON). */
export function stillCleared<V>(state: DraftState, taken: TakenFields<V>): boolean {
  return JSON.stringify(pick(state, taken.after)) === JSON.stringify(taken.after);
}

/**
 * Undo for a box's Clear: the fields come back until the next edit or run in the box (the fields no longer hold what Clear left)
 * or another Clear. Every Clear-and-Undo of a text box goes through one of these.
 */
export class ClearUndo<V = null> {
  #taken = $state<TakenFields<V> | null>(null);

  /** Whether Undo is offered now. */
  get offered(): boolean {
    return this.#taken !== null && stillCleared(draft.state, this.#taken);
  }

  /** Set *cleared* in the draft and remember how to put it back, with the *view* the box showed. */
  clear(cleared: Partial<DraftState>, view: V): void {
    this.#taken = takeFields(draft.state, cleared, view);
    draft.patch(cleared);
  }

  /** Put the fields back; returns the view the box showed, or null when Undo was not offered. */
  undo(): { view: V } | null {
    const taken = this.#taken;
    this.#taken = null;
    if (taken === null || !stillCleared(draft.state, taken)) return null;
    draft.patch(taken.before);
    return { view: taken.view };
  }

  /** Drop Undo: the box was edited or run. */
  forget(): void {
    this.#taken = null;
  }
}
