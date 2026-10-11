import { ApiError, errorText } from '$lib/api/client';
import { inspectExtendSource } from '$lib/api/extend';
import type { WorkspaceContext } from '$lib/types';
import { draft } from './draft.svelte';
import type { ExtendSourceStatus } from './extend';

// How the last inspection ended, and for which path; a status only describes the path it was made for.
let _status = $state<ExtendSourceStatus>('idle');
let _path = $state<string | null>(null);
let _failure = $state<string | null>(null);
// The inputs of the latest request, so an unchanged state is never inspected twice.
let _requestedKey: string | null = null;
let _requestedPath: string | null = null;
let _sequence = 0;
// The inspection in flight, so a caller can wait for its answer.
let _pending: Promise<void> | null = null;

function _keyFor(): string {
  const { extendSourcePath, extendFrames, model, lowMemory } = draft.state;
  return JSON.stringify([extendSourcePath, extendFrames, model, lowMemory]);
}

async function _inspect(ctx: WorkspaceContext | null, options: { replaceText?: boolean }): Promise<void> {
  const state = draft.state;
  const path = state.extendSourcePath;
  const request = ++_sequence;
  _requestedKey = _keyFor();
  _requestedPath = path;
  if (!path) {
    _status = 'idle';
    _path = null;
    _failure = null;
    draft.clearExtendSource();
    return;
  }
  _path = path;
  _status = 'checking';
  _failure = null;
  try {
    const info = await inspectExtendSource({ path, extend_frames: state.extendFrames, model: state.model || null, low_memory: state.lowMemory });
    if (request !== _sequence || draft.state.extendSourcePath !== path) return;
    draft.applyExtendSource(info, ctx, options);
    _status = 'ready';
    // The answer sized itself for the inputs it was asked about; filling the 2 s default length is not a change worth asking again.
    const after = draft.state;
    if (after.extendSourcePath === path && after.model === state.model && after.lowMemory === state.lowMemory && (state.extendFrames === null || after.extendFrames === state.extendFrames)) {
      _requestedKey = _keyFor();
    }
  } catch (error) {
    if (request !== _sequence || draft.state.extendSourcePath !== path) return;
    draft.clearExtendSource();
    if (error instanceof ApiError && error.status === 404) {
      _status = 'not_found';
    } else {
      _status = 'failed';
      _failure = errorText(error, 'The video could not be checked.');
    }
  }
}

/** Inspection of the extend source: asks the server about the chosen video and hands its answer to the draft. */
export const extendInspection = {
  /** The status for *path*; `idle` when the last inspection was for another path. */
  statusFor(path: string | null): ExtendSourceStatus {
    return path !== null && path === _path ? _status : 'idle';
  },

  /** Why the server could not read the source, when it failed for a reason other than a missing file. */
  failureFor(path: string | null): string | null {
    return path !== null && path === _path && _status === 'failed' ? _failure : null;
  },

  /** The path last inspected, so a change of length or model for the same source can wait for typing to settle. */
  requestedPath(): string | null {
    return _requestedPath;
  },

  /** Whether the draft's source, length, model and low-memory setting are the ones last inspected. */
  isCurrent(): boolean {
    return _requestedKey === _keyFor();
  },

  /**
   * Inspect the draft's source and apply the answer.
   *
   * Pass `replaceText` when the user just chose a new source, so its Scene and Prompt replace both boxes; a
   * re-inspection of the same source (page load, a new length, model or memory setting) refreshes the facts only.
   * A newer inspection supersedes an older one still in flight.
   */
  inspect(ctx: WorkspaceContext | null, options: { replaceText?: boolean } = {}): Promise<void> {
    const run = _inspect(ctx, options);
    const tracked: Promise<void> = run.finally(() => {
      if (_pending === tracked) _pending = null;
    });
    _pending = tracked;
    return tracked;
  },

  /** Resolve once the inspection in flight, if any, has been answered. */
  settled(): Promise<void> {
    return _pending ?? Promise.resolve();
  },

  /** Forget everything, for example when the draft is reset. */
  reset(): void {
    _sequence++;
    _status = 'idle';
    _path = null;
    _failure = null;
    _requestedKey = null;
    _requestedPath = null;
    _pending = null;
  },
};
