import { checkVideoMemoryFit } from '$lib/api/videoMemory';
import type { ClipMemoryFit } from '$lib/types';
import { INPUT_DEBOUNCE_MS } from './debounce';
import type { ClipMemoryRequest } from './memoryFit';

// The latest answer; null while unknown, after an error, and whenever the clip is not one to check.
let _fit = $state<ClipMemoryFit | null>(null);
// The inputs of the latest request (pending or answered), so unchanged inputs are never asked twice.
let _requestedKey: string | null = null;
let _requestedModel: string | null = null;
let _sequence = 0;
let _timer: ReturnType<typeof setTimeout> | undefined;

function _forget(): void {
  clearTimeout(_timer);
  _sequence++;
  _requestedKey = null;
  _requestedModel = null;
  _fit = null;
}

async function _ask(request: ClipMemoryRequest, sequence: number): Promise<void> {
  try {
    const answer = await checkVideoMemoryFit(request);
    if (sequence === _sequence) _fit = answer.memory_fit;
  } catch {
    // An incomplete or invalid clip, or a server that cannot answer: show nothing.
    if (sequence === _sequence) _fit = null;
  }
}

/** Whether the Text to Video or Image to Video clip being set up fits this machine's memory, as the server estimates it. */
export const clipMemory = {
  /** The latest fit for the clip last passed to `watch`; null when unknown. */
  get fit(): ClipMemoryFit | null {
    return _fit;
  },

  /**
   * Follow the clip the form would submit.
   *
   * Pass null when there is nothing to check (another workflow, an incomplete size): the fit is dropped at once. A
   * changed model drops the fit at once too, since it described another model; any other change asks the server
   * after the inputs have been still for a moment, and keeps the previous fit until it answers. Unchanged inputs ask
   * nothing, and an answer to an earlier request never replaces a later one.
   */
  watch(request: ClipMemoryRequest | null): void {
    if (request === null) {
      _forget();
      return;
    }
    const key = JSON.stringify(request);
    if (key === _requestedKey) return;
    clearTimeout(_timer);
    if (request.model !== _requestedModel) _fit = null;
    _requestedKey = key;
    _requestedModel = request.model;
    const sequence = ++_sequence;
    _timer = setTimeout(() => void _ask(request, sequence), INPUT_DEBOUNCE_MS);
  },

  /** Forget everything, for example when the draft is reset. */
  reset(): void {
    _forget();
  },
};
