import { checkMemory } from '$lib/api/memory';
import type { JobMemory } from '$lib/types';
import { INPUT_DEBOUNCE_MS } from './debounce';
import type { MemoryRequest } from './memoryRequest';

/** How long after a failed reply the same request is asked once more. */
export const MEMORY_RETRY_MS = 5000;

// The latest answer; null while unknown, after an error, and whenever there is nothing to check.
let _answer = $state<JobMemory | null>(null);
// True from a change of model or of kind of job until the reply for the latest request arrives: `_answer` is then still the previous job's.
let _pending = $state(false);
// True while the form cannot yet say what it would submit (a reference picture's size is being read): `_answer` is then of the job as it was.
let _held = $state(false);
// The inputs of the latest request (pending or answered), so unchanged inputs are never asked twice.
let _requestedKey: string | null = null;
let _requestedModel: string | null = null;
let _requestedKind: string | null = null;
// The body the form last passed to `watch`, so `refresh` can ask the same question again.
let _latestBody: MemoryRequest | null = null;
// The key whose failed reply has already been followed by its one retry; cleared by a change of key or a good reply.
let _retriedKey: string | null = null;
let _sequence = 0;
let _timer: ReturnType<typeof setTimeout> | undefined;

/**
 * What makes a request its own question. Quant and Low memory are left out: an answer carries a row for each level of them, so
 * changing either is read from the rows the tag already holds and asks nothing.
 */
function _keyOf(body: MemoryRequest): string {
  const { quantize: _quantize, low_memory: _lowMemory, ...rest } = body;
  return JSON.stringify(rest);
}

function _forget(): void {
  clearTimeout(_timer);
  _sequence++;
  _requestedKey = null;
  _requestedModel = null;
  _requestedKind = null;
  _latestBody = null;
  _retriedKey = null;
  _answer = null;
  _pending = false;
  _held = false;
}

/** Ask the failed request once more after a short wait; a second failure of the same key is final until the inputs change. */
function _retryOnce(body: MemoryRequest): void {
  const key = _keyOf(body);
  if (_retriedKey === key) return;
  _retriedKey = key;
  const sequence = ++_sequence;
  _timer = setTimeout(() => void _ask(body, sequence), MEMORY_RETRY_MS);
}

async function _ask(body: MemoryRequest, sequence: number): Promise<void> {
  try {
    const answer = await checkMemory(body);
    if (sequence === _sequence) {
      _answer = answer ?? null;
      _pending = false;
      _retriedKey = null;
    }
  } catch {
    // An incomplete or invalid request, or a server that cannot answer: show nothing, and ask once more shortly in case it was a blip.
    if (sequence === _sequence) {
      _answer = null;
      _pending = false;
      _retryOnce(body);
    }
  }
}

/** How the job being set up (an image, an edit, a video clip or an extension) fits this machine's memory, as the server judges it. */
export const memoryCheck = {
  /** The latest answer for the request last passed to `watch`; null when unknown. */
  get answer(): JobMemory | null {
    return _answer;
  },

  /**
   * Whether `answer` still describes the previous model or kind of job: true from the change until the reply for the latest request arrives,
   * so the tag can show it dimmed instead of vanishing. False once the answer is of the current model and kind.
   */
  get pending(): boolean {
    return _pending || _held;
  },

  /**
   * Mark the shown answer as not current, because the form cannot yet say what it would submit (a reference picture's size is still being read).
   * Nothing is asked; the next `watch` or `reset` ends it.
   */
  hold(): void {
    _held = true;
  },

  /**
   * Follow the request the form would submit.
   *
   * Pass a null body when there is nothing to check (another workflow, an incomplete size): the answer is dropped at once. A different
   * kind of job (another tab) or a changed model keeps the previous answer but marks it `pending`, since it describes another job or
   * model, until the reply arrives. Every change asks the server after the inputs have been still for a moment, and keeps the previous
   * answer until it replies; a failed or empty reply leaves no answer, and a failed one is asked once more after a few seconds. Unchanged
   * inputs ask nothing (changing only the Quant level or Low memory counts as unchanged; `refresh` asks again on purpose), and a reply to
   * an earlier request never replaces a later one.
   */
  watch(body: MemoryRequest | null): void {
    if (body === null) {
      _forget();
      return;
    }
    _latestBody = body;
    _held = false;
    const key = _keyOf(body);
    if (key === _requestedKey) return;
    clearTimeout(_timer);
    _retriedKey = null;
    const kind = `${body.media}:${body.kind}`;
    if (kind !== _requestedKind || body.model !== _requestedModel) _pending = true;
    _requestedKey = key;
    _requestedModel = body.model;
    _requestedKind = kind;
    const sequence = ++_sequence;
    _timer = setTimeout(() => void _ask(body, sequence), INPUT_DEBOUNCE_MS);
  },

  /**
   * Ask the latest request again although its inputs are unchanged, because what the server answers from can have changed (a model
   * finished downloading, a LoRA was added or removed). The model and kind are the same, so the answer stays shown, not pending,
   * until the new reply replaces it. Does nothing while there is nothing to check.
   */
  refresh(): void {
    if (_latestBody === null) return;
    clearTimeout(_timer);
    const body = _latestBody;
    const sequence = ++_sequence;
    _timer = setTimeout(() => void _ask(body, sequence), INPUT_DEBOUNCE_MS);
  },

  /** Forget everything, for example when the draft is reset. */
  reset(): void {
    _forget();
  },
};
