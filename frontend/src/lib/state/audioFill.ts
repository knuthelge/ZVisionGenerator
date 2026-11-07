import type { FillEnhanceRequest } from '$lib/api/promptEnhance';
import { DEFAULT_SPEECH_DELIVERY, DELIVERY_FIELDS } from './audioDraft';
import { LYRICS_CONTEXT_CHARS } from './audioEnhance';
import { songHasWords } from './lyricsHelper';
import { pushVersion } from './lyricsVersions';
import { sameDelivery } from './speechDelivery';
import { parseSong } from './songBlocks';
import type {
  AudioDescribe,
  AudioSettingsVocabulary,
  DescribeKind,
  DraftState,
  FillField,
  FillSnapshot,
  SpeechDeliveryValues,
} from '$lib/types';

/**
 * Describe it and Fill in, as pure rules over the draft.
 *
 * Describe it is the only place that writes from an idea. Fill in fills the fields the user has left empty, and the fields it filled before while
 * they are still as written (marked ✦); it never touches a field that is the user's. A field the user edits after the fill is theirs again:
 * a mark lasts exactly as long as the field holds the value Fill in gave it.
 */

/** The fields each kind's Fill in can write, in the order the summary names them. */
export const FILL_FIELDS: Record<DescribeKind, readonly FillField[]> = {
  music: ['style', 'lyrics', 'tempo', 'key'],
  speech: ['text', 'vibe'],
};

/** How a field is named in the summary. */
export const FILL_NAMES: Record<FillField, string> = { style: 'Style', lyrics: 'Lyrics', tempo: 'Tempo', key: 'Key', text: 'Text', vibe: 'Vibe' };

/** The longest idea the server takes. */
export const DESCRIBE_MAX_CHARS = 1000;

/** The controls the open row shows, which decide what Fill in may write (Stable Audio has no lyrics; a recording has no Delivery). */
export interface FillShown {
  showLyrics: boolean;
  showTempo: boolean;
  showKey: boolean;
  showDelivery: boolean;
}

/** Whether *kind* has a Describe it box. */
export function hasDescribe(kind: string | null): kind is DescribeKind {
  return kind === 'music' || kind === 'speech';
}

/** The fields Fill in could write for *kind* on the open row: those whose control the row shows. */
export function fillable(kind: DescribeKind, shown: FillShown, recording = false): FillField[] {
  return FILL_FIELDS[kind].filter((field) => {
    if (field === 'lyrics') return shown.showLyrics;
    if (field === 'tempo') return shown.showTempo;
    if (field === 'key') return shown.showKey;
    if (field === 'vibe') return shown.showDelivery && !recording;
    return true;
  });
}

/**
 * Whether the user has left *field* empty: Style = the Style text is blank, Lyrics = no words
 * (blank, whitespace or only `[section]` tags: the sections are kept for Fill in to write into), Tempo and Key = Auto, Vibe = all five Delivery values at their defaults, Text = blank.
 */
export function isEmpty(state: DraftState, field: FillField, defaults: SpeechDeliveryValues = DEFAULT_SPEECH_DELIVERY): boolean {
  switch (field) {
    case 'style':
      return state.audioStyle.trim() === '';
    case 'lyrics':
      return !songHasWords(parseSong(state.lyrics));
    case 'tempo':
      return state.audioTempo === null;
    case 'key':
      return state.audioKey === null || state.audioKey === '';
    case 'vibe':
      return sameDelivery(state.speechDelivery, defaults);
    case 'text':
      return state.speechText.trim() === '';
  }
}

/** The five delivery values as one string, so a Vibe filled in can be compared with what the Delivery section holds now. */
function deliveryKey(values: SpeechDeliveryValues): string {
  return DELIVERY_FIELDS.map((field) => values[field]).join('|');
}

/** Whether *field* still holds the value Fill in gave it. */
function holdsFilled(state: DraftState, field: FillField, filled: string): boolean {
  switch (field) {
    case 'style':
      return state.audioStyle === filled;
    case 'lyrics':
      return state.lyrics === filled;
    case 'tempo':
      return state.audioTempo !== null && String(state.audioTempo) === filled;
    case 'key':
      return state.audioKey === filled;
    case 'vibe':
      return deliveryKey(state.speechDelivery) === filled;
    case 'text':
      return state.speechText === filled;
  }
}

/**
 * The Describe it box with the marks of the fields the user has edited since taken off: a field that no longer holds what Fill in wrote is the
 * user's. The record of what was written goes with the mark.
 */
export function unmarkOnEdit(describe: AudioDescribe, state: DraftState): AudioDescribe {
  const marked = describe.marked.filter((field) => {
    const filled = describe.filled[field];
    return filled !== undefined && holdsFilled(state, field, filled);
  });
  if (marked.length === describe.marked.length) return describe;
  return { ...describe, marked, filled: Object.fromEntries(marked.map((field) => [field, describe.filled[field]])) };
}

/** The fields of *kind* that carry a ✦ now. */
export function markedFields(state: DraftState, kind: DescribeKind): FillField[] {
  return unmarkOnEdit(state.audioDescribe[kind], state).marked;
}

/** The requested fields Fill in would write now: those that are empty or still marked ✦, in the order of the summary. */
export function fillTargets(state: DraftState, kind: DescribeKind, requested: readonly FillField[], defaults: SpeechDeliveryValues = DEFAULT_SPEECH_DELIVERY): FillField[] {
  const marked = markedFields(state, kind);
  return FILL_FIELDS[kind].filter((field) => requested.includes(field) && (marked.includes(field) || isEmpty(state, field, defaults)));
}

/** The requested fields Fill in would leave alone: the user's own. */
export function keptFields(state: DraftState, kind: DescribeKind, requested: readonly FillField[], defaults: SpeechDeliveryValues = DEFAULT_SPEECH_DELIVERY): FillField[] {
  const targets = fillTargets(state, kind, requested, defaults);
  return FILL_FIELDS[kind].filter((field) => requested.includes(field) && !targets.includes(field));
}

/** *fields* as the names the summary uses: "Style, Lyrics and Tempo". */
export function fieldList(fields: readonly FillField[]): string {
  const names = fields.map((field) => FILL_NAMES[field]);
  return names.length < 2 ? names.join('') : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

/** What the box says before Fill in runs: what it will write and what it keeps. */
export function fillHint(targets: readonly FillField[], kept: readonly FillField[]): string {
  if (targets.length === 0) return 'Every field is yours, so Fill in has nothing to write. Clear one to have it filled.';
  return `Will fill ${fieldList(targets)}${kept.length > 0 ? ` · keeps your ${fieldList(kept)}` : ''}.`;
}

/** What the box says after Fill in ran: the fields still marked ✦, which are what Undo fill restores, as "Filled Style, Lyrics and Tempo". */
export function fillSummary(marked: readonly FillField[]): string {
  return marked.length > 0 ? `Filled ${fieldList(marked)}` : 'Nothing was filled';
}

/** Whether the box shows as one line: folded by hand, or never opened while the Lyrics or Text already have words. */
export function describeFolded(describe: AudioDescribe, state: DraftState, kind: DescribeKind): boolean {
  if (describe.folded !== null) return describe.folded;
  return kind === 'music' ? songHasWords(parseSong(state.lyrics)) : state.speechText.trim() !== '';
}

/** What Fill in is told besides the idea: the fields the user already set, so the new ones suit them. */
export function fillContext(state: DraftState, kind: DescribeKind, shown: FillShown): NonNullable<FillEnhanceRequest['context']> {
  if (kind === 'speech') return state.speechText.trim() === '' ? {} : { text: state.speechText.slice(0, LYRICS_CONTEXT_CHARS) };
  return {
    ...(state.audioStyle.trim() === '' ? {} : { style: state.audioStyle }),
    ...(shown.showLyrics && state.lyrics.trim() !== '' ? { lyrics: state.lyrics.slice(0, LYRICS_CONTEXT_CHARS) } : {}),
    ...(shown.showTempo && state.audioTempo !== null ? { tempo: state.audioTempo } : {}),
    ...(shown.showKey && state.audioKey ? { key: state.audioKey } : {}),
  };
}

/** The Fill in request for the idea *describe*: the fields still to write and what is already set. */
export function fillRequest(model: string, kind: DescribeKind, describe: string, fields: readonly FillField[], context: FillEnhanceRequest['context']): FillEnhanceRequest {
  return { mode: 'audio', task: 'fill', model, kind, prompt: describe, settings: { fields: [...fields] }, ...(context && Object.keys(context).length > 0 ? { context } : {}) };
}

/** The draft keys each field lives in, which Undo fill puts back. */
const FIELD_KEYS: Record<FillField, readonly (keyof DraftState)[]> = {
  style: ['audioStyle'],
  lyrics: ['lyrics', 'lyricsVersions', 'lyricsCurrent'],
  tempo: ['audioTempo'],
  key: ['audioKey'],
  vibe: ['speechDelivery'],
  text: ['speechText'],
};

/** A fill's result for the draft: the fields to set, which fields it wrote, and the Describe it box after it. */
export interface AppliedFill {
  patch: Partial<DraftState>;
  wrote: FillField[];
}

/** What a finished Fill in hands over: the fields it wrote (as the server's frame holds them) and the vocabulary to read a Vibe in. */
export interface FillOutcome {
  fields: Record<string, string>;
  vocabulary?: Pick<AudioSettingsVocabulary, 'vibes'>;
}

/**
 * Write what Fill in returned into the draft, for the *targets* only.
 *
 * - Style is written into the Style box as your own text, like Speech's Text.
 * - Lyrics replace an empty song, or add a version to the history of a song Fill in wrote.
 * - Tempo and Key are set; Text goes in the Text box; a Vibe sets the five delivery values.
 *
 * The result holds the Describe it box too: the written fields marked, folded to one line, with a snapshot of everything it changed.
 * A field the answer lacks or cannot be read is left as it was.
 */
export function applyFill(state: DraftState, kind: DescribeKind, targets: readonly FillField[], outcome: FillOutcome): AppliedFill {
  const patch: Partial<DraftState> = {};
  const wrote: FillField[] = [];
  const filled: Partial<Record<FillField, string>> = {};
  const values: Partial<DraftState> = {};
  const snapshot = (field: FillField): void => {
    for (const key of FIELD_KEYS[field]) (values as Record<string, unknown>)[key] = state[key];
  };
  const set = (field: FillField, part: Partial<DraftState>, value: string): void => {
    snapshot(field);
    Object.assign(patch, part);
    wrote.push(field);
    filled[field] = value;
  };

  for (const field of FILL_FIELDS[kind]) {
    const value = outcome.fields[field];
    if (!targets.includes(field) || value === undefined || value.trim() === '') continue;
    if (field === 'style') {
      set(field, { audioStyle: value }, value);
    } else if (field === 'lyrics') {
      set(field, { lyrics: value, ...pushVersion(state, value) }, value);
    } else if (field === 'tempo') {
      const tempo = Number(value);
      if (Number.isInteger(tempo)) set(field, { audioTempo: tempo }, String(tempo));
    } else if (field === 'key') {
      set(field, { audioKey: value }, value);
    } else if (field === 'vibe') {
      const vibe = outcome.vocabulary?.vibes.find((option) => option.id === value);
      if (vibe) set(field, { speechDelivery: { ...vibe.delivery } }, deliveryKey(vibe.delivery));
    } else {
      set(field, { speechText: value }, value);
    }
  }

  const describe = state.audioDescribe[kind];
  const before = unmarkOnEdit(describe, state);
  const undo: FillSnapshot | null = wrote.length === 0 ? describe.undo : { values, marked: before.marked, filled: before.filled };
  patch.audioDescribe = {
    ...state.audioDescribe,
    [kind]: {
      ...describe,
      marked: [...before.marked.filter((field) => !wrote.includes(field)), ...wrote],
      filled: { ...before.filled, ...filled },
      undo,
      folded: true,
    },
  };
  return { patch, wrote };
}

/** Whether Undo fill has something to put back: a fill was made and at least one field it wrote is still as written. */
export function canUndoFill(state: DraftState, kind: DescribeKind): boolean {
  return state.audioDescribe[kind].undo !== null && markedFields(state, kind).length > 0;
}

/**
 * The fields to set to put back what the last fill replaced, exactly as it was: only for the fields still marked, since a field the user
 * has edited since is theirs and stays. Null when there is nothing to undo.
 */
export function undoFill(state: DraftState, kind: DescribeKind): Partial<DraftState> | null {
  const describe = state.audioDescribe[kind];
  const snapshot = describe.undo;
  if (snapshot === null || !canUndoFill(state, kind)) return null;
  const restore = markedFields(state, kind).filter((field) => FIELD_KEYS[field].every((key) => key in snapshot.values));
  const patch: Partial<DraftState> = {};
  for (const field of restore) {
    for (const key of FIELD_KEYS[field]) (patch as Record<string, unknown>)[key] = snapshot.values[key];
  }
  // The fields put back are marked as they were before the fill; the others keep the marks they have.
  const live = unmarkOnEdit(describe, state);
  const stays = live.marked.filter((field) => !restore.includes(field));
  const was = snapshot.marked.filter((field) => restore.includes(field));
  patch.audioDescribe = {
    ...state.audioDescribe,
    [kind]: {
      ...describe,
      marked: [...stays, ...was],
      filled: { ...Object.fromEntries(stays.map((field) => [field, live.filled[field]])), ...Object.fromEntries(was.map((field) => [field, snapshot.filled[field]])) },
      undo: null,
    },
  };
  return patch;
}
