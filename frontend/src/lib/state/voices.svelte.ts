// Module-level store of the saved voices, using Svelte 5 runes (the pattern of presets.svelte.ts).
// The Workspace reads the list when it mounts and again when the Voice section opens, so voices saved or removed elsewhere
// (another tab) show up. The mutations throw: the caller shows the server's reason.
//
// `loaded` turns true with the first successful read. Until then the list says nothing about which voices exist, so a draft
// that names a saved voice is not called deleted. Once it is true, a list without the chosen voice is the view's error; only
// this tab's own `remove` of the chosen voice moves the draft back to the default built-in voice.

import { clearCandidates, deleteVoice, getDesignState, listVoices, renameVoice, saveRecordingVoice, saveCandidate as saveCandidateVoice } from '$lib/api/voices';
import type { SavedVoice, VoiceDesignState } from '$lib/types';
import { draft } from './draft.svelte';

function byName(a: SavedVoice, b: SavedVoice): number {
  return a.name.toLowerCase().localeCompare(b.name.toLowerCase()) || a.slug.localeCompare(b.slug);
}

const NO_DESIGN: VoiceDesignState = { batches: [], unpicked: 0 };

let _list = $state<SavedVoice[]>([]);
let _loaded = $state(false);
let _design = $state<VoiceDesignState>(NO_DESIGN);

/** Replace the voice with the same slug, or add it. */
function upsert(voice: SavedVoice): void {
  _list = [..._list.filter((other) => other.slug !== voice.slug), voice].sort(byName);
}

export const voicesStore = {
  /** Every saved voice, by name. */
  get list(): SavedVoice[] { return _list; },

  /** The list has been read from the server at least once. */
  get loaded(): boolean { return _loaded; },

  /** Read the list. A failed read keeps the list as it was and leaves `loaded` as it was. */
  async load(): Promise<void> {
    try {
      _list = [...(await listVoices())].sort(byName);
      _loaded = true;
    } catch {
      // The picker still works from what it has; the next open reads again.
    }
  },

  /** The voice designer's kept candidates, newest batch first; they stay until Clear voices. */
  get design(): VoiceDesignState { return _design; },

  /** How many candidates wait to be picked: the unsaved ones of the batches made since the last save or Clear voices. */
  get unpicked(): number { return _design.unpicked; },

  /** Read the kept candidates. A failed read keeps what the store has. */
  async loadDesign(): Promise<void> {
    try {
      _design = await getDesignState();
    } catch {
      // The marker and the New voice tab work from what they have; the next open reads again.
    }
  },

  /** Forget the list, the candidates and that the list was read. For tests. */
  reset(): void {
    _list = [];
    _loaded = false;
    _design = NO_DESIGN;
  },

  /** Save a recording as a voice; the form carries the fields of `POST /api/voices`. */
  async saveRecording(form: FormData): Promise<SavedVoice> {
    const saved = await saveRecordingVoice(form);
    upsert(saved);
    return saved;
  },

  /** Save the candidate *candidateId* as a voice named *name*. The candidate stays in its batch, marked as saved, the others stay unpicked, and the ready marker clears. */
  async saveCandidate(candidateId: string, name: string): Promise<SavedVoice> {
    const saved = await saveCandidateVoice(candidateId, name);
    upsert(saved);
    const batches = _design.batches.map((batch) => ({
      ...batch,
      candidates: batch.candidates.map((candidate) => (candidate.id === candidateId ? { ...candidate, saved_as: saved.name } : candidate)),
    }));
    _design = { batches, unpicked: 0 };
    return saved;
  },

  /** Remove every kept candidate, the saved ones' rows with them (the voices stay). The server refuses while a design job runs. */
  async clearCandidates(): Promise<void> {
    await clearCandidates();
    _design = NO_DESIGN;
  },

  /** Rename a voice; its slug stays, so a draft that follows it keeps following it. */
  async rename(slug: string, name: string): Promise<SavedVoice> {
    const renamed = await renameVoice(slug, name);
    upsert(renamed);
    return renamed;
  },

  /** Delete a voice. When it is the chosen one, the draft goes back to the model's default built-in voice. */
  async remove(slug: string): Promise<void> {
    await deleteVoice(slug);
    _list = _list.filter((voice) => voice.slug !== slug);
    draft.dropSavedVoice(slug);
  },
};
