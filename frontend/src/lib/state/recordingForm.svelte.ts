// The From recording tab's form, held in memory: it survives closing the Voices window and is lost with a page reload.
// The recording, its words and the tick given here go to the draft with Use once, or into the library with Save voice; either empties the form.

import { hasPath } from './imageSlots';

class RecordingForm {
  /** The recording as a path on the host, or null. */
  path = $state<string | null>(null);
  /** The recording as a file from this device, or null; it wins over the path. */
  file = $state.raw<File | null>(null);
  text = $state('');
  /** The speaker's confirmation that the voice may be cloned. */
  ticked = $state(false);
  name = $state('');
  /** The recording the form was filled with from the one-off voice, to tell a different one from it; null while it was never filled. */
  #base = $state.raw<{ path: string | null; file: File | null } | null>(null);
  /** The recording the form was filled with needs the tick (a take that is in a built-in voice does not). */
  #baseNeedsConsent = $state(true);

  /** A recording is chosen. */
  get recorded(): boolean {
    return this.file !== null || hasPath(this.path);
  }

  /** The tick is asked for: a recording is chosen and it needs consent, or it is not the one the form was filled with. */
  get needsTick(): boolean {
    return this.recorded && (this.#baseNeedsConsent || this.path !== this.#base?.path || this.file !== this.#base?.file);
  }

  /** Take another recording; the tick belongs to the recording it was given for, so it clears. */
  setRecording(path: string | null, file: File | null): void {
    this.path = path;
    this.file = file;
    this.ticked = false;
  }

  /** Fill the form from the one-off voice now in use, so it can be kept as a voice. The name and the tick start empty: the speaker gives the tick here. */
  fillFrom(voice: { path: string | null; file: File | null; text: string; needsConsent: boolean }): void {
    this.path = voice.file === null ? voice.path : null;
    this.file = voice.file;
    this.text = voice.text;
    this.ticked = false;
    this.name = '';
    this.#base = { path: this.path, file: this.file };
    this.#baseNeedsConsent = voice.needsConsent;
  }

  /** The form was last filled from this recording (the one-off voice's path or file), whatever was typed since. */
  filledFrom(voice: { path: string | null; file: File | null }): boolean {
    return this.#base !== null && (voice.file === null ? voice.path : null) === this.#base.path && voice.file === this.#base.file;
  }

  /** Empty the form. */
  clear(): void {
    this.path = null;
    this.file = null;
    this.text = '';
    this.ticked = false;
    this.name = '';
    this.#base = null;
    this.#baseNeedsConsent = true;
  }

  /** The fields of `POST /api/voices`, for the speech model *model*. */
  formData(model: string): FormData {
    const form = new FormData();
    form.set('model', model);
    form.set('name', this.name.trim());
    if (this.file !== null) form.set('voice_file', this.file);
    else if (this.path !== null) form.set('voice_path', this.path);
    form.set('voice_text', this.text);
    if (this.needsTick && this.ticked) form.set('voice_consent', 'on');
    return form;
  }
}

export const recordingForm = new RecordingForm();
