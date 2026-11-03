/**
 * The speaker's consent to clone a voice, and whether the voice needs it, held in memory only.
 *
 * The tick is never saved with the draft, never put in an address and never read back from a job's settings: a new
 * recording, a Reuse, a Load settings, a Reset and a reload all leave it unticked, so each clone is confirmed by a person
 * who has just chosen the voice. A voice that is itself a take the model chose the voice of (an *Any* take, used as the
 * voice) needs no consent, which `exempt` records; a reload forgets that too, so the tick is asked for once more.
 */
class VoiceConsent {
  /** The tick is set. */
  ticked = $state(false);
  /** The voice is an *Any* take: it is nobody's voice, so no tick is asked for. */
  exempt = $state(false);

  /** Tick or untick the box. */
  set(ticked: boolean): void {
    this.ticked = ticked;
  }

  /** Forget the tick and the exemption: the voice is new, and its consent is not known. */
  clear(): void {
    this.ticked = false;
    this.exempt = false;
  }

  /** Forget the tick, and record whether the new voice needs none (a take made in the model's own voice). */
  reset(exempt: boolean): void {
    this.ticked = false;
    this.exempt = exempt;
  }
}

export const voiceConsent = new VoiceConsent();
