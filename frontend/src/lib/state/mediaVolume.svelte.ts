// The volume and mute of the player, shared by every player of the session: the viewer builds a new element for each asset
// it pages to, and each should sound as the last one did. Not stored; a reload starts at full volume, unmuted.

let _volume = $state(1);
let _muted = $state(false);

export const mediaVolume = {
  /** The volume, 0 to 1. */
  get volume(): number { return _volume; },
  get muted(): boolean { return _muted; },
  /** Set the volume, 0 to 1; raising it from silence unmutes. */
  setVolume(value: number): void {
    _volume = Math.min(1, Math.max(0, value));
    if (_volume > 0) _muted = false;
  },
  setMuted(value: boolean): void {
    _muted = value;
  },
  /** Back to full volume, unmuted. For tests. */
  reset(): void {
    _volume = 1;
    _muted = false;
  },
};
