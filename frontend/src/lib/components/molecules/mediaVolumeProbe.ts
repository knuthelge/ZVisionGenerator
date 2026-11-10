/** Whether a page can set a media element's volume. */

type VolumeProbe = { volume: number };

/** Return whether *probe* keeps a volume a page gives it: iPhone and iPad Safari do not (only the device's buttons set the level there, and the element reads 1 whatever it is given). */
export function holdsVolume(probe: VolumeProbe): boolean {
  try {
    probe.volume = 0.5;
    return Math.abs(probe.volume - 0.5) < 0.01;
  } catch {
    return false;
  }
}

let settable: boolean | null = null;

/** Return whether this browser lets a page set a media element's volume, asked once. */
export function canSetVolume(): boolean {
  settable ??= holdsVolume(document.createElement('audio'));
  return settable;
}
