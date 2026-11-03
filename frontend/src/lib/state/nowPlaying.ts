// Only one take plays at a time: a player that starts pauses the one that was playing.

let current: HTMLMediaElement | null = null;

/** Make *element* the one playing, pausing the element that was. */
export function claimPlayback(element: HTMLMediaElement): void {
  if (current && current !== element) current.pause();
  current = element;
}

/** Let go of *element*, if it is the one playing. */
export function releasePlayback(element: HTMLMediaElement): void {
  if (current === element) current = null;
}
