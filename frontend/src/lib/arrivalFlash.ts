/** What an action filled in the Workspace: the First frame slot, the Last frame slot, or the Source video section. */
export type Arrival = 'first-frame' | 'last-frame' | 'source-video';

// Where each shows: the sidebar's group or section, the open settings popover's group, and on a phone the chip that opens it.
const TARGETS: Record<Arrival, string> = {
  'first-frame': '[aria-labelledby="ws-start-group"], [aria-labelledby="pop-start-group"], [data-chip="images"]',
  'last-frame': '[aria-labelledby="ws-last-group"], [aria-labelledby="pop-last-group"], [data-chip="images"]',
  'source-video': '[data-section="source-video"], [data-chip="source"]',
};

/** How long the ring takes to fade; the stylesheet's animation lasts as long. */
export const ARRIVAL_MS = 900;

/** Flash a ring on whatever shows *arrival* now. Nothing flashes under reduced motion. */
export function flashArrival(arrival: Arrival, root: ParentNode = document): void {
  if (typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  for (const element of root.querySelectorAll<HTMLElement>(TARGETS[arrival])) {
    // Taking the mark off first restarts the animation when the same element is flashed twice in a row.
    element.removeAttribute('data-arrived');
    void element.offsetWidth;
    element.setAttribute('data-arrived', '');
    setTimeout(() => element.removeAttribute('data-arrived'), ARRIVAL_MS);
  }
}

/** Return what a Workspace prefill fills when it is a Use as for a video's frames or an Extend; null for any other. */
export function arrivalFor(params: Readonly<Record<string, unknown>>): Arrival | null {
  if (params.use_as === 'start' && params.workflow === 'img2vid') return 'first-frame';
  if (params.use_as === 'last') return 'last-frame';
  if (params.workflow === 'extend' && params.source_video) return 'source-video';
  return null;
}
