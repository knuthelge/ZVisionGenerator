/** A file name split at the cut: what may be shortened, and the tail that always stays. */
export interface NameParts {
  start: string;
  tail: string;
}

// The tail is the last `_segment` and the extension: for the app's `{set}_{date}_{time}.ext` names that is the time, and for
// a counted name (`…_09-12-33_2.png`) the counter.
const TAIL = /^(.+?)((?:_[^_]*)?\.[^.]+)$/;

/** Split a file name into the part that may be cut and the tail that stays visible; a name with no extension has no tail. */
export function nameParts(name: string): NameParts {
  const match = TAIL.exec(name);
  return match ? { start: match[1], tail: match[2] } : { start: name, tail: '' };
}

/**
 * Cut a file name in the middle so it fits *width*: the start, an ellipsis, then the tail. A name that already fits is
 * returned whole. *measure* returns the width of a string in the font the name is drawn in.
 */
export function fitName(name: string, width: number, measure: (text: string) => number): string {
  if (measure(name) <= width) return name;
  const { start, tail } = nameParts(name);
  let low = 0;
  let high = start.length;
  while (low < high) {
    const mid = (low + high + 1) >> 1;
    if (measure(`${start.slice(0, mid)}…${tail}`) <= width) low = mid;
    else high = mid - 1;
  }
  return `${start.slice(0, low)}…${tail}`;
}

let canvasContext: CanvasRenderingContext2D | null | undefined;

/** Return a function that measures text in *element*'s computed font, or null where there is no canvas to measure with. */
export function measurerFor(element: HTMLElement): ((text: string) => number) | null {
  if (canvasContext === undefined) canvasContext = typeof document === 'undefined' ? null : document.createElement('canvas').getContext('2d');
  const context = canvasContext;
  if (!context) return null;
  const style = getComputedStyle(element);
  context.font = `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
  return (text) => context.measureText(text).width;
}

// One observer serves every name on screen; each name registers the function that fits it again.
const refits = new Map<Element, () => void>();
let observer: ResizeObserver | null = null;

/** Call *refit* now and whenever *element*'s width changes; returns the function that stops it. */
export function watchName(element: HTMLElement, refit: () => void): () => void {
  if (typeof ResizeObserver !== 'undefined') {
    observer ??= new ResizeObserver((entries) => entries.forEach((entry) => refits.get(entry.target)?.()));
    observer.observe(element);
  }
  refits.set(element, refit);
  refit();
  return () => {
    refits.delete(element);
    observer?.unobserve(element);
  };
}
