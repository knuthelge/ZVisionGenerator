/** A box on screen, in viewport pixels. */
export interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** How long the image takes to fly between a tile and the viewer. */
export const FLIGHT_MS = 240;
export const FLIGHT_EASING = 'cubic-bezier(0.2, 0, 0, 1)';

export function rectOf(box: { left: number; top: number; width: number; height: number }): Rect {
  return { left: box.left, top: box.top, width: box.width, height: box.height };
}

/** Return the largest box of the given aspect that fits inside `container`, centred in it (object-fit: contain). */
export function fitRect(container: Rect, aspect: number): Rect {
  const ratio = aspect > 0 && Number.isFinite(aspect) ? aspect : 1;
  const width = Math.min(container.width, container.height * ratio);
  const height = width / ratio;
  return {
    left: container.left + (container.width - width) / 2,
    top: container.top + (container.height - height) / 2,
    width,
    height,
  };
}

/**
 * Return the keyframes that move an element laid out at `to` so that it starts over `from`, as a FLIP: only the
 * transform animates (translate and scale from the top-left corner), so the browser never lays out mid-flight.
 */
export function flightKeyframes(from: Rect, to: Rect): Keyframe[] {
  const scaleX = to.width > 0 ? from.width / to.width : 1;
  const scaleY = to.height > 0 ? from.height / to.height : 1;
  return [
    { transform: `translate(${from.left - to.left}px, ${from.top - to.top}px) scale(${scaleX}, ${scaleY})` },
    { transform: 'translate(0px, 0px) scale(1, 1)' },
  ];
}

/** Whether a box is worth flying from or to: on screen and not collapsed. */
export function usableRect(rect: Rect | null | undefined, viewport: { width: number; height: number }): rect is Rect {
  if (!rect || rect.width <= 0 || rect.height <= 0) return false;
  return rect.left < viewport.width && rect.top < viewport.height && rect.left + rect.width > 0 && rect.top + rect.height > 0;
}
