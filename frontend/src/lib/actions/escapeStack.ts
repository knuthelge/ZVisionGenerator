/**
 * One Escape for the innermost overlay.
 *
 * Every open overlay (sheet, popover, drawer, dialog) pushes a layer; Escape answers only the top one, so a path
 * browser opened from a popover, or a menu opened from a sheet, closes alone. The single listener is installed when
 * this module loads, before any page registers its own, so the page handlers see `defaultPrevented` and stand aside.
 */

interface Layer {
  onescape: () => void;
}

const layers: Layer[] = [];

function onKeydown(event: KeyboardEvent): void {
  if (event.key !== 'Escape' || event.defaultPrevented) return;
  const top = layers[layers.length - 1];
  if (!top) return;
  event.preventDefault();
  top.onescape();
}

if (typeof document !== 'undefined') document.addEventListener('keydown', onKeydown);

/** Open a layer that Escape closes while it is the innermost one; call the returned function when it closes. */
export function pushEscapeLayer(onescape: () => void): () => void {
  const layer: Layer = { onescape };
  layers.push(layer);
  return () => {
    const index = layers.indexOf(layer);
    if (index >= 0) layers.splice(index, 1);
  };
}

/** Whether any overlay layer is open. */
export function escapeLayerOpen(): boolean {
  return layers.length > 0;
}
