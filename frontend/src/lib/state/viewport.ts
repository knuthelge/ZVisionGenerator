/** The part of `window.visualViewport` the app sizes itself by. */
export interface VisualViewportLike {
  height: number;
  offsetTop: number;
  scale: number;
}

/** Where the app sits: the visible area's height and its offset from the top of the layout viewport, in px. */
export interface ViewportBox {
  height: number;
  top: number;
}

/** Zoom within this of 1 counts as unzoomed. */
const SCALE_TOLERANCE = 0.01;

/**
 * Return the box the app should fill for this visual viewport, or null while the page is pinch-zoomed.
 *
 * Zoomed in, the visual viewport is a window onto the page rather than the screen, so the app keeps its last size
 * instead of shrinking to the zoomed area.
 */
export function viewportBox(viewport: VisualViewportLike): ViewportBox | null {
  if (Math.abs(viewport.scale - 1) > SCALE_TOLERANCE) return null;
  return { height: Math.round(viewport.height), top: Math.max(0, Math.round(viewport.offsetTop)) };
}

/** Whether focus is in a field that brings up an on-screen keyboard. */
function editing(doc: Document): boolean {
  const el = doc.activeElement;
  if (!el) return false;
  if (el instanceof HTMLTextAreaElement || el instanceof HTMLSelectElement) return true;
  if (el instanceof HTMLInputElement) return !['checkbox', 'radio', 'range', 'button', 'submit', 'reset', 'file', 'color'].includes(el.type);
  return el instanceof HTMLElement && el.isContentEditable;
}

/**
 * Keep the app the size of the visible area, and the page under it unscrolled.
 *
 * Sets `--app-height` and `--app-top` on the root element from `window.visualViewport`, which `.ui-viewport` layers use.
 * The app is then always exactly what is visible: above an on-screen keyboard (iOS Safari ignores
 * `interactive-widget=resizes-content`), and in step with a browser toolbar that comes and goes, instead of settling
 * only on the next scroll. When no field is being edited, a page that the browser scrolled (to show a field above the
 * keyboard, say) is put back at the top, so no gap is left under the app.
 */
export function watchViewport(win: Window): () => void {
  const viewport = win.visualViewport;
  if (!viewport) return () => {};
  const rootStyle = win.document.documentElement.style;

  function apply(): void {
    const box = viewportBox(viewport!);
    if (box) {
      rootStyle.setProperty('--app-height', `${box.height}px`);
      rootStyle.setProperty('--app-top', `${box.top}px`);
    }
    if (!editing(win.document) && (win.scrollY !== 0 || win.scrollX !== 0)) win.scrollTo(0, 0);
  }

  // The keyboard closes a moment after focus leaves the field; settle once it has.
  let settle: ReturnType<typeof setTimeout> | undefined;
  function onFocusOut(): void {
    clearTimeout(settle);
    settle = setTimeout(apply, 120);
  }

  apply();
  viewport.addEventListener('resize', apply);
  viewport.addEventListener('scroll', apply);
  win.addEventListener('scroll', apply);
  win.document.addEventListener('focusout', onFocusOut);
  return () => {
    clearTimeout(settle);
    viewport.removeEventListener('resize', apply);
    viewport.removeEventListener('scroll', apply);
    win.removeEventListener('scroll', apply);
    win.document.removeEventListener('focusout', onFocusOut);
  };
}
