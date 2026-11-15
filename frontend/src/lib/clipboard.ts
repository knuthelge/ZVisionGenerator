/**
 * Copy *text* to the clipboard; resolve false instead of throwing when that is not possible.
 *
 * The Clipboard API exists only in secure contexts, so a page opened over plain http on the LAN has none. There, and when the
 * API refuses, a hidden text field is selected and copied with `execCommand`, which works inside a user gesture. The fallback
 * runs without waiting, so call this straight from the click.
 */
export async function copyText(text: string): Promise<boolean> {
  if (typeof window === 'undefined') return false;
  if (window.isSecureContext && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // fall through to the text field
    }
  }
  return copyWithTextField(text);
}

/** Copy *text* through an off-screen text field, leaving the focus and the selection as they were. */
function copyWithTextField(text: string): boolean {
  if (typeof document.execCommand !== 'function') return false;
  const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  const selection = document.getSelection();
  const ranges = selection ? Array.from({ length: selection.rangeCount }, (_, index) => selection.getRangeAt(index)) : [];
  const field = document.createElement('textarea');
  field.value = text;
  field.setAttribute('readonly', '');
  field.setAttribute('aria-hidden', 'true');
  field.style.cssText = 'position:fixed;top:0;left:-9999px;opacity:0;pointer-events:none';
  // Inside the open dialog, if there is one: a dialog's focus trap would pull the focus back out of a field in the body.
  (previous?.closest('[role="dialog"]') ?? document.body).appendChild(field);
  try {
    field.focus({ preventScroll: true });
    field.select();
    // iOS Safari copies only what a range names.
    field.setSelectionRange(0, text.length);
    return document.execCommand('copy');
  } catch {
    return false;
  } finally {
    field.remove();
    previous?.focus?.({ preventScroll: true });
    if (selection) {
      selection.removeAllRanges();
      ranges.forEach((range) => selection.addRange(range));
    }
  }
}

/** Select all the text inside *element*, so the user can copy it with the keyboard. */
export function selectElementText(element: HTMLElement): void {
  const selection = window.getSelection();
  if (!selection) return;
  const range = document.createRange();
  range.selectNodeContents(element);
  selection.removeAllRanges();
  selection.addRange(range);
}
