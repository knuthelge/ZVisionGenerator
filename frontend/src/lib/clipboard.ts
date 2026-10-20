/**
 * Copy *text* to the clipboard; resolve false instead of throwing when that is not possible.
 *
 * The Clipboard API exists only in secure contexts, so a page opened over plain http on the LAN has none.
 */
export async function copyText(text: string): Promise<boolean> {
  if (typeof window === 'undefined' || !window.isSecureContext || !navigator.clipboard?.writeText) return false;
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
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
