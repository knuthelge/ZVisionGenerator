/**
 * Make *textarea* hold *next* and put its caret at *caret*, as one edit the person can undo.
 *
 * Only the stretch that differs is replaced, through the browser's own text insertion, so Undo takes it back and the `input` event
 * the page listens to fires as for typing. Where the browser has no such insertion, or *focus* is off (a phone would raise its
 * keyboard), the value is set and an `input` event is sent, which cannot be undone.
 */
export function editTextarea(textarea: HTMLTextAreaElement, next: string, caret: number, { focus = true }: { focus?: boolean } = {}): void {
  const previous = textarea.value;
  let head = 0;
  while (head < previous.length && head < next.length && previous[head] === next[head]) head++;
  let tail = 0;
  while (tail < previous.length - head && tail < next.length - head && previous[previous.length - 1 - tail] === next[next.length - 1 - tail]) tail++;
  let done = false;
  if (focus && typeof document.execCommand === 'function') {
    textarea.focus({ preventScroll: true });
    textarea.setSelectionRange(head, previous.length - tail);
    done = document.execCommand('insertText', false, next.slice(head, next.length - tail)) && textarea.value === next;
  }
  if (!done) {
    textarea.value = next;
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    if (focus) textarea.focus({ preventScroll: true });
  }
  textarea.setSelectionRange(caret, caret);
}
