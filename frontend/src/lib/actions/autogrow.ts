/** Grow a textarea with its content; CSS `max-height` caps it, after which it scrolls. */
export function autogrow(node: HTMLTextAreaElement, _value?: unknown): { update: () => void; destroy: () => void } {
  function resize(): void {
    node.style.height = 'auto';
    node.style.height = `${node.scrollHeight}px`;
  }
  node.addEventListener('input', resize);
  queueMicrotask(resize);
  return {
    // Re-measure when the bound value changes from outside (reuse, enhance, tab switch).
    update(): void { queueMicrotask(resize); },
    destroy(): void { node.removeEventListener('input', resize); },
  };
}
