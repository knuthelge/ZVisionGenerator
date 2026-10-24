/**
 * Scroll every scrolled container inside `root` back to its top, and return how many there were.
 *
 * A phone's tap on the status bar scrolls to the top; here the tap is on the page's title in the top strip, and the
 * page's own scrolling container (the Gallery grid, a settings page) is the one that moves.
 */
export function scrollToTop(root: ParentNode): number {
  let scrolled = 0;
  for (const el of root.querySelectorAll<HTMLElement>('*')) {
    if (!(el.scrollTop > 0)) continue;
    scrolled += 1;
    if (typeof el.scrollTo === 'function') el.scrollTo({ top: 0, behavior: 'smooth' });
    else el.scrollTop = 0;
  }
  return scrolled;
}
