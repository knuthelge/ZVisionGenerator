/**
 * Publish an element's height as a CSS custom property on the page root, for things that must stay clear of it.
 *
 * The phone's controls (settings strip, composer and any message above Generate) change height with their content;
 * toasts read the property to sit above them, so a toast never covers a strip or a message however tall the dock is.
 * The property is removed when the element goes away.
 */
export function dockHeight(node: HTMLElement, name: string): { destroy: () => void } {
  const root = document.documentElement;
  const publish = (): void => {
    root.style.setProperty(name, `${Math.ceil(node.getBoundingClientRect().height)}px`);
  };
  publish();
  const observer = new ResizeObserver(publish);
  observer.observe(node);
  return {
    destroy() {
      observer.disconnect();
      root.style.removeProperty(name);
    },
  };
}
