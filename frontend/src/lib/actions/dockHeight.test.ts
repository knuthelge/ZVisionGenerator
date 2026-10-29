import { afterEach, describe, expect, it, vi } from 'vitest';
import { dockHeight } from './dockHeight';

describe('dockHeight', () => {
  const observers: { callback: () => void; disconnect: ReturnType<typeof vi.fn> }[] = [];

  afterEach(() => {
    vi.unstubAllGlobals();
    observers.length = 0;
    document.documentElement.style.removeProperty('--dock');
  });

  function setup(height: number): HTMLElement {
    vi.stubGlobal('ResizeObserver', class {
      callback: () => void;
      disconnect = vi.fn();
      constructor(callback: () => void) { this.callback = callback; observers.push(this); }
      observe(): void {}
    });
    const node = document.createElement('div');
    node.getBoundingClientRect = () => ({ height }) as DOMRect;
    return node;
  }

  it('publishes the height at once and again whenever it changes', () => {
    let height = 120.2;
    const node = setup(0);
    node.getBoundingClientRect = () => ({ height }) as DOMRect;

    dockHeight(node, '--dock');
    expect(document.documentElement.style.getPropertyValue('--dock')).toBe('121px');

    height = 190;
    observers[0].callback();
    expect(document.documentElement.style.getPropertyValue('--dock')).toBe('190px');
  });

  it('removes the property and stops watching when the element goes away', () => {
    const action = dockHeight(setup(100), '--dock');

    action.destroy();

    expect(document.documentElement.style.getPropertyValue('--dock')).toBe('');
    expect(observers[0].disconnect).toHaveBeenCalled();
  });
});
