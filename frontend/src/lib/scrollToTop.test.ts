import { describe, expect, it, vi } from 'vitest';

import { scrollToTop } from './scrollToTop';

describe('scrollToTop', () => {
  it('scrolls the scrolled containers under the root back up, and leaves the rest alone', () => {
    const root = document.createElement('div');
    const grid = document.createElement('section');
    const still = document.createElement('section');
    root.append(grid, still);
    Object.defineProperty(grid, 'scrollTop', { value: 420, writable: true });
    grid.scrollTo = vi.fn();
    still.scrollTo = vi.fn();

    expect(scrollToTop(root)).toBe(1);
    expect(grid.scrollTo).toHaveBeenCalledWith({ top: 0, behavior: 'smooth' });
    expect(still.scrollTo).not.toHaveBeenCalled();
  });
});
