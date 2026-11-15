import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { mountComponent } from '../../../test-utils/mount';
import FileName from './FileName.svelte';

const NAME = 'web_2026-11-14_09-12-33.png';

describe('FileName', () => {
  let width = 0;

  beforeEach(() => {
    // jsdom has no layout and no canvas: the element is `width` wide and each character is one unit.
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(() => width);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(
      () => ({ measureText: (text: string) => ({ width: text.length }), font: '' }) as unknown as CanvasRenderingContext2D,
    );
  });

  afterEach(() => vi.restoreAllMocks());

  const shown = (target: HTMLElement) => target.querySelector('[aria-hidden="true"]')?.textContent;

  it('shows a name that fits whole', () => {
    width = 60;
    const { target } = mountComponent(FileName, { name: NAME });
    expect(shown(target)).toBe(NAME);
  });

  it('cuts a long name in the middle, keeps its tail and carries the full name in the title and the accessible text', () => {
    width = 23;
    const { target } = mountComponent(FileName, { name: NAME });
    expect(shown(target)).toBe('web_2026-…_09-12-33.png');
    expect(shown(target)?.endsWith('_09-12-33.png')).toBe(true);
    const root = target.querySelector('.file-name') as HTMLElement;
    expect(root.title).toBe(NAME);
    expect(root.querySelector('.sr-only')?.textContent).toBe(NAME);
  });

  it('shows the whole name where the element has no width to measure', () => {
    width = 0;
    const { target } = mountComponent(FileName, { name: NAME });
    expect(shown(target)).toBe(NAME);
  });
});
