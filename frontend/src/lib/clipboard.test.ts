import { afterEach, describe, expect, it, vi } from 'vitest';

import { copyText, selectElementText } from './clipboard';

function stubClipboard(writeText: unknown, secure = true): void {
  vi.stubGlobal('isSecureContext', secure);
  Object.defineProperty(navigator, 'clipboard', { value: writeText === undefined ? undefined : { writeText }, configurable: true });
}

describe('copyText', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    Object.defineProperty(navigator, 'clipboard', { value: undefined, configurable: true });
  });

  it('writes the text and resolves true', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    stubClipboard(writeText);
    expect(await copyText('boom')).toBe(true);
    expect(writeText).toHaveBeenCalledWith('boom');
  });

  it('resolves false when the page is not a secure context', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    stubClipboard(writeText, false);
    expect(await copyText('boom')).toBe(false);
    expect(writeText).not.toHaveBeenCalled();
  });

  it('resolves false when the browser has no clipboard', async () => {
    stubClipboard(undefined);
    expect(await copyText('boom')).toBe(false);
  });

  it('resolves false when the write is refused', async () => {
    stubClipboard(vi.fn().mockRejectedValue(new Error('denied')));
    expect(await copyText('boom')).toBe(false);
  });
});

describe('selectElementText', () => {
  it('selects the whole text of the element', () => {
    const element = document.createElement('p');
    element.textContent = 'CUDA out of memory';
    document.body.appendChild(element);
    selectElementText(element);
    expect(window.getSelection()?.toString()).toBe('CUDA out of memory');
    element.remove();
  });
});
