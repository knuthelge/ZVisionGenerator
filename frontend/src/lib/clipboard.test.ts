import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { copyText, selectElementText } from './clipboard';

function stubClipboard(writeText: unknown, secure = true): void {
  vi.stubGlobal('isSecureContext', secure);
  Object.defineProperty(navigator, 'clipboard', { value: writeText === undefined ? undefined : { writeText }, configurable: true });
}

describe('copyText', () => {
  const execCommand = vi.fn();

  beforeEach(() => {
    execCommand.mockReset().mockReturnValue(true);
    Object.defineProperty(document, 'execCommand', { value: execCommand, configurable: true });
  });

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

  it('copies through a hidden text field where the page is not a secure context, without the Clipboard API', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    stubClipboard(writeText, false);
    let copied = '';
    execCommand.mockImplementation(() => {
      copied = (document.activeElement as HTMLTextAreaElement).value;
      return true;
    });

    expect(await copyText('boom')).toBe(true);

    expect(writeText).not.toHaveBeenCalled();
    expect(copied).toBe('boom');
    expect(document.querySelector('textarea')).toBeNull();
  });

  it('copies through the text field when the browser has no clipboard, and when the write is refused', async () => {
    stubClipboard(undefined);
    expect(await copyText('boom')).toBe(true);
    stubClipboard(vi.fn().mockRejectedValue(new Error('denied')));
    expect(await copyText('boom')).toBe(true);
    expect(execCommand).toHaveBeenCalledTimes(2);
  });

  it('gives the focus back to what had it, and leaves no text field behind', async () => {
    stubClipboard(undefined, false);
    const button = document.createElement('button');
    document.body.appendChild(button);
    button.focus();

    await copyText('boom');

    expect(document.activeElement).toBe(button);
    expect(document.querySelector('textarea')).toBeNull();
    button.remove();
  });

  it('selects the whole text with a range, for iOS Safari', async () => {
    stubClipboard(undefined, false);
    const setRange = vi.spyOn(HTMLTextAreaElement.prototype, 'setSelectionRange');

    await copyText('boom');

    expect(setRange).toHaveBeenCalledWith(0, 4);
    setRange.mockRestore();
  });

  it('puts the text field inside the open dialog, so its focus trap leaves it alone, and not in the body', async () => {
    stubClipboard(undefined, false);
    const dialog = document.createElement('div');
    dialog.setAttribute('role', 'dialog');
    const button = document.createElement('button');
    dialog.appendChild(button);
    document.body.appendChild(dialog);
    button.focus();
    let parent: Element | null = null;
    execCommand.mockImplementation(() => {
      parent = document.activeElement?.parentElement ?? null;
      return true;
    });

    await copyText('boom');

    expect(parent).toBe(dialog);
    expect(dialog.querySelector('textarea')).toBeNull();
    expect(document.activeElement).toBe(button);
    dialog.remove();
  });

  it('resolves false only when both ways fail', async () => {
    stubClipboard(vi.fn().mockRejectedValue(new Error('denied')));
    execCommand.mockReturnValue(false);
    expect(await copyText('boom')).toBe(false);
    execCommand.mockImplementation(() => { throw new Error('blocked'); });
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
