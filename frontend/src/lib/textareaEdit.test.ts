import { afterEach, describe, expect, it, vi } from 'vitest';

import { editTextarea } from './textareaEdit';

function box(value: string): HTMLTextAreaElement {
  const textarea = document.createElement('textarea');
  textarea.value = value;
  document.body.append(textarea);
  return textarea;
}

describe('editTextarea', () => {
  afterEach(() => {
    document.body.innerHTML = '';
    Reflect.deleteProperty(document, 'execCommand');
  });

  it('sets the value, tells the page with an input event, focuses the box and places the caret', () => {
    const textarea = box('Hello there');
    const seen = vi.fn();
    textarea.addEventListener('input', seen);

    editTextarea(textarea, 'Hello… there', 6);

    expect(textarea.value).toBe('Hello… there');
    expect(seen).toHaveBeenCalledTimes(1);
    expect(document.activeElement).toBe(textarea);
    expect([textarea.selectionStart, textarea.selectionEnd]).toEqual([6, 6]);
  });

  it('inserts only the stretch that differs through the browser so Undo can take it back', () => {
    const textarea = box('Hi. You called.');
    const insert = vi.fn((_command: string, _ui: boolean, text: string) => {
      textarea.setRangeText(text, textarea.selectionStart, textarea.selectionEnd, 'end');
      return true;
    });
    Object.defineProperty(document, 'execCommand', { value: insert, configurable: true });

    editTextarea(textarea, 'Hi. [sad] You called.', 10);

    expect(insert).toHaveBeenCalledWith('insertText', false, '[sad] ');
    expect(textarea.value).toBe('Hi. [sad] You called.');
    expect(textarea.selectionStart).toBe(10);
  });

  it('falls back to setting the value when the browser does not insert what was asked', () => {
    const textarea = box('abc');
    Object.defineProperty(document, 'execCommand', { value: () => false, configurable: true });

    editTextarea(textarea, 'abXc', 3);

    expect(textarea.value).toBe('abXc');
  });

  it('leaves focus alone when asked, so a phone keeps its keyboard down', () => {
    const textarea = box('abc');
    const insert = vi.fn(() => true);
    Object.defineProperty(document, 'execCommand', { value: insert, configurable: true });

    editTextarea(textarea, 'abXc', 3, { focus: false });

    expect(insert).not.toHaveBeenCalled();
    expect(document.activeElement).not.toBe(textarea);
    expect(textarea.value).toBe('abXc');
  });
});
