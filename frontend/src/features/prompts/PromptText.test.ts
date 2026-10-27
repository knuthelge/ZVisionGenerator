import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import PromptText from './PromptText.svelte';
import { buttonByText as button, mountComponent, settle } from '../../test-utils/mount';

const TEXT = 'a {red|blue} car';

describe('PromptText choices while typing', () => {
  let target: HTMLElement;
  let onreplace: ReturnType<typeof vi.fn>;
  let onendedit: ReturnType<typeof vi.fn>;
  let textarea: HTMLTextAreaElement;

  beforeEach(async () => {
    onreplace = vi.fn();
    onendedit = vi.fn();
    target = mountComponent(PromptText, { text: TEXT, editing: true, label: 'Prompt', snippets: [], onstartedit: vi.fn(), onendedit, ontype: vi.fn(), onreplace }).target;
    await settle();
    textarea = target.querySelector('textarea')!;
  });

  afterEach(() => {
    document.querySelectorAll('.choice-editor').forEach((node) => node.remove());
  });

  async function setCaret(start: number, end = start): Promise<void> {
    textarea.focus();
    textarea.setSelectionRange(start, end);
    textarea.dispatchEvent(new Event('select', { bubbles: true }));
    textarea.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true }));
    await settle();
  }

  async function applyOptions(lines: string): Promise<void> {
    const box = document.querySelector<HTMLTextAreaElement>('.choice-editor textarea')!;
    box.value = lines;
    box.dispatchEvent(new Event('input', { bubbles: true }));
    await settle();
    button('Apply').click();
    await settle();
  }

  it('edits the choice around the caret with ⌥↵, staying in edit mode', async () => {
    await setCaret(5);
    textarea.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', altKey: true, bubbles: true }));
    await settle();

    expect(document.querySelector<HTMLTextAreaElement>('.choice-editor textarea')?.value).toBe('red\nblue');
    await applyOptions('red\nblue\ngreen');

    expect(onreplace).toHaveBeenCalledWith('a {red|blue|green} car');
    expect(onendedit).not.toHaveBeenCalled();
    expect(document.querySelector('.choice-editor')).toBeNull();
  });

  it('turns selected words into a new choice', async () => {
    await setCaret(13, 16);
    button('Make a random choice').click();
    await settle();

    expect(document.querySelector<HTMLTextAreaElement>('.choice-editor textarea')?.value).toBe('car');
    expect(Array.from(document.querySelectorAll('.choice-editor button')).some((item) => item.textContent?.includes('Remove choice'))).toBe(false);
    await applyOptions('car\nbus');

    expect(onreplace).toHaveBeenCalledWith('a {red|blue} {car|bus}');
    expect(onendedit).not.toHaveBeenCalled();
  });

  it('starts a new choice at the caret with ⌥↵', async () => {
    await setCaret(13);
    textarea.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', altKey: true, bubbles: true }));
    await settle();

    expect(document.querySelector<HTMLTextAreaElement>('.choice-editor textarea')?.value).toBe('');
    await applyOptions('big\nsmall');

    expect(onreplace).toHaveBeenCalledWith('a {red|blue} {big|small}car');
    expect(onendedit).not.toHaveBeenCalled();
  });

  it('works from the keyboard alone: focus moves to the options, ⌘↵ applies and returns to the prompt', async () => {
    await setCaret(13);
    textarea.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', altKey: true, bubbles: true }));
    await settle();

    const options = document.querySelector<HTMLTextAreaElement>('.choice-editor textarea')!;
    expect(document.activeElement).toBe(options);
    options.value = 'big\nsmall';
    options.dispatchEvent(new Event('input', { bubbles: true }));
    options.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', metaKey: true, bubbles: true }));
    await settle();

    expect(onreplace).toHaveBeenCalledWith('a {red|blue} {big|small}car');
    expect(document.querySelector('.choice-editor')).toBeNull();
    expect(document.activeElement).toBe(target.querySelector('textarea'));
    expect(target.querySelector('textarea')!.selectionStart).toBe('a {red|blue} {big|small}'.length);
  });

  it('inserts nothing when a new choice is applied empty', async () => {
    await setCaret(16);
    button('New random choice').click();
    await settle();
    await applyOptions('');

    expect(onreplace).toHaveBeenCalledWith(TEXT);
  });
});
