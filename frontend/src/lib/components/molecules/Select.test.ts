// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import PromptFileField from '$lib/components/molecules/PromptFileField.svelte';
import type { PromptFileContract } from '$lib/types';
import { openOptions, optionValues, pickOption } from '../../../test-utils/select';
import Select from './Select.svelte';

const promptFileContract: PromptFileContract = {
  accepted_extensions: ['.yaml'],
  selection_required: false,
  trust_boundary: {
    scope: 'server_host_only',
    manual_entry: 'submitted_value_kept_until_backend_validation',
    picker: 'in_app_browser',
    read_write: 'yaml_files_only',
  },
  help: {
    path: 'Path', option_required: 'Required', option_optional: 'Optional',
    empty_options: 'Empty', stale_selection: 'Stale', loaded: 'Loaded',
    ignored_negative_video: 'Ignored', ignored_negative_unsupported: 'Unsupported',
  },
};

const FRUIT = [
  { value: 'apple', label: 'Apple' },
  { value: 'banana', label: 'Banana', disabled: true },
  { value: 'cherry', label: 'Cherry' },
  { value: 'date', label: 'Date' },
];

function key(element: Element, name: string): void {
  flushSync(() => element.dispatchEvent(new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true })));
}

describe('Select', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    document.body.innerHTML = '';
  });

  function mountSelect(props: Record<string, unknown>): { box: HTMLButtonElement; form: HTMLFormElement } {
    const form = document.createElement('form');
    target.appendChild(form);
    app = flushSync(() => mount(Select, { target: form, props: { id: 'fruit', name: 'fruit', options: FRUIT, ...props } }));
    return { box: form.querySelector('#fruit') as HTMLButtonElement, form };
  }

  describe('native required semantics', () => {
    it('is optional by default for existing consumers that do not opt into required', () => {
      const { form } = mountSelect({ options: [{ value: 'choice', label: 'Choice' }] });

      const select = form.querySelector('select[name="fruit"]') as HTMLSelectElement;
      expect(select.required).toBe(false);
      expect(select.validity.valid).toBe(true);
      expect(form.checkValidity()).toBe(true);
    });

    it('fails validation until a required select with a placeholder has a choice', () => {
      const { box, form } = mountSelect({ required: true, placeholder: 'Choose a fruit' });

      const select = form.querySelector('select[name="fruit"]') as HTMLSelectElement;
      expect(select.required).toBe(true);
      expect(select.validity.valueMissing).toBe(true);
      expect(form.checkValidity()).toBe(false);
      expect(box.textContent).toContain('Choose a fruit');
      expect(optionValues(box)).toEqual(['apple', 'banana', 'cherry', 'date']);

      pickOption(box, 'cherry');
      expect(form.checkValidity()).toBe(true);
      expect(new FormData(form).get('fruit')).toBe('cherry');
    });

    it('keeps the existing optional prompt-option consumer optional', () => {
      const form = document.createElement('form');
      target.appendChild(form);
      app = flushSync(() => mount(PromptFileField, {
        target: form,
        props: {
          contract: promptFileContract,
          promptSource: 'inline',
          path: null,
          selectedOptionIds: [],
          workflowMode: 'image',
          negativePromptSupported: true,
          onPathChange: () => undefined,
          onOptionChange: () => undefined,
        },
      }));

      expect(form.querySelector('input[required]')).toBeNull();
      expect(form.checkValidity()).toBe(true);
    });
  });

  it('shows the choice, and picking an option submits it and reports it once', () => {
    const onchange = vi.fn();
    const { box, form } = mountSelect({ value: 'apple', onchange });
    expect(box.getAttribute('role')).toBe('combobox');
    expect(box.textContent).toContain('Apple');

    pickOption(box, 'date');
    expect(onchange).toHaveBeenCalledExactlyOnceWith('date');
    expect(box.dataset.value).toBe('date');
    expect(box.getAttribute('aria-expanded')).toBe('false');
    expect(new FormData(form).get('fruit')).toBe('date');
  });

  it('marks the chosen option and ignores a disabled one', () => {
    const onchange = vi.fn();
    const { box } = mountSelect({ value: 'cherry', onchange });
    const options = openOptions(box);

    expect(options.find((option) => option.getAttribute('aria-selected') === 'true')?.dataset.value).toBe('cherry');
    const banana = options.find((option) => option.dataset.value === 'banana')!;
    expect(banana.getAttribute('aria-disabled')).toBe('true');
    flushSync(() => banana.click());
    expect(onchange).not.toHaveBeenCalled();
    expect(box.getAttribute('aria-expanded')).toBe('true');
  });

  it('opens from the keyboard, skips disabled options, jumps by typing and picks with Enter', () => {
    const onchange = vi.fn();
    const { box } = mountSelect({ value: 'apple', onchange });

    key(box, 'ArrowDown');
    const list = document.getElementById(box.getAttribute('aria-controls')!)!;
    expect(list.getAttribute('role')).toBe('listbox');
    const activeValue = (): string | undefined => document.getElementById(list.getAttribute('aria-activedescendant')!)?.dataset.value;
    expect(activeValue()).toBe('apple');

    key(list, 'ArrowDown');
    expect(activeValue()).toBe('cherry');
    key(list, 'End');
    expect(activeValue()).toBe('date');
    key(list, 'Home');
    expect(activeValue()).toBe('apple');
    key(list, 'c');
    expect(activeValue()).toBe('cherry');

    key(list, 'Enter');
    expect(onchange).toHaveBeenCalledExactlyOnceWith('cherry');
    expect(box.getAttribute('aria-expanded')).toBe('false');
  });

  it('picks with Space on release, so the release does not land on the button', () => {
    const onchange = vi.fn();
    const { box } = mountSelect({ value: 'apple', onchange });
    openOptions(box);
    const list = document.getElementById(box.getAttribute('aria-controls')!)!;

    key(list, 'ArrowDown');
    key(list, ' ');
    expect(onchange).not.toHaveBeenCalled();
    flushSync(() => list.dispatchEvent(new KeyboardEvent('keyup', { key: ' ', bubbles: true, cancelable: true })));
    expect(onchange).toHaveBeenCalledExactlyOnceWith('cherry');
  });

  it('closes on Escape without changing the value', () => {
    const onchange = vi.fn();
    const { box } = mountSelect({ value: 'apple', onchange });
    openOptions(box);
    const list = document.getElementById(box.getAttribute('aria-controls')!)!;

    key(list, 'ArrowDown');
    key(document.body, 'Escape');
    expect(box.getAttribute('aria-expanded')).toBe('false');
    expect(onchange).not.toHaveBeenCalled();
    expect(box.dataset.value).toBe('apple');
  });

  it('keeps the submitted value on the shown choice when the form is reset', async () => {
    const { box, form } = mountSelect({ value: 'apple' });
    pickOption(box, 'date');

    form.reset();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(box.dataset.value).toBe('date');
    expect(new FormData(form).get('fruit')).toBe('date');
  });

  it('hands focus to the button when the browser focuses the hidden field, as for a required field left empty', () => {
    const { box, form } = mountSelect({ required: true, placeholder: 'Choose a fruit' });
    (form.querySelector('select[name="fruit"]') as HTMLSelectElement).focus();
    expect(document.activeElement).toBe(box);
  });

  it('returns focus to the button when Tab leaves the open list', () => {
    const { box } = mountSelect({ value: 'apple' });
    openOptions(box);
    key(document.getElementById(box.getAttribute('aria-controls')!)!, 'Tab');
    expect(box.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(box);
  });

  it('does not open while disabled', () => {
    const { box } = mountSelect({ value: 'apple', disabled: true });
    flushSync(() => box.click());
    expect(box.getAttribute('aria-expanded')).toBe('false');
  });
});
