import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const layoutMock = vi.hoisted(() => ({ compact: false, mode: 'wide', coarse: false }));

vi.mock('$lib/state/layoutMode.svelte', () => ({ layout: layoutMock }));

import { mountComponent, settle } from '../../../test-utils/mount';
import { openOptions, optionValues, pickOption } from '../../../test-utils/select';
import Select from './Select.svelte';

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
  beforeEach(() => {
    Object.assign(layoutMock, { compact: false, mode: 'wide', coarse: false });
  });

  function mountSelect(props: Record<string, unknown>): { box: HTMLButtonElement; form: HTMLFormElement } {
    const form = document.createElement('form');
    mountComponent(Select, { id: 'fruit', name: 'fruit', options: FRUIT, ...props }, form);
    return { box: form.querySelector('#fruit') as HTMLButtonElement, form };
  }

  describe('native required semantics', () => {
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
    await settle();
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

  it('opens the list at the field on a phone, not as a sheet, and picks with a tap', () => {
    Object.assign(layoutMock, { compact: true, mode: 'compact', coarse: true });
    const onchange = vi.fn();
    const { box } = mountSelect({ ariaLabel: 'Sort', value: 'apple', onchange });

    const list = openOptions(box)[0].closest('[role="listbox"]') as HTMLElement;
    expect(list.getAttribute('aria-label')).toBe('Sort');
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();

    pickOption(box, 'date');
    expect(onchange).toHaveBeenCalledExactlyOnceWith('date');
  });

  describe('grouped columns', () => {
    const KEYS = [
      { value: '', label: 'Auto', hint: 'The model picks' },
      { value: 'C', label: 'C major', group: 'Major' },
      { value: 'D', label: 'D major', group: 'Major' },
      { value: 'Cm', label: 'C minor', group: 'Minor' },
      { value: 'Dm', label: 'D minor', group: 'Minor' },
    ];

    it('lists the options that have no group first, then one titled column per group, with the hint under the label', () => {
      const { box } = mountSelect({ options: KEYS, value: '', groupColumns: true });
      const options = openOptions(box);

      expect(options.map((option) => option.dataset.value)).toEqual(['', 'C', 'D', 'Cm', 'Dm']);
      const columns = Array.from(document.querySelectorAll('.ui-select-column'));
      expect(columns.map((column) => column.getAttribute('aria-label'))).toEqual(['Major', 'Minor']);
      expect(columns[1].querySelectorAll('[role="option"]')).toHaveLength(2);
      expect(options[0].textContent).toContain('The model picks');
      expect(options[0].closest('.ui-select-column')).toBeNull();
    });

    it('moves Left and Right to the same row of the next column, and Down along the order', () => {
      const { box } = mountSelect({ options: KEYS, value: 'D', groupColumns: true });
      const list = document.getElementById(openOptions(box)[0].closest('[role="listbox"]')!.id)!;
      const active = () => list.querySelector('[data-active="true"]')?.getAttribute('data-value');

      expect(active()).toBe('D');
      key(list, 'ArrowRight');
      expect(active()).toBe('Dm');
      key(list, 'ArrowRight');
      expect(active()).toBe('Dm');
      key(list, 'ArrowLeft');
      expect(active()).toBe('D');
      key(list, 'ArrowDown');
      expect(active()).toBe('Cm');
    });

    it('lists the options in one column without the prop, and picks from the columns', () => {
      const flat = mountSelect({ options: KEYS, value: '' });
      openOptions(flat.box);
      expect(document.querySelector('.ui-select-column')).toBeNull();
      flushSync(() => flat.box.click());

      const onchange = vi.fn();
      const { box } = mountSelect({ options: KEYS, value: '', groupColumns: true, onchange });
      pickOption(box, 'Dm');
      expect(onchange).toHaveBeenCalledExactlyOnceWith('Dm');
    });
  describe('stacked groups', () => {
    const VOICES = [
      { value: 'ryan', label: 'Ryan · English', group: 'Built-in' },
      { value: 'aiden', label: 'Aiden · English', group: 'Built-in' },
      { value: 'saved:morgan', label: 'Morgan', group: 'Saved' },
    ];

    it('lists each run of a group under its heading, in the order given', () => {
      const { box } = mountSelect({ options: VOICES, value: 'ryan' });
      const options = openOptions(box);

      expect(options.map((option) => option.dataset.value)).toEqual(['ryan', 'aiden', 'saved:morgan']);
      const groups = Array.from(document.querySelectorAll('[role="listbox"] [role="group"]'));
      expect(groups.map((group) => group.getAttribute('aria-label'))).toEqual(['Built-in', 'Saved']);
      expect(groups[1].querySelectorAll('[role="option"]')).toHaveLength(1);
    });

    it('picks from under a heading and walks the options with the keys across the headings', () => {
      const onchange = vi.fn();
      const { box } = mountSelect({ options: VOICES, value: 'aiden', onchange });
      const list = document.getElementById(openOptions(box)[0].closest('[role="listbox"]')!.id)!;
      const active = () => list.querySelector('[data-active="true"]')?.getAttribute('data-value');

      key(list, 'ArrowDown');
      expect(active()).toBe('saved:morgan');
      key(list, 'ArrowUp');
      expect(active()).toBe('aiden');
      key(list, 'ArrowDown');
      expect(active()).toBe('saved:morgan');
      flushSync(() => box.click());

      pickOption(box, 'saved:morgan');
      expect(onchange).toHaveBeenCalledExactlyOnceWith('saved:morgan');
    });

    it('shows no groups when no option has one', () => {
      const { box } = mountSelect({});
      openOptions(box);

      expect(document.querySelector('[role="listbox"] [role="group"]')).toBeNull();
    });
  });
  });
});
