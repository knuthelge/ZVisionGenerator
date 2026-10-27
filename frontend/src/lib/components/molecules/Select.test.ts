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
});
