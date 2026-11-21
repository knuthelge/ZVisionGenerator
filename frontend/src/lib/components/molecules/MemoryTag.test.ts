// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { JobMemory } from '$lib/types';
import { imageMemory, memoryRow, singleMemory } from '../../../test-utils/memory';
import MemoryTag from './MemoryTag.svelte';

describe('MemoryTag', () => {
  let target: HTMLDivElement;
  let component: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(() => {
    if (component) {
      unmount(component);
      component = null;
    }
    target.remove();
  });

  function render(memory: JobMemory, props: Record<string, unknown> = {}): HTMLElement {
    component = mount(MemoryTag, { target, props: { memory, ...props } });
    flushSync();
    return target.querySelector('[data-testid="memory-tag"]') as HTMLElement;
  }

  const text = (tag: HTMLElement, id: string) => tag.querySelector(`[data-testid="${id}"]`)?.textContent;
  const head = (tag: HTMLElement) => Array.from(tag.querySelector('.mem-row-head')!.children).map((cell) => cell.textContent?.trim()).filter(Boolean);
  const rows = (tag: HTMLElement) => Array.from(tag.querySelectorAll<HTMLElement>('[data-testid="memory-row"]'));

  it.each([['none', 'tight', 'Tight'], ['8', 'fits', 'Fits'], ['4', 'too_large', 'Too large']] as const)('says the word of the current level (%s)', (level, status, word) => {
    const tag = render(imageMemory(level));

    expect(tag.querySelector('[data-status]')?.getAttribute('data-status')).toBe(status);
    expect(tag.querySelector('.ui-badge')?.textContent?.trim()).toBe(word);
  });

  it('is a tab stop whose popover opens on focus and is described to a screen reader in words', () => {
    const tag = render(imageMemory('4'));
    const bubble = tag.querySelector('[role="tooltip"]') as HTMLElement;

    expect(tag.getAttribute('tabindex')).toBe('0');
    expect(bubble.dataset.open).toBe('false');
    tag.dispatchEvent(new FocusEvent('focus'));
    flushSync();
    expect(bubble.dataset.open).toBe('true');
    const memory = imageMemory('4');
    const spoken = bubble.querySelector('.sr-only')?.textContent ?? '';
    expect(spoken).toContain(memory.title);
    expect(spoken).toContain(memory.rows[2].sentence);
    expect(spoken).toContain('Quant q4 (selected): Too large, GPU 10.4, RAM 5.9 GB.');
  });

  it('lays out the title, what is judged, the sentence, the label and one row per quant with the current one marked', () => {
    const memory = imageMemory('8');
    const tag = render(memory);

    expect(text(tag, 'memory-title')).toBe(memory.title);
    expect(text(tag, 'memory-judged')).toBe(memory.rows[1].details);
    expect(text(tag, 'memory-sentence')).toBe(memory.rows[1].sentence);
    expect(text(tag, 'memory-options-label')).toBe(memory.table_heading);
    expect(rows(tag).map((row) => [row.dataset.key, row.hasAttribute('data-current'), row.textContent?.replace(/\s+/g, ' ').trim()])).toEqual([
      ['none', false, 'None Tight 8.0 32.5'],
      ['8', true, 'q8 Fits 8.0 17.0'],
      ['4', false, 'q4 Too large 10.4 5.9'],
    ]);
    expect(head(tag)).toEqual(['Quant', 'GPU · 10 GB', 'RAM · 30.3 GB']);
  });

  it('judges the level it is asked for, not the answer\'s', () => {
    const tag = render(imageMemory('none'), { level: '4' });

    expect(tag.querySelector('[data-status]')?.getAttribute('data-status')).toBe('too_large');
    expect(rows(tag).find((row) => row.hasAttribute('data-current'))?.dataset.key).toBe('4');
  });

  it('shows the sentence of the row as the server sent it, and the Mac\'s one memory column', () => {
    const ram = imageMemory('none', { rows: [{ ...memoryRow('none', 'too_large', 4, 40), resource: 'ram', sentence: 'A sentence about RAM.' }], control: null });
    const mac: JobMemory = { ...ram, kind: 'unified', budget: { gpu_gb: 21.3, ram_gb: null }, rows: [{ ...memoryRow('none', 'too_large', 40, null), resource: 'unified', sentence: 'A sentence about the Mac.' }] };

    expect(text(render(ram), 'memory-sentence')).toBe('A sentence about RAM.');
    unmount(component!);
    component = null;
    const macTag = render(mac);
    expect(text(macTag, 'memory-sentence')).toBe('A sentence about the Mac.');
    expect(head(macTag)).toEqual(['Memory · 21.3 GB']);
  });

  it('shows a model that is not downloaded with the download it would need', () => {
    const tag = render(imageMemory('8', { download_size_label: '17 GB' }));

    expect(text(tag, 'memory-download')).toContain('Not downloaded · about 17 GB');
  });

  it('has one row and no label for a model with a single level, and no RAM column for audio', () => {
    const tag = render(singleMemory('tight'));

    expect(rows(tag)).toHaveLength(1);
    expect(tag.querySelector('[data-testid="memory-options-label"]')).toBeNull();
    expect(head(tag)).toEqual(['GPU · 10 GB']);
    expect(text(tag, 'memory-sentence')).toBe(singleMemory('tight').rows[0].sentence);
  });

  it('carries no buttons, links or meters while it is for reading', () => {
    const tag = render(imageMemory('4'));

    expect(tag.querySelector('[role="tooltip"] button, [role="tooltip"] a, [role="tooltip"] progress, [role="tooltip"] [role="meter"]')).toBeNull();
  });

  describe('while pending', () => {
    it('dims the badge and marks it busy, with no new text', () => {
      const idle = render(imageMemory('8'));
      const idleSpan = idle.querySelector('[data-status]')!;
      expect(idleSpan.hasAttribute('aria-busy')).toBe(false);
      expect(idleSpan.classList.contains('opacity-50')).toBe(false);
      const idleText = idle.textContent;
      unmount(component!);
      component = null;

      const tag = render(imageMemory('8'), { pending: true });
      const span = tag.querySelector('[data-status]')!;

      expect(span.getAttribute('aria-busy')).toBe('true');
      expect(span.classList.contains('opacity-50')).toBe(true);
      expect(tag.textContent).toBe(idleText);
    });

    it('keeps the rows from being chosen, even with a callback', () => {
      const onselect = vi.fn();
      const tag = render(imageMemory('8'), { onselect, selectable: ['none', '8', '4'], pending: true });

      expect(tag.querySelectorAll('button')).toHaveLength(0);
      tag.querySelector<HTMLElement>('[data-testid="memory-row"][data-key="4"]')!.click();
      expect(onselect).not.toHaveBeenCalled();
    });
  });

  describe('choosing a level', () => {
    const buttons = (tag: HTMLElement) => Array.from(tag.querySelectorAll<HTMLButtonElement>('button[data-testid="memory-row"]'));
    const button = (tag: HTMLElement, key: string) => tag.querySelector<HTMLButtonElement>(`[data-testid="memory-row"][data-key="${key}"]`)!;

    it('leaves the rows plain without a callback', () => {
      const tag = render(imageMemory('8'));

      expect(buttons(tag)).toHaveLength(0);
      expect(tag.querySelector('[data-testid="memory-row"][tabindex]')).toBeNull();
    });

    it('makes the rows plain buttons, with the current level pressed and no radio or tooltip roles around them', () => {
      const tag = render(imageMemory('8'), { onselect: vi.fn() });

      expect(buttons(tag).map((row) => [row.dataset.key, row.type, row.getAttribute('aria-pressed'), row.disabled])).toEqual([
        ['none', 'button', 'false', false],
        ['8', 'button', 'true', false],
        ['4', 'button', 'false', false],
      ]);
      expect(tag.querySelector('[role="radio"], [role="radiogroup"], [role="tooltip"]')).toBeNull();
      expect(button(tag, '4').getAttribute('aria-label')).toBe('Quant q4: Too large, GPU 10.4, RAM 5.9 GB.');
      expect(button(tag, '8').getAttribute('aria-label')).toBe('Quant q8: Fits, GPU 8.0, RAM 17.0 GB.');
    });

    it('names the Low memory control for a Low memory answer', () => {
      const lowMemory = singleMemory('fits', { control: 'low_memory', control_title: 'Low memory', current: 'on', rows: [memoryRow('on', 'tight', 6, 20), memoryRow('off', 'too_large', 14, 25)] });
      const tag = render(lowMemory, { onselect: vi.fn() });

      expect(buttons(tag).map((row) => [row.dataset.key, row.getAttribute('aria-pressed')])).toEqual([['on', 'true'], ['off', 'false']]);
      expect(button(tag, 'off').getAttribute('aria-label')).toBe('Low memory Off: Too large, GPU 14.0, RAM 25.0 GB.');
    });

    it('calls back with the key of a clicked row, and does nothing for the current one', () => {
      const onselect = vi.fn();
      const tag = render(imageMemory('8'), { onselect });

      button(tag, '4').click();
      button(tag, 'none').click();
      button(tag, '8').click();

      expect(onselect.mock.calls).toEqual([['4'], ['none']]);
    });

    it('keeps rows that are not selectable disabled and silent', () => {
      const onselect = vi.fn();
      const tag = render(imageMemory('8'), { onselect, selectable: ['none', '8'] });

      expect(button(tag, '4').disabled).toBe(true);
      expect(button(tag, '4').tabIndex).toBe(0);
      button(tag, '4').click();
      button(tag, 'none').click();

      expect(onselect.mock.calls).toEqual([['none']]);
    });

    it('offers nothing when no row is selectable, or the answer has a single level', () => {
      const none = render(imageMemory('8'), { onselect: vi.fn(), selectable: [] });
      expect(buttons(none)).toHaveLength(0);
      unmount(component!);
      component = null;

      const single = render(singleMemory('tight'), { onselect: vi.fn() });
      expect(buttons(single)).toHaveLength(0);
    });

    it('does not read the rows twice: the sentence leaves out what the buttons name', () => {
      const memory = imageMemory('4');
      const tag = render(memory, { onselect: vi.fn() });
      const spoken = tag.querySelector('[data-open] .sr-only')?.textContent ?? '';

      expect(spoken).toContain(memory.rows[2].sentence);
      expect(spoken).toContain('Budget: GPU 10 GB, RAM 30.3 GB.');
      expect(spoken).not.toContain('Quant q4');
    });

    it('opens as a group that takes the pointer, not a tooltip around its buttons', () => {
      const tag = render(imageMemory('8'), { onselect: vi.fn() });
      const bubble = tag.querySelector('[data-open]') as HTMLElement;

      expect(bubble.getAttribute('role')).toBe('group');
      tag.dispatchEvent(new FocusEvent('focus'));
      flushSync();
      expect(getComputedStyle(bubble).pointerEvents).toBe('auto');
    });

    it('closes on Escape pressed on a row and puts focus back on the tag', () => {
      const tag = render(imageMemory('8'), { onselect: vi.fn() });
      const bubble = tag.querySelector('[data-open]') as HTMLElement;
      tag.dispatchEvent(new FocusEvent('focus'));
      flushSync();
      button(tag, '4').focus();

      button(tag, '4').dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      flushSync();

      expect(document.activeElement).toBe(tag);
      expect(bubble.dataset.open).toBe('false');
    });

    it('stays open for a tap on a row, which chooses it, and closes for a tap on the tag again', () => {
      const onselect = vi.fn();
      const tag = render(imageMemory('8'), { onselect });
      const bubble = tag.querySelector('[data-open]') as HTMLElement;
      const touch = (element: Element) => {
        element.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerType: 'touch' }));
        flushSync();
      };

      touch(tag);
      expect(bubble.dataset.open).toBe('true');
      touch(button(tag, '4'));
      button(tag, '4').click();

      expect(bubble.dataset.open).toBe('true');
      expect(onselect.mock.calls).toEqual([['4']]);
    });
  });

  describe('a popover for reading', () => {
    it('closes on a tap inside it instead of letting the tap reach what lies beneath', () => {
      const tag = render(imageMemory('8'));
      const bubble = tag.querySelector('[data-open]') as HTMLElement;
      const touch = (element: Element) => {
        element.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerType: 'touch' }));
        flushSync();
      };

      touch(tag);
      expect(bubble.dataset.open).toBe('true');
      expect(getComputedStyle(bubble).pointerEvents).toBe('auto');
      touch(bubble.querySelector('[data-testid="memory-sentence"]')!);

      expect(bubble.dataset.open).toBe('false');
    });
  });
});
