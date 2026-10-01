// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { createRawSnippet } from 'svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import Tooltip from './Tooltip.svelte';
import * as atoms from './index';

const label = createRawSnippet(() => ({ render: () => '<span>Trigger</span>' }));

describe('Tooltip', () => {
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

  function render(props: Record<string, unknown>): { trigger: HTMLElement; bubble: HTMLElement } {
    component = mount(Tooltip, { target, props: { text: 'Helpful text', children: label, ...props } });
    flushSync();
    const trigger = target.firstElementChild as HTMLElement;
    return { trigger, bubble: trigger.querySelector('[role="tooltip"]') as HTMLElement };
  }

  function placeTrigger(trigger: HTMLElement, rect: Partial<DOMRect>): void {
    trigger.getBoundingClientRect = () => ({ top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0, x: 0, y: 0, toJSON: () => ({}), ...rect }) as DOMRect;
  }

  it('is exported from the atoms barrel', () => {
    expect(atoms.Tooltip).toBe(Tooltip);
  });

  it('describes its trigger and opens on hover and focus, closing on leave, blur and Escape', () => {
    const { trigger, bubble } = render({});
    expect(trigger.getAttribute('aria-describedby')).toBe(bubble.id);
    expect(trigger.getAttribute('tabindex')).toBe('0');
    expect(bubble.dataset.open).toBe('false');

    trigger.dispatchEvent(new Event('pointerenter'));
    flushSync();
    expect(bubble.dataset.open).toBe('true');
    trigger.dispatchEvent(new Event('pointerleave'));
    flushSync();
    expect(bubble.dataset.open).toBe('false');

    trigger.dispatchEvent(new FocusEvent('focus'));
    flushSync();
    expect(bubble.dataset.open).toBe('true');
    trigger.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    flushSync();
    expect(bubble.dataset.open).toBe('false');
  });

  it('is not a tab stop when focusable is off', () => {
    const { trigger } = render({ focusable: false });

    expect(trigger.hasAttribute('tabindex')).toBe(false);
  });

  it('positions against the viewport and flips below when there is no room above', () => {
    const { trigger, bubble } = render({ placement: 'top' });
    Object.defineProperty(bubble, 'offsetHeight', { configurable: true, value: 80 });

    placeTrigger(trigger, { top: 300, bottom: 320, left: 40, right: 100 });
    trigger.dispatchEvent(new Event('pointerenter'));
    flushSync();
    expect(bubble.style.top).toBe(`${300 - 6 - 80}px`);
    expect(bubble.style.left).toBe('40px');

    trigger.dispatchEvent(new Event('pointerleave'));
    placeTrigger(trigger, { top: 20, bottom: 40, left: 40, right: 100 });
    trigger.dispatchEvent(new Event('pointerenter'));
    flushSync();
    expect(bubble.style.top).toBe(`${40 + 6}px`);
  });

  it('aligns to the requested edge but stays inside the viewport', () => {
    const { trigger, bubble } = render({ align: 'end' });
    Object.defineProperty(bubble, 'offsetWidth', { configurable: true, value: 200 });
    Object.defineProperty(bubble, 'offsetHeight', { configurable: true, value: 40 });

    placeTrigger(trigger, { top: 300, bottom: 320, left: 400, right: 500 });
    trigger.dispatchEvent(new Event('pointerenter'));
    flushSync();
    expect(bubble.style.left).toBe('300px'); // right edges line up

    trigger.dispatchEvent(new Event('pointerleave'));
    placeTrigger(trigger, { top: 300, bottom: 320, left: 10, right: 60 });
    trigger.dispatchEvent(new Event('pointerenter'));
    flushSync();
    expect(bubble.style.left).toBe('8px'); // clamped instead of running off the left edge
  });

  it('toggles on tap for touch, ignoring the pointerleave that follows', () => {
    const { trigger, bubble } = render({});
    const touch = (type: string) => {
      const event = new Event(type, { bubbles: true }) as Event & { pointerType: string };
      Object.defineProperty(event, 'pointerType', { value: 'touch' });
      trigger.dispatchEvent(event);
      flushSync();
    };

    touch('pointerenter');
    expect(bubble.dataset.open).toBe('false');
    touch('pointerdown');
    expect(bubble.dataset.open).toBe('true');
    touch('pointerleave');
    expect(bubble.dataset.open).toBe('true');
    touch('pointerdown');
    expect(bubble.dataset.open).toBe('false');
  });

  it('closes when the page scrolls, since a fixed tooltip would drift from its trigger', () => {
    const { trigger, bubble } = render({});
    trigger.dispatchEvent(new Event('pointerenter'));
    flushSync();

    window.dispatchEvent(new Event('scroll'));
    flushSync();

    expect(bubble.dataset.open).toBe('false');
  });
});
