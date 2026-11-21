// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { createRawSnippet } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import Tooltip from './Tooltip.svelte';

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
    return { trigger, bubble: trigger.querySelector('[data-open]') as HTMLElement };
  }

  function placeTrigger(trigger: HTMLElement, rect: Partial<DOMRect>): void {
    trigger.getBoundingClientRect = () => ({ top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0, x: 0, y: 0, toJSON: () => ({}), ...rect }) as DOMRect;
  }

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
    trigger.dispatchEvent(new FocusEvent('blur'));
    flushSync();
    expect(bubble.dataset.open).toBe('false');
    trigger.dispatchEvent(new FocusEvent('focus'));
    flushSync();
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
      trigger.dispatchEvent(new PointerEvent(type, { bubbles: true, pointerType: 'touch' }));
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

  describe('interactive', () => {
    const inside = createRawSnippet(() => ({ render: () => '<button type="button" data-testid="inner">Pick</button>' }));
    const touch = (target: Element, type: string) => {
      target.dispatchEvent(new PointerEvent(type, { bubbles: true, pointerType: 'touch' }));
      flushSync();
    };

    afterEach(() => {
      vi.useRealTimers();
    });

    it('stays open while the pointer crosses from the trigger to the content, and closes once it has left both', () => {
      vi.useFakeTimers();
      const { trigger, bubble } = render({ interactive: true, content: inside });

      trigger.dispatchEvent(new Event('pointerenter'));
      flushSync();
      trigger.dispatchEvent(new Event('pointerleave'));
      vi.advanceTimersByTime(50);
      trigger.dispatchEvent(new Event('pointerenter'));
      vi.advanceTimersByTime(500);
      flushSync();
      expect(bubble.dataset.open).toBe('true');

      trigger.dispatchEvent(new Event('pointerleave'));
      vi.advanceTimersByTime(500);
      flushSync();
      expect(bubble.dataset.open).toBe('false');
    });

    it('closes at once on leave when it is not interactive', () => {
      const { trigger, bubble } = render({});

      trigger.dispatchEvent(new Event('pointerenter'));
      flushSync();
      trigger.dispatchEvent(new Event('pointerleave'));
      flushSync();

      expect(bubble.dataset.open).toBe('false');
    });

    it('stays open while focus moves into its content, and closes when focus leaves the trigger and the content', () => {
      const { trigger, bubble } = render({ interactive: true, content: inside });
      const inner = bubble.querySelector('[data-testid="inner"]') as HTMLElement;

      trigger.dispatchEvent(new FocusEvent('focus'));
      flushSync();
      trigger.dispatchEvent(new FocusEvent('focusout', { bubbles: true, relatedTarget: inner }));
      flushSync();
      expect(bubble.dataset.open).toBe('true');

      inner.dispatchEvent(new FocusEvent('focusout', { bubbles: true, relatedTarget: document.body }));
      flushSync();
      expect(bubble.dataset.open).toBe('false');
    });

    it('does not close on a tap inside the content, but a tap on the trigger still toggles it', () => {
      const { trigger, bubble } = render({ interactive: true, content: inside });
      const inner = bubble.querySelector('[data-testid="inner"]') as HTMLElement;

      touch(trigger, 'pointerdown');
      expect(bubble.dataset.open).toBe('true');
      touch(inner, 'pointerdown');
      expect(bubble.dataset.open).toBe('true');
      touch(trigger, 'pointerdown');
      expect(bubble.dataset.open).toBe('false');
    });

    it('ignores a pointerdown inside non-interactive content, and closes on the click that follows, which goes no further', () => {
      const { trigger, bubble } = render({ content: inside });
      const beneath = vi.fn();
      target.addEventListener('click', beneath);

      touch(trigger, 'pointerdown');
      expect(bubble.dataset.open).toBe('true');
      touch(bubble, 'pointerdown');
      expect(bubble.dataset.open).toBe('true');

      const click = new MouseEvent('click', { bubbles: true, cancelable: true });
      bubble.dispatchEvent(click);
      flushSync();

      expect(bubble.dataset.open).toBe('false');
      expect(click.defaultPrevented).toBe(true);
      expect(beneath).not.toHaveBeenCalled();
      target.removeEventListener('click', beneath);
    });

    it('leaves a click inside interactive content alone: it is not closed, stopped or prevented', () => {
      const { trigger, bubble } = render({ interactive: true, content: inside });
      const inner = bubble.querySelector('[data-testid="inner"]') as HTMLElement;
      const beneath = vi.fn();
      target.addEventListener('click', beneath);

      touch(trigger, 'pointerdown');
      const click = new MouseEvent('click', { bubbles: true, cancelable: true });
      inner.dispatchEvent(click);
      flushSync();

      expect(bubble.dataset.open).toBe('true');
      expect(click.defaultPrevented).toBe(false);
      expect(beneath).toHaveBeenCalledOnce();
      target.removeEventListener('click', beneath);
    });

    it('is not a tooltip around controls: interactive content is a group, plain content a tooltip', () => {
      expect(render({ interactive: true, content: inside }).bubble.getAttribute('role')).toBe('group');
      unmount(component!);
      component = null;
      expect(render({ content: inside }).bubble.getAttribute('role')).toBe('tooltip');
    });
  });

  describe('laid-out content', () => {
    const inside = createRawSnippet(() => ({ render: () => '<button type="button" data-testid="inner">Pick</button>' }));
    const pointerEvents = (bubble: HTMLElement) => getComputedStyle(bubble).pointerEvents;

    it('takes pointer events while open, so a tap lands on it and not on the control beneath, and none while closed', () => {
      const { trigger, bubble } = render({ content: inside });
      expect(pointerEvents(bubble)).toBe('none');

      trigger.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerType: 'touch' }));
      flushSync();
      expect(pointerEvents(bubble)).toBe('auto');

      bubble.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerType: 'touch' }));
      flushSync();
      expect(bubble.dataset.open).toBe('true');
      expect(pointerEvents(bubble)).toBe('auto');

      bubble.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
      flushSync();
      expect(bubble.dataset.open).toBe('false');
      expect(pointerEvents(bubble)).toBe('none');
    });

    it('does the same when it is interactive', () => {
      const { trigger, bubble } = render({ content: inside, interactive: true });
      expect(pointerEvents(bubble)).toBe('none');

      trigger.dispatchEvent(new FocusEvent('focus'));
      flushSync();

      expect(pointerEvents(bubble)).toBe('auto');
    });

    it('leaves a text tooltip out of the way of the pointer', () => {
      const { bubble } = render({});

      expect(bubble.classList.contains('pointer-events-none')).toBe(true);
    });

    it('moves focus back to the trigger before it closes on Escape pressed inside it', () => {
      const { trigger, bubble } = render({ content: inside, interactive: true });
      const inner = bubble.querySelector('[data-testid="inner"]') as HTMLElement;
      trigger.dispatchEvent(new FocusEvent('focus'));
      flushSync();
      inner.focus();
      expect(document.activeElement).toBe(inner);

      inner.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      flushSync();

      expect(document.activeElement).toBe(trigger);
      expect(bubble.dataset.open).toBe('false');
    });

    it('leaves focus where it is when Escape is pressed on the trigger', () => {
      const { trigger, bubble } = render({ content: inside, interactive: true });
      trigger.focus();
      flushSync();
      expect(bubble.dataset.open).toBe('true');

      trigger.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      flushSync();

      expect(document.activeElement).toBe(trigger);
      expect(bubble.dataset.open).toBe('false');
    });
  });
});
