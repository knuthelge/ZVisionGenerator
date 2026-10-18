// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { HOLD_DELAY_MS, HOLD_INTERVAL_MS, HOLD_SLOP_PX } from '$lib/actions/holdRepeat';
import InspectorNumber from './InspectorNumber.svelte';

function press(button: HTMLButtonElement): void {
  button.dispatchEvent(new PointerEvent('pointerdown', { button: 0, bubbles: true }));
}

function release(button: HTMLButtonElement): void {
  button.dispatchEvent(new PointerEvent('pointerup', { button: 0, bubbles: true }));
}

function finger(button: HTMLButtonElement, type: string, x = 10, y = 10): void {
  button.dispatchEvent(new PointerEvent(type, { button: 0, pointerType: 'touch', clientX: x, clientY: y, bubbles: true }));
}

describe('InspectorNumber', () => {
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
    target.remove();
    vi.useRealTimers();
  });

  it('renders a plain field with a fine pointer', () => {
    app = flushSync(() => mount(InspectorNumber, { target, props: { value: 20, step: 1, min: 1, max: 60, onchange: vi.fn() } }));
    expect(target.querySelector('input[type="number"]')).not.toBeNull();
    expect(target.querySelectorAll('button')).toHaveLength(0);
  });

  it('shows − and + when asked and nudges by one step within bounds', () => {
    const onchange = vi.fn();
    app = flushSync(() => mount(InspectorNumber, { target, props: { value: 60, step: 1, min: 1, max: 60, stepper: true, ariaLabel: 'Steps', onchange } }));

    const down = target.querySelector<HTMLButtonElement>('[data-step="down"]')!;
    const up = target.querySelector<HTMLButtonElement>('[data-step="up"]')!;
    expect(down.getAttribute('aria-label')).toBe('Decrease Steps');
    expect(up.getAttribute('aria-label')).toBe('Increase Steps');

    press(down);
    release(down);
    expect(onchange).toHaveBeenLastCalledWith(59);

    // Already at the maximum: + changes nothing.
    onchange.mockClear();
    press(up);
    release(up);
    expect(onchange).not.toHaveBeenCalled();
  });

  it('steps from the minimum when the value is empty', () => {
    const onchange = vi.fn();
    app = flushSync(() => mount(InspectorNumber, { target, props: { value: null, step: 0.1, min: 0, max: 10, nullable: true, stepper: true, onchange } }));
    const up = target.querySelector<HTMLButtonElement>('[data-step="up"]')!;
    press(up);
    release(up);
    expect(onchange).toHaveBeenLastCalledWith(0.1);
  });

  it('repeats while held and stops on release', () => {
    vi.useFakeTimers();
    const onchange = vi.fn();
    app = flushSync(() => mount(InspectorNumber, { target, props: { value: 10, step: 1, min: 1, max: 60, stepper: true, onchange } }));
    const up = target.querySelector<HTMLButtonElement>('[data-step="up"]')!;

    press(up);
    expect(onchange).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(HOLD_DELAY_MS + HOLD_INTERVAL_MS * 3);
    expect(onchange.mock.calls.length).toBeGreaterThanOrEqual(3);

    const calls = onchange.mock.calls.length;
    release(up);
    vi.advanceTimersByTime(HOLD_INTERVAL_MS * 5);
    expect(onchange).toHaveBeenCalledTimes(calls);
  });

  it('steps a finger on release, and not when the press turns into a scroll', () => {
    vi.useFakeTimers();
    const onchange = vi.fn();
    app = flushSync(() => mount(InspectorNumber, { target, props: { value: 10, step: 1, min: 1, max: 60, stepper: true, onchange } }));
    const up = target.querySelector<HTMLButtonElement>('[data-step="up"]')!;

    finger(up, 'pointerdown');
    expect(onchange).not.toHaveBeenCalled();
    finger(up, 'pointerup');
    expect(onchange).toHaveBeenCalledTimes(1);

    // The browser takes the pan over.
    onchange.mockClear();
    finger(up, 'pointerdown');
    finger(up, 'pointercancel');
    vi.advanceTimersByTime(HOLD_DELAY_MS * 2);
    expect(onchange).not.toHaveBeenCalled();

    // The finger moves on the button, as when pulling a sheet down.
    finger(up, 'pointerdown');
    finger(up, 'pointermove', 10, 10 + HOLD_SLOP_PX + 4);
    finger(up, 'pointerup', 10, 10 + HOLD_SLOP_PX + 4);
    vi.advanceTimersByTime(HOLD_DELAY_MS * 2);
    expect(onchange).not.toHaveBeenCalled();

    // A held finger still repeats.
    finger(up, 'pointerdown');
    vi.advanceTimersByTime(HOLD_DELAY_MS + HOLD_INTERVAL_MS * 3);
    expect(onchange.mock.calls.length).toBeGreaterThanOrEqual(3);
    const calls = onchange.mock.calls.length;
    finger(up, 'pointerup');
    expect(onchange).toHaveBeenCalledTimes(calls);
  });

  it('keeps a plain field when the stepper is turned off', () => {
    app = flushSync(() => mount(InspectorNumber, { target, props: { value: 1024, step: 16, stepper: false, onchange: vi.fn() } }));
    expect(target.querySelectorAll('button')).toHaveLength(0);
  });

  it('does nothing while disabled', () => {
    const onchange = vi.fn();
    app = flushSync(() => mount(InspectorNumber, { target, props: { value: 5, step: 1, stepper: true, disabled: true, onchange } }));
    const up = target.querySelector<HTMLButtonElement>('[data-step="up"]')!;
    expect(up.disabled).toBe(true);
    press(up);
    release(up);
    expect(onchange).not.toHaveBeenCalled();
  });
});
