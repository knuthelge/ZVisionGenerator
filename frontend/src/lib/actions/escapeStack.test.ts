import { afterEach, describe, expect, it, vi } from 'vitest';
import { escapeLayerOpen, pushEscapeLayer } from './escapeStack';

function pressEscape(prevented = false): KeyboardEvent {
  const event = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
  if (prevented) event.preventDefault();
  document.body.dispatchEvent(event);
  return event;
}

describe('escape stack', () => {
  const releases: Array<() => void> = [];
  afterEach(() => { releases.splice(0).forEach((release) => release()); });

  it('closes only the innermost layer', () => {
    const outer = vi.fn();
    const inner = vi.fn();
    releases.push(pushEscapeLayer(outer), pushEscapeLayer(inner));
    const event = pressEscape();
    expect(inner).toHaveBeenCalledTimes(1);
    expect(outer).not.toHaveBeenCalled();
    expect(event.defaultPrevented).toBe(true);
  });

  it('hands Escape to the outer layer once the inner one is released', () => {
    const outer = vi.fn();
    releases.push(pushEscapeLayer(outer));
    const releaseInner = pushEscapeLayer(vi.fn());
    releaseInner();
    pressEscape();
    expect(outer).toHaveBeenCalledTimes(1);
  });

  it('leaves Escape to the page when no layer is open', () => {
    expect(escapeLayerOpen()).toBe(false);
    expect(pressEscape().defaultPrevented).toBe(false);
  });

  it('ignores an Escape a field inside the overlay already answered', () => {
    const onescape = vi.fn();
    releases.push(pushEscapeLayer(onescape));
    pressEscape(true);
    expect(onescape).not.toHaveBeenCalled();
  });
});
