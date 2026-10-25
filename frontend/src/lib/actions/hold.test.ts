import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hold, HOLD_MS } from './hold';

/** Dispatch a pointer event; jsdom has no PointerEvent constructor with pointerType, so the fields are set by hand. */
function pointer(node: Element, type: string, pointerType = 'touch', x = 10, y = 10, pointerId = 1, isPrimary = true): void {
  const event = new MouseEvent(type, { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0 });
  Object.defineProperty(event, 'pointerType', { value: pointerType });
  Object.defineProperty(event, 'pointerId', { value: pointerId });
  Object.defineProperty(event, 'isPrimary', { value: isPrimary });
  node.dispatchEvent(event);
}

describe('hold', () => {
  let node: HTMLButtonElement;
  let stop: () => void = () => {};

  beforeEach(() => {
    vi.useFakeTimers();
    node = document.createElement('button');
    document.body.append(node);
  });

  afterEach(() => {
    stop();
    node.remove();
    vi.useRealTimers();
  });

  function attach(onhold = vi.fn(), ignore?: string) {
    stop = hold(node, { onhold, ignore }).destroy;
    return onhold;
  }

  it('fires once a finger has held long enough, showing the hold meanwhile, and swallows the click after it', () => {
    const onhold = attach();
    const onclick = vi.fn();
    node.addEventListener('click', onclick);
    pointer(node, 'pointerdown');
    expect(node.dataset.holding).toBe('');
    vi.advanceTimersByTime(HOLD_MS);
    expect(onhold).toHaveBeenCalledOnce();
    expect(node.dataset.holding).toBeUndefined();
    pointer(node, 'pointerup');
    node.click();
    expect(onclick).not.toHaveBeenCalled();
  });

  it('swallows the click the lifting finger makes wherever it lands, then clicks are ordinary again', () => {
    attach();
    const item = document.createElement('button');
    document.body.append(item);
    const onItem = vi.fn();
    item.addEventListener('click', onItem);
    pointer(node, 'pointerdown');
    vi.advanceTimersByTime(HOLD_MS);
    // The menu the hold opened has an item under the finger when it lifts.
    pointer(node, 'pointerup');
    item.click();
    expect(onItem).not.toHaveBeenCalled();

    vi.advanceTimersByTime(150);
    item.click();
    expect(onItem).toHaveBeenCalledOnce();
    // The held node is ordinary again too: a click with no press before it (a keyboard Enter) goes through.
    const onNode = vi.fn();
    node.addEventListener('click', onNode);
    node.click();
    expect(onNode).toHaveBeenCalledOnce();
    item.remove();
  });

  it('makes the node ordinary again when no click follows the lift, as on iOS', () => {
    attach();
    const onclick = vi.fn();
    node.addEventListener('click', onclick);
    pointer(node, 'pointerdown');
    vi.advanceTimersByTime(HOLD_MS);
    pointer(node, 'pointerup');
    vi.advanceTimersByTime(150);
    // A click with no press before it, as a keyboard activation makes.
    node.click();
    expect(onclick).toHaveBeenCalledOnce();
  });

  it('is a plain tap when the finger lifts early', () => {
    const onhold = attach();
    const onclick = vi.fn();
    node.addEventListener('click', onclick);
    pointer(node, 'pointerdown');
    vi.advanceTimersByTime(HOLD_MS / 2);
    pointer(node, 'pointerup');
    vi.advanceTimersByTime(HOLD_MS);
    node.click();
    expect(onhold).not.toHaveBeenCalled();
    expect(onclick).toHaveBeenCalledOnce();
  });

  it('cancels when the finger moves, as a scroll starting on it does', () => {
    const onhold = attach();
    pointer(node, 'pointerdown');
    pointer(node, 'pointermove', 'touch', 10, 40);
    vi.advanceTimersByTime(HOLD_MS);
    expect(onhold).not.toHaveBeenCalled();
  });

  it('leaves a mouse alone, and presses on ignored parts', () => {
    const onhold = attach(vi.fn(), '.inner');
    pointer(node, 'pointerdown', 'mouse');
    vi.advanceTimersByTime(HOLD_MS);
    const inner = document.createElement('span');
    inner.className = 'inner';
    node.append(inner);
    pointer(inner, 'pointerdown');
    vi.advanceTimersByTime(HOLD_MS);
    expect(onhold).not.toHaveBeenCalled();
  });

  /** A second finger, down while the first still is. */
  function second(target: Element, type: string, x = 200, y = 200): void {
    pointer(target, type, 'touch', x, y, 2, false);
  }

  it('cancels when a second finger comes down, as a pinch starts, however little the fingers move', () => {
    const onhold = attach();
    pointer(node, 'pointerdown', 'touch', 100, 100);
    second(node, 'pointerdown');
    pointer(node, 'pointermove', 'touch', 102, 102);
    second(node, 'pointermove', 203, 203);
    vi.advanceTimersByTime(HOLD_MS * 2);
    expect(onhold).not.toHaveBeenCalled();
    expect(node.dataset.holding).toBeUndefined();

    // Both fingers lift; the next press is a hold again.
    second(node, 'pointerup');
    pointer(node, 'pointerup', 'touch', 100, 100);
    pointer(node, 'pointerdown', 'touch', 10, 10, 3);
    vi.advanceTimersByTime(HOLD_MS);
    expect(onhold).toHaveBeenCalledOnce();
  });

  it('cancels when the second finger comes down off the node', () => {
    const onhold = attach();
    const elsewhere = document.createElement('div');
    document.body.append(elsewhere);
    pointer(node, 'pointerdown');
    second(elsewhere, 'pointerdown');
    vi.advanceTimersByTime(HOLD_MS);
    expect(onhold).not.toHaveBeenCalled();
    elsewhere.remove();
  });

  it('does not hold for a second finger when the first went down on an ignored part', () => {
    const onhold = attach(vi.fn(), '.inner');
    const inner = document.createElement('span');
    inner.className = 'inner';
    node.append(inner);
    pointer(inner, 'pointerdown');
    second(node, 'pointerdown');
    vi.advanceTimersByTime(HOLD_MS);
    expect(onhold).not.toHaveBeenCalled();
  });

  it('keeps swallowing the release when a second finger comes down after the hold fired', () => {
    attach();
    const onclick = vi.fn();
    node.addEventListener('click', onclick);
    pointer(node, 'pointerdown');
    vi.advanceTimersByTime(HOLD_MS);
    second(node, 'pointerdown');
    second(node, 'pointerup');
    pointer(node, 'pointerup');
    node.click();
    expect(onclick).not.toHaveBeenCalled();
    // No click follows a lift on iOS: once the first finger lifted, clicks are ordinary again.
    vi.advanceTimersByTime(150);
    node.click();
    expect(onclick).toHaveBeenCalledOnce();
  });

  it('starts afresh on the next press when a finger\'s lift never arrived', () => {
    const onhold = attach();
    const onclick = vi.fn();
    node.addEventListener('click', onclick);
    pointer(node, 'pointerdown');
    vi.advanceTimersByTime(HOLD_MS);
    // The lift of finger 1 is lost; a new first finger taps.
    pointer(node, 'pointerdown', 'touch', 10, 10, 2);
    pointer(node, 'pointerup', 'touch', 10, 10, 2);
    node.click();
    expect(onclick).toHaveBeenCalledOnce();
    pointer(node, 'pointerdown', 'touch', 10, 10, 3);
    vi.advanceTimersByTime(HOLD_MS);
    expect(onhold).toHaveBeenCalledTimes(2);
  });
});
