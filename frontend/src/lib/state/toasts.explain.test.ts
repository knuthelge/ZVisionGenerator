import { beforeEach, describe, expect, it } from 'vitest';

import { explainOff, toasts } from './toasts.svelte';

describe('explainOff', () => {
  beforeEach(() => toasts.splice(0, toasts.length));

  it('leaves a mouse to the tooltip', () => {
    explainOff('The model has no last frame', 'mouse');
    expect(toasts).toHaveLength(0);
  });

  it.each(['touch', 'pen', 'keyboard', ''])('toasts the reason for a %s', (pointerType) => {
    explainOff('The model has no last frame', pointerType);
    expect(toasts.map((toast) => toast.message)).toEqual(['The model has no last frame']);
  });

  it('says nothing without a reason', () => {
    explainOff(undefined, 'touch');
    expect(toasts).toHaveLength(0);
  });
});
