import { describe, expect, it } from 'vitest';

import { hasPendingReference, requestReference, takeReference } from './pendingReference.svelte';

describe('pendingReference', () => {
  it('hands a requested image out once', () => {
    expect(hasPendingReference()).toBe(false);
    requestReference('/outputs/a.png');
    expect(hasPendingReference()).toBe(true);
    expect(takeReference()).toBe('/outputs/a.png');
    expect(hasPendingReference()).toBe(false);
    expect(takeReference()).toBeNull();
  });

  it('keeps only the latest request', () => {
    requestReference('/outputs/a.png');
    requestReference('/outputs/b.png');
    expect(takeReference()).toBe('/outputs/b.png');
    expect(takeReference()).toBeNull();
  });
});
