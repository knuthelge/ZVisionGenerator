import { describe, expect, it } from 'vitest';

import { shouldApplyStartupView } from './startupRouting';

describe('shouldApplyStartupView', () => {
  it('applies the configured view when the address names no page', () => {
    expect(shouldApplyStartupView('')).toBe(true);
  });

  it('keeps an address that names a page', () => {
    expect(shouldApplyStartupView('#/gallery')).toBe(false);
    expect(shouldApplyStartupView('#/workspace?prompt=fox')).toBe(false);
  });
});
