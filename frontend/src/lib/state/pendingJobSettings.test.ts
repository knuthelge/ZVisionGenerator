import { describe, expect, it } from 'vitest';

import { hasPendingJobSettings, requestJobSettings, takeJobSettings } from './pendingJobSettings.svelte';

describe('pendingJobSettings', () => {
  it('hands a requested set of settings out once', () => {
    expect(hasPendingJobSettings()).toBe(false);
    requestJobSettings({ model: 'zit', steps: '9' });
    expect(hasPendingJobSettings()).toBe(true);
    expect(takeJobSettings()).toEqual({ model: 'zit', steps: '9' });
    expect(hasPendingJobSettings()).toBe(false);
    expect(takeJobSettings()).toBeNull();
  });

  it('keeps only the latest request', () => {
    requestJobSettings({ model: 'first' });
    requestJobSettings({ model: 'second' });
    expect(takeJobSettings()).toEqual({ model: 'second' });
    expect(takeJobSettings()).toBeNull();
  });
});
