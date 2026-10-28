import { describe, expect, it, vi } from 'vitest';
import type { LoraInfo } from '$lib/types';
import { LISTING_DELAYS_MS, waitForListing } from './loraListing';

const file = (name: string): LoraInfo => ({ name, path: `/loras/${name}.safetensors` });

describe('waitForListing', () => {
  it('reads again after a missed read and after a read that fails, until the file is listed', async () => {
    const sleep = vi.fn().mockResolvedValue(undefined);
    const read = vi.fn().mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce([file('other')]).mockResolvedValue([file('other'), file('style')]);

    const result = await waitForListing('style', read, { sleep, delays: [10, 20, 40, 80] });

    expect(result).toBe('listed');
    expect(read).toHaveBeenCalledTimes(3);
    expect(sleep.mock.calls.map(([ms]) => ms)).toEqual([10, 20, 40]);
  });

  it('gives up after the last pause, so Generate is never held for good', async () => {
    const sleep = vi.fn().mockResolvedValue(undefined);
    const read = vi.fn().mockRejectedValue(new Error('offline'));

    const result = await waitForListing('style', read, { sleep, delays: [1, 2, 3] });

    expect(result).toBe('gave-up');
    expect(read).toHaveBeenCalledTimes(3);
  });

  it('stops reading when the page no longer cares', async () => {
    const read = vi.fn().mockResolvedValue([]);
    let stopped = false;
    const sleep = vi.fn(async () => { stopped = true; });

    expect(await waitForListing('style', read, { sleep, stopped: () => stopped })).toBe('stopped');
    expect(read).not.toHaveBeenCalled();
  });

  it('waits longer after each miss and for about a minute in all', () => {
    expect([...LISTING_DELAYS_MS]).toEqual([...LISTING_DELAYS_MS].sort((a, b) => a - b));
    const total = LISTING_DELAYS_MS.reduce((sum, ms) => sum + ms, 0);
    expect(total).toBeGreaterThanOrEqual(30_000);
    expect(total).toBeLessThanOrEqual(120_000);
  });
});
