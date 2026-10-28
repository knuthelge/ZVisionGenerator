import type { LoraInfo } from '$lib/types';

/** Pauses between reads of the LoRA list while a finished download's file is not in it: about a minute in all. */
export const LISTING_DELAYS_MS: readonly number[] = [1000, 2000, 4000, 8000, 15000, 30000];

export type ListingResult = 'listed' | 'gave-up' | 'stopped';

export interface ListingOptions {
  delays?: readonly number[];
  sleep?: (ms: number) => Promise<void>;
  /** True once the page no longer cares, so the loop ends. */
  stopped?: () => boolean;
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Read the LoRA list until the file *name* is in it, pausing longer after each miss or failed read.
 * Gives up after the last pause, so a flaky connection cannot hold Generate back for good.
 *
 * @param read Reads the list and puts it where the page shows it; a rejection is a missed read.
 */
export async function waitForListing(name: string, read: () => Promise<readonly LoraInfo[]>, options: ListingOptions = {}): Promise<ListingResult> {
  const { delays = LISTING_DELAYS_MS, sleep = wait, stopped = () => false } = options;
  for (const delay of delays) {
    await sleep(delay);
    if (stopped()) return 'stopped';
    try {
      const loras = await read();
      if (loras.some((lora) => lora.name === name)) return 'listed';
    } catch {
      // A missed read says nothing about the file: read again after the next pause.
    }
  }
  return 'gave-up';
}
