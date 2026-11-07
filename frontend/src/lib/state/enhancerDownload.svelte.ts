import { SvelteSet } from 'svelte/reactivity';
import { ApiError } from '$lib/api/client';
import { getDownloadPlan } from '$lib/api/models';
import { confirmDownloads } from './confirmDownloads';
import { uncheckedJobPlan } from './downloadConfirm';
import type { EnhancerTier, PromptEnhancerContract } from '$lib/types';

/** Enhancer models downloaded during this session, so their first-use size stops showing. */
export const downloadedTiers = new SvelteSet<EnhancerTier>();

/** Whether the model of *tier* is missing from the disk, so the first enhancement would download it. */
export function tierMissing(contract: PromptEnhancerContract, tier: EnhancerTier): boolean {
  if (downloadedTiers.has(tier)) return false;
  return tier === 'smart' ? contract.smart?.downloaded === false : !contract.downloaded;
}

/**
 * Ask before an enhancement downloads its model. `confirmed` is true when nothing is missing or the user agrees; `error` says why when the plan
 * can't be read.
 */
export async function confirmEnhancerDownload(contract: PromptEnhancerContract, tier: EnhancerTier): Promise<{ confirmed: boolean; error: string | null }> {
  if (!tierMissing(contract, tier)) return { confirmed: true, error: null };
  // A custom model (no Fast/Smart choice) is the Models page's `custom` row.
  const name = tier === 'smart' ? 'smart' : contract.smart === null ? 'custom' : 'fast';
  try {
    return { confirmed: await confirmDownloads(await getDownloadPlan('enhancer', name)), error: null };
  } catch (error) {
    // Hugging Face can't be reached to size it: still ask, and let the user go ahead.
    if (error instanceof ApiError && error.status === 503) return { confirmed: await confirmDownloads(uncheckedJobPlan(`${name.charAt(0).toUpperCase()}${name.slice(1)} enhancer model`, name)), error: null };
    return { confirmed: false, error: error instanceof Error ? error.message : 'Could not check the enhancer model download.' };
  }
}
