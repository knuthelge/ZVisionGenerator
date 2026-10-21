import type { DownloadPlan } from '$lib/types';
import { sizeLabel } from './downloadProgress';

/** What the Download confirmation says about a plan, and what its confirm button is called. */
export interface DownloadConfirmText {
  info: string;
  confirmLabel: 'Download' | 'Download anyway';
}

/** Describe a non-empty download plan: its size, parts and the free space on the download drive. */
export function downloadConfirmText(plan: DownloadPlan): DownloadConfirmText {
  const { space, parts, total_bytes: total } = plan;
  const inParts = parts.length > 1 ? ` in ${parts.length} parts` : '';
  if (space && !space.enough) {
    const needed = sizeLabel(total ?? space.needed_bytes);
    return {
      info: `It needs about ${needed}${inParts} (an estimate), but only ${sizeLabel(space.free_bytes)} is free on ${space.path}. The download may run out of space.`,
      confirmLabel: 'Download anyway',
    };
  }
  const size = total === null ? 'Size unknown.' : `${sizeLabel(total)}${inParts}.`;
  return {
    info: space ? `${size} ${sizeLabel(space.free_bytes)} free on ${space.path}.` : size,
    confirmLabel: 'Download',
  };
}
