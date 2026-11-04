import type { JobDownloadPlan } from '$lib/types';
import { sizeLabel } from './downloadProgress';

/** What the Download confirmation says about a plan, and what its confirm button is called. */
export interface DownloadConfirmText {
  info: string;
  confirmLabel: 'Download' | 'Download anyway';
}

/** What the dialog says when a part could not be sized because Hugging Face could not be reached. */
export const UNCHECKED_TEXT: DownloadConfirmText = {
  info: "Couldn't check the size: Hugging Face can't be reached. If the download can't run, the job fails.",
  confirmLabel: 'Download anyway',
};

/** Describe a non-empty download plan: its size, parts and the free space on the download drive. */
export function downloadConfirmText(plan: Pick<JobDownloadPlan, 'parts' | 'total_bytes' | 'space'>): DownloadConfirmText {
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

/** A job plan for a model whose size could not be read, so the dialog still asks and offers Download anyway. */
export function uncheckedJobPlan(label: string, model: string): JobDownloadPlan {
  return { parts: [{ label, model, bytes: null }], total_bytes: null, space: null, unchecked: true, licences: [] };
}
