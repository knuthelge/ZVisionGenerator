import type { DownloadPlan, JobDownloadPlan } from '$lib/types';
import { sizeLabel } from './downloadProgress';

/** What the Download confirmation says about a plan, and what its confirm button is called. */
export interface DownloadConfirmText {
  info: string;
  confirmLabel: 'Download' | 'Download anyway';
}

/** The part of a plan the text reads; a Models-page plan counts its repos as parts, a job plan counts its models. */
type SizedPlan = Pick<DownloadPlan, 'total_bytes' | 'space'> & { parts: readonly unknown[] };

/** What the dialog says when a part could not be sized because Hugging Face could not be reached. */
export const UNCHECKED_TEXT: DownloadConfirmText = {
  info: "Couldn't check the size: Hugging Face can't be reached. If the download can't run, the job fails.",
  confirmLabel: 'Download anyway',
};

/** Describe a non-empty download plan: its size, parts and the free space on the download drive. */
export function downloadConfirmText(plan: SizedPlan): DownloadConfirmText {
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

/** Describe a Models-page row's plan as a job plan of one part, so the one dialog asks about both. */
export function jobPlanOfRow(plan: DownloadPlan): JobDownloadPlan {
  if (plan.parts.length === 0) return { parts: [], total_bytes: 0, space: null, unchecked: false, licences: [] };
  const sizes = plan.parts.map((part) => part.bytes);
  const bytes = sizes.some((size) => size === null) ? null : sizes.reduce<number>((sum, size) => sum + (size ?? 0), 0);
  const files = plan.parts.reduce((sum, part) => sum + part.files, 0);
  return { parts: [{ label: plan.label, model: plan.name, repo: plan.parts[0].repo, files, bytes }], total_bytes: plan.total_bytes, space: plan.space, unchecked: false, licences: plan.licences ?? [] };
}

/** A job plan for a model whose size could not be read, so the dialog still asks and offers Download anyway. */
export function uncheckedJobPlan(label: string, model: string): JobDownloadPlan {
  return { parts: [{ label, model, repo: '', files: 0, bytes: null }], total_bytes: null, space: null, unchecked: true, licences: [] };
}
