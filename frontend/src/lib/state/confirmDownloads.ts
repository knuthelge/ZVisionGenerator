import { requestConfirm } from '$lib/components/molecules/confirm.svelte';
import type { ConfirmPart } from '$lib/components/molecules/ConfirmDialog.svelte';
import type { JobDownloadLicence, JobDownloadPlan } from '$lib/types';
import { downloadConfirmText, UNCHECKED_TEXT } from './downloadConfirm';
import { sizeLabel } from './downloadProgress';

/** The lines under a part: one per licence record that covers its model, each ending in its links. */
function licenceLines(licences: JobDownloadLicence[], model: string): ConfirmPart['lines'] {
  return licences
    .filter((licence) => licence.model === model)
    .map((licence) => ({
      text: `${licence.name} (${licence.supplier}): ${licence.summary}`,
      links: [
        { label: `${licence.url_label} ↗`, href: licence.url, ariaLabel: `${licence.name} ${licence.url_label.toLowerCase()} (opens in a new tab)` },
        ...(licence.notice_url ? [{ label: 'NOTICE ↗', href: licence.notice_url, ariaLabel: `${licence.name} NOTICE (opens in a new tab)` }] : []),
      ],
    }));
}

/**
 * Ask the user to approve the downloads of *plan* before a job (or a Models-page download) starts them.
 *
 * Resolves true at once, with no dialog, when the plan has no parts. Otherwise it shows one dialog with a block per
 * part, its size, the terms of its model and the free space on the download drive; resolves true on Download, false on Cancel.
 */
export async function confirmDownloads(plan: JobDownloadPlan): Promise<boolean> {
  if (plan.parts.length === 0) return true;
  const question = plan.parts.length === 1 ? `Download ${plan.parts[0].label}?` : `Download ${plan.parts.length} models?`;
  const { info, confirmLabel } = plan.unchecked ? UNCHECKED_TEXT : downloadConfirmText(plan);
  return requestConfirm({
    question,
    parts: plan.parts.map((part) => ({ label: part.label, size: part.bytes === null ? null : sizeLabel(part.bytes), lines: licenceLines(plan.licences, part.model) })),
    info,
    infoTone: plan.unchecked ? 'warning' : undefined,
    confirmLabel,
    danger: false,
  });
}
