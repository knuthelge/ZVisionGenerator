import { canUseAsReference, type AssetActionHandlers, type ReferenceTarget } from '$lib/state/assetActions';
import type { GalleryAsset } from '$lib/types';
import type { ActionMenuEntry } from './ActionMenu.svelte';

/** Build the secondary actions for an asset: reference, download, delete. */
export function assetMenuEntries(
  asset: GalleryAsset,
  handlers: AssetActionHandlers,
  deleting = false,
  referenceUnavailable: Partial<Record<ReferenceTarget, string>> = {}
): ActionMenuEntry[] {
  const entries: ActionMenuEntry[] = [];
  const { onreference, ondelete } = handlers;
  if (onreference && canUseAsReference(asset)) {
    entries.push(
      { kind: 'heading', label: 'Use as reference' },
      ...referenceEntries(asset, onreference, referenceUnavailable),
      { kind: 'separator' }
    );
  }
  entries.push({ kind: 'item', id: 'download', label: 'Download', href: asset.url, download: asset.filename });
  if (ondelete) {
    entries.push(
      { kind: 'separator' },
      { kind: 'item', id: 'delete', label: deleting ? 'Deleting…' : 'Delete', danger: true, disabled: deleting, onselect: () => ondelete(asset) }
    );
  }
  return entries;
}

/** Build the "use as reference" items; a target the current model can't use is disabled with its reason. */
export function referenceEntries(
  asset: GalleryAsset,
  onreference: NonNullable<AssetActionHandlers['onreference']>,
  unavailable: Partial<Record<ReferenceTarget, string>> = {}
): ActionMenuEntry[] {
  const targets: { target: ReferenceTarget; id: string; label: string }[] = [
    { target: 'image', id: 'reference-image', label: 'For image (img2img)' },
    { target: 'video', id: 'reference-video', label: 'For video (img2vid)' },
  ];
  return targets.map(({ target, id, label }) => ({
    kind: 'item' as const,
    id,
    label,
    disabled: Boolean(unavailable[target]),
    title: unavailable[target],
    onselect: () => onreference(asset, target),
  }));
}
