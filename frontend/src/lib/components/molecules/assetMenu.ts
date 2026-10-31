import { canExtend, canUseAsReference, extendReason, seedvr2Options, upscaleFactors, type AssetActionHandlers, type ReferenceTarget, type UseAsState } from '$lib/state/assetActions';
import type { AssetUpscaleFactor, GalleryAsset } from '$lib/types';
import type { ActionMenuEntry } from './ActionMenu.svelte';

/**
 * Build the actions for an asset's menu: reuse, use as, extend, upscale, download, delete.
 *
 * Reuse leads when offered: on a touch screen a grid tile shows only the menu button, so the menu is where the
 * tile's quick action lives.
 */
export function assetMenuEntries(
  asset: GalleryAsset,
  handlers: AssetActionHandlers,
  deleting = false,
  useAs: UseAsState = {}
): ActionMenuEntry[] {
  const entries: ActionMenuEntry[] = [];
  const { onreuse, onreference, onextend, onupscale, ondelete } = handlers;
  if (onreuse) {
    const reusable = asset.has_reusable_config === true;
    entries.push(
      { kind: 'item', id: 'reuse', label: 'Reuse settings', disabled: !reusable, title: reusable ? undefined : 'Reusable settings unavailable', onselect: () => onreuse(asset) },
      { kind: 'separator' }
    );
  }
  if (onreference && canUseAsReference(asset)) {
    entries.push(
      { kind: 'heading', label: 'Use as' },
      ...referenceEntries(asset, onreference, useAs),
      { kind: 'separator' }
    );
  }
  // A video that offers extending lists it, disabled with the reason when it can't be extended right now.
  if (onextend && asset.media_type === 'video' && asset.extend) {
    entries.push(
      { kind: 'item', id: 'extend', label: 'Extend', disabled: !canExtend(asset), title: extendReason(asset) ?? undefined, onselect: () => onextend(asset) },
      { kind: 'separator' }
    );
  }
  const upscaleItems = onupscale ? upscaleEntries(asset, onupscale) : [];
  if (upscaleItems.length > 0) {
    entries.push({ kind: 'heading', label: 'Upscale' }, ...upscaleItems, { kind: 'separator' });
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

/**
 * Build the upscale items: the SeedVR2 ones where offered (`2× → 1664×2432`), a separator, then the creative ones
 * (`2× creative → 1664×2432`). A SeedVR2 item is never disabled: over the memory estimate it is marked and explained
 * in its tooltip. A creative factor over a limit is disabled with its reason.
 */
export function upscaleEntries(asset: GalleryAsset, onupscale: NonNullable<AssetActionHandlers['onupscale']>): ActionMenuEntry[] {
  const seedvr2 = asset.upscale?.seedvr2;
  const plain = seedvr2
    ? seedvr2Options(asset).map((option) => ({
        kind: 'item' as const,
        id: `upscale-seedvr2-${option.factor}`,
        label: `${sizeLabel(option, '')}${warningMarker(option)}`,
        title: plainTitle(option, seedvr2.download_size_label),
        onselect: () => onupscale(asset, option.factor, 'seedvr2'),
      }))
    : [];
  const creative = upscaleFactors(asset).map((option) => ({
    kind: 'item' as const,
    id: `upscale-creative-${option.factor}`,
    label: sizeLabel(option, ' creative'),
    disabled: !option.allowed,
    title: option.allowed ? undefined : (option.reason ?? undefined),
    onselect: () => onupscale(asset, option.factor, 'creative'),
  }));
  return plain.length > 0 && creative.length > 0 ? [...plain, { kind: 'separator' as const }, ...creative] : [...plain, ...creative];
}

const HIGH_MEMORY_MARKER = ' · high memory';
const ONE_FRAME_MARKER = ' · very slow';

/** The label suffix for a warned SeedVR2 option: what the warning is about, from its code; none without a warning. */
function warningMarker(option: AssetUpscaleFactor): string {
  if (!option.warning) return '';
  return option.warning_code === 'one_frame_batch' ? ONE_FRAME_MARKER : HIGH_MEMORY_MARKER;
}

function sizeLabel(option: AssetUpscaleFactor, suffix: string): string {
  return `${option.factor}×${suffix} → ${option.width}×${option.height}`;
}

/** The tooltip of a SeedVR2 item: its memory warning, then the first-use download size while the model is not on disk. */
function plainTitle(option: AssetUpscaleFactor, downloadSizeLabel: string | null): string | undefined {
  const lines = [option.warning, downloadSizeLabel ? `First use downloads ${downloadSizeLabel}` : null].filter(Boolean);
  return lines.length > 0 ? lines.join('\n') : undefined;
}

const REFERENCE_TARGETS: readonly ReferenceTarget[] = ['image', 'reference', 'video', 'last'];
const REFERENCE_LABELS: Record<ReferenceTarget, string> = { image: 'Start image', reference: 'Reference', video: 'First frame', last: 'Last frame' };

/**
 * Build the "Use as" items: the start image, a reference, a video's first frame and its last frame. A target the model it
 * opens on can't use is disabled; its hint (or reason) shows beside the label and as the tooltip.
 */
export function referenceEntries(
  asset: GalleryAsset,
  onreference: NonNullable<AssetActionHandlers['onreference']>,
  useAs: UseAsState = {}
): ActionMenuEntry[] {
  return REFERENCE_TARGETS.map((target) => {
    const state = useAs[target];
    const disabled = state?.disabled === true;
    return {
      kind: 'item' as const,
      id: `reference-${target === 'last' ? 'last-frame' : target}`,
      label: REFERENCE_LABELS[target],
      hint: state?.hint,
      disabled,
      title: disabled ? state?.hint : undefined,
      onselect: () => onreference(asset, target),
    };
  });
}
