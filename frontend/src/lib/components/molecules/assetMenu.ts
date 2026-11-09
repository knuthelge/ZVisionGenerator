import {
  AUDIO_TRACK_HINT, SAVE_VOICE_HINT, VOICE_HINT, canExtend, canUseAsAudioTrack, canUseAsReference, canUseAsVoice, extendReason, plainUpscaleMethod, upscaleFactors, voiceTooLongReason,
  type AssetActionHandlers, type ReferenceTarget, type UseAsState,
} from '$lib/state/assetActions';
import type { AssetUpscaleFactor, GalleryAsset } from '$lib/types';
import type { ActionMenuEntry } from './ActionMenu.svelte';

/**
 * Build the actions for an asset's menu: reuse, use as, extend, upscale, move, download, delete.
 *
 * `openMove` opens the Move to project list; the Move item is offered when it is given.
 *
 * Reuse leads when offered: on a touch screen a grid tile shows only the menu button, so the menu is where the
 * tile's quick action lives.
 */
export function assetMenuEntries(
  asset: GalleryAsset,
  handlers: AssetActionHandlers,
  deleting = false,
  useAs: UseAsState = {},
  openMove?: () => void
): ActionMenuEntry[] {
  const entries: ActionMenuEntry[] = [];
  const { onreuse, onreference, onextend, onupscale, onuseaudio, ondelete } = handlers;
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
  if (onuseaudio && asset.media_type === 'audio') {
    const audioItems = audioUseEntries(asset, onuseaudio, useAs);
    if (audioItems.length > 0) entries.push({ kind: 'heading', label: 'Use as' }, ...audioItems, { kind: 'separator' });
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
  // Move opens the project list, which the menu's owner holds: this menu closes first, so it cannot host the list.
  if (openMove) entries.push({ kind: 'item', id: 'move', label: 'Move to project…', disabled: deleting, onselect: openMove });
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
 * Build the upscale items: the plain ones where offered (SeedVR2, or LTX for a video on macOS: `2× → 1664×2432`), a
 * separator, then the creative ones (`2× creative → 1664×2432`). A plain item is never disabled: over the memory estimate
 * it is marked and explained in its tooltip. A creative factor over a limit is disabled with its reason.
 */
export function upscaleEntries(asset: GalleryAsset, onupscale: NonNullable<AssetActionHandlers['onupscale']>): ActionMenuEntry[] {
  const method = plainUpscaleMethod(asset);
  const block = method === 'seedvr2' ? asset.upscale?.seedvr2 : method === 'ltx' ? asset.upscale?.ltx : null;
  const plain = block
    ? block.options.map((option) => ({
        kind: 'item' as const,
        id: `upscale-${method}-${option.factor}`,
        label: `${sizeLabel(option, '')}${warningMarker(option)}`,
        title: plainTitle(option, block.download_size_label),
        onselect: () => onupscale(asset, option.factor, method),
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

/** The label suffix for a warned plain option: what the warning is about, from its code; none without a warning. */
function warningMarker(option: AssetUpscaleFactor): string {
  if (!option.warning) return '';
  return option.warning_code === 'one_frame_batch' ? ONE_FRAME_MARKER : HIGH_MEMORY_MARKER;
}

function sizeLabel(option: AssetUpscaleFactor, suffix: string): string {
  return `${option.factor}×${suffix} → ${option.width}×${option.height}`;
}

/** The tooltip of a plain item: its memory warning, then the first-use download size while the model is not on disk. */
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

/**
 * Build the "Use as" items of an audio take: the Video tab's audio track, and for a speech take the Speech tab's voice and a
 * saved voice. The hint shows beside the label of an enabled item. A voice item the machine can't take is greyed with no hint:
 * its reason is the tooltip, and a touch shows it as a toast. A speech take longer than the speech model's voice limit is
 * greyed with "Use a take of <limit> s or less as a voice".
 */
export function audioUseEntries(asset: GalleryAsset, onuseaudio: NonNullable<AssetActionHandlers['onuseaudio']>, useAs: UseAsState = {}): ActionMenuEntry[] {
  const entries: ActionMenuEntry[] = [];
  if (canUseAsAudioTrack(asset)) {
    const state = useAs.audio_track;
    const disabled = state?.disabled === true;
    const hint = disabled ? state?.hint : (state?.hint ?? AUDIO_TRACK_HINT);
    entries.push({ kind: 'item', id: 'use-audio-track', label: 'Audio track', hint, disabled, title: disabled ? hint : undefined, onselect: () => onuseaudio(asset, 'audio_track') });
  }
  if (canUseAsVoice(asset)) {
    const tooLong = voiceTooLongReason(asset, useAs);
    for (const [target, id, label, fallback] of [['voice', 'use-voice', 'Voice', VOICE_HINT], ['save_voice', 'use-save-voice', 'Saved voice', SAVE_VOICE_HINT]] as const) {
      const state = useAs[target];
      const reason = tooLong ?? (state?.disabled === true ? state.hint : undefined);
      entries.push({ kind: 'item', id, label, hint: reason === undefined ? (state?.hint ?? fallback) : undefined, disabled: reason !== undefined, title: reason, onselect: () => onuseaudio(asset, target) });
    }
  }
  return entries;
}
