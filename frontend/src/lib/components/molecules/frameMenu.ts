import { canExtend, canUseFrame, extendLabel, extendNote, extendPointReason, extendReason, type AssetActionHandlers, type FrameTarget, type UseAsState } from '$lib/state/assetActions';
import { busyFor } from '$lib/state/frameGrab.svelte';
import type { FramePoint } from '$lib/state/framePoint';
import { videoFacts } from '$lib/state/videoFacts.svelte';
import type { GalleryAsset } from '$lib/types';
import type { ActionMenuItem } from './ActionMenu.svelte';

/** What the row of the frame that is being extracted reads. */
export const FRAME_BUSY_LABEL = 'Extracting the frame…';

const FRAME_ROWS: ReadonlyArray<{ target: FrameTarget; id: string; label: string; note: string; icon: 'frame-first' | 'frame-last' }> = [
  { target: 'video', id: 'frame-first', label: 'First frame', note: 'Start a new clip on this frame', icon: 'frame-first' },
  { target: 'last', id: 'frame-last', label: 'Last frame', note: 'End a new clip on this frame', icon: 'frame-last' },
];

/**
 * Build the Extend row of a video's menu, or null when the video does not offer extending or there is no handler.
 *
 * With a *point* it reads `Extend from 3.2 s` before the end and `Extend the clip` at it, and carries a note of what it keeps;
 * without one it is plain `Extend`. A clip that can't be extended, or a frame too early, keeps the row disabled with its reason.
 */
export function extendEntry(asset: GalleryAsset, point: FramePoint | null, onextend: AssetActionHandlers['onextend']): ActionMenuItem | null {
  if (!onextend || asset.media_type !== 'video' || !asset.extend) return null;
  const available = canExtend(asset);
  const reason = extendReason(asset) ?? extendPointReason(point) ?? undefined;
  const clipSeconds = asset.file_path ? videoFacts.lengthOf(asset.file_path) : null;
  return {
    kind: 'item',
    id: 'extend',
    label: available ? extendLabel(point) : 'Extend',
    icon: 'extend',
    description: point && !reason ? extendNote(point, clipSeconds) : undefined,
    hint: reason,
    disabled: !available || point?.tooShort === true,
    onselect: () => (point ? onextend(asset, point) : onextend(asset)),
  };
}

/**
 * Build the First frame and Last frame rows of a paused frame.
 *
 * A target the model can't use keeps its row, disabled with the model's hint. While a frame is being extracted for the
 * clip, every row is disabled and only the running one reads as busy.
 */
export function frameEntries(asset: GalleryAsset, point: FramePoint, onuseframe: NonNullable<AssetActionHandlers['onuseframe']>, useAs: UseAsState = {}): ActionMenuItem[] {
  const running = busyFor(asset.id);
  return FRAME_ROWS.map(({ target, id, label, note, icon }) => {
    const state = useAs[target];
    const reason = state?.disabled === true ? (state.hint ?? `${label} is not available`) : undefined;
    const busy = running === target;
    return {
      kind: 'item' as const,
      id,
      label: busy ? FRAME_BUSY_LABEL : label,
      icon,
      description: reason ? undefined : note,
      hint: reason,
      disabled: reason !== undefined || running !== null,
      busy,
      onselect: () => onuseframe(asset, point, target),
    };
  });
}

/**
 * Build the rows of the menu a paused frame's time chip opens: Extend, then the first frame and the last frame.
 *
 * A handler that is absent leaves its rows out; a video not on the host has no frame rows.
 */
export function frameMenuEntries(asset: GalleryAsset, point: FramePoint, handlers: Pick<AssetActionHandlers, 'onextend' | 'onuseframe'>, useAs: UseAsState = {}): ActionMenuItem[] {
  const { onextend, onuseframe } = handlers;
  const extend = extendEntry(asset, point, onextend);
  return [...(extend ? [extend] : []), ...(onuseframe && canUseFrame(asset, point) ? frameEntries(asset, point, onuseframe, useAs) : [])];
}
