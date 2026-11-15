import {
  AUDIO_TRACK_HINT, SAVE_VOICE_HINT, VOICE_HINT, canEdit, canUseAsAudioTrack, canUseAsReference, canUseAsVoice, canUseFrame, plainUpscaleMethod, upscaleFactors, voiceTooLongReason,
  type AssetActionHandlers, type ReferenceTarget, type UseAsState,
} from '$lib/state/assetActions';
import { canAdjust } from '$lib/state/adjustState';
import { kindFact } from '$lib/state/audioAsset';
import { formatClock } from '$lib/state/clock';
import { copyPrompt, copyableText } from '$lib/state/copyPrompt';
import { pointLabel, type FramePoint } from '$lib/state/framePoint';
import type { AssetUpscaleFactor, GalleryAsset } from '$lib/types';
import type { ActionMenuItem } from './ActionMenu.svelte';
import { formatSeconds } from './assetDetails';
import { extendEntry, frameEntries } from './frameMenu';

/** One action of the file panel: a choice (two lines, its name and what it makes) or a file cell (an icon over its label). */
export interface PanelItem extends ActionMenuItem {
  /** The teal primary: Reuse settings. */
  primary?: boolean;
  /** The only choice of its section: it spans the whole row. */
  span?: boolean;
  /** A warning on the name, in amber: `high memory`, `very slow`. */
  warn?: string;
  /** The accessible name, where the label and hint read together are not enough. */
  name?: string;
}

/** A titled group of choices: Use as, Use frame at 3.2 s as, Upscale. */
export interface PanelSection {
  id: 'use-as' | 'frames' | 'upscale';
  label: string;
  items: PanelItem[];
}

/** What the file panel shows for one file, in the order it shows it. */
export interface AssetPanel {
  /** The head's controls: View where a click on the picture doesn't open the viewer, and the star. */
  head: { view: PanelItem | null; star: PanelItem | null };
  /** Reuse settings, then the kind's own way to make something new: Edit, Extend or Audio track. */
  make: PanelItem[];
  sections: PanelSection[];
  /** Download, Move, Copy prompt, then Delete. */
  file: PanelItem[];
}

export interface PanelOptions {
  deleting?: boolean;
  /** What the Use as choices show: which targets are disabled, and their reasons. */
  useAs?: UseAsState;
  /** Opens the Move to project list; Move is offered when it is given. */
  openMove?: () => void;
  /**
   * The frame a paused video is on: Extend then reads `Extend from 3.2 s` (or is refused with the reason when the frame is
   * too early), and the frame's own choices, First frame and Last frame, join the panel. No point, no frame choices.
   */
  point?: FramePoint | null;
  /** Opens the viewer; View is in the head when it is given. */
  onview?: () => void;
}

const REUSE_HINT = 'Load its settings';
const REUSE_UNAVAILABLE = 'Reusable settings unavailable';
const EDIT_HINT = 'Paint over part of it';
const ADJUST_HINT = 'Crop, light and colour';
const NO_PROMPT = 'No prompt recorded';

/**
 * Build the file panel of an asset: the head's controls, Make (Reuse settings and the kind's own action), the choices
 * (Use as, the paused frame, Upscale) and the file row (Download, Move, Copy prompt, Delete).
 *
 * An action whose handler is absent is left out. A choice the model it opens on can't take stays, greyed, with its reason as
 * its hint; the reason is also its tooltip, which a touch shows as a toast. Move is offered when `openMove` is given.
 */
export function assetPanel(asset: GalleryAsset, handlers: AssetActionHandlers, options: PanelOptions = {}): AssetPanel {
  const { onreuse, onreference, onextend, onedit, onadjust, onuseframe, onupscale, onuseaudio, ondelete, onstar } = handlers;
  const { deleting = false, useAs = {}, openMove, point = null, onview } = options;

  const make: PanelItem[] = [];
  if (onreuse) {
    const reusable = asset.has_reusable_config === true;
    make.push({
      kind: 'item', id: 'reuse', label: 'Reuse settings', icon: 'reuse', primary: true, hint: reusable ? REUSE_HINT : REUSE_UNAVAILABLE, disabled: !reusable,
      title: reusable ? undefined : REUSE_UNAVAILABLE, onselect: () => onreuse(asset),
    });
  }
  const extend = extendEntry(asset, point, onextend);
  if (extend) make.push(extend);
  if (onedit && canEdit(asset)) make.push({ kind: 'item', id: 'edit', label: 'Edit', icon: 'edit', hint: EDIT_HINT, onselect: () => onedit(asset) });
  // An image can also be adjusted by hand: cropped, rotated, flipped, or changed in light and colour.
  if (onadjust && canAdjust(asset)) make.push({ kind: 'item', id: 'adjust', label: 'Adjust', icon: 'sliders', hint: ADJUST_HINT, onselect: () => onadjust(asset) });
  if (onuseaudio && asset.media_type === 'audio') {
    const track = audioTrackItem(asset, onuseaudio, useAs);
    if (track) make.push(track);
  }

  const sections: PanelSection[] = [];
  const useAsItems = [
    ...(onreference && canUseAsReference(asset) ? referenceItems(asset, onreference, useAs) : []),
    ...(onuseaudio && asset.media_type === 'audio' ? voiceItems(asset, onuseaudio, useAs) : []),
  ];
  if (useAsItems.length > 0) sections.push({ id: 'use-as', label: 'Use as', items: useAsItems });
  if (onuseframe && point && canUseFrame(asset, point)) {
    sections.push({ id: 'frames', label: `Use frame at ${pointLabel(point)} as`, items: frameEntries(asset, point, onuseframe, useAs) });
  }
  const upscale = onupscale ? upscaleItems(asset, onupscale) : [];
  if (upscale.length > 0) sections.push({ id: 'upscale', label: 'Upscale', items: upscale });

  const starred = asset.starred === true;
  const text = copyableText(asset);
  const file: PanelItem[] = [{ kind: 'item', id: 'download', label: 'Download', icon: 'download', href: asset.url, download: asset.filename }];
  if (openMove) file.push({ kind: 'item', id: 'move', label: 'Move', icon: 'folder-move', disabled: deleting, onselect: openMove });
  file.push({ kind: 'item', id: 'copy-prompt', label: 'Copy prompt', icon: 'copy', disabled: text === '', title: text === '' ? NO_PROMPT : undefined, onselect: () => void copyPrompt(text) });
  if (ondelete) file.push({ kind: 'item', id: 'delete', label: deleting ? 'Deleting…' : 'Delete', icon: 'trash', danger: true, disabled: deleting, onselect: () => ondelete(asset) });

  return {
    head: {
      view: onview ? { kind: 'item', id: 'view', label: 'View', icon: 'expand', onselect: onview } : null,
      star: onstar ? { kind: 'item', id: 'star', label: starred ? 'Unstar' : 'Star', icon: 'star', checked: starred, onselect: () => onstar(asset, !starred) } : null,
    },
    make,
    sections,
    file,
  };
}

/** Describe a file in the panel's head: its kind, its size or length. `Image · 1216 × 832`, `Video · 768 × 448 · 5.0 s`, `Speech · 0:06`. */
export function assetKindLine(asset: GalleryAsset): string {
  const size = asset.width && asset.height ? `${asset.width} × ${asset.height}` : null;
  if (asset.media_type === 'audio') {
    return [kindFact(asset), asset.duration != null ? formatClock(asset.duration) : null].filter(Boolean).join(' · ');
  }
  if (asset.media_type === 'video') return ['Video', size, asset.duration != null ? formatSeconds(asset.duration) : null].filter(Boolean).join(' · ');
  return ['Image', size].filter(Boolean).join(' · ');
}

const sentence = (text: string) => text.charAt(0).toLowerCase() + text.slice(1);

const REFERENCE_TARGETS: readonly ReferenceTarget[] = ['image', 'reference', 'video', 'last'];
const REFERENCE_CHOICES: Record<ReferenceTarget, { label: string; icon: NonNullable<PanelItem['icon']>; hint: string }> = {
  image: { label: 'Start image', icon: 'image', hint: 'A new image from it' },
  reference: { label: 'Reference', icon: 'image-plus', hint: 'Guide a new image' },
  video: { label: 'First frame', icon: 'frame-first', hint: 'Start a new clip on it' },
  last: { label: 'Last frame', icon: 'frame-last', hint: 'End a new clip on it' },
};

/**
 * Build the Use as choices of an image: the start image, a reference, a video's first frame and its last frame. A target the
 * model it opens on can't use is greyed, and its reason is its hint. A reference adds which slot it fills (`as image 1`).
 */
export function referenceItems(asset: GalleryAsset, onreference: NonNullable<AssetActionHandlers['onreference']>, useAs: UseAsState = {}): PanelItem[] {
  return REFERENCE_TARGETS.map((target) => {
    const state = useAs[target];
    const disabled = state?.disabled === true;
    const { label, icon, hint } = REFERENCE_CHOICES[target];
    const shown = disabled ? state?.hint : state?.hint ? `${hint} · ${state.hint}` : hint;
    return {
      kind: 'item' as const,
      id: `reference-${target === 'last' ? 'last-frame' : target}`,
      label, icon, hint: shown, disabled,
      title: disabled ? state?.hint : undefined,
      // A built-in hint reads as part of the sentence; a reason comes from the model and is passed through as it is.
      name: `Use as ${sentence(label)}, ${disabled ? (shown ?? hint) : sentence(shown ?? hint)}`,
      onselect: () => onreference(asset, target),
    };
  });
}

/**
 * Build the Audio track choice of an audio take: the Video tab's audio track, the take's way to make something new. It is
 * greyed, with its reason, where the Video model can't follow a track; null for a take that is not a file on the host.
 */
function audioTrackItem(asset: GalleryAsset, onuseaudio: NonNullable<AssetActionHandlers['onuseaudio']>, useAs: UseAsState = {}): PanelItem | null {
  if (!canUseAsAudioTrack(asset)) return null;
  const state = useAs.audio_track;
  const disabled = state?.disabled === true;
  const hint = disabled ? state?.hint : AUDIO_TRACK_HINT;
  return { kind: 'item', id: 'use-audio-track', label: 'Audio track', icon: 'waveform', hint, disabled, title: disabled ? hint : undefined, onselect: () => onuseaudio(asset, 'audio_track') };
}

/**
 * Build the Use as choices of a speech take: Voice and Saved voice. Music and sound have none. A voice the machine can't take
 * is greyed with its reason; a take longer than the speech model's voice limit says `Use a take of <limit> s or less as a voice`.
 */
export function voiceItems(asset: GalleryAsset, onuseaudio: NonNullable<AssetActionHandlers['onuseaudio']>, useAs: UseAsState = {}): PanelItem[] {
  if (!canUseAsVoice(asset)) return [];
  const tooLong = voiceTooLongReason(asset, useAs);
  const targets = [['voice', 'use-voice', 'Voice', 'speech', VOICE_HINT], ['save_voice', 'use-save-voice', 'Saved voice', 'bookmark', SAVE_VOICE_HINT]] as const;
  return targets.map(([target, id, label, icon, fallback]) => {
    const state = useAs[target];
    const reason = tooLong ?? (state?.disabled === true ? state.hint : undefined);
    return { kind: 'item' as const, id, label, icon, hint: reason ?? fallback, disabled: reason !== undefined, title: reason, onselect: () => onuseaudio(asset, target) };
  });
}

const HIGH_MEMORY_MARKER = 'high memory';
const ONE_FRAME_MARKER = 'very slow';

/** The marker on a warned plain option's name: what the warning is about, from its code; none without a warning. */
function warningMarker(option: AssetUpscaleFactor): string | undefined {
  if (!option.warning) return undefined;
  return option.warning_code === 'one_frame_batch' ? ONE_FRAME_MARKER : HIGH_MEMORY_MARKER;
}

const sizeHint = (option: AssetUpscaleFactor) => `${option.width} × ${option.height}`;

/**
 * Build the Upscale choices: the plain ones where offered (SeedVR2, or LTX for a video on macOS), named by the choice
 * (`2×`, `4×`) with the output size as the hint, then the creative ones (`2× creative`). A plain choice is never greyed: over
 * the memory estimate it is marked on its name and explained in its tooltip, and while the model is not on disk its hint says what
 * the first use downloads. A creative factor over a limit is greyed, with its reason as the hint.
 *
 * A file that offers one choice gets that choice alone, named in full (`Upscale 2×`), spanning its row.
 */
export function upscaleItems(asset: GalleryAsset, onupscale: NonNullable<AssetActionHandlers['onupscale']>): PanelItem[] {
  const method = plainUpscaleMethod(asset);
  const block = method === 'seedvr2' ? asset.upscale?.seedvr2 : method === 'ltx' ? asset.upscale?.ltx : null;
  const plain: PanelItem[] = block
    ? block.options.map((option) => {
        const warn = warningMarker(option);
        const download = block.download_size_label ? ` · first use downloads ${block.download_size_label}` : '';
        return {
          kind: 'item' as const,
          id: `upscale-${method}-${option.factor}`,
          label: `${option.factor}×`,
          icon: 'upscale' as const,
          hint: `${sizeHint(option)}${download}`,
          warn,
          title: option.warning ?? undefined,
          name: `Upscale ${option.factor}×, ${option.width} by ${option.height}${warn ? `, ${warn}` : ''}${download ? `, first use downloads ${block.download_size_label}` : ''}`,
          onselect: () => onupscale(asset, option.factor, method),
        };
      })
    : [];
  const creative: PanelItem[] = upscaleFactors(asset).map((option) => ({
    kind: 'item' as const,
    id: `upscale-creative-${option.factor}`,
    label: `${option.factor}× creative`,
    icon: 'upscale' as const,
    hint: option.allowed ? sizeHint(option) : (option.reason ?? sizeHint(option)),
    disabled: !option.allowed,
    title: option.allowed ? undefined : (option.reason ?? undefined),
    name: `Upscale ${option.factor}× creative, ${option.width} by ${option.height}`,
    onselect: () => onupscale(asset, option.factor, 'creative'),
  }));
  const items = [...plain, ...creative];
  return items.length === 1 ? [{ ...items[0], label: `Upscale ${items[0].label}`, span: true }] : items;
}
