import type { GalleryAsset } from '$lib/types';
import type { AssetPanel, PanelItem } from './assetMenu';

/** One action of the viewer's toolbar or dock: the same list at mouse size and at finger size, each wrapping an item of the file panel. */
export interface ViewerAct {
  /** The `data-action` of its button: the item's id, except for Use as (`use-as`) and Upscale (`upscale`, direct or with a list). */
  id: string;
  /** The panel item it runs and draws: its icon, label, warning, and whether it is off with a reason. A choice that opens a list wraps a stand-in. */
  item: PanelItem;
  /** The tooltip, with the action's key. */
  title: string;
  /** Opens a choice (Use as, Upscale with several sizes, Move): it takes the chevron and the open tint. */
  menu?: 'useas' | 'upscale' | 'move';
  /** The toolbar's label steps drop whole groups after Reuse: the rest of Make, then File. */
  group?: 'make' | 'file';
  /** The end of the label that gives way first, when the bar is short of room: `Reuse settings` reads `Reuse`. */
  tail?: string;
  /** Refused outright while a delete runs (Move, Delete): the button is disabled, where any other action that is off is greyed with its reason. */
  disabled?: boolean;
  /** The icon is filled (the dock's More). */
  solid?: boolean;
}

export interface ViewerActs {
  /** Reuse settings, then the kind's own action, Use as, and Upscale. Closes up: a kind without an action has no hole. */
  make: ViewerAct[];
  /** Download, Move and Copy prompt. */
  file: ViewerAct[];
  del: ViewerAct | null;
}

export interface ActContext {
  asset: GalleryAsset;
  panel: AssetPanel;
  /** What the X chord is followed by: `2 or 4`, or `2`. */
  upscaleKeys: string;
}

/** The key that runs each action in the viewer, for its tooltip. */
const KEYS: Record<string, string> = { reuse: 'R', edit: 'M', adjust: 'A', extend: 'E', 'use-audio-track': 'E', download: 'D', 'copy-prompt': 'C', delete: 'Del' };

/** The tooltip of an item: its reason while it is off, else its name and key. */
function titleOf(item: PanelItem): string {
  if (item.disabled && item.title) return item.title;
  const key = KEYS[item.id];
  return key ? `${item.label} (${key})` : item.label;
}

/**
 * Build the toolbar's and the dock's actions from the file panel, in its order: Make, then Use as and Upscale, then the file row.
 * Every surface lists the same actions, and each runs the panel item it wraps.
 *
 * A kind's own action is Edit (image), Extend (video) or Audio track (a take). Use as is for images and speech takes; a file that
 * offers exactly one upscale gets a direct `Upscale 2×` button, and several sizes one `Upscale` that opens them.
 */
export function viewerActs({ asset, panel, upscaleKeys }: ActContext): ViewerActs {
  const make: ViewerAct[] = panel.make.map((item) => ({
    id: item.id,
    item,
    title: titleOf(item),
    // Reuse leads and is dropped last; the rest of Make gives way together.
    group: item.primary ? undefined : 'make',
    tail: item.primary ? item.label.slice(item.label.indexOf(' ')) : undefined,
  }));

  if (panel.sections.some((section) => section.id === 'use-as')) {
    make.push({
      id: 'use-as', item: { kind: 'item', id: 'use-as', label: 'Use as', icon: 'use-as' }, menu: 'useas', group: 'make',
      title: asset.media_type === 'audio' ? 'Use as a voice' : 'Use as start image, reference or frame (E)',
    });
  }
  const upscale = panel.sections.find((section) => section.id === 'upscale');
  if (upscale) {
    const direct = upscale.items.length === 1 ? upscale.items[0] : null;
    make.push(direct
      ? {
          id: 'upscale', item: direct, group: 'make',
          // The warning and the first-use download size sit in the tooltip, above the key.
          title: direct.disabled && direct.title ? direct.title : [...new Set([direct.title, direct.hint, `${direct.label} (X then ${upscaleKeys})`].filter(Boolean))].join('\n'),
        }
      : { id: 'upscale', item: { kind: 'item', id: 'upscale', label: 'Upscale', icon: 'upscale' }, menu: 'upscale', group: 'make', title: `Upscale (X then ${upscaleKeys})` });
  }

  const file: ViewerAct[] = panel.file.filter((item) => item.id !== 'delete').map((item) => ({
    id: item.id,
    item,
    title: item.id === 'move' ? 'Move to project' : titleOf(item),
    group: 'file',
    menu: item.id === 'move' ? 'move' : undefined,
    disabled: item.id === 'move' ? item.disabled : undefined,
  }));

  const remove = panel.file.find((item) => item.id === 'delete');
  return { make, file, del: remove ? { id: 'delete', item: remove, title: 'Delete (Del)', disabled: remove.disabled } : null };
}

/** The phone's four slots: Reuse, the kind's own action, Download and More. Use as and Upscale would still cost two taps there, so More holds them. */
export function phoneActs(acts: ViewerActs): ViewerAct[] {
  const own = acts.make.find((act) => ['edit', 'extend', 'use-audio-track'].includes(act.id));
  const slots = [acts.make.find((act) => act.item.primary), own, acts.file.find((act) => act.id === 'download')];
  return [
    ...slots.flatMap((act) => (act ? [act] : [])),
    { id: 'more', item: { kind: 'item', id: 'more', label: 'More', icon: 'more-dots' }, title: 'More actions', solid: true },
  ];
}

/** Which of the viewer's controls a layout shows: a mouse the toolbar, a finger a dock, chosen by the pointer. */
export type ViewerChrome = 'toolbar' | 'dock' | 'phone' | 'rail';

/**
 * Choose the viewer's chrome. A mouse always gets the toolbar, so a window's size never moves an action to another place. A finger
 * gets the full dock, except where a menu is a sheet already (`compact`): the phone's four slots, as a rail on the right when the
 * phone is held sideways.
 */
export function chromeFor(layout: { coarse: boolean; compact: boolean; short: boolean }): ViewerChrome {
  if (!layout.coarse) return 'toolbar';
  if (!layout.compact) return 'dock';
  return layout.short ? 'rail' : 'phone';
}
