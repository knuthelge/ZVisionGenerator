<script lang="ts">
  import type { GalleryAsset } from '$lib/types';
  import type { PopoverAlign, PopoverSide } from '$lib/actions/popover';
  import ActionMenu, { type MenuContext } from './ActionMenu.svelte';
  import FilePanel from './FilePanel.svelte';
  import PanelControls from './PanelControls.svelte';
  import PanelFile from './PanelFile.svelte';
  import { assetKindLine, type AssetPanel, type PanelSection } from './assetMenu';

  interface Props {
    open: boolean;
    anchor: HTMLElement | null;
    asset: GalleryAsset;
    /** The file's panel, built once by the surface (`assetPanel`): its actions, its View in the head where it has one, its Move. */
    panel: AssetPanel;
    /** What follows the kind and size in the head, such as the place in the set. */
    metaSuffix?: string;
    /**
     * Show only one section's choices, with no head and no file row: the viewer's Use as and Upscale buttons open the panel's
     * own choices in a popover.
     */
    only?: PanelSection['id'];
    align?: PopoverAlign;
    side?: PopoverSide;
    onclose: () => void;
  }

  let { open, anchor, asset, panel: full, metaSuffix = '', only, align = 'end', side = 'below', onclose }: Props = $props();

  const panel = $derived<AssetPanel>(only
    ? { head: { view: null, star: null }, make: [], sections: full.sections.filter((section) => section.id === only), file: [] }
    : full);
  const meta = $derived([assetKindLine(asset), metaSuffix].filter(Boolean).join(' · '));

</script>

{#snippet sheetHead()}<PanelFile {asset} {meta} />{/snippet}
{#snippet sheetControls(ctx: MenuContext)}<PanelControls view={panel.head.view} star={panel.head.star} select={ctx.select} />{/snippet}
{#snippet body(ctx: MenuContext)}
  <FilePanel {panel} {asset} {meta} {ctx} head={!only && (!ctx.sheet || ctx.wide)} />
{/snippet}

<ActionMenu
  {open}
  {anchor}
  label={only ? (panel.sections[0]?.label ?? asset.filename) : asset.filename}
  {align}
  {side}
  content={body}
  head={only ? undefined : sheetHead}
  headActions={only ? undefined : sheetControls}
  {onclose}
/>
