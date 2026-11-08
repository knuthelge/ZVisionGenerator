<script lang="ts">
  import { layout } from '$lib/state/layoutMode.svelte';
  import ProjectMenu from './ProjectMenu.svelte';

  interface Props {
    open: boolean;
    anchor: HTMLElement | null;
    align?: 'start' | 'end';
    /** How many files move: the label says so for more than one. */
    count?: number;
    /** The files' shared location: a project's name, null for none, undefined for a mix. It is checked and disabled. */
    current: string | null | undefined;
    /** A project was chosen or created: its name, or null for No project. */
    onmove: (project: string | null) => void;
    onclose: () => void;
  }

  let { open, anchor, align = 'start', count = 1, current, onmove, onclose }: Props = $props();

  const title = $derived(count > 1 ? `Move ${count} files to` : 'Move to');
</script>

<!-- A phone's sheet names itself, so the heading is for the popover only. -->
<ProjectMenu
  {open}
  {anchor}
  {align}
  label="{title} project"
  heading={layout.compact ? undefined : title}
  noneLabel="No project"
  createLabel="Create and move"
  {current}
  disableCurrent
  onchoose={onmove}
  {onclose}
/>
