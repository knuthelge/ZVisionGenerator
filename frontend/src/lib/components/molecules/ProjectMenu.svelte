<script lang="ts">
  import { untrack } from 'svelte';
  import { projectsStore } from '$lib/state/projects.svelte';
  import ActionMenu from './ActionMenu.svelte';
  import ProjectNameDialog from './ProjectNameDialog.svelte';
  import { projectMenuEntries } from './projectMenu';

  interface Props {
    open: boolean;
    anchor: HTMLElement | null;
    /** Names the list: the popover's `aria-label` and the phone sheet's title. */
    label: string;
    align?: 'start' | 'end';
    /** The first row: `All files` in the picker, `No project` in a Move list. */
    noneLabel: string;
    /** A heading above the rows (popover only is the caller's choice). */
    heading?: string;
    /** The main button of the New project dialog. */
    createLabel?: string;
    /** The checked row: a project's name, null for the none row, undefined for no check. */
    current: string | null | undefined;
    /** Disable the checked row, for a Move list where it is the file's own location. */
    disableCurrent?: boolean;
    /** A row was chosen, or a project was created: a project's name, or null for the none row. */
    onchoose: (name: string | null) => void;
    /** Both the list and the dialog have closed. */
    onclose: () => void;
  }

  let { open, anchor, label, align = 'start', noneLabel, heading, createLabel = 'Create', current, disableCurrent = false, onchoose, onclose }: Props = $props();

  let dialogOpen = $state(false);

  // The list is read again each time it opens, so projects made or removed elsewhere show up.
  $effect(() => {
    if (open) untrack(() => void projectsStore.load());
  });

  // The dialog closing ends the whole interaction, whether it created a project or was cancelled.
  function setDialogOpen(next: boolean): void {
    dialogOpen = next;
    if (!next) onclose();
  }

  const items = $derived(projectMenuEntries(projectsStore.list, current, {
    noneLabel,
    heading,
    disableCurrent,
    onchoose,
    onnew: () => { dialogOpen = true; },
  }));

  // Choosing `New project…` closes the list first; the dialog is what is left open, so the caller hears nothing yet.
  function closeList(): void {
    queueMicrotask(() => { if (!dialogOpen) onclose(); });
  }
</script>

<ActionMenu open={open && !dialogOpen} {anchor} {align} {label} {items} onclose={closeList} />
<ProjectNameDialog bind:open={() => dialogOpen, setDialogOpen} {createLabel} oncreated={onchoose} />
