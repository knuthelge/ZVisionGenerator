<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { projectPick } from '$lib/state/projectPick.svelte';
  import { projectsStore } from '$lib/state/projects.svelte';
  import ProjectMenu from './ProjectMenu.svelte';

  interface Props {
    /** `nav` is the top bar's button; `drawer` is the navigation drawer's row. */
    variant: 'nav' | 'drawer';
    /** Cap the name's width to the room the Workspace's workflow tabs leave (`nav` only). */
    tight?: boolean;
    /** A project was picked, or created and picked. */
    onpicked?: () => void;
  }

  let { variant, tight = false, onpicked }: Props = $props();

  let open = $state(false);
  let anchor = $state<HTMLButtonElement | null>(null);

  const pick = $derived(projectPick.current);
  const name = $derived(pick ?? 'All files');

  function choose(next: string | null): void {
    projectsStore.choose(next);
    onpicked?.();
  }
</script>

{#if variant === 'drawer'}
  <button
    type="button"
    bind:this={anchor}
    class="drawer-project"
    data-picked={pick !== null ? '' : undefined}
    aria-haspopup="menu"
    aria-expanded={open}
    aria-label="Project: {name}"
    onclick={() => { open = !open; }}
  >
    <Icon name="folder" size={18} />
    <span class="drawer-project-text">
      <span class="ui-area-label">Project</span>
      <span class="drawer-project-name">{name}</span>
    </span>
    <Icon name="chevdown" size={14} class="opacity-70" />
  </button>
{:else}
  <button
    type="button"
    bind:this={anchor}
    class="ui-btn project-trigger"
    class:ui-btn-on={pick !== null}
    class:tight
    aria-haspopup="menu"
    aria-expanded={open}
    aria-label="Project: {name}"
    title={pick ?? undefined}
    onclick={() => { open = !open; }}
  >
    <Icon name="folder" size={14} />
    <span class="project-name">{name}</span>
    <Icon name="chevdown" size={12} class="opacity-70" />
  </button>
{/if}

<ProjectMenu
  {open}
  {anchor}
  label="Project"
  align="start"
  noneLabel="All files"
  current={pick}
  onchoose={choose}
  onclose={() => { open = false; }}
/>

<style>
  .project-trigger { align-self: center; max-width: 200px; }
  .project-trigger .project-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  @media (hover: hover) { .project-trigger.ui-btn-on:hover:not(:disabled) { background: color-mix(in srgb, var(--color-primary-main) 24%, transparent); } }
  @media (max-width: 1000px) {
    /* `tight` is the Workspace only, where the workflow tabs sit beside it; other pages keep the 200px cap. */
    .project-trigger.tight { max-width: clamp(110px, calc(100vw - 630px), 200px); }
    .project-trigger > :global(svg:first-child) { display: none; }
  }
  @media (max-width: 739px) {
    .project-trigger { margin-left: 8px; margin-right: auto; }
    .project-trigger.tight { max-width: 160px; }
  }

  .drawer-project { display: flex; width: 100%; align-items: center; gap: 12px; min-height: 56px; margin-bottom: 12px; padding: 8px 12px; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); color: var(--color-text-primary); text-align: left; }
  .drawer-project:active { background: var(--color-bg-control-hover); }
  .drawer-project > :global(svg:first-child) { flex: none; color: var(--color-text-muted); }
  .drawer-project[data-picked] > :global(svg:first-child) { color: var(--color-primary-main); }
  .drawer-project-text { display: flex; flex: 1; min-width: 0; flex-direction: column; gap: 2px; }
  .drawer-project-name { overflow: hidden; font-size: 15px; font-weight: 600; line-height: 20px; text-overflow: ellipsis; white-space: nowrap; }
</style>
