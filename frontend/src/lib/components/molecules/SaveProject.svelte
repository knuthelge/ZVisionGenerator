<script lang="ts">
  import { Icon } from '$lib/components/atoms';

  interface Props {
    /** The project a job saves to. */
    name: string;
    testid?: string;
  }

  let { name, testid }: Props = $props();
</script>

<div class="save-project" title="Saves to {name}" data-testid={testid}><Icon name="folder" size={12} /><span class="save-project-name">{name}</span></div>

<style>
  .save-project { display: flex; align-items: center; gap: 4px; min-width: 0; font-size: var(--text-meta); color: var(--color-text-muted); }
  .save-project :global(svg) { flex: none; }
  .save-project-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
