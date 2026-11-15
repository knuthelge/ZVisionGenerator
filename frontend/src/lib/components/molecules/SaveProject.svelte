<script lang="ts">
  import { Icon } from '$lib/components/atoms';

  interface Props {
    /** The project a job saves to. */
    name: string;
    testid?: string;
  }

  let { name, testid }: Props = $props();
</script>

<span class="save-project" title="Saves to {name}" data-testid={testid}><Icon name="folder" size={12} /><span class="sr-only">Saves to</span><span class="save-project-name">{name}</span></span>

<style>
  .save-project { display: inline-flex; align-items: center; gap: 4px; min-width: 0; max-width: 100%; font-size: var(--text-meta); line-height: 16px; color: var(--color-text-secondary); white-space: nowrap; }
  .save-project :global(svg) { flex: none; color: var(--color-text-muted); }
  .save-project-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
</style>
