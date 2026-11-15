<script lang="ts">
  // The files a job starts from, one line per role: an icon, a secondary label and the file's name.
  import { Icon } from '$lib/components/atoms';
  import { inputGroups } from '$lib/state/jobInputs';
  import type { JobInput } from '$lib/types';

  interface Props {
    inputs: readonly JobInput[];
  }

  let { inputs }: Props = $props();

  const groups = $derived(inputGroups(inputs));
</script>

{#each groups as group (group.key)}
  <p class="job-input" data-testid="job-input" data-role={group.key}>
    <Icon name={group.kind === 'video' ? 'film' : 'file'} size={12} />
    <span class="label">{group.label}</span>
    <span class="name" title={group.names.join(', ')}>{group.names.join(', ')}</span>
  </p>
{/each}

<style>
  .job-input { display: flex; align-items: center; gap: 6px; min-width: 0; margin-top: 4px; font-size: var(--text-meta); line-height: 15px; color: var(--color-text-muted); }
  .job-input :global(svg) { flex: none; }
  .label { flex: none; color: var(--color-text-secondary); }
  .name { min-width: 0; overflow: hidden; font-family: var(--font-mono); white-space: nowrap; text-overflow: ellipsis; }
</style>
