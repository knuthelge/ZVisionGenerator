<script lang="ts">
  import { Icon } from '$lib/components/atoms';

  interface Props {
    /** The words beside the sparkle ("From your idea"); without them the mark is the sparkle alone, named for assistive technology. */
    label?: string;
  }

  let { label }: Props = $props();

  const TITLE = 'Written for you from Describe it. Edit it and it becomes yours; Fill in again rewrites it while it is marked.';
</script>

<!-- The ✦ mark of a field Fill in wrote. -->
<span class="filled-mark" class:with-label={label !== undefined} data-testid="filled-mark" title={TITLE} role={label === undefined ? 'img' : undefined} aria-label={label === undefined ? 'Filled from your idea' : undefined}>
  <Icon name="sparkle" size={11} />{#if label}<span>{label}</span>{/if}
</span>

<style>
  .filled-mark { display: inline-flex; flex: none; align-items: center; gap: 4px; color: var(--color-primary-main); }
  .filled-mark.with-label { height: var(--spacing-badge); padding: 0 6px; border-radius: var(--radius-xs); background: color-mix(in srgb, var(--color-primary-main) 14%, transparent); font-size: var(--text-meta); font-weight: 600; white-space: nowrap; }
</style>
