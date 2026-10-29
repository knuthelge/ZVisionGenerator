<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { SectionFilter } from './discoverView';

  interface Props {
    /** What the search is over, as "Filter 85 curated LoRAs" says. */
    placeholder: string;
    label: string;
    query: string;
    onquery: (query: string) => void;
    filter: SectionFilter;
    onfilter: (filter: SectionFilter) => void;
  }

  let { placeholder, label, query, onquery, filter, onfilter }: Props = $props();

  const FILTERS: { value: SectionFilter; label: string }[] = [
    { value: 'all', label: 'All' },
    { value: 'style', label: 'Styles' },
    { value: 'tool', label: 'Tools' },
  ];
</script>

<!-- Search is always on top, with the kind filter beside it; both stay in view while the list scrolls. -->
<div class="disc-tools">
  <label class="disc-filter">
    <Icon name="search" size={14} />
    <input class="ui-field" type="search" {placeholder} aria-label={label} value={query} oninput={(e) => onquery(e.currentTarget.value)}>
  </label>
  <div class="ui-segmented ui-segmented-sm" role="group" aria-label="Show">
    {#each FILTERS as option (option.value)}
      <button type="button" aria-pressed={filter === option.value} onclick={() => onfilter(option.value)}>{option.label}</button>
    {/each}
  </div>
</div>
