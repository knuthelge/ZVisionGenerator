<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { CuratedEntry } from '$lib/types';
  import { curatedTile, filterCurated, loraStatus, type SectionFilter, type UseFacts } from './discoverView';
  import ListTools from './ListTools.svelte';
  import LoraTile from './LoraTile.svelte';

  interface Props {
    entries: CuratedEntry[];
    facts: UseFacts;
    /** The search text and the kind filter; the drawer keeps them so they survive a visit to a detail. */
    query: string;
    onquery: (query: string) => void;
    filter: SectionFilter;
    onfilter: (filter: SectionFilter) => void;
    onopen: (id: string, trigger: HTMLElement) => void;
    /** Go to the Hugging Face tab, with these words in its search ('' for none). */
    onsearchhf: (words: string) => void;
  }

  let { entries, facts, query, onquery, filter, onfilter, onopen, onsearchhf }: Props = $props();

  const shown = $derived(filterCurated(entries, query, filter));
  const words = $derived(query.trim().replace(/\s+/g, ' '));
</script>

<ListTools placeholder="Filter {entries.length} curated LoRAs" label="Filter curated LoRAs" {query} {onquery} {filter} {onfilter} />

{#if shown.length === 0}
  <div class="disc-none">
    {#if words}<p>Nothing curated matches “{words}”.</p>{/if}
    <button type="button" class="ui-btn" onclick={() => onsearchhf(words)}><Icon name="search" size={14} />Search Hugging Face{words ? ` for “${words}”` : ''}</button>
  </div>
{:else}
  <div class="disc-grid">
    {#each shown as entry (entry.id)}<LoraTile tile={curatedTile(entry, loraStatus(entry.name, entry.state, facts))} {onopen} />{/each}
  </div>
  <button type="button" class="disc-more" onclick={() => onsearchhf('')}>More on Hugging Face<Icon name="chevright" size={14} /></button>
{/if}
