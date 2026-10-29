<script lang="ts">
  import type { CuratedEntry, LoraInfo } from '$lib/types';
  import { Icon } from '$lib/components/atoms';
  import { filterInstalled, installedTile, loraStatus, type SectionFilter, type UseFacts } from './discoverView';
  import ListTools from './ListTools.svelte';
  import LoraTile from './LoraTile.svelte';

  interface Props {
    /** The files that fit the model, and the files whose model nobody could tell. */
    main: LoraInfo[];
    other: LoraInfo[];
    /** The curated entries of the model, to describe the files that are one. */
    entries: CuratedEntry[];
    facts: UseFacts;
    modelName: string;
    query: string;
    onquery: (query: string) => void;
    filter: SectionFilter;
    onfilter: (filter: SectionFilter) => void;
    onopen: (name: string, trigger: HTMLElement) => void;
    /** Go to the Curated tab. */
    oncurated: () => void;
  }

  let { main, other, entries, facts, modelName, query, onquery, filter, onfilter, onopen, oncurated }: Props = $props();

  let othersOpen = $state(false);
  const shown = $derived(filterInstalled(main, entries, query, filter));
  const shownOther = $derived(filterInstalled(other, entries, query, filter));
  const tile = (lora: LoraInfo) => installedTile(lora, entries, loraStatus(lora.name, null, facts));
</script>

{#if main.length === 0}
  <div class="disc-none">
    <p>No LoRAs for {modelName} yet.</p>
    <button type="button" class="ui-btn" onclick={oncurated}>Curated</button>
  </div>
{:else}
  <ListTools placeholder="Filter {main.length} installed LoRAs" label="Filter installed LoRAs" {query} {onquery} {filter} {onfilter} />
  {#if shown.length === 0}
    <div class="disc-none"><p>Nothing installed matches “{query.trim()}”.</p></div>
  {:else}
    <div class="disc-grid">
      {#each shown as lora (lora.name)}<LoraTile tile={tile(lora)} hideInstalled {onopen} />{/each}
    </div>
  {/if}
{/if}
{#if other.length > 0}
  <button type="button" class="disc-others" aria-expanded={othersOpen} onclick={() => { othersOpen = !othersOpen; }}>
    <Icon name={othersOpen ? 'chevdown' : 'chevright'} size={14} />Other models ({other.length})
  </button>
  {#if othersOpen}
    <div class="disc-grid">
      {#each shownOther as lora (lora.name)}<LoraTile tile={tile(lora)} hideInstalled {onopen} />{/each}
    </div>
  {/if}
{/if}
