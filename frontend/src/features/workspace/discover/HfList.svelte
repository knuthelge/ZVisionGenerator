<script lang="ts">
  import { untrack } from 'svelte';
  import { Icon, Spinner } from '$lib/components/atoms';
  import { ActionMenu, Alert, Select, type ActionMenuEntry } from '$lib/components/molecules';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { errorText, isRetryable } from '$lib/api/client';
  import { getHfList, type DiscoverModel } from '$lib/api/discover';
  import { INPUT_DEBOUNCE_MS } from '$lib/state/debounce';
  import type { HfItem, HfSort } from '$lib/types';
  import HfRow from './HfRow.svelte';

  interface Props {
    model: DiscoverModel;
    /** The search text; the drawer keeps it, so Curated's search can hand its words over. */
    query: string;
    onquery: (query: string) => void;
    onopen: (item: HfItem, trigger: HTMLElement) => void;
    /** Turning the results off is being saved. */
    saving: boolean;
    /** Why turning the results off failed. */
    error: string | null;
    onturnoff: () => void;
  }

  let { model, query, onquery, onopen, saving, error: turnOffError, onturnoff }: Props = $props();

  const SORTS: { value: HfSort; label: string }[] = [
    { value: 'downloads', label: 'Most downloaded' },
    { value: 'likes', label: 'Most liked' },
    { value: 'trending', label: 'Trending' },
    { value: 'newest', label: 'Newest' },
  ];

  let sort = $state<HfSort>('downloads');
  let sortOpen = $state(false);
  let sortButton = $state<HTMLButtonElement | null>(null);
  const sortItems = $derived<ActionMenuEntry[]>(SORTS.map((option) => ({ kind: 'item', id: `sort-${option.value}`, label: option.label, checked: sort === option.value, onselect: () => { sort = option.value; } })));
  let items = $state<HfItem[]>([]);
  let cursor = $state<string | null>(null);
  let loading = $state(false);
  let loaded = $state(false);
  let error = $state<{ text: string; retry: boolean } | null>(null);
  // Only the newest request may fill the list, so a slow earlier answer cannot replace a later one.
  let latest = 0;

  /** The rows with *added* after them, keeping the first row of each repository so the keyed list never sees one twice. */
  function mergeItems(have: HfItem[], added: HfItem[]): HfItem[] {
    const seen = new Set(have.map((item) => item.repo));
    return [...have, ...added.filter((item) => !seen.has(item.repo) && seen.add(item.repo))];
  }

  async function load(more: boolean): Promise<void> {
    const request = ++latest;
    loading = true;
    error = null;
    try {
      const page = await getHfList(model, { q: query.trim(), sort, cursor: more ? cursor : null });
      if (request !== latest) return;
      items = mergeItems(more ? items : [], page.items);
      cursor = page.next_cursor;
      loaded = true;
    } catch (err) {
      if (request !== latest) return;
      error = { text: errorText(err, 'Could not reach Hugging Face.'), retry: isRetryable(err) };
    } finally {
      if (request === latest) loading = false;
    }
  }

  // The list follows the search and the sort: typing waits for a pause, a sort or a model change asks at once.
  let firstRun = true;
  $effect(() => {
    void query;
    void sort;
    void model.family;
    void model.size;
    void model.version;
    const wait = firstRun ? 0 : INPUT_DEBOUNCE_MS;
    firstRun = false;
    const timer = setTimeout(() => untrack(() => void load(false)), wait);
    return () => clearTimeout(timer);
  });
</script>

<div class="disc-search">
  <label class="disc-filter">
    <Icon name="search" size={14} />
    <input class="ui-field" type="search" placeholder="Search Hugging Face" aria-label="Search Hugging Face" value={query} oninput={(e) => onquery(e.currentTarget.value)}>
  </label>
  {#if layout.compact}
    <!-- On a phone the sort menu is an icon button. -->
    <button bind:this={sortButton} type="button" class="ui-btn ui-btn-icon" aria-label="Sort by" aria-haspopup="menu" aria-expanded={sortOpen} title={SORTS.find((option) => option.value === sort)?.label} onclick={() => { sortOpen = !sortOpen; }}><Icon name="sliders" size={16} /></button>
    <ActionMenu open={sortOpen} anchor={sortButton} items={sortItems} label="Sort by" onclose={() => { sortOpen = false; }} />
  {:else}
    <Select ariaLabel="Sort by" value={sort} options={SORTS} class="w-44 shrink-0" onchange={(value) => { sort = value as HfSort; }} />
  {/if}
</div>
<div class="disc-hfnote">
  <Icon name="info" size={13} />
  <span>Showing Hugging Face Community models</span>
  <button type="button" class="ui-btn ui-btn-quiet ui-btn-sm" disabled={saving} aria-busy={saving || undefined} onclick={onturnoff}>Turn off</button>
</div>
{#if turnOffError}<div style="padding-bottom: 8px"><Alert tone="error" live>{turnOffError}</Alert></div>{/if}

{#if error}
  <Alert tone="error" live>
    {error.text}
    {#if error.retry}<button type="button" class="ui-btn ui-btn-sm" style="margin-left: 8px" onclick={() => void load(false)}>Retry</button>{/if}
  </Alert>
{:else if !loaded}
  <p class="disc-loading" role="status"><Spinner size="sm" />Asking Hugging Face…</p>
{:else if items.length === 0}
  <div class="disc-none"><p>{query.trim() ? `Nothing on Hugging Face matches “${query.trim()}”.` : 'No LoRAs found.'}</p></div>
{:else}
  <div class="disc-list" aria-busy={loading || undefined}>
    {#each items as item (item.repo)}<HfRow {item} {onopen} />{/each}
  </div>
  {#if cursor}
    <div style="display: flex; justify-content: center; margin-top: 12px">
      <button type="button" class="ui-btn" disabled={loading} aria-busy={loading || undefined} onclick={() => void load(true)}>{loading ? 'Loading…' : 'Show more'}</button>
    </div>
  {/if}
{/if}
