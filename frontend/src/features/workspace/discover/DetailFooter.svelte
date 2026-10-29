<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import Toggle from '$lib/components/atoms/Toggle.svelte';
  import type { FooterView } from './discoverView';

  interface Props {
    view: FooterView;
    /** Why the last request failed, shown above the buttons. */
    error?: string | null;
    onaction: () => void;
    /** The switch was turned on or off. */
    onuse: (on: boolean) => void;
    /** Changes when a switch request ended without the state the switch shows, so it is drawn again from `view`. */
    epoch?: number;
  }

  let { view, error = null, onaction, onuse, epoch = 0 }: Props = $props();
</script>

<div class="disc-foot">
  <div class="contents" aria-live="polite">
    {#if error}<p class="disc-why error" role="alert"><Icon name="alert" size={12} /><span>{error}</span></p>
    {:else if view.note}<p class="disc-why"><span>{view.note}</span></p>{/if}
  </div>
  {#if view.action.kind === 'delete'}
    <button type="button" class="ui-btn ui-btn-quiet disc-delete" disabled={view.action.disabled} onclick={onaction}>{view.action.label}</button>
  {:else}
    <button type="button" class="ui-btn" disabled={view.action.disabled} aria-busy={view.action.kind === 'downloading' || undefined} onclick={onaction}>{view.action.label}</button>
  {/if}
  <span class="disc-spacer"></span>
  {#key epoch}
    <Toggle label="Use in LoRAs" checked={view.use.on} disabled={view.use.disabled} onchange={(event) => onuse((event.currentTarget as HTMLInputElement).checked)} />
  {/key}
</div>
