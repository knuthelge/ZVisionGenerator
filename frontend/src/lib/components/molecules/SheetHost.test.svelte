<script lang="ts">
  import Sheet from './Sheet.svelte';

  interface Props {
    open?: boolean;
    keepMounted?: boolean;
    /** Give the sheet its own head content, in place of the title. */
    ownHead?: boolean;
    wide?: boolean;
    onclose?: () => void;
  }

  let { open = $bindable(false), keepMounted = false, ownHead = false, wide = false, onclose }: Props = $props();
</script>

{#snippet head()}<span data-testid="own-head">A file</span>{/snippet}

<button type="button" data-testid="open" onclick={() => { open = true; }}>Open</button>
<Sheet bind:open title="All settings" {keepMounted} {onclose} head={ownHead ? head : undefined} {wide}>
  <input name="steps" value="32">
</Sheet>
