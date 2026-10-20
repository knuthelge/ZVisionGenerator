<script lang="ts">
  import Sheet from './Sheet.svelte';

  interface Props {
    open?: boolean;
    keepMounted?: boolean;
    onclose?: () => void;
  }

  let { open = $bindable(false), keepMounted = false, onclose }: Props = $props();
</script>

<button type="button" data-testid="open" onclick={() => { open = true; }}>Open</button>
<Sheet bind:open title="All settings" {keepMounted} {onclose}>
  <input name="steps" value="32">
</Sheet>
