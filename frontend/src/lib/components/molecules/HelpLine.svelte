<script lang="ts">
  interface Props {
    /** An error: shown in the error colour and announced at once. */
    error?: string | null;
    /** The last result's note. An error, when there is one, is shown instead. */
    note?: string | null;
    testid?: string;
    /** A Clear that can be undone: the line says "Cleared." with an Undo. It comes before a note, after an error. */
    undo?: { what: string; id: string; onundo: () => void } | null;
  }

  let { error = null, note = null, testid = 'help-line', undo = null }: Props = $props();
</script>

{#if error}
  <p class="ui-help ui-help-error help-line" role="alert" data-testid={testid}>{error}</p>
{:else if undo}
  <p class="ui-help help-line" role="status" data-testid="clear-undo">Cleared. <button type="button" id={undo.id} class="undo" aria-label="Undo clearing the {undo.what}" onclick={undo.onundo}>Undo</button></p>
{:else if note}
  <p class="ui-help help-line" role="status" data-testid={testid}>{note}</p>
{/if}

<style>
  /* The one line under a box; it sits up against the box's own edge. */
  .help-line { margin: -2px 0 0; padding: 0 4px; }
  .undo { font-weight: 600; text-decoration: underline; color: var(--color-text-primary); }
  /* Touch: a target a finger can hit, without moving the text. */
  @media (pointer: coarse) {
    .undo { display: inline-flex; align-items: center; min-height: 32px; padding: 0 6px; margin-left: -6px; }
  }
</style>
