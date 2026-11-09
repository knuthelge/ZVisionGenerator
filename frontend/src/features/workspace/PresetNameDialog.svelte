<script lang="ts">
  import { errorText } from '$lib/api/client';
  import Modal from '$lib/components/molecules/Modal.svelte';

  interface Props {
    open?: boolean;
    title: string;
    /** The name the field starts with. */
    initialName?: string;
    /** The main button's label. */
    confirmLabel: string;
    /** Save the preset under the chosen name; a rejection is shown in the dialog and keeps it open. */
    onsubmit: (name: string) => Promise<void>;
    /** What the dialog says when a rejection carries no reason of its own. */
    failureText?: string;
    /** Stack above another dialog that asks for the name. */
    elevated?: boolean;
  }

  let { open = $bindable(false), title, initialName = '', confirmLabel, onsubmit, failureText = 'Could not save the preset.', elevated = false }: Props = $props();

  let name = $state('');
  let saving = $state(false);
  let serverError = $state<string | null>(null);

  const trimmed = $derived(name.trim());
  const canSave = $derived(trimmed !== '' && !saving);

  $effect(() => {
    if (open) {
      name = initialName;
      serverError = null;
    }
  });

  async function submit(): Promise<void> {
    if (!canSave) return;
    saving = true;
    serverError = null;
    try {
      await onsubmit(trimmed);
      open = false;
    } catch (err) {
      serverError = errorText(err, failureText);
    } finally {
      saving = false;
    }
  }
</script>

<Modal bind:open {title} size="sm" {elevated}>
  {#snippet children()}
    <form class="preset-name" onsubmit={(event) => { event.preventDefault(); void submit(); }}>
      <label class="ui-label" for="preset-name">Name</label>
      <input id="preset-name" class="ui-field" bind:value={name} spellcheck="false" autocomplete="off">
      {#if serverError}<p class="ui-help ui-help-error" role="alert">{serverError}</p>{/if}
    </form>
  {/snippet}
  {#snippet footer()}
    <button type="button" class="ui-btn" onclick={() => { open = false; }}>Cancel</button>
    <button type="button" class="ui-btn ui-btn-primary" data-action="confirm-preset-name" disabled={!canSave} onclick={() => void submit()}>{confirmLabel}</button>
  {/snippet}
</Modal>

<style>
  .preset-name { display: flex; flex-direction: column; gap: 8px; }
</style>
