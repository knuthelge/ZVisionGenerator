<script lang="ts" module>
  import type { KeyValueItem } from './KeyValueList.svelte';

  /** Where the server reports a token saved on its computer (by this dialog or `hf auth login`). */
  const SAVED_TOKEN_SOURCE = 'saved';

  /**
   * Build the Hugging Face token row for a token *source*: none, saved on this computer, or an environment variable.
   * Only a token the app can change gets a button; one from the environment wins over a saved one.
   */
  export function hfTokenItem(source: string | null | undefined, onAdd: () => void): KeyValueItem {
    const label = 'Hugging Face token';
    if (!source) return { label, value: 'Not set', tone: 'muted', hint: 'Gives higher download limits and access to gated models.', action: { id: 'hf-token', label: 'Add token', run: onAdd } };
    if (source === SAVED_TOKEN_SOURCE) return { label, value: 'Available', tone: 'success', hint: 'Saved on this computer.', action: { id: 'hf-token', label: 'Replace token', run: onAdd } };
    return { label, value: 'Available', tone: 'success', hint: `From ${source}. Unset it to use a token saved here.` };
  }
</script>

<script lang="ts">
  import { errorText } from '$lib/api/client';
  import { saveHfToken } from '$lib/api/config';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import Modal from './Modal.svelte';

  interface Props {
    open: boolean;
    /** The dialog was closed: cancelled, dismissed or saved. */
    onclose: () => void;
    /** A token was saved; *source* is where the server now finds the token. */
    onsaved: (source: string | null) => void;
  }

  let { open, onclose, onsaved }: Props = $props();

  let token = $state('');
  let saving = $state(false);
  let failure = $state<string | null>(null);

  const canSave = $derived(token.trim() !== '' && !saving);

  // Every opening starts with an empty field and no error. A check still running from an earlier opening keeps Save busy until it ends.
  $effect(() => {
    if (!open) return;
    token = '';
    failure = null;
  });

  async function save(): Promise<void> {
    if (!canSave) return;
    saving = true;
    failure = null;
    try {
      const result = await saveHfToken(token);
      addToast('Hugging Face token saved', 'success');
      onsaved(result.huggingface_token_source);
      onclose();
    } catch (err) {
      failure = errorText(err, 'Could not save the token.');
    } finally {
      saving = false;
    }
  }
</script>

<Modal {open} title="Hugging Face token" size="sm" {onclose}>
  {#snippet children()}
    <form class="hft-body" data-compact={layout.compact} onsubmit={(event) => { event.preventDefault(); void save(); }}>
      <p class="ui-help">A token does two things:</p>
      <ul class="ui-help hft-uses">
        <li><strong>Higher download limits.</strong> Hugging Face allows more downloads with a token, so big downloads are less likely to be slowed down or stopped.</li>
        <li><strong>Gated models.</strong> Models that ask you to accept a license download once the token's account has accepted it.</li>
      </ul>
      <p class="ui-help">
        <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">Create a token on Hugging Face</a> and choose the <strong>Read</strong> token type.
      </p>
      <div class="hft-field">
        <label class="ui-label" for="hft-token">Token</label>
        <input
          id="hft-token"
          class="ui-field"
          type="text"
          placeholder="hf_…"
          autocomplete="off"
          autocapitalize="off"
          spellcheck="false"
          aria-invalid={failure ? 'true' : undefined}
          aria-describedby={failure ? 'hft-error' : undefined}
          bind:value={token}
          oninput={() => { failure = null; }}
        >
        {#if failure}<p id="hft-error" class="ui-help ui-help-error" role="alert">{failure}</p>{/if}
      </div>
    </form>
  {/snippet}
  {#snippet footer()}
    {#if !layout.compact}<button type="button" class="ui-btn" onclick={onclose}>Cancel</button>{/if}
    <button type="button" class="ui-btn ui-btn-primary hft-save" class:wide={layout.compact} data-action="save-hf-token" disabled={!canSave} aria-busy={saving} onclick={() => void save()}>{saving ? 'Checking…' : 'Save'}</button>
  {/snippet}
</Modal>

<style>
  .hft-body { display: flex; flex-direction: column; gap: 12px; }
  .hft-uses { margin: 0; padding-left: 1.2em; list-style: disc; display: flex; flex-direction: column; gap: 4px; }
  .hft-uses strong { color: var(--color-text-primary); font-weight: 600; }
  .hft-field { display: flex; flex-direction: column; gap: 4px; }
  .hft-body a { color: var(--color-primary-main); text-decoration: underline; text-underline-offset: 2px; }
  @media (pointer: coarse) { .hft-body a { display: inline-block; padding-block: 6px; } }
  /* A phone: a 44 px field and one full-width button in the foot. */
  .hft-body[data-compact='true'] :global(input.ui-field) { height: 44px; font-size: 16px; }
  .hft-save.wide { flex: 1 1 100%; height: 44px; font-size: 15px; }
</style>
