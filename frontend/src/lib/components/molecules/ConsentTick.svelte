<script lang="ts">
  import { VOICE_CONSENT_TEXT } from '$lib/state/voiceConsent.svelte';

  interface Props {
    /** The speaker's confirmation is given. */
    checked: boolean;
    /** The tick changed. */
    onchange: (checked: boolean) => void;
    /** The id of the checkbox. */
    id?: string;
    /** Always use the finger-sized look, as a phone sheet does, not only on a touch screen. */
    large?: boolean;
  }

  let { checked, onchange, id, large = false }: Props = $props();
</script>

<label class="consent-tick" class:ticked={checked} data-large={large}>
  <input type="checkbox" class="ui-checkbox" {id} {checked} onchange={(event) => onchange(event.currentTarget.checked)}>
  <span>{VOICE_CONSENT_TEXT}</span>
</label>

<style>
  .consent-tick { display: flex; align-items: flex-start; gap: 8px; min-height: 30px; font-size: var(--text-ui); line-height: 1.4; color: var(--color-text-secondary); cursor: pointer; }
  .consent-tick.ticked { color: var(--color-text-primary); }
  .consent-tick input { margin-top: 2px; }
  @media (pointer: coarse) {
    .consent-tick { min-height: 44px; font-size: 15px; }
    .consent-tick input { width: 20px; height: 20px; }
  }
  .consent-tick[data-large='true'] { gap: 10px; min-height: 44px; font-size: 15px; }
  .consent-tick[data-large='true'] input { flex: none; width: 20px; height: 20px; margin-top: 1px; }
</style>
