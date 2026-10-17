<script lang="ts">
  import type { EnhanceParts } from '$lib/types';
  import EnhanceChoice from './EnhanceChoice.svelte';

  interface Props {
    /** The choice the run will use. */
    value: EnhanceParts;
    /** "Action only" waits for a scene: with none, Enhance splits the text into both parts. */
    actionOnlyBlocked?: boolean;
    disabled?: boolean;
    /** Prefix of the element ids, as in `EnhanceOptions`. */
    idPrefix?: string;
    onchange: (parts: EnhanceParts) => void;
  }

  let { value, actionOnlyBlocked = false, disabled = false, idPrefix = 'ws-enhance', onchange }: Props = $props();

  const BLOCKED_REASON = 'Write a scene first. With no scene, Enhance splits your text into both.';
  const options = $derived([
    { value: 'both' as const, label: 'Scene and action', help: 'Enhance rewrites both from your text.' },
    { value: 'action' as const, label: 'Action only', help: 'Your scene is kept as written.', unavailable: actionOnlyBlocked ? BLOCKED_REASON : null },
  ]);
</script>

<EnhanceChoice label="Rewrite" {options} {value} {disabled} idPrefix={`${idPrefix}-rewrite`} {onchange} />
