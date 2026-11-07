<script lang="ts">
  import { InspectorSection } from '$lib/components/molecules';
  import { draft } from '$lib/state/draft.svelte';
  import { deliverySummary } from '$lib/state/speechDelivery';
  import type { AudioSettingsVocabulary } from '$lib/types';
  import DeliveryControls from './DeliveryControls.svelte';

  interface Props {
    /** The delivery vocabulary of the Workspace payload. */
    vocabulary: AudioSettingsVocabulary;
    /** The voice follows a recording: the section holds one line and nothing is sent. */
    recording: boolean;
    /** The delivery differs from its defaults: the section's changed mark and its reset. */
    changed: boolean;
    /** Put all five values back. */
    onreset: () => void;
  }

  let { vocabulary, recording, changed, onreset }: Props = $props();

  // A voice read as recorded has no delivery to summarise.
  const summary = $derived(recording ? '' : deliverySummary(draft.state.speechDelivery, vocabulary));
</script>

<InspectorSection id="delivery" title="Delivery" {summary} active={!recording && changed}>
  <DeliveryControls {vocabulary} {recording} {onreset} />
</InspectorSection>
