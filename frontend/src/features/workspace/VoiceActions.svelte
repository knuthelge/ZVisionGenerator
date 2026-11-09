<script lang="ts">
  import type { AudioModelOption } from '$lib/types';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { newVoiceTab, voiceWindow } from '$lib/state/voiceWindow.svelte';
  import type { VoiceView } from './audioView';

  interface Props {
    voice: Pick<VoiceView, 'recording' | 'pending'>;
    /** The open speech model: it says whether the New voice tab exists. */
    model: AudioModelOption | null;
  }

  let { voice, model }: Props = $props();

  const canDesign = $derived(model?.kind === 'speech' && model.can_design);

  function openNew(): void {
    voiceWindow.show(newVoiceTab({ canDesign, oneOff: voice.recording, pending: voice.pending }));
  }
</script>

<div class="voice-actions" data-compact={layout.compact}>
  <button type="button" class="ui-btn" data-testid="your-voices-button" aria-haspopup="dialog" onclick={() => voiceWindow.show('yours')}>Your voices</button>
  <button
    type="button"
    class="ui-btn ui-btn-add"
    data-testid="new-voice-button"
    aria-haspopup="dialog"
    aria-label={voice.pending > 0 ? `New voice, ${voice.pending} to pick from` : undefined}
    onclick={openNew}
  >New voice{#if voice.pending > 0}<span class="ui-badge ui-badge-info voice-badge" data-testid="voice-pending">{voice.pending}</span>{/if}</button>
</div>

<style>
  .voice-actions { display: flex; gap: 6px; padding: 4px 8px 6px 12px; }
  .voice-actions .ui-btn { flex: 1 1 0; min-width: 0; }
  .voice-badge { height: 16px; padding: 0 4px; font-family: var(--font-mono); font-weight: 500; line-height: 16px; }
  /* A phone: the buttons are 44 px. */
  .voice-actions[data-compact='true'] { gap: 8px; }
  .voice-actions[data-compact='true'] .ui-btn { height: 44px; padding: 0 10px; font-size: 14px; }
</style>
