<script lang="ts">
  import { Modal } from '$lib/components/molecules';
  import { draft } from '$lib/state/draft.svelte';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { recordingForm } from '$lib/state/recordingForm.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import { voiceWindow, type VoicesTab } from '$lib/state/voiceWindow.svelte';
  import type { AudioModelOption, AudioSettingsVocabulary, SavedVoice } from '$lib/types';
  import type { VoiceView } from './audioView';
  import NewVoiceTab from './NewVoiceTab.svelte';
  import RecordingTab from './RecordingTab.svelte';
  import YourVoicesTab from './YourVoicesTab.svelte';

  interface Props {
    /** The voice of the Speech tab: a one-off recording in use is what the From recording tab starts from. */
    voice: VoiceView;
    /** The open speech model; the New voice tab exists only when it can design voices. */
    model: AudioModelOption | null;
    /** How fast speech is read: the New voice tab tells how long its preview line takes. */
    vocabulary: AudioSettingsVocabulary | undefined;
  }

  let { voice, model, vocabulary }: Props = $props();

  const canDesign = $derived(model?.kind === 'speech' && model.can_design);
  const tabs = $derived<{ id: VoicesTab; label: string }[]>([
    { id: 'yours', label: 'Your voices' },
    ...(canDesign ? [{ id: 'new' as const, label: 'New voice' }] : []),
    { id: 'recording', label: 'From recording' },
  ]);
  // A tab that the model does not offer shows the nearest one that it does.
  const tab = $derived(tabs.some((entry) => entry.id === voiceWindow.tab) ? voiceWindow.tab : 'recording');

  // The window opens with a one-off recording in use: the From recording tab starts from it, so it can be kept as a voice.
  // A recording the form was already filled from keeps what was typed since, its tick included; the Voice section's tick is not carried over.
  let wasOpen = false;
  $effect(() => {
    const open = voiceWindow.open;
    if (open && !wasOpen && voice.recording && voice.set) {
      if (!recordingForm.filledFrom(voice)) recordingForm.fillFrom({ path: voice.path, file: voice.file, text: voice.text, needsConsent: voice.needsConsent });
    }
    wasOpen = open;
  });

  // The window goes with the Speech tab: it does not wait, open, for the next visit.
  $effect(() => () => { voiceWindow.open = false; });

  function close(): void {
    voiceWindow.open = false;
  }

  /** Choose a voice from the library and close the window. */
  function use(saved: SavedVoice): void {
    draft.setSavedVoice(saved.slug);
    close();
  }

  /** A recording or a candidate was saved as a voice and chosen. */
  function saved(made: SavedVoice): void {
    close();
    addToast(`Saved voice "${made.name}"`, 'success');
  }

  /** Arrow keys, Home and End move between the tabs and open the one they land on. */
  function onTabKeydown(event: KeyboardEvent): void {
    const index = tabs.findIndex((entry) => entry.id === tab);
    const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index - 1 + tabs.length) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : -1;
    if (next < 0) return;
    event.preventDefault();
    voiceWindow.tab = tabs[next].id;
    document.getElementById(`voices-tab-${tabs[next].id}`)?.focus();
  }
</script>

<Modal bind:open={voiceWindow.open} title="Voices" size="lg" fill class={layout.mode === 'tall' ? 'voices-window voices-window-tall' : 'voices-window'}>
  {#snippet children()}
    <div class="voices-head" data-compact={layout.compact}>
      <div class="ui-segmented voices-tabs" role="tablist" aria-label="Voices">
        {#each tabs as entry (entry.id)}
          <button
            type="button"
            role="tab"
            id="voices-tab-{entry.id}"
            aria-selected={tab === entry.id}
            aria-controls="voices-panel"
            tabindex={tab === entry.id ? 0 : -1}
            onclick={() => { voiceWindow.tab = entry.id; }}
            onkeydown={onTabKeydown}
          >{entry.label}</button>
        {/each}
      </div>
    </div>
    <div id="voices-panel" class="voices-panel" role="tabpanel" aria-labelledby="voices-tab-{tab}">
      {#if tab === 'yours'}
        <YourVoicesTab inUse={voice.saved ? draft.state.savedVoice : null} onuse={use} onnew={() => { voiceWindow.tab = canDesign ? 'new' : 'recording'; }} />
      {:else if tab === 'new'}
        <NewVoiceTab {model} {vocabulary} onsaved={saved} />
      {:else}
        <RecordingTab {model} onuse={close} onsaved={saved} />
      {/if}
    </div>
  {/snippet}
</Modal>

<style>
  :global(.voices-window) { height: min(480px, 100%); }
  :global(.voices-window-tall) { height: min(540px, 100%); }
  .voices-head { flex: none; padding: 10px 16px 0; }
  .voices-head[data-compact='true'] { padding: 8px 12px 0; }
  .voices-head[data-compact='true'] .voices-tabs { display: flex; }
  .voices-head[data-compact='true'] .voices-tabs > button { flex: 1; height: 44px; padding: 0 4px; font-size: 14px; }
  .voices-panel { display: flex; flex: 1; min-height: 0; flex-direction: column; }
</style>
