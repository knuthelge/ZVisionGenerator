<script lang="ts">
  import { errorText } from '$lib/api/client';
  import { voiceAudioUrl } from '$lib/api/voices';
  import { Icon } from '$lib/components/atoms';
  import { requestConfirm, VoicePlay } from '$lib/components/molecules';
  import { formatTimestamp } from '$lib/state/audioTrack';
  import { draft } from '$lib/state/draft.svelte';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import { voicesStore } from '$lib/state/voices.svelte';
  import type { SavedVoice } from '$lib/types';
  import PresetNameDialog from './PresetNameDialog.svelte';

  interface Props {
    /** The slug of the saved voice the take follows now, or null: its row says *In use* in place of Use. */
    inUse: string | null;
    /** Choose a voice: the window closes. */
    onuse: (voice: SavedVoice) => void;
    /** The empty state's button: move to the New voice tab. */
    onnew: () => void;
  }

  let { inUse, onuse, onnew }: Props = $props();

  let renaming = $state<SavedVoice | null>(null);
  let renameOpen = $state(false);

  /** Rename the voice; a rejection (the name is taken) is shown in the dialog. */
  async function rename(name: string): Promise<void> {
    if (renaming !== null) await voicesStore.rename(renaming.slug, name);
  }

  /** Delete the voice after asking; the server's refusal (a job is running) is a toast. */
  async function remove(voice: SavedVoice): Promise<void> {
    const confirmed = await requestConfirm({ question: `Delete "${voice.name}"?`, info: 'Takes you made with it keep their sound.', confirmLabel: 'Delete', danger: true });
    if (!confirmed) return;
    try {
      await voicesStore.remove(voice.slug);
    } catch (err) {
      addToast(errorText(err, 'Could not delete the voice.'), 'error');
    }
  }
</script>

{#if voicesStore.list.length === 0}
  <div class="yv-empty ui-empty" data-testid="your-voices-empty">
    <p class="ui-empty-title">No voices yet</p>
    <p>Voices you make or save appear here, ready for any take.</p>
    <button type="button" class="ui-btn ui-btn-primary" data-testid="your-voices-new" onclick={onnew}>New voice</button>
  </div>
{:else}
  <ul class="yv-list" data-compact={layout.compact} data-testid="your-voices-list">
    {#each voicesStore.list as voice (voice.slug)}
      <li class="yv-row" data-testid="your-voice" data-slug={voice.slug}>
        <VoicePlay src={voiceAudioUrl(voice.slug)} label={voice.name} class="yv-play" />
        <span class="yv-text">
          <span class="yv-name" title={voice.name}>{voice.name}</span>
          <span class="yv-len">{formatTimestamp(Math.round(voice.seconds))}</span>
        </span>
        <button type="button" class="ui-btn ui-btn-row" aria-label="Rename {voice.name}" title="Rename" data-action="voice-rename" onclick={() => { renaming = voice; renameOpen = true; }}><Icon name="edit" size={14} /></button>
        <button type="button" class="ui-btn ui-btn-row ui-btn-danger" aria-label="Delete {voice.name}" title="Delete" data-action="voice-delete" onclick={() => void remove(voice)}><Icon name="trash" size={14} /></button>
        {#if voice.slug === inUse}
          <span class="yv-inuse" data-testid="voice-in-use">In use</span>
        {:else}
          <button type="button" class="ui-btn ui-btn-quiet ui-btn-sm yv-use" aria-label="Use {voice.name}" data-action="voice-use" onclick={() => onuse(voice)}>Use</button>
        {/if}
      </li>
    {/each}
  </ul>
{/if}
<PresetNameDialog bind:open={renameOpen} elevated title="Rename voice" initialName={renaming?.name ?? ''} confirmLabel="Rename" failureText="Could not rename the voice." onsubmit={rename} />

<style>
  .yv-list { flex: 1; min-height: 0; margin: 0; padding: 4px 8px 8px; list-style: none; overflow-y: auto; }
  .yv-row { display: flex; align-items: center; gap: 4px; min-height: 44px; padding: 2px 6px 2px 8px; border-bottom: 1px solid var(--color-border-subtle); }
  .yv-row:last-child { border-bottom: 0; }
  .yv-row :global(.yv-play) { width: 28px; height: 28px; margin-right: 6px; }
  .yv-text { display: flex; flex: 1; min-width: 0; align-items: center; gap: 6px; }
  .yv-name { flex: 1; min-width: 0; overflow: hidden; padding: 0 4px; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-content); font-weight: 600; color: var(--color-text-primary); cursor: default; }
  .yv-len { flex: none; margin-right: 6px; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  .yv-use, .yv-inuse { flex: none; width: 44px; min-width: 44px; padding: 0; justify-content: center; text-align: center; }
  .yv-inuse { font-size: var(--text-meta); color: var(--color-text-muted); }
  /* A mouse sees Use on hover and on focus; touch always. */
  @media (hover: hover) and (pointer: fine) {
    .yv-use { opacity: 0; }
    .yv-row:hover .yv-use, .yv-row:focus-within .yv-use { opacity: 1; }
    .yv-row:hover { background: color-mix(in srgb, var(--color-bg-surface-hover) 60%, transparent); }
  }
  .yv-empty { flex: 1; justify-content: center; }
  .yv-empty .ui-empty-title { margin: 0; }
  @media (pointer: coarse) {
    .yv-row { min-height: 52px; padding-left: 12px; }
    .yv-row :global(.yv-play), .yv-row .ui-btn-row { width: 40px; height: 40px; }
    .yv-use { height: 44px; }
  }
  /* A phone: 60 px rows with 44 px controls; the length sits under the name so names have room. */
  .yv-list[data-compact='true'] .yv-row { min-height: 60px; padding: 4px 4px 4px 12px; }
  .yv-list[data-compact='true'] .yv-row :global(.yv-play), .yv-list[data-compact='true'] .yv-row .ui-btn-row { width: 44px; height: 44px; }
  .yv-list[data-compact='true'] .yv-text { flex-direction: column; align-items: flex-start; gap: 0; }
  .yv-list[data-compact='true'] .yv-name { flex: none; max-width: 100%; font-size: 15px; }
  .yv-list[data-compact='true'] .yv-len { margin: 0; padding: 0 4px; font-size: 13px; }
</style>
