<script lang="ts">
  import { InspectorNumber } from '$lib/components/atoms';
  import { InspectorRow, InspectorSection } from '$lib/components/molecules';
  import { formatTimestamp, startNote } from '$lib/state/audioTrack';
  import { draft } from '$lib/state/draft.svelte';
  import type { AudioTrackView } from './audioTrackView';
  import ReferenceSlot from './ReferenceSlot.svelte';

  interface Props {
    /** The audio track view of the open tab, computed once by the Workspace. */
    track: AudioTrackView;
  }

  let { track }: Props = $props();

  // `visible` is only true in Video and Extend Video, so the tab is known wherever the section renders.
  const tab = $derived(track.tab ?? 'video');
</script>

{#if track.visible}
  <InspectorSection id="audio-track" title="Audio track" summary={track.summary.text} active={track.summary.tone === 'active'} error={track.summary.tone === 'error'}>
    <ReferenceSlot
      idBase="ws-audio-track"
      kind="audio"
      label="Track"
      names={{ browse: 'Browse for an audio track', upload: 'Upload an audio track from this device', clear: 'Clear audio track', preview: 'Audio track' }}
      emptyText="Browse or drop"
      clearable={track.set}
      path={track.path}
      file={track.file}
      previewUrl={null}
      name={track.name}
      detail={track.length === null ? '' : formatTimestamp(track.length)}
      dropText={track.set ? 'Drop to replace' : 'Drop an audio file'}
      missing={track.gone}
      error={track.block ?? undefined}
      onfilechange={(file) => { if (file !== null) draft.setAudioTrack(tab, { file }); }}
      onpathchange={(path) => draft.setAudioTrack(tab, path === null ? null : { path })}
    />
    {#if track.set}
      <div class="start-row">
        <InspectorRow label="Start at (s)" forId="ws-audio-track-start" changed={track.start !== 0} onreset={() => draft.setAudioTrackStart(tab, 0)}>
          <InspectorNumber id="ws-audio-track-start" ariaLabel="Start at, in seconds" value={track.start} step={1} min={0} onchange={(v) => draft.setAudioTrackStart(tab, v ?? 0)} />
          <span class="start-note" data-testid="audio-track-start-note">{startNote(track.start)}</span>
        </InspectorRow>
      </div>
    {/if}
  </InspectorSection>
{/if}

<style>
  /* A finger gets 44px targets on this row: the field, its − and +, and the reset mark. */
  @media (pointer: coarse) {
    .start-row :global(.inspector-number), .start-row :global(.inspector-step) { height: 44px; }
    .start-row :global(.inspector-step) { width: 44px; }
    .start-row :global(.inspector-reset) { width: 44px; height: 44px; }
  }
  .start-note { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); white-space: nowrap; }
</style>
