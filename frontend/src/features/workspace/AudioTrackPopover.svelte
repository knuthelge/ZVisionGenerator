<script lang="ts">
  import { Icon, InspectorNumber } from '$lib/components/atoms';
  import { PathInput } from '$lib/components/molecules';
  import { formatTimestamp, startNote, takeAudioFiles } from '$lib/state/audioTrack';
  import { draft } from '$lib/state/draft.svelte';
  import type { AudioTrackView } from './audioTrackView';
  import ImageSources from './ImageSources.svelte';

  interface Props {
    /** The audio track view of the open tab, computed once by the Workspace. */
    track: AudioTrackView;
  }

  let { track }: Props = $props();

  const tab = $derived(track.tab ?? 'video');

  // The path field commits on change (Enter or leaving the field), so a half-typed path never reaches the draft.
  let pathText = $state('');
  $effect(() => {
    pathText = track.path ?? '';
  });

  function takeFiles(files: File[]): void {
    takeAudioFiles(files, (file) => draft.setAudioTrack(tab, { file }));
  }

  function commitPath(): void {
    const path = pathText.trim();
    if (path === '') {
      if (track.path !== null) draft.setAudioTrack(tab, null);
    } else if (path !== track.path) draft.setAudioTrack(tab, { path });
  }

  // Enter takes the typed path at once; it must not submit the form.
  function onPathKeydown(event: KeyboardEvent): void {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    commitPath();
  }
</script>

<div class="track-pop">
  <div class="pop-group" role="group" aria-labelledby="pop-track-group">
    <h4 id="pop-track-group" class="ui-area-label pop-group-label">Track</h4>
    <div class="picker-row">
      <span class="picker-thumb" class:missing={track.gone}><Icon name={track.gone ? 'alert' : 'waveform'} size={20} /></span>
      <span class="picker-name" class:muted={!track.name} class:missing={track.gone} title={track.name}>{track.name || 'No audio track yet'}</span>
      {#if track.length !== null}<span class="picker-length" data-testid="pop-track-length">{formatTimestamp(track.length)}</span>{/if}
      {#if track.set}<button type="button" class="ui-btn ui-btn-icon" aria-label="Clear audio track" onclick={() => draft.setAudioTrack(tab, null)}><Icon name="close" size={16} /></button>{/if}
    </div>
    {#if track.block}<p class="ui-help ui-help-error">{track.block}</p>{/if}
    <ImageSources kind="audio" browseFrom={track.path} onfiles={takeFiles} onpath={(path) => draft.setAudioTrack(tab, { path })} />
    <PathInput id="ws-pop-track-path" purpose="audio_track" class="ui-field pop-path" placeholder="/path/to/song.mp3" value={pathText} oninput={(text) => { pathText = text; }} onchange={commitPath} onkeydown={onPathKeydown} onpick={(path) => draft.setAudioTrack(tab, { path })} />
  </div>

  {#if track.set}
    <div class="pop-group" role="group" aria-labelledby="pop-start-at-group">
      <h4 id="pop-start-at-group" class="ui-area-label pop-group-label">Start at</h4>
      <InspectorNumber ariaLabel="Start at, in seconds" value={track.start} step={1} min={0} stepper size="lg" onchange={(v) => draft.setAudioTrackStart(tab, v ?? 0)} />
      <p class="ui-help">{startNote(track.start)}. Seconds into the track where the clip's sound starts.</p>
    </div>
  {/if}
</div>

<style>
  .track-pop { display: flex; flex-direction: column; gap: 20px; }
  .pop-group { display: flex; flex-direction: column; gap: 12px; padding: 0 2px; }
  .pop-group-label { padding: 0; }
  .picker-row { display: flex; align-items: center; gap: 12px; min-height: 56px; font-size: 15px; }
  .picker-thumb { display: grid; flex: none; place-items: center; width: 56px; height: 56px; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); color: var(--color-text-muted); }
  .picker-thumb.missing { background: var(--color-error-surface); color: var(--color-error); }
  .picker-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .picker-length { flex: none; font-family: var(--font-mono); font-size: 13px; color: var(--color-text-muted); white-space: nowrap; }
  .picker-name.muted { color: var(--color-text-muted); }
  .picker-name.missing { color: var(--color-error); }
  .pop-group :global(.pop-path) { height: 40px; font-family: var(--font-mono); font-size: 13px; }
</style>
