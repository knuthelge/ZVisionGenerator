<script lang="ts">
  import type { WriterResult } from '$lib/state/writerRun.svelte';
  import { WriterRun } from '$lib/state/writerRun.svelte';
  import { DEFAULT_PICKS, NOTE_MAX_CHARS, helperView, lyricsRequest, type LyricsTarget } from '$lib/state/lyricsHelper';
  import type { LyricsImprovement, PromptEnhancerContract } from '$lib/types';
  import HelperPanel, { type HelperJob } from './HelperPanel.svelte';

  interface Props {
    contract: PromptEnhancerContract;
    /** What the helper is open on (the whole song, or one section to improve or write); null is closed. */
    target: LyricsTarget | null;
    /** The element a popover opens beside (a sheet has none). */
    anchor: HTMLElement | null;
    /** The song as the Lyrics box holds it. */
    lyrics: string;
    /** The style that will be sent with the take, so the words suit it. */
    style: string;
    /** The audio model's id. */
    model: string;
    /** A job is being submitted. */
    busy: boolean;
    /** A take is running or queued; the enhancer would need its memory. */
    jobsActive: boolean;
    /** The helper is writing: the box it works on cannot be cleared meanwhile. */
    running?: boolean;
    onclose: () => void;
    /** Fill in the Lyrics from Describe it: offered where the song is empty, since new lyrics come from there. */
    onfill?: () => void;
    /** The helper wrote the whole song again; *warning* is the enhancer's notice about it (a changed quote). */
    onresult: (text: string, warning: string | null) => void;
  }

  let { contract, target, anchor, lyrics, style, model, busy, jobsActive, running = $bindable(false), onclose, onfill, onresult }: Props = $props();

  const run = new WriterRun();
  $effect(() => { running = run.running; });
  // The whole song keeps its picks and direction between runs; a section starts clean each time it is opened. Tighten starts ticked.
  const whole = $state<{ picks: LyricsImprovement[]; note: string }>({ picks: [...DEFAULT_PICKS], note: '' });
  const part = $state<{ picks: LyricsImprovement[]; note: string }>({ picks: [...DEFAULT_PICKS], note: '' });

  const form = $derived(target !== null && target.section !== null ? part : whole);
  const view = $derived(target === null ? null : helperView(lyrics, target, style));
  const noun = $derived(target?.write ? 'Write' : 'Improve');
  const problem = $derived(view !== null && view.problem !== null ? { text: view.problem, icon: view.emptySong, fill: view.emptySong } : null);

  // A new target starts a section with no picks or direction.
  $effect(() => {
    void target;
    if (target !== null && target.section !== null) {
      part.picks = [...DEFAULT_PICKS];
      part.note = '';
    }
  });

  function toggle(slug: LyricsImprovement): void {
    form.picks = form.picks.includes(slug) ? form.picks.filter((item) => item !== slug) : [...form.picks, slug];
  }

  /** What the panel runs: the request for the open target, applied only while the lyrics are still those it was made from. */
  function onrun(): HelperJob {
    const base = lyrics;
    return {
      request: lyricsRequest(model, base, style, target as LyricsTarget, form.picks, form.note),
      field: 'lyrics',
      changed: () => lyrics !== base,
      changedNote: 'Your lyrics changed while the helper was working, so nothing was applied. Try again.',
      apply: (result: WriterResult) => onresult(result.text, result.warning),
    };
  }
</script>

{#snippet body()}
  {#if view}
    {#if view.options.length > 0}
      <fieldset class="lh-chips" disabled={run.running}>
        <legend class="ui-area-label">Improve · pick any</legend>
        <div class="lh-chip-list" role="group" aria-label="Improve">
          {#each view.options as option (option.slug)}
            <button type="button" class="ui-chip lh-chip" aria-pressed={form.picks.includes(option.slug)} title={option.help} data-improve={option.slug} onclick={() => toggle(option.slug)}>{option.label}</button>
          {/each}
        </div>
      </fieldset>
    {/if}
    <input
      id="ws-lyrics-helper-note"
      type="text"
      class="ui-field lh-note"
      maxlength={NOTE_MAX_CHARS}
      placeholder={target?.write ? 'Direction (optional): turn toward hope' : 'Direction (optional): make it more hopeful'}
      aria-label="Direction for the lyrics helper"
      disabled={run.running}
      bind:value={form.note}
    />
    <p class="ui-help lh-context" data-testid="lyrics-helper-context">{view.context}</p>
  {/if}
{/snippet}

<HelperPanel
  {contract}
  {run}
  open={target !== null}
  {anchor}
  title={view?.title ?? 'Improve lyrics'}
  idPrefix="lyrics-helper"
  smartHelp="Follows your options and wording more closely."
  runLabel={view?.runLabel ?? noun}
  ready={view !== null && view.problem === null}
  {problem}
  quote={view === null || view.emptySong ? null : view.quote}
  session={target}
  {busy}
  {jobsActive}
  {onclose}
  {onfill}
  {onrun}
  {body}
/>

<style>
  .lh-chips { display: grid; gap: 6px; min-width: 0; margin: 0; padding: 0; border: 0; }
  .lh-chips legend { padding: 0; }
  .lh-chip-list { display: flex; flex-wrap: wrap; gap: 4px; }
  .lh-chip { cursor: pointer; }
  .lh-chip:disabled { cursor: not-allowed; opacity: 0.4; }
  .lh-context { margin: 0; }
</style>
