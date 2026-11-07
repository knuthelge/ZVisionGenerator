<script lang="ts">
  import { untrack } from 'svelte';
  import { Icon } from '$lib/components/atoms';
  import { RECORDING_NOTE, SCRIPT_ACTIONS, type ScriptResult, SCRIPT_SECONDS_MAX, SCRIPT_SECONDS_MIN, defaultFitSeconds, parseFitSeconds, scriptRequest, scriptRunLabel, scriptSettings, scriptView } from '$lib/state/scriptHelper';
  import { formatReadingTime } from '$lib/state/speechDelivery';
  import { WriterRun } from '$lib/state/writerRun.svelte';
  import type { AudioSettingsVocabulary, PromptEnhancerContract, ScriptAction } from '$lib/types';
  import HelperPanel, { type HelperJob } from './HelperPanel.svelte';

  interface Props {
    contract: PromptEnhancerContract;
    /** The panel is open. */
    open: boolean;
    /** The element a popover opens beside (a sheet has none). */
    anchor: HTMLElement | null;
    /** The script the helper works on: your Text, never the Written text. */
    text: string;
    /** Why the Written text is out of date, in a sentence ("Your Text changed since this was written."); null while it is not. The panel says it beside Run. */
    staleReason?: string | null;
    /** The Pace the script is read at (Normal for a voice that follows a recording), which sets how many characters a second holds. */
    pace: string;
    /** The voice follows a recording, which is read as recorded: delivery marks are left out. */
    recording: boolean;
    /** The audio model's id. */
    model: string;
    vocabulary?: AudioSettingsVocabulary;
    /** A job is being submitted. */
    busy: boolean;
    /** A take is running or queued; the enhancer would need its memory. */
    jobsActive: boolean;
    /** The helper is writing: the box it works on cannot be cleared meanwhile. */
    running?: boolean;
    onclose: () => void;
    /** Fill in the Text from Describe it: offered where there is no script, since a script is written from there. */
    onfill?: () => void;
    onresult: (result: ScriptResult) => void;
  }

  let { contract, open, anchor, text, staleReason = null, pace, recording, model, vocabulary, busy, jobsActive, running = $bindable(false), onclose, onfill, onresult }: Props = $props();

  const run = new WriterRun();
  $effect(() => { running = run.running; });
  /** The actions picked, any of them; a run does them in one request, fit first. Null until the user picks: Add delivery, or Fit alone where it is not offered. */
  let picked = $state<ScriptAction[] | null>(null);
  let fitValue = $state('');

  const view = $derived(scriptView(text, pace, recording, vocabulary));
  /** The picks the script can take now (Add delivery is not offered for a voice from a recording). */
  const chosen = $derived((picked ?? [view.actions.includes('delivery') ? 'delivery' : 'fit']).filter((action) => view.actions.includes(action)));
  const fits = $derived(chosen.includes('fit'));
  const marks = $derived(chosen.includes('delivery'));
  const seconds = $derived(parseFitSeconds(fitValue));
  const fitProblem = $derived(fits && seconds === null ? `Fit takes ${SCRIPT_SECONDS_MIN} to ${SCRIPT_SECONDS_MAX} seconds.` : null);
  const options = $derived(SCRIPT_ACTIONS.filter((item) => view.actions.includes(item.action)));
  const readingNow = $derived(view.characters === 0 ? null : formatReadingTime(view.seconds));

  // Opening the panel starts the time to fit to at the script's own reading time, rounded.
  // Only opening does it: a change elsewhere on the page while the panel is open must not wipe the error of the run it shows.
  $effect(() => {
    if (!open) return;
    fitValue = String(defaultFitSeconds(untrack(() => view.seconds)));
  });

  /** Pick or unpick *action*. */
  function toggle(action: ScriptAction): void {
    picked = chosen.includes(action) ? chosen.filter((item) => item !== action) : [...chosen, action];
  }

  /** What the panel runs: the chosen actions on the script, applied only while the script is still the one it was made from. */
  function onrun(): HelperJob {
    const base = text;
    const settings = scriptSettings(chosen, seconds ?? SCRIPT_SECONDS_MIN, pace);
    return {
      request: scriptRequest(model, base, settings),
      field: 'script',
      changed: () => text !== base,
      changedNote: 'Your script changed while the helper was working, so nothing was applied. Try again.',
      apply: (result) => onresult({ ...result, settings }),
    };
  }
</script>

{#snippet body()}
  <p class="sh-from"><Icon name="edit" size={11} />Starts from your Text · pick any</p>
  <div class="sh-picks" role="group" aria-label="Script helper" data-testid="script-helper-actions">
    {#each options as option (option.action)}
      <button
        type="button"
        class="ui-chip"
        aria-pressed={chosen.includes(option.action)}
        disabled={run.running}
        data-value={option.action}
        onclick={() => toggle(option.action)}
      >{option.label}</button>
    {/each}
  </div>
  {#if fits}
    <div class="sh-fit">
      <label for="ws-script-helper-seconds" class="sh-fit-label">Fit to</label>
      <input
        id="ws-script-helper-seconds"
        type="number"
        class="ui-field sh-seconds"
        inputmode="numeric"
        min={SCRIPT_SECONDS_MIN}
        max={SCRIPT_SECONDS_MAX}
        step="1"
        aria-invalid={fitProblem !== null ? 'true' : undefined}
        disabled={run.running}
        value={fitValue}
        oninput={(event) => { fitValue = event.currentTarget.value; }}
      />
      <span class="sh-fit-unit">seconds</span>
    </div>
    <p class="ui-help sh-help" data-testid="script-helper-help">
      {readingNow === null ? '' : `Now about ${readingNow}${pace === 'normal' ? '' : ` at the ${pace.replace('-', ' ')} pace`}. `}Shortens or lengthens the script to fit. Names and quoted text stay as written.
    </p>
  {/if}
  {#if marks}
    <p class="ui-help sh-help" data-testid="script-helper-delivery-help">Adds pauses and delivery marks where the meaning asks for them. It never changes a word, and uses only marks the app knows, so nothing is read aloud by mistake.</p>
  {/if}
  {#if recording && fits}<p class="ui-help sh-help" data-testid="script-helper-recording">{RECORDING_NOTE}</p>{/if}
  {#if fitProblem !== null}<p class="ui-help ui-help-error sh-help" role="alert">{fitProblem}</p>{/if}
{/snippet}

<HelperPanel
  {contract}
  {run}
  {open}
  {anchor}
  title="Script helper"
  idPrefix="script-helper"
  smartHelp="Follows your wording more closely."
  runLabel={scriptRunLabel(chosen)}
  ready={view.problem === null && chosen.length > 0 && fitProblem === null}
  problem={view.problem === null ? null : { text: view.problem, icon: true, fill: true }}
  quote={view.problem === null ? { label: 'Script', text } : null}
  {staleReason}
  session={open}
  {busy}
  {jobsActive}
  {onclose}
  {onfill}
  {onrun}
  {body}
/>

<style>
  .sh-from { display: flex; align-items: center; gap: 6px; margin: 0; font-size: var(--text-meta); color: var(--color-text-muted); }
  .sh-picks { display: flex; flex-wrap: wrap; gap: 6px; }
  .sh-help { margin: 0; }
  .sh-fit { display: flex; align-items: center; gap: 8px; }
  .sh-fit-label, .sh-fit-unit { font-size: var(--text-ui); color: var(--color-text-secondary); }
  .sh-seconds { width: 5.5em; font-family: var(--font-mono); text-align: right; }
</style>
