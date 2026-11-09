<script lang="ts">
  import { InspectorRow, Segmented, Select } from '$lib/components/molecules';
  import { markedFields } from '$lib/state/audioFill';
  import { draft } from '$lib/state/draft.svelte';
  import { vibeDetail, vibeOf } from '$lib/state/speechDelivery';
  import type { AudioSettingsVocabulary, SpeechDeliveryValues } from '$lib/types';
  import FilledMark from './FilledMark.svelte';
  import { rangeFill } from './settingsSpec';

  interface Props {
    /** The delivery vocabulary the Workspace payload serves: the Vibes and the choices of every setting. */
    vocabulary: AudioSettingsVocabulary;
    /** The voice is a saved voice or a one-off recording: the controls give way to one line, and nothing is sent. */
    recording: boolean;
    /** Put all five values back (the section's reset); omitted where the container has its own Reset. */
    onreset?: () => void;
    /** The controls sit in a phone's sheet while the Settings pane is in the page too: their ids and test ids are their own. */
    popover?: boolean;
  }

  let { vocabulary, recording, onreset, popover = false }: Props = $props();

  const idBase = $derived(popover ? 'ws-pop-delivery' : 'ws-delivery');
  const testBase = $derived(popover ? 'pop-delivery' : 'delivery');

  const values = $derived(draft.state.speechDelivery);
  const defaults = $derived(vocabulary.delivery_defaults);
  const vibe = $derived(vibeOf(values, vocabulary.vibes));
  /** Describe it picked the Vibe and the five values are still as it set them. */
  const filled = $derived(markedFields(draft.state, 'speech').includes('vibe'));
  const changedAny = $derived((Object.keys(defaults) as (keyof SpeechDeliveryValues)[]).some((field) => values[field] !== defaults[field]));
  const emotionOptions = $derived(
    vocabulary.emotions.map((emotion) => ({ value: emotion.id, label: emotion.label, ...(emotion.group === null ? {} : { group: emotion.group }) })),
  );
  const paceIndex = $derived(Math.max(0, vocabulary.paces.findIndex((pace) => pace.id === values.pace)));
  const paceLabel = $derived(vocabulary.paces[paceIndex]?.label ?? '');
  const levelOptions = $derived(vocabulary.levels.map((level) => ({ value: level.id, label: level.label })));
  const mannerOptions = $derived(vocabulary.manners.map((manner) => ({ value: manner.id, label: manner.label })));

  const set = (field: keyof SpeechDeliveryValues, value: string): void => draft.update('speechDelivery', { ...values, [field]: value });
  const changed = (field: keyof SpeechDeliveryValues): boolean => values[field] !== defaults[field];
  const reset = (field: keyof SpeechDeliveryValues): void => set(field, defaults[field]);
</script>

{#if recording}
  <p class="delivery-note" data-testid="{testBase}-recording-note">Your voices read as they were made. Delivery works with the default voices.</p>
{:else}
  <div class="delivery" data-testid="{testBase}-controls">
    {#if onreset}
      <InspectorRow label="Vibe" changed={changedAny} {onreset}>
        <span class="vibe-hint">Fills the five values below</span>
      </InspectorRow>
    {/if}
    {#if filled}<p class="vibe-filled" data-testid="{testBase}-vibe-filled"><FilledMark label={vibe ? `${vibe.label} · from your idea` : 'From your idea'} /></p>{/if}
    <div class="vibes" role="group" aria-label="Vibe">
      {#each vocabulary.vibes as option (option.id)}
        <button type="button" class="vibe" aria-pressed={vibe?.id === option.id} data-vibe={option.id} onclick={() => draft.update('speechDelivery', { ...option.delivery })}>
          <span class="vibe-name">{option.label}</span>
          <span class="vibe-detail">{vibeDetail(option, vocabulary)}</span>
        </button>
      {/each}
    </div>
    <InspectorRow label="Emotion" forId="{idBase}-emotion" changed={changed('emotion')} onreset={() => reset('emotion')}>
      <Select
        id="{idBase}-emotion"
        variant="bare"
        class="flex-1"
        triggerClass="delivery-select"
        label="Emotion"
        testId="{testBase}-emotion"
        value={values.emotion}
        options={emotionOptions}
        groupColumns
        onchange={(emotion) => set('emotion', emotion)}
      />
    </InspectorRow>
    <InspectorRow label="Manner" changed={changed('manner')} onreset={() => reset('manner')}>
      <Segmented label="Manner" size="sm" class="delivery-seg" testId="{testBase}-manner" options={mannerOptions} value={values.manner} onchange={(manner) => set('manner', manner)} />
    </InspectorRow>
    <InspectorRow label="Pace" forId="{idBase}-pace" changed={changed('pace')} onreset={() => reset('pace')}>
      <div class="pace">
        <input
          id="{idBase}-pace"
          class="ui-range"
          type="range"
          min={0}
          max={vocabulary.paces.length - 1}
          step={1}
          value={paceIndex}
          aria-label="Pace"
          aria-valuetext={paceLabel}
          data-testid="{testBase}-pace"
          style:--fill={rangeFill(paceIndex, { min: 0, max: vocabulary.paces.length - 1 })}
          oninput={(e) => set('pace', vocabulary.paces[Number(e.currentTarget.value)].id)}
        >
        <span class="pace-word" data-testid="{testBase}-pace-word">{paceLabel}</span>
      </div>
    </InspectorRow>
    <InspectorRow label="Pitch" changed={changed('pitch')} onreset={() => reset('pitch')}>
      <Segmented label="Pitch" size="sm" class="delivery-seg" testId="{testBase}-pitch" options={levelOptions} value={values.pitch} onchange={(pitch) => set('pitch', pitch)} />
    </InspectorRow>
    <InspectorRow label="Expressiveness" changed={changed('expressiveness')} onreset={() => reset('expressiveness')}>
      <Segmented label="Expressiveness" size="sm" class="delivery-seg" testId="{testBase}-expressiveness" options={levelOptions} value={values.expressiveness} onchange={(level) => set('expressiveness', level)} />
    </InspectorRow>
  </div>
{/if}

<style>
  .delivery-note { margin: 0; padding: 4px 12px 6px; font-size: var(--text-meta); line-height: 1.45; color: var(--color-text-muted); }
  .vibe-filled { margin: 0; padding: 4px 12px 0; }
  .vibe-hint { flex: 1; min-width: 0; font-size: var(--text-meta); color: var(--color-text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  /* Two columns of tiles; the picked one takes the primary tint. */
  .vibes { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; padding: 4px 8px 6px 12px; }
  .vibe { display: grid; gap: 1px; min-width: 0; padding: 5px 8px 6px; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); text-align: left; }
  .vibe:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 1px; }
  .vibe-name { overflow: hidden; font-size: var(--text-ui); font-weight: 600; color: var(--color-text-primary); white-space: nowrap; text-overflow: ellipsis; }
  .vibe-detail { overflow: hidden; min-height: 1.3em; font-size: var(--text-meta); color: var(--color-text-muted); white-space: nowrap; text-overflow: ellipsis; }
  .vibe[aria-pressed='true'] { background: color-mix(in srgb, var(--color-primary-main) 12%, var(--color-bg-base)); box-shadow: inset 0 0 0 1px var(--color-primary-border); }
  .vibe[aria-pressed='true'] .vibe-name { color: var(--color-primary-main); }
  @media (hover: hover) { .vibe:not([aria-pressed='true']):hover { background: var(--color-bg-surface); } }
  .pace { display: flex; flex: 1; min-width: 0; align-items: center; gap: 8px; }
  .pace .ui-range { flex: 1; min-width: 0; }
  .pace-word { flex: none; width: 62px; font-size: var(--text-ui); color: var(--color-text-primary); white-space: nowrap; }
  .delivery :global(.delivery-seg) { display: flex; flex: 1; min-width: 0; }
  .delivery :global(.delivery-seg > button) { flex: 1; min-width: 0; padding: 0 2px; }
  .delivery :global(.delivery-select) { height: 24px; padding: 0 6px; border: 1px solid transparent; border-radius: 4px; background: transparent; font-size: var(--text-ui); color: var(--color-text-primary); }
  @media (hover: hover) { .delivery :global(.delivery-select:hover) { background: var(--color-bg-surface-hover); } }
  .delivery :global(.delivery-select:focus-visible), .delivery :global(.delivery-select[aria-expanded='true']) { outline: 2px solid var(--color-primary-main); outline-offset: 2px; background: var(--color-bg-surface); }
  @media (pointer: coarse) {
    /* "Expressiveness" at the touch size needs more than the usual label column. */
    .delivery { --inspector-label-width: 116px; }
    .vibes { gap: 6px; }
    .vibe { padding: 8px 10px 9px; }
    .vibe-name { font-size: 14px; }
    .vibe-detail { font-size: 12px; }
    .pace .ui-range { height: 28px; }
    .pace-word { width: 72px; font-size: 14px; }
    .delivery :global(.delivery-select) { height: 32px; padding: 0 8px; background: var(--color-bg-surface-hover); font-size: 14px; }
  }
</style>
