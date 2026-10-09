<script lang="ts">
  import { draft, MAX_SHARPEN_AMOUNT, offeredSizes, settingDefaultsFor, type SettingKey } from '$lib/state/draft.svelte';
  import { Icon, InspectorNumber, Toggle } from '$lib/components/atoms';
  import { ActionBar, InspectorRow, InspectorSection } from '$lib/components/molecules';
  import type { NumberSpec, ScrubOptions } from '$lib/actions/scrub';
  import type { WorkspaceContext } from '$lib/types';
  import { workspaceCapabilities } from './capabilities';
  import { randomSeed } from './seed';

  interface Props {
    context: WorkspaceContext;
    /** A submit is in flight. */
    busy: boolean;
    /** A job is running or queued: Generate adds to the queue. */
    jobsActive?: boolean;
    imageFile: File | null;
    /** Preview for a reference path that points at a known asset. */
    referencePreviewUrl: string | null;
    /** Seed of the most recent output, reused when the seed gets locked. */
    lastSeed: number | null;
    onImageFileChange: (file: File | null) => void;
  }

  let { context, busy, jobsActive = false, imageFile, referencePreviewUrl, lastSeed, onImageFileChange }: Props = $props();

  const s = $derived(draft.state);
  const caps = $derived(workspaceCapabilities(context, s));
  // Narrow dependencies, so typing in the prompt does not re-resolve the defaults.
  const workflow = $derived(s.workflow);
  const model = $derived(s.model);
  const defaults = $derived(settingDefaultsFor(context, { workflow, model }));

  const SPEC = {
    runs: { step: 1, min: 1, max: 100 },
    frames: { step: 1, min: 1, max: 256 },
    steps: { step: 1, min: 1, max: 60 },
    guidance: { step: 0.1, min: 0, max: 10 },
    strength: { step: 0.01, min: 0, max: 1 },
    seed: { step: 1, min: 0, max: 2 ** 32 - 1 },
    firstSigma: { step: 0.001, min: 0.001, max: 2 },
    sharpen: { step: 0.05, min: 0, max: MAX_SHARPEN_AMOUNT },
    contrast: { step: 0.05, min: 0.5, max: 2 },
    saturation: { step: 0.05, min: 0.5, max: 2 },
    upscaleDenoise: { step: 0.01, min: 0, max: 1 },
    upscaleSteps: { step: 1, min: 1, max: 60 },
    upscaleGuidance: { step: 0.1, min: 0, max: 20 },
  } satisfies Record<string, NumberSpec>;

  // --- Dimensions -----------------------------------------------------------
  const dimensionMode = $derived(s.dimensionMode);
  const ratios = $derived(caps.isImageMode ? context.image_ratios : context.video_ratios);
  const sizesFor = (ratio: string): string[] => offeredSizes(context, workflow, model, ratio);
  const sizeOptions = $derived(sizesFor(s.ratio));
  const presetDims = $derived<[number, number]>(
    (caps.isImageMode ? context.image_size_dimensions : (context.video_size_dimensions ?? {}))[s.ratio]?.[s.size] ?? [s.width, s.height]
  );
  const shownWidth = $derived(dimensionMode === 'custom' ? s.width : presetDims[0]);
  const shownHeight = $derived(dimensionMode === 'custom' ? s.height : presetDims[1]);
  const dimensionSpec = $derived<NumberSpec>({ step: caps.dimensionStep, min: caps.dimensionMin, max: caps.dimensionMax ?? undefined });

  function pickRatio(ratio: string): void {
    const valid = sizesFor(ratio);
    draft.patch({ ratio, dimensionMode: 'ratio', ...(valid.includes(s.size) ? {} : { size: valid[0] ?? '' }) });
  }

  function pickSize(size: string): void {
    draft.patch({ size, dimensionMode: 'ratio' });
  }

  // Typing a dimension switches to custom W×H, starting from the preset's other side.
  function setDimension(key: 'width' | 'height', value: number | null): void {
    if (value === null) return;
    draft.patch({ width: shownWidth, height: shownHeight, [key]: value, dimensionMode: 'custom' });
  }

  // --- Changes from defaults ------------------------------------------------
  function changed(...keys: SettingKey[]): boolean {
    return keys.some((key) => s[key] !== defaults[key]);
  }

  function reset(...keys: SettingKey[]): void {
    draft.patch(Object.fromEntries(keys.map((key) => [key, defaults[key]])));
  }

  const changedRows = $derived(
    [
      caps.showDimensions && changed('ratio'),
      caps.showDimensions && changed('size'),
      caps.showDimensions && changed('dimensionMode'),
      caps.showRuns && changed('runs'),
      caps.showFrameCount && changed('frameCount'),
      caps.showSteps && changed('steps'),
      caps.showGuidance && changed('guidance'),
      caps.showI2IStrength && changed('referenceImageStrength'),
      caps.showSeed && changed('seed'),
      caps.showScheduler && changed('scheduler'),
      caps.supportsFirstSigma && changed('firstSigma'),
      caps.showPostprocessSharpen && changed('postprocessSharpenEnabled', 'postprocessSharpenAmount'),
      caps.showPostprocessContrast && changed('postprocessContrastEnabled', 'postprocessContrastAmount'),
      caps.showPostprocessSaturation && changed('postprocessSaturationEnabled', 'postprocessSaturationAmount'),
      caps.showImageUpscale && changed('upscaleEnabled', 'upscaleFactor'),
      caps.showUpscaleDenoise && s.upscaleEnabled && changed('upscaleDenoise'),
      caps.showUpscaleSteps && s.upscaleEnabled && changed('upscaleSteps'),
      caps.showUpscaleGuidance && s.upscaleEnabled && changed('upscaleGuidance'),
      caps.showUpscaleSharpen && s.upscaleEnabled && changed('upscaleSharpen'),
      caps.showAudio && changed('audio'),
      caps.showLowMemory && changed('lowMemory'),
      caps.showVideoUpscale && changed('videoUpscaleEnabled', 'videoUpscaleFactor'),
    ].filter(Boolean).length
  );

  // --- Section summaries ----------------------------------------------------
  const sizeSummary = $derived(`${shownWidth} × ${shownHeight}${dimensionMode === 'custom' ? ' · custom' : ''}`);
  const samplingSummary = $derived(
    [
      caps.showRuns ? `${s.runs}×` : null,
      caps.showSteps ? `${s.steps} steps` : null,
      caps.showGuidance ? `cfg ${s.guidance}` : null,
      caps.showSeed ? (s.seed !== null ? `seed ${s.seed}` : 'random seed') : null,
    ].filter(Boolean).join(' · ')
  );
  const postprocessOn = $derived(
    [s.postprocessSharpenEnabled, s.postprocessContrastEnabled, s.postprocessSaturationEnabled].filter(Boolean).length
  );
  const showPostprocess = $derived(caps.showPostprocessSharpen || caps.showPostprocessContrast || caps.showPostprocessSaturation);
  const showVideo = $derived(caps.showAudio || caps.showLowMemory || caps.showVideoUpscale);
  const videoSummary = $derived(
    [s.audio ? 'Audio' : null, s.lowMemory ? 'Low memory' : null, s.videoUpscaleEnabled ? `${s.videoUpscaleFactor}× upscale` : null]
      .filter(Boolean).join(' · ') || 'Off'
  );

  // --- Seed -----------------------------------------------------------------
  // A locked seed repeats every run; unlocked (empty) picks a new random seed each run.
  function toggleSeedLock(): void {
    draft.update('seed', s.seed !== null ? null : (lastSeed ?? randomSeed()));
  }

  // --- Reference image ------------------------------------------------------
  let dragOver = $state(false);
  let filePreviewUrl = $state<string | null>(null);

  // Revoke the previous object URL whenever the file changes or the pane unmounts.
  $effect(() => {
    const url = imageFile ? URL.createObjectURL(imageFile) : null;
    filePreviewUrl = url;
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  });

  let fileInput = $state<HTMLInputElement | null>(null);

  // Forget a browsed file once it stops being the reference, so choosing it again fires a change.
  $effect(() => {
    if (!imageFile && fileInput) fileInput.value = '';
  });

  const referencePreview = $derived(filePreviewUrl ?? referencePreviewUrl);
  const referenceName = $derived(imageFile?.name ?? (s.referenceImagePath ? s.referenceImagePath.split('/').pop() ?? '' : ''));

  function clearImage(): void {
    onImageFileChange(null);
    draft.update('referenceImagePath', null);
  }

  function handleDrop(event: DragEvent): void {
    event.preventDefault();
    dragOver = false;
    const file = event.dataTransfer?.files[0] ?? null;
    if (file && file.type.startsWith('image/')) onImageFileChange(file);
  }

  // --- Footer ---------------------------------------------------------------
  const promptFileMode = $derived(s.promptSource === 'file');
  const submitDisabled = $derived(busy || (promptFileMode && (!s.promptFilePath || s.promptFileOptionIds.length === 0)));

  function resetAll(): void {
    draft.resetSelections(context);
    onImageFileChange(null);
  }

  function scrubber(value: number | null, spec: NumberSpec, onchange: (value: number) => void, disabled = false): ScrubOptions {
    return { ...spec, value, onchange, disabled: disabled || busy };
  }
</script>

<section class="settings-pane" aria-labelledby="ws-settings-title">
  <div class="settings-head">
    <h2 id="ws-settings-title" class="ui-area-label">Settings</h2>
    <span class="settings-hint" title="Drag a label left or right to change its value; arrow keys nudge (Shift ×10)">Drag labels to scrub</span>
  </div>

  <div class="settings-scroll">
    {#if caps.showDimensions}
      <InspectorSection id="size" title="Size" summary={sizeSummary}>
        <InspectorRow label="Ratio" changed={changed('ratio')} onreset={() => reset('ratio', 'size', 'dimensionMode')}>
          <div class="ui-segmented ui-segmented-sm ui-segmented-mono mini-seg" role="group" aria-label="Aspect ratio">
            {#each ratios as ratio (ratio)}
              <button type="button" aria-pressed={dimensionMode === 'ratio' && s.ratio === ratio} onclick={() => pickRatio(ratio)}>{ratio}</button>
            {/each}
          </div>
        </InspectorRow>
        <InspectorRow label="Resolution" changed={changed('size')} onreset={() => reset('size', 'dimensionMode')}>
          <div class="ui-segmented ui-segmented-sm ui-segmented-mono mini-seg" role="group" aria-label="Resolution">
            {#each sizeOptions as size (size)}
              <button type="button" aria-pressed={dimensionMode === 'ratio' && s.size === size} aria-label="Resolution {size}" onclick={() => pickSize(size)}>{size.toUpperCase()}</button>
            {/each}
          </div>
        </InspectorRow>
        <InspectorRow label="W × H" forId="ws-width" changed={changed('dimensionMode')} onreset={() => reset('ratio', 'size', 'dimensionMode')}>
          <InspectorNumber id="ws-width" name={dimensionMode === 'custom' ? 'width' : undefined} ariaLabel="Width" value={shownWidth} {...dimensionSpec} onchange={(v) => setDimension('width', v)} />
          <span class="dim-times" aria-hidden="true">×</span>
          <InspectorNumber id="ws-height" name={dimensionMode === 'custom' ? 'height' : undefined} ariaLabel="Height" value={shownHeight} {...dimensionSpec} onchange={(v) => setDimension('height', v)} />
        </InspectorRow>
        {#if dimensionMode === 'ratio'}
          <input type="hidden" name="ratio" value={s.ratio}>
          <input type="hidden" name="size" value={s.size}>
        {/if}
      </InspectorSection>
    {/if}

    <InspectorSection id="sampling" title="Sampling" summary={samplingSummary}>
      {#if caps.showRuns}
        <InspectorRow label="Batch size" forId="ws-runs" scrubber={scrubber(s.runs, SPEC.runs, (v) => draft.update('runs', v))} changed={changed('runs')} onreset={() => reset('runs')}>
          <InspectorNumber id="ws-runs" name="runs" value={s.runs} {...SPEC.runs} onchange={(v) => v !== null && draft.update('runs', v)} />
        </InspectorRow>
      {/if}
      {#if caps.showFrameCount}
        <InspectorRow label="Frames" forId="ws-frames" scrubber={scrubber(s.frameCount, SPEC.frames, (v) => draft.update('frameCount', v))} changed={changed('frameCount')} onreset={() => reset('frameCount')}>
          <InspectorNumber id="ws-frames" name="frames" value={s.frameCount} {...SPEC.frames} onchange={(v) => v !== null && draft.update('frameCount', v)} />
        </InspectorRow>
      {/if}
      {#if caps.showSteps}
        <InspectorRow label="Steps" forId="ws-steps" scrubber={scrubber(s.steps, SPEC.steps, (v) => draft.update('steps', v))} changed={changed('steps')} onreset={() => reset('steps')}>
          <InspectorNumber id="ws-steps" name="steps" value={s.steps} {...SPEC.steps} onchange={(v) => v !== null && draft.update('steps', v)} />
          <input class="mini-range accent-primary-main" type="range" tabindex="-1" aria-hidden="true" {...SPEC.steps} value={s.steps} oninput={(e) => draft.update('steps', Number(e.currentTarget.value))}>
        </InspectorRow>
      {/if}
      {#if caps.showGuidance}
        <InspectorRow label="Guidance" forId="ws-guidance" scrubber={scrubber(s.guidance, SPEC.guidance, (v) => draft.update('guidance', v))} changed={changed('guidance')} onreset={() => reset('guidance')}>
          <InspectorNumber id="ws-guidance" name="guidance" value={s.guidance} {...SPEC.guidance} onchange={(v) => v !== null && draft.update('guidance', v)} />
          <input class="mini-range accent-primary-main" type="range" tabindex="-1" aria-hidden="true" {...SPEC.guidance} value={s.guidance} oninput={(e) => draft.update('guidance', Number(e.currentTarget.value))}>
        </InspectorRow>
      {/if}
      {#if caps.showI2IStrength}
        <InspectorRow label="Img strength" forId="ws-image-strength" scrubber={scrubber(s.referenceImageStrength, SPEC.strength, (v) => draft.update('referenceImageStrength', v))} changed={changed('referenceImageStrength')} onreset={() => reset('referenceImageStrength')}>
          <InspectorNumber id="ws-image-strength" name="image_strength" value={s.referenceImageStrength} {...SPEC.strength} onchange={(v) => v !== null && draft.update('referenceImageStrength', v)} />
          <input class="mini-range accent-primary-main" type="range" tabindex="-1" aria-hidden="true" {...SPEC.strength} value={s.referenceImageStrength} oninput={(e) => draft.update('referenceImageStrength', Number(e.currentTarget.value))}>
        </InspectorRow>
      {/if}
      {#if caps.showSeed}
        <InspectorRow label="Seed" forId="ws-seed" changed={changed('seed')} onreset={() => reset('seed')}>
          <InspectorNumber id="ws-seed" name="seed" value={s.seed} {...SPEC.seed} nullable placeholder="Random each run" onchange={(v) => draft.update('seed', v)} />
          <button
            type="button"
            class="ui-btn ui-btn-row"
            aria-pressed={s.seed !== null}
            aria-label={s.seed !== null ? 'Unlock seed (random each run)' : 'Lock seed (reuse it every run)'}
            title={s.seed !== null ? 'Seed locked: click for a random seed each run' : 'Random each run: click to lock'}
            onclick={toggleSeedLock}
          ><Icon name={s.seed !== null ? 'lock' : 'unlock'} size={14} /></button>
          <button type="button" class="ui-btn ui-btn-row" aria-label="Pick a new random seed" title="New random seed" onclick={() => draft.update('seed', randomSeed())}>
            <Icon name="dice" size={14} />
          </button>
        </InspectorRow>
      {/if}
      {#if caps.showScheduler}
        <InspectorRow label="Scheduler" forId="ws-scheduler" changed={changed('scheduler')} onreset={() => reset('scheduler')}>
          <select
            id="ws-scheduler"
            name="scheduler"
            class="inspector-select"
            value={s.scheduler ?? ''}
            onchange={(e) => draft.update('scheduler', e.currentTarget.value || null)}
          >
            <option value="">Auto (model default)</option>
            {#each context.scheduler_options as option (option)}
              <option value={option}>{option}</option>
            {/each}
          </select>
        </InspectorRow>
      {/if}
      {#if caps.supportsFirstSigma}
        <InspectorRow label="First sigma" forId="ws-first-sigma" scrubber={scrubber(s.firstSigma, SPEC.firstSigma, (v) => draft.update('firstSigma', v))} changed={changed('firstSigma')} onreset={() => reset('firstSigma')}>
          <InspectorNumber id="ws-first-sigma" value={s.firstSigma} {...SPEC.firstSigma} nullable placeholder="1.004" onchange={(v) => draft.update('firstSigma', v)} />
          {#if s.firstSigma !== null}
            <input type="hidden" name="first_sigma" value={String(s.firstSigma)}>
          {/if}
        </InspectorRow>
        <p class="row-hint">Best-effort: a smaller first-step sigma may reduce the grey "blocked by safety filter" frame. Not guaranteed.</p>
      {/if}
    </InspectorSection>

    {#if caps.showRefImage}
      <InspectorSection id="reference" title="Reference image" summary={referenceName || 'None'} active={Boolean(referenceName)}>
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div
          class="reference-drop"
          data-dragover={dragOver}
          ondragover={(e) => { e.preventDefault(); dragOver = true; }}
          ondragleave={() => { dragOver = false; }}
          ondrop={handleDrop}
        >
          <InspectorRow label="Image">
            <div class="reference-thumb">
              {#if referencePreview}
                <img src={referencePreview} alt="Reference preview">
              {:else}
                <Icon name="reference" size={16} />
              {/if}
            </div>
            <span class="reference-name" title={referenceName}>{referenceName || 'Drop or browse'}</span>
            <!-- No name: the chosen file is attached on submit only while it is the reference (see WorkspacePage). -->
            <input id="ws-image-file" bind:this={fileInput} type="file" accept="image/png,image/jpeg,image/webp" class="hidden" onchange={(e) => onImageFileChange(e.currentTarget.files?.[0] ?? null)}>
            <label for="ws-image-file" class="ui-btn ui-btn-row" title="Browse for an image" aria-label="Browse for a reference image"><Icon name="plus" size={14} /></label>
            <button type="button" class="ui-btn ui-btn-row" aria-label="Clear reference image" title="Clear" onclick={clearImage}><Icon name="close" size={14} /></button>
          </InspectorRow>
          <InspectorRow label="Path" forId="ws-image-path">
            <input
              id="ws-image-path"
              name="image_path"
              type="text"
              class="inspector-text"
              placeholder="/path/to/reference.png"
              value={s.referenceImagePath ?? ''}
              oninput={(e) => draft.update('referenceImagePath', e.currentTarget.value || null)}
            >
          </InspectorRow>
        </div>
      </InspectorSection>
    {/if}

    {#if showPostprocess}
      <InspectorSection id="postprocess" title="Post-processing" summary={postprocessOn ? `${postprocessOn} on` : 'Off'} active={postprocessOn > 0}>
        {#if caps.showPostprocessSharpen}
          <InspectorRow
            label="Sharpen"
            forId="ws-pp-sharpen-amount"
            scrubber={scrubber(s.postprocessSharpenAmount, SPEC.sharpen, (v) => draft.update('postprocessSharpenAmount', v), !s.postprocessSharpenEnabled)}
            changed={changed('postprocessSharpenEnabled', 'postprocessSharpenAmount')}
            onreset={() => reset('postprocessSharpenEnabled', 'postprocessSharpenAmount')}
          >
            <Toggle id="ws-pp-sharpen" ariaLabel="Sharpen" checked={s.postprocessSharpenEnabled} disabled={busy} onchange={(e) => draft.update('postprocessSharpenEnabled', (e.currentTarget as HTMLInputElement).checked)} />
            <InspectorNumber id="ws-pp-sharpen-amount" ariaLabel="Sharpen amount" value={s.postprocessSharpenAmount} {...SPEC.sharpen} nullable placeholder="auto" disabled={!s.postprocessSharpenEnabled} onchange={(v) => draft.update('postprocessSharpenAmount', v)} />
          </InspectorRow>
          <input type="hidden" name="sharpen_enabled" value={s.postprocessSharpenEnabled ? 'true' : 'false'}>
          {#if s.postprocessSharpenEnabled && s.postprocessSharpenAmount !== null}
            <input type="hidden" name="sharpen_amount" value={String(s.postprocessSharpenAmount)}>
          {/if}
        {/if}
        {#if caps.showPostprocessContrast}
          <InspectorRow
            label="Contrast"
            forId="ws-pp-contrast-amount"
            scrubber={scrubber(s.postprocessContrastAmount, SPEC.contrast, (v) => draft.update('postprocessContrastAmount', v), !s.postprocessContrastEnabled)}
            changed={changed('postprocessContrastEnabled', 'postprocessContrastAmount')}
            onreset={() => reset('postprocessContrastEnabled', 'postprocessContrastAmount')}
          >
            <Toggle id="ws-pp-contrast" ariaLabel="Contrast" checked={s.postprocessContrastEnabled} disabled={busy} onchange={(e) => draft.update('postprocessContrastEnabled', (e.currentTarget as HTMLInputElement).checked)} />
            <InspectorNumber id="ws-pp-contrast-amount" ariaLabel="Contrast amount" value={s.postprocessContrastAmount} {...SPEC.contrast} disabled={!s.postprocessContrastEnabled} onchange={(v) => v !== null && draft.update('postprocessContrastAmount', v)} />
          </InspectorRow>
          <input type="hidden" name="contrast_enabled" value={s.postprocessContrastEnabled ? 'true' : 'false'}>
          {#if s.postprocessContrastEnabled}
            <input type="hidden" name="contrast_amount" value={String(s.postprocessContrastAmount)}>
          {/if}
        {/if}
        {#if caps.showPostprocessSaturation}
          <InspectorRow
            label="Saturation"
            forId="ws-pp-saturation-amount"
            scrubber={scrubber(s.postprocessSaturationAmount, SPEC.saturation, (v) => draft.update('postprocessSaturationAmount', v), !s.postprocessSaturationEnabled)}
            changed={changed('postprocessSaturationEnabled', 'postprocessSaturationAmount')}
            onreset={() => reset('postprocessSaturationEnabled', 'postprocessSaturationAmount')}
          >
            <Toggle id="ws-pp-saturation" ariaLabel="Saturation" checked={s.postprocessSaturationEnabled} disabled={busy} onchange={(e) => draft.update('postprocessSaturationEnabled', (e.currentTarget as HTMLInputElement).checked)} />
            <InspectorNumber id="ws-pp-saturation-amount" ariaLabel="Saturation amount" value={s.postprocessSaturationAmount} {...SPEC.saturation} disabled={!s.postprocessSaturationEnabled} onchange={(v) => v !== null && draft.update('postprocessSaturationAmount', v)} />
          </InspectorRow>
          <input type="hidden" name="saturation_enabled" value={s.postprocessSaturationEnabled ? 'true' : 'false'}>
          {#if s.postprocessSaturationEnabled}
            <input type="hidden" name="saturation_amount" value={String(s.postprocessSaturationAmount)}>
          {/if}
        {/if}
      </InspectorSection>
    {/if}

    {#if caps.showImageUpscale}
      <InspectorSection id="upscale" title="Upscale" summary={s.upscaleEnabled ? `${s.upscaleFactor}×` : 'Off'} active={s.upscaleEnabled}>
        <InspectorRow label="Upscale" changed={changed('upscaleEnabled', 'upscaleFactor')} onreset={() => reset('upscaleEnabled', 'upscaleFactor')}>
          <Toggle id="ws-upscale-enabled" ariaLabel="Enable upscale" checked={s.upscaleEnabled} disabled={busy} onchange={(e) => draft.update('upscaleEnabled', (e.currentTarget as HTMLInputElement).checked)} />
          <div class="ui-segmented ui-segmented-sm ui-segmented-mono mini-seg factor" role="group" aria-label="Upscale factor">
            {#each [2, 4] as factor (factor)}
              <button type="button" aria-pressed={s.upscaleFactor === factor} disabled={!s.upscaleEnabled} onclick={() => draft.update('upscaleFactor', factor)}>{factor}×</button>
            {/each}
          </div>
          {#if s.upscaleEnabled}
            <input type="hidden" name="upscale" value={String(s.upscaleFactor)}>
          {/if}
        </InspectorRow>
        {#if s.upscaleEnabled}
          {#if caps.showUpscaleDenoise}
            <InspectorRow sub label="Denoise" forId="ws-upscale-denoise" scrubber={scrubber(s.upscaleDenoise, SPEC.upscaleDenoise, (v) => draft.update('upscaleDenoise', v))} changed={changed('upscaleDenoise')} onreset={() => reset('upscaleDenoise')}>
              <InspectorNumber id="ws-upscale-denoise" name="upscale_denoise" value={s.upscaleDenoise} {...SPEC.upscaleDenoise} nullable placeholder="auto" onchange={(v) => draft.update('upscaleDenoise', v)} />
            </InspectorRow>
          {/if}
          {#if caps.showUpscaleSteps}
            <InspectorRow sub label="Steps" forId="ws-upscale-steps" scrubber={scrubber(s.upscaleSteps, SPEC.upscaleSteps, (v) => draft.update('upscaleSteps', v))} changed={changed('upscaleSteps')} onreset={() => reset('upscaleSteps')}>
              <InspectorNumber id="ws-upscale-steps" name="upscale_steps" value={s.upscaleSteps} {...SPEC.upscaleSteps} nullable placeholder="auto" onchange={(v) => draft.update('upscaleSteps', v)} />
            </InspectorRow>
          {/if}
          {#if caps.showUpscaleGuidance}
            <InspectorRow sub label="Guidance" forId="ws-upscale-guidance" scrubber={scrubber(s.upscaleGuidance, SPEC.upscaleGuidance, (v) => draft.update('upscaleGuidance', v))} changed={changed('upscaleGuidance')} onreset={() => reset('upscaleGuidance')}>
              <InspectorNumber id="ws-upscale-guidance" name="upscale_guidance" value={s.upscaleGuidance} {...SPEC.upscaleGuidance} nullable placeholder="auto" onchange={(v) => draft.update('upscaleGuidance', v)} />
            </InspectorRow>
          {/if}
          {#if caps.showUpscaleSharpen}
            <InspectorRow sub label="Sharpen" changed={changed('upscaleSharpen')} onreset={() => reset('upscaleSharpen')}>
              <Toggle id="ws-upscale-sharpen" ariaLabel="Upscale sharpen" checked={s.upscaleSharpen} disabled={busy} onchange={(e) => draft.update('upscaleSharpen', (e.currentTarget as HTMLInputElement).checked)} />
              <input type="hidden" name="upscale_sharpen" value={s.upscaleSharpen ? 'true' : 'false'}>
            </InspectorRow>
          {/if}
        {/if}
      </InspectorSection>
    {/if}

    {#if showVideo}
      <InspectorSection id="video" title="Video" summary={videoSummary} active={videoSummary !== 'Off'}>
        {#if caps.showAudio}
          <InspectorRow label="Audio" changed={changed('audio')} onreset={() => reset('audio')}>
            <Toggle id="ws-audio" name="audio" ariaLabel="Audio" checked={s.audio} onchange={(e) => draft.update('audio', (e.currentTarget as HTMLInputElement).checked)} />
            <input type="hidden" name="audio" value="false">
          </InspectorRow>
        {/if}
        {#if caps.showLowMemory}
          <InspectorRow label="Low memory" changed={changed('lowMemory')} onreset={() => reset('lowMemory')}>
            <Toggle id="ws-low-memory" name="low_memory" ariaLabel="Low memory mode" checked={s.lowMemory} onchange={(e) => draft.update('lowMemory', (e.currentTarget as HTMLInputElement).checked)} />
            <input type="hidden" name="low_memory" value="false">
          </InspectorRow>
        {/if}
        {#if caps.showVideoUpscale}
          <InspectorRow label="Upscale" changed={changed('videoUpscaleEnabled', 'videoUpscaleFactor')} onreset={() => reset('videoUpscaleEnabled', 'videoUpscaleFactor')}>
            <Toggle id="ws-video-upscale" ariaLabel="Video upscale" checked={s.videoUpscaleEnabled} disabled={busy} onchange={(e) => draft.update('videoUpscaleEnabled', (e.currentTarget as HTMLInputElement).checked)} />
            {#if s.videoUpscaleEnabled}
              {#if caps.showVideoUpscaleFactor}
                <select
                  id="ws-video-upscale-factor"
                  class="inspector-select factor"
                  aria-label="Video upscale factor"
                  value={String(s.videoUpscaleFactor)}
                  onchange={(e) => draft.update('videoUpscaleFactor', Number(e.currentTarget.value))}
                >
                  <option value="2">2×</option>
                </select>
              {/if}
              <input type="hidden" name="upscale" value={String(s.videoUpscaleFactor)}>
              <input type="hidden" name="video_upscale_factor" value={String(s.videoUpscaleFactor)}>
            {/if}
          </InspectorRow>
        {/if}
      </InspectorSection>
    {/if}
  </div>

  <div class="ui-pane-footer settings-footer">
    {#if jobsActive}
      <p id="ws-busy-note" class="ui-alert ui-alert-info footer-note">
        A job is running. New runs join the queue.
      </p>
    {/if}
    {#if promptFileMode && s.promptFileOptionIds.length === 0}
      <p class="ui-alert ui-alert-warning footer-note">{context.prompt_file.help.option_required}</p>
    {/if}
    <ActionBar>
      <button
        id="ws-reset"
        type="button"
        disabled={busy}
        onclick={resetAll}
        title="Reset settings to model defaults (keeps model, LoRAs, quantization, and prompt)"
        aria-label="Reset settings"
        class="ui-btn"
      >
        <Icon name="reset" size={14} />Reset
        {#if changedRows > 0}<span class="reset-count" data-testid="changed-count">{changedRows}</span>{/if}
      </button>
      <button
        id="ws-submit"
        type="submit"
        disabled={submitDisabled}
        class="ui-btn ui-btn-primary ui-btn-main"
      >
        <Icon name="bolt" size={16} />
        <span>{busy ? 'Submitting…' : jobsActive ? 'Add to queue' : 'Generate'}</span>
        <kbd>⌘↵</kbd>
      </button>
    </ActionBar>
  </div>
</section>

<style>
  .settings-pane { display: flex; flex: 1; min-height: 0; flex-direction: column; }
  .settings-head { display: flex; flex: none; align-items: center; justify-content: space-between; gap: 8px; padding: 10px 12px 8px; border-bottom: 1px solid var(--color-border-subtle); }
  .settings-hint { font-size: var(--text-meta); color: var(--color-text-muted); }
  .settings-scroll { flex: 1; min-height: 0; overflow-y: auto; }

  /* Presets fill the value column; each option takes an equal share. */
  .mini-seg { display: flex; flex: 1; }
  .mini-seg.factor { flex: none; width: 90px; }
  .mini-seg > button { flex: 1; min-width: 0; padding: 0 2px; }
  .mini-range { flex: none; width: 90px; }
  .dim-times { font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }

  .inspector-select, .inspector-text { width: 100%; min-width: 0; height: 24px; padding: 0 6px; border: 1px solid transparent; border-radius: 4px; background: transparent; font-size: var(--text-ui); color: var(--color-text-primary); }
  .inspector-text { font-family: var(--font-mono); }
  .inspector-select.factor { flex: none; width: 70px; }
  .inspector-select:hover, .inspector-text:hover { border-color: var(--color-border-strong); background: var(--color-bg-surface); }
  .inspector-select:focus, .inspector-text:focus { outline: 2px solid var(--color-primary-main); outline-offset: 2px; background: var(--color-bg-surface); }
  .inspector-select option { background: var(--color-bg-surface); }
  .row-hint { padding: 0 12px 6px 116px; font-size: var(--text-meta); color: var(--color-text-muted); }

  .reference-drop { border-radius: var(--radius-sm); }
  .reference-drop[data-dragover='true'] { outline: 2px dashed var(--color-primary-main); outline-offset: -2px; }
  .reference-thumb { display: grid; flex: none; place-items: center; width: 40px; height: 40px; margin: 4px 0; overflow: hidden; border: 1px solid var(--color-border-strong); border-radius: var(--radius-sm); background: var(--color-bg-surface); color: var(--color-text-muted); }
  .reference-thumb img { width: 100%; height: 100%; object-fit: cover; }
  .reference-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-ui); }

  .settings-footer { flex: none; padding: 10px 12px; }
  .footer-note { margin-bottom: 8px; }
  .reset-count { font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-primary-main); }
</style>
