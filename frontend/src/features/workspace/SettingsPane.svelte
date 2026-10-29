<script lang="ts">
  import { draft, extendStepFor, offeredSizes, settingDefaultsFor, snapSteps, stepOptionsFor, videoTimingFor, type SettingKey } from '$lib/state/draft.svelte';
  import { rangeFill, SPEC, stepsSpec } from './settingsSpec';
  import { Icon, InspectorNumber, Toggle } from '$lib/components/atoms';
  import { ActionBar, InspectorRow, InspectorSection, Select } from '$lib/components/molecules';
  import { fileName } from '$lib/components/molecules/assetDetails';
  import { extendGuard, extendModelSupport } from '$lib/state/extend';
  import { clipMemory } from '$lib/state/clipMemory.svelte';
  import { MEMORY_FIT_CONSEQUENCES, clipTimeNote, memoryFitNote, type MemoryNote } from '$lib/state/memoryFit';
  import { extendInspection } from '$lib/state/extendInspection.svelte';
  import type { NumberSpec, ScrubOptions } from '$lib/actions/scrub';
  import type { WorkspaceContext } from '$lib/types';
  import { workspaceCapabilities } from './capabilities';
  import { randomSeed } from './seed';
  import ReferenceSlot from './ReferenceSlot.svelte';
  import ReferencesSection from './ReferencesSection.svelte';
  import SourceVideoSection from './SourceVideoSection.svelte';
  import { defaultExtendFrames, durationSpec, extendFramesFromSeconds, extendTotal, framesToSeconds, roundSeconds, secondsToFrames } from '$lib/state/videoDuration';

  interface Props {
    context: WorkspaceContext;
    /** A submit is in flight. */
    busy: boolean;
    /** A job is running or queued: Generate adds to the queue. */
    jobsActive?: boolean;
    /** Why the server refused the last Generate, shown above the button. */
    submitError?: string | null;
    imageFile: File | null;
    /** Preview for a reference path that points at a known asset. */
    referencePreviewUrl: string | null;
    /** Seed of the most recent output, reused when the seed gets locked. */
    lastSeed: number | null;
    onImageFileChange: (file: File | null) => void;
    /** An image uploaded as the last frame (Image to Video, on a model that has one). */
    endImageFile?: File | null;
    /** Preview for a last-frame path that points at a known asset. */
    endPreviewUrl?: string | null;
    onEndImageFileChange?: (file: File | null) => void;
    /** The images uploaded as references, by row (Text to Image and Image to Image, on a model that takes references). */
    referenceFiles?: (File | null)[];
    /** Why a reference blocks Generate (a row whose image is gone), computed once by the Workspace; null when none does. */
    referenceBlock?: string | null;
    onReferenceFilesChange?: (files: (File | null)[]) => void;
    /** Copy the CLI command for the current settings; the button is hidden without it. */
    oncopycli?: () => void;
    /** Off: the footer closes the pane (`ondone`) instead of generating, where another button owns Generate. */
    submit?: boolean;
    ondone?: () => void;
  }

  let { context, busy, jobsActive = false, submitError = null, imageFile, referencePreviewUrl, lastSeed, onImageFileChange, endImageFile = null, endPreviewUrl = null, onEndImageFileChange = () => {}, referenceFiles = [], referenceBlock = null, onReferenceFilesChange = () => {}, oncopycli, submit = true, ondone }: Props = $props();

  const s = $derived(draft.state);
  const caps = $derived(workspaceCapabilities(context, s));
  // Narrow dependencies, so typing in the prompt does not re-resolve the defaults.
  const workflow = $derived(s.workflow);
  const model = $derived(s.model);
  const defaults = $derived(settingDefaultsFor(context, { workflow, model }));

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
  // --- Duration -------------------------------------------------------------
  // The clip length is edited in seconds but submitted as frames, snapped to a count the model accepts.
  const timing = $derived(videoTimingFor(context, model));
  const secondsSpec = $derived(durationSpec(timing.fps, timing.frameRule));
  const shownSeconds = $derived(roundSeconds(framesToSeconds(s.frameCount, timing.fps)));

  function setSeconds(seconds: number | null): void {
    if (seconds !== null) draft.update('frameCount', secondsToFrames(seconds, timing.fps, timing.frameRule));
  }

  // --- Steps: a model that offers only some counts (4 or 8) steps between them ---
  const stepOptions = $derived(caps.isImageMode ? null : stepOptionsFor(context, model));
  const stepsRange = $derived(stepOptions ? stepsSpec(stepOptions) : SPEC.steps);

  function setSteps(steps: number): void {
    draft.update('steps', stepOptions ? snapSteps(steps, stepOptions) : steps);
  }

  // --- Extend: the length of the new frames (the source itself is chosen in SourceVideoSection) ---
  const info = $derived(draft.extendSource);
  const sourceStatus = $derived(extendInspection.statusFor(s.extendSourcePath));
  const sourceGuard = $derived(extendGuard(s, info, sourceStatus, extendModelSupport(context, model)));
  const sourceFps = $derived(info?.fps ?? null);
  const silentSource = $derived(s.workflow === 'extend' && info?.has_audio === false);
  const extendStep = $derived(extendStepFor(context, model));
  const extendSecondsSpec = $derived(sourceFps ? durationSpec(sourceFps, { alignment: extendStep }) : { step: 0.1, min: 0.1 });
  const shownExtendSeconds = $derived(sourceFps && s.extendFrames !== null ? roundSeconds(s.extendFrames / sourceFps) : null);
  const extendLength = $derived(info && sourceFps && info.frame_count !== null && s.extendFrames !== null ? extendTotal(info.frame_count, s.extendFrames, sourceFps) : null);
  const extendDefaultFrames = $derived(sourceFps ? defaultExtendFrames(sourceFps, extendStep) : null);
  const memoryNote = $derived(memoryFitNote(info?.memory_fit ?? null, 'extension'));
  const clipNote = $derived(memoryFitNote(clipMemory.fit, 'clip'));
  const clipTime = $derived(clipTimeNote(clipMemory.fit));
  const extendTime = $derived(clipTimeNote(info?.memory_fit ?? null));

  function setExtendSeconds(seconds: number | null): void {
    if (seconds !== null && sourceFps) draft.update('extendFrames', extendFramesFromSeconds(seconds, sourceFps, extendStep));
  }

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
      caps.showExtendLength && extendDefaultFrames !== null && s.extendFrames !== extendDefaultFrames,
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
      caps.showUpscaleReview && s.upscaleEnabled && changed('upscaleReview'),
      caps.showUpscaleDenoise && s.upscaleEnabled && changed('upscaleDenoise'),
      caps.showUpscaleSteps && s.upscaleEnabled && changed('upscaleSteps'),
      caps.showUpscaleGuidance && s.upscaleEnabled && changed('upscaleGuidance'),
      caps.showUpscaleSharpen && s.upscaleEnabled && changed('upscaleSharpen'),
      caps.showAudio && changed('audio'),
      caps.showLowMemory && changed('lowMemory'),
      caps.showVideoUpscale && changed('videoUpscaleEnabled', 'videoUpscaleFactor'),
      caps.showVideoUpscaleReview && s.videoUpscaleEnabled && changed('videoUpscaleReview'),
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
  const upscaleReviewOn = $derived(caps.showUpscaleReview && s.upscaleEnabled && s.upscaleReview);
  const upscaleSummary = $derived(
    s.upscaleEnabled ? [`${s.upscaleFactor}×`, upscaleReviewOn ? 'Review' : null].filter(Boolean).join(' · ') : 'Off'
  );
  const showVideo = $derived(caps.showAudio || caps.showLowMemory || caps.showVideoUpscale);
  const videoUpscaleReviewOn = $derived(caps.showVideoUpscaleReview && s.videoUpscaleEnabled && s.videoUpscaleReview);
  const videoSummary = $derived(
    [
      s.audio && !silentSource ? 'Audio' : null,
      caps.showLowMemory && s.lowMemory ? 'Low memory' : null,
      caps.showVideoUpscale && s.videoUpscaleEnabled ? `${s.videoUpscaleFactor}× upscale` : null,
      videoUpscaleReviewOn ? 'Review' : null,
    ]
      .filter(Boolean).join(' · ') || 'Off'
  );

  // --- Seed -----------------------------------------------------------------
  // A locked seed repeats every run; unlocked (empty) picks a new random seed each run.
  function toggleSeedLock(): void {
    draft.update('seed', s.seed !== null ? null : (lastSeed ?? randomSeed()));
  }

  // --- Start image ----------------------------------------------------------
  const referenceName = $derived(imageFile?.name ?? (s.referenceImagePath ? fileName(s.referenceImagePath) : ''));
  const endName = $derived(endImageFile?.name ?? (s.endImagePath ? fileName(s.endImagePath) : ''));
  // The summary reads "start.png → end.png" once both are set, and the first alone (or None) otherwise.
  const referenceSummary = $derived(caps.showEndImage && endName ? `${referenceName || 'None'} → ${endName}` : referenceName || 'None');

  // --- Footer ---------------------------------------------------------------
  const promptFileMode = $derived(s.promptSource === 'file');
  const promptChoiceMissing = $derived(promptFileMode && s.promptFileOptionIds.length === 0);
  const submitDisabled = $derived(busy || (promptFileMode && (!s.promptFilePath || s.promptFileOptionIds.length === 0)) || sourceGuard.blocked || referenceBlock !== null);
  // One reason at a time, linked to Generate: the source comes first, then the references, then the prompt file.
  const generateReason = $derived(sourceGuard.note ?? referenceBlock ?? (promptChoiceMissing ? context.prompt_file.help.option_required : null));

  function resetAll(): void {
    draft.resetSelections(context);
    onImageFileChange(null);
    onEndImageFileChange(null);
    onReferenceFilesChange([]);
  }

  function scrubber(value: number | null, spec: NumberSpec, onchange: (value: number) => void, disabled = false): ScrubOptions {
    return { ...spec, value, onchange, disabled: disabled || busy };
  }
</script>

{#snippet memoryNoteLine(note: MemoryNote, testid: string)}
  <p class="row-hint ui-help-warning memory-note" data-testid={testid} data-status={note.status} title="{note.text}&#10;{MEMORY_FIT_CONSEQUENCES.unified[note.status]}">
    <Icon name="alert" size={12} />{note.text}
  </p>
{/snippet}

{#snippet timeNoteLine(text: string, testid: string)}
  <p class="row-hint" data-testid={testid}>{text}</p>
{/snippet}

<section class="settings-pane" aria-labelledby="ws-settings-title">
  <div class="settings-head">
    <h2 id="ws-settings-title" class="ui-area-label">Settings</h2>
    <div class="settings-head-end">
      <span class="settings-hint" title="Drag a label left or right to change its value; arrow keys nudge (Shift ×10)">Drag labels to scrub</span>
      {#if oncopycli}
        <button
          id="ws-copy-cli"
          type="button"
          class="ui-btn ui-btn-sm ui-btn-icon"
          title="Copy these settings as a ziv CLI command"
          aria-label="Copy CLI command"
          onclick={oncopycli}
        ><Icon name="terminal" size={14} /></button>
      {/if}
    </div>
  </div>

  <div class="settings-scroll">
    {#if caps.showSourceVideo}
      <SourceVideoSection {context} {busy} guard={sourceGuard} />
    {/if}

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
          <InspectorNumber id="ws-width" name={dimensionMode === 'custom' ? 'width' : undefined} ariaLabel="Width" value={shownWidth} {...dimensionSpec} stepper={false} onchange={(v) => setDimension('width', v)} />
          <span class="dim-times" aria-hidden="true">×</span>
          <InspectorNumber id="ws-height" name={dimensionMode === 'custom' ? 'height' : undefined} ariaLabel="Height" value={shownHeight} {...dimensionSpec} stepper={false} onchange={(v) => setDimension('height', v)} />
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
        <InspectorRow label="Duration (s)" forId="ws-duration" scrubber={scrubber(shownSeconds, secondsSpec, setSeconds)} changed={changed('frameCount')} onreset={() => reset('frameCount')}>
          <InspectorNumber id="ws-duration" value={shownSeconds} {...secondsSpec} commitOnly onchange={setSeconds} />
          <span class="frame-count">{s.frameCount} frames</span>
          <input type="hidden" name="frames" value={s.frameCount}>
        </InspectorRow>
        <div aria-live="polite">
          {#if clipTime}{@render timeNoteLine(clipTime, 'clip-time-note')}{/if}
          {#if clipNote}{@render memoryNoteLine(clipNote, 'clip-memory-note')}{/if}
        </div>
      {/if}
      {#if caps.showExtendLength}
        <InspectorRow
          label="Extend by (s)"
          forId="ws-extend-seconds"
          scrubber={scrubber(shownExtendSeconds, extendSecondsSpec, setExtendSeconds, sourceFps === null)}
          changed={extendDefaultFrames !== null && s.extendFrames !== extendDefaultFrames}
          onreset={() => extendDefaultFrames !== null && draft.update('extendFrames', extendDefaultFrames)}
        >
          <InspectorNumber id="ws-extend-seconds" value={shownExtendSeconds} {...extendSecondsSpec} disabled={sourceFps === null} title={sourceFps === null ? 'Choose a video to extend first.' : undefined} commitOnly onchange={setExtendSeconds} />
          <span class="frame-count" data-testid="extend-frames">{sourceFps !== null && s.extendFrames !== null ? `+${s.extendFrames} frames` : '—'}</span>
          {#if s.extendFrames !== null}<input type="hidden" name="extend_frames" value={s.extendFrames}>{/if}
        </InspectorRow>
        <InspectorRow label="Total">
          <span class="extend-total" data-testid="extend-total">{extendLength ? `${extendLength.seconds.toFixed(2)} s · ${extendLength.frames} frames` : '—'}</span>
        </InspectorRow>
        <div aria-live="polite">
          {#if extendTime}{@render timeNoteLine(extendTime, 'extend-time-note')}{/if}
          {#if memoryNote}{@render memoryNoteLine(memoryNote, 'extend-memory-note')}{/if}
        </div>
      {/if}
      {#if caps.showSteps}
        <InspectorRow label="Steps" forId="ws-steps" scrubber={scrubber(s.steps, stepsRange, setSteps)} changed={changed('steps')} onreset={() => reset('steps')}>
          <InspectorNumber id="ws-steps" name="steps" value={s.steps} {...stepsRange} onchange={(v) => v !== null && setSteps(v)} />
          <input class="mini-range ui-range" type="range" tabindex="-1" aria-hidden="true" {...stepsRange} value={s.steps} style:--fill={rangeFill(s.steps, stepsRange)} oninput={(e) => setSteps(Number(e.currentTarget.value))}>
        </InspectorRow>
      {/if}
      {#if caps.showGuidance}
        <InspectorRow label="Guidance" forId="ws-guidance" scrubber={scrubber(s.guidance, SPEC.guidance, (v) => draft.update('guidance', v))} changed={changed('guidance')} onreset={() => reset('guidance')}>
          <InspectorNumber id="ws-guidance" name="guidance" value={s.guidance} {...SPEC.guidance} onchange={(v) => v !== null && draft.update('guidance', v)} />
          <input class="mini-range ui-range" type="range" tabindex="-1" aria-hidden="true" {...SPEC.guidance} value={s.guidance} style:--fill={rangeFill(s.guidance, SPEC.guidance)} oninput={(e) => draft.update('guidance', Number(e.currentTarget.value))}>
        </InspectorRow>
      {/if}
      {#if caps.showI2IStrength}
        <InspectorRow label="Img strength" forId="ws-image-strength" scrubber={scrubber(s.referenceImageStrength, SPEC.strength, (v) => draft.update('referenceImageStrength', v))} changed={changed('referenceImageStrength')} onreset={() => reset('referenceImageStrength')}>
          <InspectorNumber id="ws-image-strength" name="image_strength" value={s.referenceImageStrength} {...SPEC.strength} onchange={(v) => v !== null && draft.update('referenceImageStrength', v)} />
          <input class="mini-range ui-range" type="range" tabindex="-1" aria-hidden="true" {...SPEC.strength} value={s.referenceImageStrength} style:--fill={rangeFill(s.referenceImageStrength, SPEC.strength)} oninput={(e) => draft.update('referenceImageStrength', Number(e.currentTarget.value))}>
        </InspectorRow>
      {/if}
      {#if caps.showSeed}
        <InspectorRow label="Seed" forId="ws-seed" changed={changed('seed')} onreset={() => reset('seed')}>
          <InspectorNumber id="ws-seed" name="seed" value={s.seed} {...SPEC.seed} nullable placeholder="Random each run" stepper={false} onchange={(v) => draft.update('seed', v)} />
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
          <Select
            id="ws-scheduler"
            name="scheduler"
            variant="bare"
            class="flex-1"
            triggerClass="inspector-select"
            label="Scheduler"
            value={s.scheduler ?? ''}
            options={[{ value: '', label: 'Auto (model default)' }, ...context.scheduler_options.map((option) => ({ value: option, label: option }))]}
            onchange={(v) => draft.update('scheduler', v || null)}
          />
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
      <InspectorSection id="reference" title={caps.showEndImage ? 'Frames' : 'Start image'} summary={referenceSummary} active={Boolean(referenceName || (caps.showEndImage && endName))}>
        <ReferenceSlot
          idBase="ws-image"
          label={caps.showEndImage ? 'First frame' : 'Image'}
          names={{ browse: 'Browse for a start image', upload: 'Upload a start image from this device', clear: 'Clear start image', preview: 'Start image preview' }}
          pathName="image_path"
          placeholder="/path/to/start-image.png"
          emptyText="Browse or drop"
          path={s.referenceImagePath}
          file={imageFile}
          previewUrl={referencePreviewUrl}
          onfilechange={onImageFileChange}
          onpathchange={(path) => draft.update('referenceImagePath', path)}
        />
        {#if caps.showEndImage}
          <ReferenceSlot
            idBase="ws-end-image"
            label="Last frame"
            names={{ browse: 'Browse for a last-frame image', upload: 'Upload a last-frame image from this device', clear: 'Clear last frame', preview: 'Last frame preview' }}
            pathName="end_image_path"
            placeholder="/path/to/last-frame.png"
            emptyText="Optional"
            emptyMuted
            path={s.endImagePath}
            file={endImageFile}
            previewUrl={endPreviewUrl}
            onfilechange={onEndImageFileChange}
            onpathchange={(path) => draft.update('endImagePath', path)}
          />
          {#if endName}<p class="row-hint">The clip ends on this image; different scenes dissolve into each other.</p>{/if}
        {/if}
      </InspectorSection>
    {/if}

    {#if caps.showReferences}
      <ReferencesSection max={caps.maxReferences} files={referenceFiles} onfileschange={onReferenceFilesChange} />
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
      <InspectorSection id="upscale" title="Upscale" summary={upscaleSummary} active={s.upscaleEnabled}>
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
          {#if caps.showUpscaleReview}
            <InspectorRow sub label="Review" changed={changed('upscaleReview')} onreset={() => reset('upscaleReview')}>
              <Toggle
                id="ws-upscale-review"
                ariaLabel="Review before upscaling"
                title="Pauses after the small render. Approve to upscale it or Discard to skip it."
                checked={s.upscaleReview}
                disabled={busy}
                onchange={(e) => draft.update('upscaleReview', (e.currentTarget as HTMLInputElement).checked)}
              />
              {#if s.upscaleReview}
                <input type="hidden" name="review" value="true">
              {/if}
            </InspectorRow>
          {/if}
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
            <!-- A silent source gives a silent clip; the preference is kept and audio=false is what gets submitted. -->
            <Toggle
              id="ws-audio"
              name="audio"
              ariaLabel={silentSource ? 'Audio (the source has no audio)' : 'Audio'}
              title={silentSource ? 'Source has no audio' : undefined}
              checked={s.audio && !silentSource}
              disabled={silentSource}
              onchange={(e) => draft.update('audio', (e.currentTarget as HTMLInputElement).checked)}
            />
            {#if silentSource}<span class="frame-count">Source has no audio</span>{/if}
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
                <Select
                  id="ws-video-upscale-factor"
                  variant="bare"
                  class="factor"
                  triggerClass="inspector-select"
                  ariaLabel="Video upscale factor"
                  value={String(s.videoUpscaleFactor)}
                  options={[{ value: '2', label: '2×' }]}
                  onchange={(v) => draft.update('videoUpscaleFactor', Number(v))}
                />
              {/if}
              <input type="hidden" name="upscale" value={String(s.videoUpscaleFactor)}>
              <input type="hidden" name="video_upscale_factor" value={String(s.videoUpscaleFactor)}>
            {/if}
          </InspectorRow>
          {#if s.videoUpscaleEnabled && caps.showVideoUpscaleReview}
            <InspectorRow sub label="Review" changed={changed('videoUpscaleReview')} onreset={() => reset('videoUpscaleReview')}>
              <Toggle
                id="ws-video-upscale-review"
                ariaLabel="Review before upscaling"
                title="Pauses after the small render. Approve to upscale it or Discard to skip it. Sound may change slightly after upscaling."
                checked={s.videoUpscaleReview}
                disabled={busy}
                onchange={(e) => draft.update('videoUpscaleReview', (e.currentTarget as HTMLInputElement).checked)}
              />
              {#if s.videoUpscaleReview}
                <input type="hidden" name="review" value="true">
              {/if}
            </InspectorRow>
          {/if}
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
    {#if generateReason}
      <p id="ws-generate-reason" class="ui-alert ui-alert-warning footer-note">{generateReason}</p>
    {/if}
    {#if submitError}
      <p class="ui-alert ui-alert-error footer-note" role="alert" data-testid="generate-error">
        <Icon name="alert" size={14} /><span class="min-w-0 break-words">{submitError}</span>
      </p>
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
      {#if submit}
        <button
          id="ws-submit"
          type="submit"
          disabled={submitDisabled}
          aria-describedby={generateReason ? 'ws-generate-reason' : undefined}
          class="ui-btn ui-btn-primary ui-btn-main"
        >
          <Icon name="bolt" size={16} />
          <span>{busy ? 'Submitting…' : jobsActive ? 'Add to queue' : 'Generate'}</span>
          <kbd>⌘↵</kbd>
        </button>
      {:else}
        <button type="button" class="ui-btn ui-btn-main" onclick={ondone}>Done</button>
      {/if}
    </ActionBar>
  </div>
</section>

<style>
  .settings-pane { display: flex; flex: 1; min-height: 0; flex-direction: column; }
  .settings-head { display: flex; flex: none; align-items: center; justify-content: space-between; gap: 8px; padding: 10px 12px 8px; border-bottom: 1px solid var(--color-border-subtle); }
  .settings-head-end { display: flex; align-items: center; gap: 6px; }
  .settings-hint { font-size: var(--text-meta); color: var(--color-text-muted); }
  /* Touch: no scrubbing to hint at. */
  @media (pointer: coarse) {
    .settings-hint { display: none; }
    .mini-range { width: 72px; }
  }
  .settings-scroll { flex: 1; min-height: 0; overflow-y: auto; }

  /* Presets fill the value column; each option takes an equal share. */
  .mini-seg { display: flex; flex: 1; }
  .mini-seg.factor { flex: none; width: 90px; }
  .mini-seg > button { flex: 1; min-width: 0; padding: 0 2px; }
  .mini-range { flex: none; width: 90px; }
  .frame-count { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); white-space: nowrap; }
  .dim-times { font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }

  .settings-pane :global(.inspector-select) { height: 24px; padding: 0 6px; border: 1px solid transparent; border-radius: 4px; background: transparent; font-size: var(--text-ui); color: var(--color-text-primary); }
  .settings-pane :global(.ui-select.factor) { flex: none; width: 70px; }
  @media (hover: hover) { .settings-pane :global(.inspector-select:hover) { background: var(--color-bg-surface-hover); } }
  .settings-pane :global(.inspector-select:focus-visible), .settings-pane :global(.inspector-select[aria-expanded='true']) { outline: 2px solid var(--color-primary-main); outline-offset: 2px; background: var(--color-bg-surface); }
  /* Touch: selects fill like the steppers beside them. After the base rule, so it wins. */
  @media (pointer: coarse) {
    .settings-pane :global(.inspector-select) { height: 32px; padding: 0 8px; background: var(--color-bg-surface-hover); font-size: 14px; }
  }
  .memory-note { display: flex; align-items: flex-start; gap: 4px; }
  .memory-note :global(svg) { margin-top: 2px; }
  .extend-total { font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-text-secondary); }

  .settings-footer { flex: none; padding: 10px 12px; }
  .footer-note { margin-bottom: 8px; }
  .reset-count { font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-primary-main); }
</style>
