<script lang="ts">
  import { Icon, InspectorNumber, Spinner, Toggle } from '$lib/components/atoms';
  import { ModelStatusBadges, PathBrowser, PathInput, Segmented } from '$lib/components/molecules';
  import { fileName } from '$lib/components/molecules/assetDetails';
  import { draft, extendStepFor, offeredSizes, settingDefaultsFor, snapSteps, stepOptionsFor, videoTimingFor, type SettingKey } from '$lib/state/draft.svelte';
  import { downloadStatusText } from '$lib/state/downloadProgress';
  import { downloadStore } from '$lib/state/downloads.svelte';
  import { galleryStore } from '$lib/state/gallery.svelte';
  import { clipMemory } from '$lib/state/clipMemory.svelte';
  import { clipTimeNote, memoryFitNote } from '$lib/state/memoryFit';
  import { familyLabel, formatLoraString, groupLorasForModel, loraLabel, modelLoraFamily, parseLoraString, recentLoraNames } from '$lib/state/loras';
  import { extendGuard, extendModelSupport, extendSourceFacts } from '$lib/state/extend';
  import { extendInspection } from '$lib/state/extendInspection.svelte';
  import { defaultExtendFrames, durationSpec, extendFramesFromSeconds, extendTotal, framesToSeconds, roundSeconds, secondsToFrames } from '$lib/state/videoDuration';
  import type { ModelOption, WorkspaceContext } from '$lib/types';
  import { workspaceCapabilities } from './capabilities';
  import { randomSeed } from './seed';
  import { rangeFill, SPEC, stepsSpec } from './settingsSpec';
  import type { ChipKey } from './settingsChips';
  import SourceVideoFeedback from './SourceVideoFeedback.svelte';
  import { pushEscapeLayer } from '$lib/actions/escapeStack';
  import { sheetDrag } from '$lib/actions/sheetDrag';
  import { fly } from 'svelte/transition';

  /** What an image picker shows and does: the first frame and the last frame use the same one. */
  interface ImagePickerOptions {
    /** The chip that opens the picker; keys its file input. */
    slot: ChipKey;
    preview: string | null;
    previewAlt: string;
    name: string | null;
    emptyText: string;
    clearLabel: string;
    onclear: () => void;
    onfile: (file: File | null) => void;
    onbrowse: () => void;
    onlatest: () => void;
    pathId: string;
    placeholder: string;
    path: string | null;
    onpath: (text: string) => void;
    onpick: (path: string) => void;
    help?: string;
  }

  interface Props {
    key: ChipKey;
    context: WorkspaceContext;
    busy: boolean;
    /** The models the workflow can use, with their status. */
    models: ModelOption[];
    /** Quantization choices for the model, or null when it has none. */
    quantizeOptions: number[] | null;
    imageFile: File | null;
    referencePreviewUrl: string | null;
    lastSeed: number | null;
    onImageFileChange: (file: File | null) => void;
    /** The last frame: an uploaded image, the preview of a known path, and the changes the page keeps. */
    endImageFile?: File | null;
    endPreviewUrl?: string | null;
    onEndImageFileChange?: (file: File | null) => void;
    /** Make the latest output the last frame. */
    onUseLatestAsEndFrame?: () => void;
    /** Pick a model: the page re-hydrates the defaults for it. */
    onModelChange: (model: string) => void;
    /** Make the latest output the reference image. */
    onUseLatestAsReference: () => void;
    onclose: () => void;
  }

  let { key, context, busy, models, quantizeOptions, imageFile, referencePreviewUrl, lastSeed, onImageFileChange, endImageFile = null, endPreviewUrl = null, onEndImageFileChange = () => {}, onUseLatestAsEndFrame = () => {}, onModelChange, onUseLatestAsReference, onclose }: Props = $props();

  const s = $derived(draft.state);
  const caps = $derived(workspaceCapabilities(context, s));
  const defaults = $derived(settingDefaultsFor(context, { workflow: s.workflow, model: s.model }));
  const TITLES: Record<ChipKey, string> = {
    seconds: 'Clip length', extendSeconds: 'Extend by', steps: 'Steps', guidance: 'Guidance', strength: 'Image strength', seed: 'Seed', runs: 'Batch size',
    reference: 'Reference image', endReference: 'Last frame', source: 'Source video', size: 'Size', audio: 'Video', loras: 'LoRAs', upscale: 'Upscale', model: 'Model',
    // The Preset chip opens the preset menu, never a popover; the entry only completes the record.
    preset: 'Preset',
  };
  // With a last frame beside it, the reference image is the first frame.
  const title = $derived(key === 'reference' && caps.showEndImage ? 'First frame' : TITLES[key]);
  /** The settings a popover's Reset puts back; none for the ones that have no default (model, LoRAs, files). */
  const RESET_KEYS: Partial<Record<ChipKey, SettingKey[]>> = {
    seconds: ['frameCount'], steps: ['steps'], guidance: ['guidance'], strength: ['referenceImageStrength'], seed: ['seed'], runs: ['runs'],
    size: ['ratio', 'size', 'dimensionMode'], audio: ['audio', 'lowMemory', 'videoUpscaleEnabled', 'videoUpscaleReview'], upscale: ['upscaleEnabled', 'upscaleFactor', 'upscaleReview'],
  };
  const resetKeys = $derived(RESET_KEYS[key] ?? []);
  const canReset = $derived(resetKeys.some((k) => s[k] !== defaults[k]));

  function reset(): void {
    draft.patch(Object.fromEntries(resetKeys.map((k) => [k, defaults[k]])));
  }

  // Escape closes, as it does every other overlay, unless a dialog opened over the popover is the innermost one.
  $effect(() => pushEscapeLayer(() => onclose()));

  // --- Size ---
  const ratios = $derived(caps.isImageMode ? context.image_ratios : context.video_ratios);
  const sizeOptions = $derived(offeredSizes(context, s.workflow, s.model, s.ratio));
  const presetDims = $derived<[number, number]>(
    (caps.isImageMode ? context.image_size_dimensions : (context.video_size_dimensions ?? {}))[s.ratio]?.[s.size] ?? [s.width, s.height]
  );
  const shownWidth = $derived(s.dimensionMode === 'custom' ? s.width : presetDims[0]);
  const shownHeight = $derived(s.dimensionMode === 'custom' ? s.height : presetDims[1]);
  const dimensionSpec = $derived({ step: caps.dimensionStep, min: caps.dimensionMin, max: caps.dimensionMax ?? undefined });
  const SIZE_NAMES: Record<string, string> = { s: 'Small', m: 'Medium', l: 'Large', xl: 'Extra large' };

  function pickRatio(ratio: string): void {
    const valid = offeredSizes(context, s.workflow, s.model, ratio);
    draft.patch({ ratio, dimensionMode: 'ratio', ...(valid.includes(s.size) ? {} : { size: valid[0] ?? '' }) });
  }

  function setDimension(dim: 'width' | 'height', value: number | null): void {
    if (value === null) return;
    draft.patch({ width: shownWidth, height: shownHeight, [dim]: value, dimensionMode: 'custom' });
  }

  // --- Length ---
  const timing = $derived(videoTimingFor(context, s.model));
  const secondsSpec = $derived(durationSpec(timing.fps, timing.frameRule));
  const shownSeconds = $derived(roundSeconds(framesToSeconds(s.frameCount, timing.fps)));
  const clipNote = $derived(memoryFitNote(clipMemory.fit, 'clip'));
  const clipTime = $derived(clipTimeNote(clipMemory.fit));

  function setSeconds(seconds: number | null): void {
    if (seconds !== null) draft.update('frameCount', secondsToFrames(seconds, timing.fps, timing.frameRule));
  }

  // --- Steps: a model that offers only some counts (4 or 8) steps between them ---
  const stepOptions = $derived(caps.isImageMode ? null : stepOptionsFor(context, s.model));
  const stepsRange = $derived(stepOptions ? stepsSpec(stepOptions) : SPEC.steps);

  function setSteps(steps: number): void {
    draft.update('steps', stepOptions ? snapSteps(steps, stepOptions) : steps);
  }

  // --- Extend ---
  const info = $derived(draft.extendSource);
  const sourceFps = $derived(info?.fps ?? null);
  const extendStep = $derived(extendStepFor(context, s.model));
  const extendSecondsSpec = $derived(sourceFps ? durationSpec(sourceFps, { alignment: extendStep }) : { step: 0.1, min: 0.1 });
  const shownExtendSeconds = $derived(sourceFps && s.extendFrames !== null ? roundSeconds(s.extendFrames / sourceFps) : null);
  const extendLength = $derived(info && sourceFps && info.frame_count !== null && s.extendFrames !== null ? extendTotal(info.frame_count, s.extendFrames, sourceFps) : null);
  const extendDefault = $derived(sourceFps ? defaultExtendFrames(sourceFps, extendStep) : null);
  const extendMemoryNote = $derived(memoryFitNote(info?.memory_fit ?? null, 'extension'));
  const extendTime = $derived(clipTimeNote(info?.memory_fit ?? null));
  const sourceGuard = $derived(extendGuard(s, info, extendInspection.statusFor(s.extendSourcePath), extendModelSupport(context, s.model)));

  function setExtendSeconds(seconds: number | null): void {
    if (seconds !== null && sourceFps) draft.update('extendFrames', extendFramesFromSeconds(seconds, sourceFps, extendStep));
  }

  // --- Seed ---
  const latestSeed = $derived(galleryStore.assets[0]?.seed ?? null);

  // --- LoRAs ---
  const loraOptions = $derived(context.loras);
  const loraChips = $derived(parseLoraString(s.loraString, loraOptions));
  const addable = $derived(loraOptions.filter((lora) => !loraChips.some((chip) => chip.name === lora.name)));
  // The LoRAs that fit the model lead; the rest fold under "Other models", open from the start only when nothing else is offered.
  const loraModel = $derived(modelLoraFamily(context, s.model));
  const loraGroups = $derived(groupLorasForModel(addable, loraModel, recentLoraNames(galleryStore.assets, loraModel?.family ?? null, loraOptions)));
  let otherToggled = $state<boolean | null>(null);
  const otherOpen = $derived(otherToggled ?? loraGroups.main.length === 0);
  $effect(() => {
    void key;
    otherToggled = null;
  });

  function addLora(name: string): void {
    draft.update('loraString', formatLoraString([...loraChips, { name, weight: 1.0 }]));
  }

  function setLoraWeight(name: string, weight: number | null): void {
    if (weight === null) return;
    draft.update('loraString', formatLoraString(loraChips.map((chip) => (chip.name === name ? { name, weight } : chip))));
  }

  // --- Reference and last frame ---
  // Both pick an image the same ways; each has its own upload, preview and server browser.
  const referenceName = $derived(imageFile?.name ?? (s.referenceImagePath ? fileName(s.referenceImagePath) : null));
  const endName = $derived(endImageFile?.name ?? (s.endImagePath ? fileName(s.endImagePath) : null));
  let browsing = $state(false);
  let browsingEnd = $state(false);
  // The hidden file input of each picker, by the chip key that opens it.
  let fileInputs = $state<Partial<Record<ChipKey, HTMLInputElement | null>>>({});
  let filePreviewUrl = $state<string | null>(null);
  let endFilePreviewUrl = $state<string | null>(null);
  // Forget a browsed file once it stops being the image, so choosing it again fires a change.
  $effect(() => {
    const input = fileInputs.reference;
    if (!imageFile && input) input.value = '';
  });
  $effect(() => {
    const input = fileInputs.endReference;
    if (!endImageFile && input) input.value = '';
  });
  $effect(() => {
    const url = imageFile ? URL.createObjectURL(imageFile) : null;
    filePreviewUrl = url;
    return () => { if (url) URL.revokeObjectURL(url); };
  });
  $effect(() => {
    const url = endImageFile ? URL.createObjectURL(endImageFile) : null;
    endFilePreviewUrl = url;
    return () => { if (url) URL.revokeObjectURL(url); };
  });
  const referencePreview = $derived(filePreviewUrl ?? referencePreviewUrl);
  const endPreview = $derived(endFilePreviewUrl ?? endPreviewUrl);

  function chooseImagePath(path: string): void {
    onImageFileChange(null);
    draft.update('referenceImagePath', path);
  }

  function chooseEndImagePath(path: string): void {
    onEndImageFileChange(null);
    draft.update('endImagePath', path);
  }

  // --- Source video ---
  // The popover only picks the video; the path field that submits it lives in the All sheet, so nothing here has a name.
  const sourcePath = $derived(s.extendSourcePath);
  const sourceName = $derived(info?.filename ?? (sourcePath ? fileName(sourcePath) : null));
  const sourceFacts = $derived(info ? extendSourceFacts(info) : []);
  const sourceThumbUrl = $derived(info?.media_url ? `${info.media_url}#t=${Math.max(0, (info.duration_seconds ?? 0) - 0.05).toFixed(2)}` : null);
  const latestVideo = $derived(galleryStore.assets.find((asset) => asset.media_type === 'video' && asset.file_path) ?? null);
  let browsingSource = $state(false);

  let sourceText = $state(draft.state.extendSourcePath ?? '');

  function chooseSource(path: string | null): void {
    sourceText = path ?? '';
    draft.setExtendSourcePath(path);
    void extendInspection.inspect(context, { replaceText: true });
  }

  function commitSourceText(): void {
    const path = sourceText.trim() || null;
    if (path !== sourcePath) chooseSource(path);
  }

  // Enter takes the typed path at once; it must not submit the form, whose Generate waits for the check.
  function onSourceKeydown(event: KeyboardEvent): void {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    commitSourceText();
  }

  const selectedModel = $derived(models.find((model) => model.id === s.model) ?? null);
</script>

{#snippet imagePicker(p: ImagePickerOptions)}
  <div class="pop-ref">
    <span class="pop-thumb">{#if p.preview}<img src={p.preview} alt={p.previewAlt}>{:else}<Icon name="reference" size={20} />{/if}</span>
    <span class="pop-item-text" class:muted={!p.name}>{p.name ?? p.emptyText}</span>
    {#if p.name}<button type="button" class="ui-btn ui-btn-icon" aria-label={p.clearLabel} onclick={p.onclear}><Icon name="close" size={16} /></button>{/if}
  </div>
  <div class="pop-list">
    <button type="button" class="pop-item" onclick={() => fileInputs[p.slot]?.click()}><Icon name="upload" size={16} />Choose from this device<small>photos or files</small></button>
    <input bind:this={fileInputs[p.slot]} type="file" accept="image/png,image/jpeg,image/webp" class="hidden" onchange={(e) => p.onfile(e.currentTarget.files?.[0] ?? null)}>
    <button type="button" class="pop-item" onclick={p.onbrowse}><Icon name="folder" size={16} />Browse the server<small>output and reference folders</small></button>
    {#if galleryStore.assets[0]?.media_type === 'image'}
      <button type="button" class="pop-item" onclick={p.onlatest}><Icon name="film" size={16} />Use the latest output</button>
    {/if}
  </div>
  <PathInput id={p.pathId} purpose="reference_image" class="ui-field pop-path" placeholder={p.placeholder} value={p.path ?? ''} oninput={p.onpath} onpick={p.onpick} />
  {#if p.help}<p class="ui-help">{p.help}</p>{/if}
{/snippet}

<div class="setting-popover ui-sheet" use:sheetDrag={{ onclose }} in:fly={{ y: 24, duration: 180 }} role="dialog" aria-label={title} data-testid="setting-popover" data-key={key}>
  <div class="ui-sheet-grab" data-sheet-handle aria-hidden="true"></div>
  <div class="ui-sheet-head" data-sheet-handle>
    <h3 class="ui-area-label">{title}</h3>
    <span class="ui-sheet-end">
      {#if resetKeys.length}
        <button type="button" class="ui-btn" disabled={!canReset} onclick={reset}><Icon name="reset" size={12} />Reset</button>
      {/if}
      <button type="button" class="ui-btn ui-btn-icon" aria-label="Close" onclick={onclose}><Icon name="close" size={16} /></button>
    </span>
  </div>
  <div class="pop-body">
    {#if key === 'steps'}
      <InspectorNumber ariaLabel="Steps" value={s.steps} {...stepsRange} stepper size="lg" onchange={(v) => v !== null && setSteps(v)} />
      <input class="ui-range pop-range" type="range" aria-label="Steps" {...stepsRange} value={s.steps} style:--fill={rangeFill(s.steps, stepsRange)} oninput={(e) => setSteps(Number(e.currentTarget.value))}>
    {:else if key === 'guidance'}
      <InspectorNumber ariaLabel="Guidance" value={s.guidance} {...SPEC.guidance} stepper size="lg" onchange={(v) => v !== null && draft.update('guidance', v)} />
      <input class="ui-range pop-range" type="range" aria-label="Guidance" {...SPEC.guidance} value={s.guidance} style:--fill={rangeFill(s.guidance, SPEC.guidance)} oninput={(e) => draft.update('guidance', Number(e.currentTarget.value))}>
    {:else if key === 'strength'}
      <InspectorNumber ariaLabel="Image strength" value={s.referenceImageStrength} {...SPEC.strength} stepper size="lg" onchange={(v) => v !== null && draft.update('referenceImageStrength', v)} />
      <input class="ui-range pop-range" type="range" aria-label="Image strength" {...SPEC.strength} value={s.referenceImageStrength} style:--fill={rangeFill(s.referenceImageStrength, SPEC.strength)} oninput={(e) => draft.update('referenceImageStrength', Number(e.currentTarget.value))}>
      <p class="ui-help">How far from the reference image the result may go.</p>
    {:else if key === 'runs'}
      <InspectorNumber ariaLabel="Batch size" value={s.runs} {...SPEC.runs} stepper size="lg" onchange={(v) => v !== null && draft.update('runs', v)} />
      <input class="ui-range pop-range" type="range" aria-label="Batch size" min={1} max={16} step={1} value={Math.min(s.runs, 16)} style:--fill={rangeFill(Math.min(s.runs, 16), { min: 1, max: 16 })} oninput={(e) => draft.update('runs', Number(e.currentTarget.value))}>
      <p class="ui-help">{caps.isImageMode ? 'Images' : 'Clips'} per run, each with its own seed.</p>
    {:else if key === 'seconds'}
      <InspectorNumber ariaLabel="Clip length in seconds" value={shownSeconds} {...secondsSpec} stepper size="lg" commitOnly onchange={setSeconds} />
      {#if secondsSpec.max !== undefined}
        <input class="ui-range pop-range" type="range" aria-label="Clip length" min={secondsSpec.min} max={secondsSpec.max} step={secondsSpec.step} value={shownSeconds} style:--fill={rangeFill(shownSeconds, { min: secondsSpec.min ?? 0, max: secondsSpec.max })} oninput={(e) => setSeconds(Number(e.currentTarget.value))}>
      {/if}
      <p class="ui-help">{s.frameCount} frames at {timing.fps} fps.</p>
      {#if clipTime}<p class="ui-help">{clipTime}</p>{/if}
      {#if clipNote}<p class="ui-help ui-help-warning"><Icon name="alert" size={12} />{clipNote.text}</p>{/if}
    {:else if key === 'extendSeconds'}
      <InspectorNumber ariaLabel="Extend by, in seconds" value={shownExtendSeconds} {...extendSecondsSpec} stepper size="lg" commitOnly disabled={sourceFps === null} onchange={setExtendSeconds} />
      <p class="ui-help">{sourceFps === null ? 'Choose a video to extend first.' : extendLength ? `Total ${extendLength.seconds.toFixed(2)} s · ${extendLength.frames} frames${s.extendFrames !== null ? ` · +${s.extendFrames} new` : ''}` : ''}</p>
      {#if extendTime}<p class="ui-help">{extendTime}</p>{/if}
      {#if extendMemoryNote}<p class="ui-help ui-help-warning"><Icon name="alert" size={12} />{extendMemoryNote.text}</p>{/if}
      {#if extendDefault !== null && s.extendFrames !== extendDefault}
        <button type="button" class="ui-btn" onclick={() => draft.update('extendFrames', extendDefault)}>Back to the default length</button>
      {/if}
    {:else if key === 'seed'}
      <InspectorNumber ariaLabel="Seed" value={s.seed} {...SPEC.seed} nullable placeholder="Random each run" stepper={false} size="lg" onchange={(v) => draft.update('seed', v)} />
      <div class="pop-list">
        <button type="button" class="pop-item" aria-pressed={s.seed !== null} onclick={() => draft.update('seed', s.seed !== null ? null : (lastSeed ?? randomSeed()))}>
          <Icon name={s.seed !== null ? 'lock' : 'unlock'} size={16} />{s.seed !== null ? 'Locked: reused every run' : 'Random each run'}
        </button>
        <button type="button" class="pop-item" onclick={() => draft.update('seed', randomSeed())}><Icon name="dice" size={16} />New random seed</button>
        {#if latestSeed !== null}
          <button type="button" class="pop-item" onclick={() => draft.update('seed', latestSeed)}><Icon name="film" size={16} />Use the latest output's seed<small>{latestSeed}</small></button>
        {/if}
      </div>
    {:else if key === 'size'}
      <Segmented label="Aspect ratio" mono class="pop-seg" options={ratios.map((ratio) => ({ value: ratio, label: ratio }))} value={s.dimensionMode === 'ratio' ? s.ratio : ''} onchange={pickRatio} />
      <Segmented label="Resolution" class="pop-seg" options={sizeOptions.map((size) => ({ value: size, label: SIZE_NAMES[size] ?? size.toUpperCase() }))} value={s.dimensionMode === 'ratio' ? s.size : ''} onchange={(size) => draft.patch({ size, dimensionMode: 'ratio' })} />
      <div class="pop-dims">
        <InspectorNumber ariaLabel="Width" value={shownWidth} {...dimensionSpec} stepper={false} size="lg" onchange={(v) => setDimension('width', v)} />
        <span class="dim-times" aria-hidden="true">×</span>
        <InspectorNumber ariaLabel="Height" value={shownHeight} {...dimensionSpec} stepper={false} size="lg" onchange={(v) => setDimension('height', v)} />
      </div>
    {:else if key === 'model'}
      <div class="pop-list" role="radiogroup" aria-label="Model">
        {#each models as model (model.id)}
          <button type="button" class="pop-item" role="radio" aria-checked={model.id === s.model} onclick={() => { if (model.id !== s.model) onModelChange(model.id); }}>
            <Icon name="cube" size={16} />
            <span class="pop-item-text">{model.label}</span>
            <ModelStatusBadges downloaded={model.downloaded} downloadStatus={downloadStatusText(downloadStore.active('model', model.id))} memoryFit={model.memory_fit} quantize={quantizeOptions ? s.quantize : null} lowMemory={!caps.showLowMemory || s.lowMemory} tooltipPlacement="top" />
          </button>
        {/each}
      </div>
      {#if selectedModel && selectedModel.downloaded === false && !downloadStore.active('model', selectedModel.id)}
        <p class="ui-help" data-testid="model-not-downloaded">{selectedModel.label} is not downloaded. The first run downloads it; the Models page shows its size.</p>
      {/if}
      {#if quantizeOptions}
        <div class="pop-row"><span>Quantization</span>
          <Segmented label="Quantization" options={[{ value: '', label: 'None' }, ...quantizeOptions.map((q) => ({ value: String(q), label: `q${q}` }))]} value={s.quantize === null ? '' : String(s.quantize)} onchange={(value) => draft.update('quantize', value ? Number(value) : null)} />
        </div>
      {/if}
    {:else if key === 'loras'}
      {#each loraChips as chip (chip.name)}
        <div class="pop-lora">
          <span class="pop-item-text" title={chip.name}>{loraLabel(chip.name)}</span>
          <InspectorNumber ariaLabel="{loraLabel(chip.name)} weight" value={chip.weight} {...SPEC.loraWeight} stepper size="lg" onchange={(v) => setLoraWeight(chip.name, v)} />
          <button type="button" class="ui-btn ui-btn-icon" aria-label="Remove {loraLabel(chip.name)}" onclick={() => draft.update('loraString', formatLoraString(loraChips.filter((c) => c.name !== chip.name)))}><Icon name="close" size={16} /></button>
        </div>
      {/each}
      {#if loraChips.length === 0}<p class="ui-help">No LoRA in use.</p>{/if}
      {#if addable.length}
        <div class="pop-list">
          {#each loraGroups.main as lora (lora.name)}
            <button type="button" class="pop-item" onclick={() => addLora(lora.name)}><Icon name="plus" size={16} />{lora.name}</button>
          {/each}
          {#if loraGroups.other.length > 0}
            <button type="button" class="pop-item pop-fold" aria-expanded={otherOpen} data-action="other-models" onclick={() => { otherToggled = !otherOpen; }}>
              <span class="pop-item-text">Other models ({loraGroups.other.length})</span>
              <Icon name={otherOpen ? 'chevdown' : 'chevright'} size={14} />
            </button>
            {#if otherOpen}
              {#each loraGroups.other as lora (lora.name)}
                <button type="button" class="pop-item" onclick={() => addLora(lora.name)}>
                  <Icon name="plus" size={16} /><span class="pop-item-text">{lora.name}</span>{#if lora.family}<small>{familyLabel(lora.family, lora.size_variant)}</small>{/if}
                </button>
              {/each}
            {/if}
          {/if}
        </div>
      {:else if loraOptions.length === 0}
        <p class="ui-help">No LoRA files in the LoRA folder.</p>
      {/if}
    {:else if key === 'upscale'}
      <label class="pop-row">Upscale after the render<Toggle ariaLabel="Enable upscale" checked={s.upscaleEnabled} disabled={busy} onchange={(e) => draft.update('upscaleEnabled', (e.currentTarget as HTMLInputElement).checked)} /></label>
      <Segmented label="Upscale factor" mono class="pop-seg" disabled={!s.upscaleEnabled} options={[{ value: '2', label: '2×' }, { value: '4', label: '4×' }]} value={String(s.upscaleFactor)} onchange={(value) => draft.update('upscaleFactor', Number(value))} />
      {#if caps.showUpscaleReview}
        <label class="pop-row">Review the small render first<Toggle ariaLabel="Review before upscaling" checked={s.upscaleReview} disabled={busy || !s.upscaleEnabled} onchange={(e) => draft.update('upscaleReview', (e.currentTarget as HTMLInputElement).checked)} /></label>
      {/if}
      <p class="ui-help">Denoise, steps, guidance and sharpen for the upscale are under All.</p>
    {:else if key === 'audio'}
      {#if caps.showAudio}
        <label class="pop-row">Audio<Toggle ariaLabel="Audio" checked={s.audio} onchange={(e) => draft.update('audio', (e.currentTarget as HTMLInputElement).checked)} /></label>
      {/if}
      {#if caps.showLowMemory}
        <label class="pop-row">Low memory<Toggle ariaLabel="Low memory mode" checked={s.lowMemory} onchange={(e) => draft.update('lowMemory', (e.currentTarget as HTMLInputElement).checked)} /></label>
      {/if}
      {#if caps.showVideoUpscale}
        <label class="pop-row">Upscale {s.videoUpscaleFactor}× after the clip<Toggle ariaLabel="Video upscale" checked={s.videoUpscaleEnabled} disabled={busy} onchange={(e) => draft.update('videoUpscaleEnabled', (e.currentTarget as HTMLInputElement).checked)} /></label>
        {#if caps.showVideoUpscaleReview && s.videoUpscaleEnabled}
          <label class="pop-row">Review the small render first<Toggle ariaLabel="Review before upscaling" checked={s.videoUpscaleReview} disabled={busy} onchange={(e) => draft.update('videoUpscaleReview', (e.currentTarget as HTMLInputElement).checked)} /></label>
        {/if}
      {/if}
    {:else if key === 'reference'}
      {@render imagePicker({
        slot: 'reference', preview: referencePreview, previewAlt: 'Reference preview', name: referenceName, emptyText: 'No reference yet', clearLabel: 'Clear reference image',
        onclear: () => { onImageFileChange(null); draft.update('referenceImagePath', null); },
        onfile: onImageFileChange, onbrowse: () => { browsing = true; },
        onlatest: onUseLatestAsReference, pathId: 'ws-pop-image-path', placeholder: '/path/to/reference.png', path: s.referenceImagePath,
        onpath: (text) => draft.update('referenceImagePath', text || null), onpick: chooseImagePath,
      })}
      <PathBrowser bind:open={browsing} purpose="reference_image" initialPath={s.referenceImagePath} onchoose={chooseImagePath} />
    {:else if key === 'endReference'}
      {@render imagePicker({
        slot: 'endReference', preview: endPreview, previewAlt: 'Last frame preview', name: endName, emptyText: 'No last frame', clearLabel: 'Clear last frame',
        onclear: () => { onEndImageFileChange(null); draft.update('endImagePath', null); },
        onfile: onEndImageFileChange, onbrowse: () => { browsingEnd = true; },
        onlatest: onUseLatestAsEndFrame, pathId: 'ws-pop-end-image-path', placeholder: '/path/to/last-frame.png', path: s.endImagePath,
        onpath: (text) => draft.update('endImagePath', text || null), onpick: chooseEndImagePath,
        help: 'Optional. The clip ends on this image; different scenes dissolve into each other.',
      })}
      <PathBrowser bind:open={browsingEnd} purpose="reference_image" initialPath={s.endImagePath} onchoose={chooseEndImagePath} />
    {:else if key === 'source'}
      <div class="pop-ref">
        <span class="pop-thumb">{#if sourceThumbUrl}<video src={sourceThumbUrl} muted preload="metadata" aria-hidden="true"></video>{:else}<Icon name="film" size={20} />{/if}</span>
        <span class="pop-item-text" class:muted={!sourceName} title={sourcePath ?? ''}>{sourceName ?? 'No source video yet'}</span>
        {#if sourcePath !== null && sourceGuard.kind === 'checking'}<Spinner size="sm" />{/if}
        {#if sourcePath !== null}<button type="button" class="ui-btn ui-btn-icon" aria-label="Clear source video" disabled={busy} onclick={() => chooseSource(null)}><Icon name="close" size={16} /></button>{/if}
      </div>
      {#if sourceFacts.length > 0}
        <p class="pop-facts">{#each sourceFacts as fact, index (fact.key)}{#if index > 0}{' · '}{/if}<span title={fact.title}>{fact.text}</span>{/each}</p>
      {/if}
      <SourceVideoFeedback />
      <div class="pop-list">
        <button type="button" class="pop-item" disabled={busy} onclick={() => { browsingSource = true; }}><Icon name="folder" size={16} />Browse the server<small>output folder</small></button>
        {#if latestVideo && latestVideo.file_path !== sourcePath}
          <button type="button" class="pop-item" disabled={busy} onclick={() => chooseSource(latestVideo.file_path ?? null)}><Icon name="film" size={16} /><span class="pop-item-text">Use the latest video</span><small>{latestVideo.filename}</small></button>
        {/if}
      </div>
      <PathInput id="ws-pop-source-path" purpose="source_video" class="ui-field pop-path" placeholder="/path/to/clip.mp4" value={sourcePath ?? ''} oninput={(text) => { sourceText = text; }} onchange={commitSourceText} onkeydown={onSourceKeydown} onpick={(path) => chooseSource(path)} />
      <PathBrowser bind:open={browsingSource} purpose="source_video" initialPath={sourcePath} onchoose={(path) => chooseSource(path)} />
    {/if}
  </div>
</div>

<style>
  .setting-popover { display: flex; flex-direction: column; }
  .pop-body { display: flex; flex-direction: column; gap: 12px; padding: 4px max(12px, env(safe-area-inset-right, 0px)) 12px max(12px, env(safe-area-inset-left, 0px)); max-height: 60dvh; overflow-y: auto; }
  .pop-range { width: 100%; height: 36px; }
  .pop-range::-webkit-slider-thumb { width: 24px; height: 24px; margin-top: -10px; }
  .pop-range::-moz-range-thumb { width: 24px; height: 24px; }
  .pop-body :global(.pop-seg) { display: flex; width: 100%; }
  .pop-body :global(.pop-seg > button) { flex: 1; height: 36px; font-size: 14px; }
  .pop-dims { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 8px; }
  .dim-times { font-family: var(--font-mono); color: var(--color-text-muted); }
  .pop-list { display: flex; flex-direction: column; gap: 2px; }
  .pop-item { display: flex; align-items: center; gap: 12px; min-height: 48px; padding: 0 12px; border-radius: var(--radius-sm); font-size: 15px; white-space: nowrap; text-align: left; color: var(--color-text-primary); }
  .pop-item:active { background: var(--color-bg-surface-hover); }
  .pop-item[aria-pressed='true'], .pop-item[aria-checked='true'] { background: color-mix(in srgb, var(--color-primary-main) 12%, transparent); color: var(--color-primary-main); }
  .pop-item small { flex: none; max-width: 42%; margin-left: auto; overflow: hidden; text-overflow: ellipsis; font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-text-muted); }
  .pop-item-text { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .pop-item-text.muted { color: var(--color-text-muted); }
  .pop-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; min-height: 44px; font-size: 15px; }
  .pop-lora { display: flex; align-items: center; gap: 10px; min-height: 48px; font-size: 15px; }
  .pop-lora :global(.inspector-number-wrap) { flex: none; width: 170px; }
  .pop-ref { display: flex; align-items: center; gap: 12px; min-height: 56px; font-size: 15px; }
  .pop-thumb { display: grid; flex: none; place-items: center; width: 56px; height: 56px; overflow: hidden; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); color: var(--color-text-muted); }
  .pop-thumb img, .pop-thumb video { width: 100%; height: 100%; object-fit: cover; }
  .pop-facts { font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  .pop-item:disabled { opacity: 0.5; }
  .pop-fold { color: var(--color-text-muted); }
  .pop-body :global(.pop-path) { height: 40px; font-family: var(--font-mono); font-size: 13px; }
  .ui-help-warning { display: flex; gap: 4px; align-items: flex-start; }
  .ui-help-warning :global(svg) { flex: none; margin-top: 2px; }
</style>
