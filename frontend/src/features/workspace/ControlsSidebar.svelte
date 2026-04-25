<script lang="ts">
  import { draft } from '$lib/state/draft.svelte';
  import { PromptFileField } from '$lib/components/molecules';
  import { Select, Toggle, RangeSlider } from '$lib/components/atoms';
  import type { WorkspaceContext, ImageModelDefaults } from '$lib/types';

  interface Props {
    context: WorkspaceContext | null;
    busy: boolean;
    imageFile: File | null;
    onImageFileChange: (file: File | null) => void;
  }

  let { context, busy, imageFile, onImageFileChange }: Props = $props();
  const authorityReady = $derived(context !== null && draft.authorityReady);

  const visibleControls = $derived(
    new Set(context?.workflow_contract.definitions[draft.state.workflow]?.visible_controls ?? [])
  );

  const isImageMode = $derived(
    context?.workflow_contract.definitions[draft.state.workflow]?.mode === 'image'
  );
  const showRefImage = $derived(
    visibleControls.has('reference_image') || visibleControls.has('reference_image_path')
  );
  const showVideoControls = $derived(
    visibleControls.has('audio') || visibleControls.has('low_memory') || visibleControls.has('video_upscale_enabled')
  );
  const showGuidance = $derived(
    visibleControls.has('guidance')
  );
  const showI2IStrength = $derived(visibleControls.has('image_strength'));
  const showPromptInline = $derived(visibleControls.has('prompt_inline'));
  const showPromptSource = $derived(visibleControls.has('prompt_source') && (context?.prompt_sources.length ?? 0) > 0);
  const showPromptFileControls = $derived(
    visibleControls.has('prompt_file_path')
      || visibleControls.has('prompt_file_option')
      || visibleControls.has('prompt_file_preview')
      || visibleControls.has('prompt_file_edit')
  );
  const showDimensions = $derived(
    visibleControls.has('ratio') || visibleControls.has('size') || visibleControls.has('custom_dimensions')
  );
  const showRuns = $derived(visibleControls.has('runs'));
  const showFrameCount = $derived(visibleControls.has('frame_count'));
  const showSteps = $derived(visibleControls.has('steps'));
  const showSeed = $derived(visibleControls.has('seed'));
  const showImageUpscale = $derived(visibleControls.has('image_upscale_enabled'));

  const showScheduler = $derived(
    isImageMode && visibleControls.has('scheduler') && (context?.scheduler_options.length ?? 0) > 0
  );
  const showPostprocess = $derived(
    isImageMode && (
      visibleControls.has('postprocess_sharpen') ||
      visibleControls.has('postprocess_contrast') ||
      visibleControls.has('postprocess_saturation')
    )
  );
  const showVideoUpscale = $derived(visibleControls.has('video_upscale_enabled'));

  // Capability flags from backend defaults for the current image model.
  const currentImageDefaults = $derived(
    isImageMode
      ? ((context?.image_model_defaults?.[draft.state.model] ?? context?.defaults) as ImageModelDefaults | undefined)
      : undefined
  );
  const supportsNegativePrompt = $derived(
    isImageMode ? (currentImageDefaults?.supports_negative_prompt ?? false) : false
  );
  const showNegativePrompt = $derived(visibleControls.has('negative_prompt') && supportsNegativePrompt);
  const promptFileMode = $derived(draft.state.promptSource === 'file');
  const submitDisabled = $derived(
    busy || (promptFileMode && (!draft.state.promptFilePath || !draft.state.promptFileOptionId))
  );

  // Size/ratio options from context only — no hard-coded fallbacks.
  const imageRatios = $derived(context?.image_ratios ?? []);
  const videoRatios = $derived(context?.video_ratios ?? []);
  const currentRatios = $derived(isImageMode ? imageRatios : videoRatios);

  const imageSizeOptions = $derived(context?.image_size_options ?? {});
  const videoSizeOptions = $derived(context?.video_size_options ?? {});

  // Size options for the currently selected ratio
  const currentSizeOptions = $derived(
    isImageMode
      ? (imageSizeOptions[draft.state.ratio] ?? [])
      : (videoSizeOptions[draft.state.ratio] ?? [])
  );

  // Dimension mode toggle
  let dimensionMode = $state<'ratio' | 'custom'>('ratio');

  // Reference image drag-over highlight state
  let dragOver = $state(false);

  // Reference image preview
  const imagePreviewUrl = $derived(
    imageFile ? URL.createObjectURL(imageFile) : null
  );
  const imageDropzoneTitle = $derived(
    imageFile ? imageFile.name : (draft.state.referenceImagePath || 'Choose a starting image')
  );

  function handleFileInput(e: Event): void {
    const input = e.currentTarget as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    onImageFileChange(file);
  }

  function clearImage(): void {
    onImageFileChange(null);
    draft.update('referenceImagePath', null);
  }

  function handleDragOver(e: DragEvent): void {
    e.preventDefault();
    dragOver = true;
  }

  function handleDragLeave(): void {
    dragOver = false;
  }

  function handleDrop(e: DragEvent): void {
    e.preventDefault();
    dragOver = false;
    const file = e.dataTransfer?.files[0] ?? null;
    if (file && file.type.startsWith('image/')) {
      onImageFileChange(file);
    }
  }

  function handleRatioChange(e: Event): void {
    const newRatio = (e.currentTarget as HTMLSelectElement).value;
    draft.update('ratio', newRatio);
    // Reset size if it's no longer valid for the new ratio
    const validSizes = isImageMode
      ? (context?.image_size_options?.[newRatio] ?? [])
      : (context?.video_size_options?.[newRatio] ?? []);
    if (!validSizes.includes(draft.state.size)) {
      draft.update('size', validSizes[0] ?? '');
    }
  }

  function randomizeSeed(): void {
    draft.update('seed', Math.floor(Math.random() * 2 ** 32));
  }
</script>

<section class="panel-shell panel-shell-left relative flex h-full w-80 flex-col">
  {#if !authorityReady || !context}
    <div class="p-4 flex-1 overflow-y-auto custom-scrollbar pb-24">
      <div class="surface-card-muted space-y-3 rounded-md p-4">
        <p class="field-label">Loading Workspace Controls</p>
        <p class="text-sm text-zinc-400">The workspace waits for backend authority before exposing editable defaults or capability-dependent controls.</p>
      </div>
    </div>

    <div class="panel-footer absolute bottom-0 left-0 right-0 z-10 w-full shrink-0 p-4 backdrop-blur-sm">
      <button
        id="ws-submit"
        type="submit"
        disabled={true}
        class="surface-button surface-button-primary surface-button-glow flex w-full items-center justify-center gap-2 rounded-md py-2.5 font-medium active:scale-[0.98]"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
        </svg>
        <span>Generate</span>
        <span class="surface-shortcut ml-2 px-1.5 py-0.5 text-xs font-mono opacity-80">⌘↵</span>
      </button>
    </div>
  {:else}
  <div class="p-4 space-y-6 flex-1 overflow-y-auto custom-scrollbar pb-24">

    <!-- Prompt Source -->
    {#if showPromptSource}
      <div>
        <label class="field-label mb-2 block" for="ws-prompt-source">
          Prompt Source
        </label>
        <select
          id="ws-prompt-source"
          name="prompt_source"
          class="surface-select w-full rounded-md focus:border-primary-main focus:ring-4 focus:ring-primary-main"
          value={draft.state.promptSource}
          onchange={(e) => draft.update('promptSource', (e.currentTarget as HTMLSelectElement).value as 'inline' | 'file')}
        >
          {#each context.prompt_sources as source}
            <option value={source}>{source === 'inline' ? 'Inline prompt' : 'Prompt file'}</option>
          {/each}
        </select>
      </div>
    {/if}

    <!-- Prompts -->
    {#if draft.state.promptSource === 'inline' && (showPromptInline || showNegativePrompt)}
    <div>
      <label class="field-label mb-2 block" for="ws-prompt">
        Prompt
      </label>
      {#if showPromptInline}
        <textarea
          id="ws-prompt"
          name="prompt"
          rows="4"
          class="surface-textarea mb-3 w-full rounded-md shadow-sm transition placeholder-zinc-600 focus:border-primary-main focus:ring-4 focus:ring-primary-main"
          placeholder="Describe the scene..."
          required
          value={draft.state.prompt}
          oninput={(e) => draft.update('prompt', (e.currentTarget as HTMLTextAreaElement).value)}
        ></textarea>
      {/if}

      <!-- Negative prompt: only visible when the current model supports it -->
      {#if showNegativePrompt}
        <div id="ws-negative-shell">
          <label class="field-label mb-2 block" for="ws-negative-prompt">
            Negative Prompt
          </label>
          <textarea
            id="ws-negative-prompt"
            name="negative_prompt"
            rows="2"
            class="surface-textarea w-full rounded-md shadow-sm transition placeholder-zinc-600 focus:border-primary-main focus:ring-4 focus:ring-primary-main"
            placeholder="What to exclude..."
            value={draft.state.negativePrompt}
            oninput={(e) => draft.update('negativePrompt', (e.currentTarget as HTMLTextAreaElement).value)}
          ></textarea>
        </div>
      {/if}
    </div>
    {/if}

    {#if promptFileMode && context?.prompt_file && showPromptFileControls}
      <PromptFileField
        contract={context.prompt_file}
        promptSource={draft.state.promptSource}
        path={draft.state.promptFilePath}
        selectedOptionId={draft.state.promptFileOptionId}
        workflowMode={isImageMode ? 'image' : 'video'}
        negativePromptSupported={supportsNegativePrompt}
        disabled={busy}
        onPathChange={(path) => draft.update('promptFilePath', path)}
        onOptionChange={(optionId) => draft.update('promptFileOptionId', optionId)}
      />
    {/if}

    <!-- Reference Image (i2i, i2v only) -->
    {#if showRefImage}
      <div class="space-y-3 border-t border-border-subtle pt-4">
        <div class="flex items-center justify-between gap-3">
          <label class="field-label block" for="ws-image-path">
            Reference / Starting Image
          </label>
          <button type="button" onclick={clearImage} class="surface-link-muted text-[11px] font-medium transition">
            Clear
          </button>
        </div>
        <input
          id="ws-image-file"
          name="image_file"
          type="file"
          accept="image/png,image/jpeg,image/webp"
          class="hidden"
          onchange={handleFileInput}
        >
        <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
        <label
          for="ws-image-file"
          class="surface-dropzone block w-full cursor-pointer px-4 py-5 text-left transition"
          data-dragover={dragOver}
          ondragover={handleDragOver}
          ondragleave={handleDragLeave}
          ondrop={handleDrop}
        >
          {#if imageFile}
            <img src={imagePreviewUrl ?? ''} alt="Preview" class="h-16 object-contain mb-2 rounded">
          {/if}
          <span class="block text-sm font-medium text-zinc-200 truncate">
            {imageDropzoneTitle}
          </span>
          <span class="mt-1 block text-xs text-zinc-500">Drop an image here or browse from disk.</span>
          <span class="surface-dropzone-badge mt-3 inline-flex items-center gap-2 px-3 py-1.5 text-xs font-medium">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path>
            </svg>
            Browse Image
          </span>
        </label>
        <div>
          <label class="field-hint-label mb-1 block" for="ws-image-path">Or paste an absolute path</label>
          <input
            id="ws-image-path"
            name="image_path"
            type="text"
            class="surface-input w-full rounded-md focus:border-primary-main focus:ring-4 focus:ring-primary-main"
            placeholder="/path/to/reference.png"
            value={draft.state.referenceImagePath ?? ''}
            oninput={(e) => draft.update('referenceImagePath', (e.currentTarget as HTMLInputElement).value || null)}
          >
        </div>
      </div>
    {/if}

    <!-- Dimensions -->
    {#if showDimensions}
    <div class="border-t border-border-subtle pt-4">
      <div class="flex items-center justify-between mb-3">
        <span class="field-label block">Dimensions</span>
        <div class="surface-toggle-group flex items-center p-0.5">
          <button
            type="button"
            class="surface-toggle-pill px-2 py-0.5 text-[10px] font-medium {dimensionMode === 'ratio' ? 'surface-toggle-pill-active' : ''}"
            onclick={() => (dimensionMode = 'ratio')}
          >Ratio</button>
          <button
            type="button"
            class="surface-toggle-pill px-2 py-0.5 text-[10px] font-medium {dimensionMode === 'custom' ? 'surface-toggle-pill-active' : ''}"
            onclick={() => (dimensionMode = 'custom')}
          >Custom W/H</button>
        </div>
      </div>

      {#if dimensionMode === 'ratio'}
        <div class="space-y-3 mb-4">
          <div>
            <label class="field-hint-label mb-1 block" for="ws-ratio">Aspect Ratio</label>
            <select
              id="ws-ratio"
              name="ratio"
              class="surface-select w-full rounded-md focus:border-primary-main focus:ring-4 focus:ring-primary-main"
              value={draft.state.ratio}
              onchange={handleRatioChange}
            >
              {#each currentRatios as ratio}
                <option value={ratio}>{ratio}</option>
              {/each}
            </select>
          </div>
          <div>
            <label class="field-hint-label mb-1 block" for="ws-size">Size / Base Resolution</label>
            <select
              id="ws-size"
              name="size"
              class="surface-select w-full rounded-md focus:border-primary-main focus:ring-4 focus:ring-primary-main"
              value={draft.state.size}
              onchange={(e) => draft.update('size', (e.currentTarget as HTMLSelectElement).value)}
            >
              {#each currentSizeOptions as size}
                <option value={size}>{size}</option>
              {/each}
            </select>
          </div>
        </div>
      {:else}
        <div class="grid grid-cols-2 gap-3 mb-4">
          <div>
            <label class="field-hint-label mb-1 block" for="ws-width">Width</label>
            <input
              id="ws-width"
              name="width"
              type="number"
              min="16"
              step="16"
              value={draft.state.width}
              class="surface-input w-full rounded-md font-mono focus:border-primary-main focus:ring-4 focus:ring-primary-main"
              oninput={(e) => draft.update('width', Number((e.currentTarget as HTMLInputElement).value))}
            >
          </div>
          <div>
            <label class="field-hint-label mb-1 block" for="ws-height">Height</label>
            <input
              id="ws-height"
              name="height"
              type="number"
              min="16"
              step="16"
              value={draft.state.height}
              class="surface-input w-full rounded-md font-mono focus:border-primary-main focus:ring-4 focus:ring-primary-main"
              oninput={(e) => draft.update('height', Number((e.currentTarget as HTMLInputElement).value))}
            >
          </div>
        </div>
      {/if}
    </div>
    {/if}

    <!-- Generation Settings -->
    <div class="space-y-4 border-t border-border-subtle pt-4">
      <span class="field-label mb-2 block">
        Generation Settings
      </span>

      <div class="flex gap-3 mb-2">
        {#if showRuns}
          <div class="flex-1">
            <label class="field-hint-label mb-1 block" for="ws-runs">Batch Size</label>
            <input
              id="ws-runs"
              type="number"
              name="runs"
              value={draft.state.runs}
              min="1"
              max="16"
              step="1"
              class="surface-input w-full rounded-md text-center font-mono focus:border-primary-main focus:ring-4 focus:ring-primary-main"
              oninput={(e) => draft.update('runs', Number((e.currentTarget as HTMLInputElement).value))}
            >
          </div>
        {/if}
        {#if showFrameCount}
          <div class="flex-1">
            <label class="field-hint-label mb-1 block" for="ws-frames">Frames</label>
            <input
              id="ws-frames"
              type="number"
              name="frames"
              value={draft.state.frameCount}
              min="1"
              max="256"
              step="1"
              class="surface-input w-full rounded-md text-center font-mono focus:border-primary-main focus:ring-4 focus:ring-primary-main"
              oninput={(e) => draft.update('frameCount', Number((e.currentTarget as HTMLInputElement).value))}
            >
          </div>
        {/if}
      </div>

      {#if showSteps}
      <div>
        <div class="flex justify-between mb-1">
          <label class="field-hint-label" for="ws-steps">Steps</label>
          <span class="field-value">{draft.state.steps}</span>
        </div>
        <input
          id="ws-steps"
          name="steps"
          type="range"
          min="1"
          max="60"
          step="1"
          value={draft.state.steps}
          class="accent-primary w-full"
          oninput={(e) => draft.update('steps', Number((e.currentTarget as HTMLInputElement).value))}
        >
      </div>
      {/if}

      {#if showGuidance}
        <div>
          <div class="flex justify-between mb-1">
            <label class="field-hint-label" for="ws-guidance">Guidance Scale</label>
            <span class="field-value">{draft.state.guidance}</span>
          </div>
          <input
            id="ws-guidance"
            name="guidance"
            type="range"
            min="0.0"
            max="10.0"
            step="0.1"
            value={draft.state.guidance}
            class="accent-primary w-full"
            oninput={(e) => draft.update('guidance', Number((e.currentTarget as HTMLInputElement).value))}
          >
        </div>
      {/if}

      {#if showI2IStrength}
        <div>
          <div class="flex justify-between mb-1">
            <label class="field-hint-label" for="ws-image-strength">Image Strength</label>
            <span class="field-value">{draft.state.referenceImageStrength}</span>
          </div>
          <input
            id="ws-image-strength"
            name="image_strength"
            type="range"
            min="0.0"
            max="1.0"
            step="0.01"
            value={draft.state.referenceImageStrength}
            class="accent-primary w-full"
            oninput={(e) => draft.update('referenceImageStrength', Number((e.currentTarget as HTMLInputElement).value))}
          >
        </div>
      {/if}

      {#if showSeed}
      <div>
        <label class="field-hint-label mb-1 block" for="ws-seed">Seed</label>
        <div class="flex gap-2">
          <input
            id="ws-seed"
            name="seed"
            type="number"
            min="0"
            placeholder="Random"
            value={draft.state.seed ?? ''}
            class="surface-input flex-1 rounded-md font-mono focus:border-primary-main focus:ring-4 focus:ring-primary-main"
            oninput={(e) => {
              const v = (e.currentTarget as HTMLInputElement).value;
              draft.update('seed', v ? Number(v) : null);
            }}
          >
          <button
            type="button"
            title="Randomize"
            class="surface-icon-button rounded-md p-2"
            onclick={randomizeSeed}
          >🎲</button>
        </div>
      </div>
      {/if}
    </div>

    <!-- Scheduler (image workflows only) -->
    {#if showScheduler}
    <div class="border-t border-border-subtle pt-4">
      <label class="field-hint-label mb-1 block" for="ws-scheduler">Scheduler</label>
      <Select
        id="ws-scheduler"
        name="scheduler"
        options={[{ value: '', label: 'Auto (model default)' }, ...(context?.scheduler_options ?? []).map(s => ({ value: s, label: s }))]}
        value={draft.state.scheduler ?? ''}
        onchange={(e) => draft.update('scheduler', (e.currentTarget as HTMLSelectElement).value || null)}
      />
    </div>
    {/if}

    <!-- Post-processing (image workflows only) -->
    {#if showPostprocess}
      <div class="space-y-3 border-t border-border-subtle pt-4">
        <span class="field-label mb-2 block">Post-processing</span>

        {#if visibleControls.has('postprocess_sharpen')}
          <div class="surface-card-muted space-y-2 rounded-md p-3">
            <div class="flex items-center justify-between">
              <span class="text-sm font-medium text-zinc-300">Sharpen</span>
              <Toggle
                id="ws-pp-sharpen"
                checked={draft.state.postprocessSharpenEnabled}
                disabled={busy}
                onchange={(e) => draft.update('postprocessSharpenEnabled', (e.currentTarget as HTMLInputElement).checked)}
              />
            </div>
            {#if draft.state.postprocessSharpenEnabled}
              <RangeSlider
                id="ws-pp-sharpen-amount"
                label="Amount"
                value={draft.state.postprocessSharpenAmount}
                min={0}
                max={2}
                step={0.05}
                formatValue={(v) => v.toFixed(2)}
                oninput={(e) => draft.update('postprocessSharpenAmount', Number((e.currentTarget as HTMLInputElement).value))}
              />
            {/if}
          </div>
          <input type="hidden" name="sharpen_enabled" value={draft.state.postprocessSharpenEnabled ? 'true' : 'false'}>
          {#if draft.state.postprocessSharpenEnabled}
            <input type="hidden" name="sharpen_amount" value={String(draft.state.postprocessSharpenAmount)}>
          {/if}
        {/if}

        {#if visibleControls.has('postprocess_contrast')}
          <div class="surface-card-muted space-y-2 rounded-md p-3">
            <div class="flex items-center justify-between">
              <span class="text-sm font-medium text-zinc-300">Contrast</span>
              <Toggle
                id="ws-pp-contrast"
                checked={draft.state.postprocessContrastEnabled}
                disabled={busy}
                onchange={(e) => draft.update('postprocessContrastEnabled', (e.currentTarget as HTMLInputElement).checked)}
              />
            </div>
            {#if draft.state.postprocessContrastEnabled}
              <RangeSlider
                id="ws-pp-contrast-amount"
                label="Amount"
                value={draft.state.postprocessContrastAmount}
                min={0.5}
                max={2}
                step={0.05}
                formatValue={(v) => v.toFixed(2)}
                oninput={(e) => draft.update('postprocessContrastAmount', Number((e.currentTarget as HTMLInputElement).value))}
              />
            {/if}
          </div>
          <input type="hidden" name="contrast_enabled" value={draft.state.postprocessContrastEnabled ? 'true' : 'false'}>
          {#if draft.state.postprocessContrastEnabled}
            <input type="hidden" name="contrast_amount" value={String(draft.state.postprocessContrastAmount)}>
          {/if}
        {/if}

        {#if visibleControls.has('postprocess_saturation')}
          <div class="surface-card-muted space-y-2 rounded-md p-3">
            <div class="flex items-center justify-between">
              <span class="text-sm font-medium text-zinc-300">Saturation</span>
              <Toggle
                id="ws-pp-saturation"
                checked={draft.state.postprocessSaturationEnabled}
                disabled={busy}
                onchange={(e) => draft.update('postprocessSaturationEnabled', (e.currentTarget as HTMLInputElement).checked)}
              />
            </div>
            {#if draft.state.postprocessSaturationEnabled}
              <RangeSlider
                id="ws-pp-saturation-amount"
                label="Amount"
                value={draft.state.postprocessSaturationAmount}
                min={0.5}
                max={2}
                step={0.05}
                formatValue={(v) => v.toFixed(2)}
                oninput={(e) => draft.update('postprocessSaturationAmount', Number((e.currentTarget as HTMLInputElement).value))}
              />
            {/if}
          </div>
          <input type="hidden" name="saturation_enabled" value={draft.state.postprocessSaturationEnabled ? 'true' : 'false'}>
          {#if draft.state.postprocessSaturationEnabled}
            <input type="hidden" name="saturation_amount" value={String(draft.state.postprocessSaturationAmount)}>
          {/if}
        {/if}
      </div>
    {/if}

    <!-- Upscale (image workflows only) -->
    {#if isImageMode && showImageUpscale}
      <div class="space-y-4 border-t border-border-subtle pb-4 pt-4">
        <span class="field-label mb-2 block">
          Enhancements &amp; System
        </span>

        <!-- Upscale toggle + factor — wired to draft state and submitted in form -->
        <div class="surface-card-muted space-y-3 p-3">
          <div class="flex items-start justify-between gap-3">
          <label class="flex items-center gap-3 cursor-pointer group" aria-label="Enable upscale">
              <div class="relative flex items-center">
                <input
                  type="checkbox"
                  class="sr-only peer"
                  checked={draft.state.upscaleEnabled}
                  onchange={(e) => draft.update('upscaleEnabled', (e.currentTarget as HTMLInputElement).checked)}
                >
                <div class="w-9 h-5 bg-zinc-800 rounded-full peer peer-focus:ring-2 peer-focus:ring-teal-500 peer-checked:after:translate-x-full peer-checked:bg-teal-500 after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:border-zinc-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all"></div>
              </div>
              <div>
                <span class="text-sm font-medium text-zinc-300 group-hover:text-zinc-100 block">Upscale</span>
                <span class="text-xs text-zinc-500 block mt-0.5">
                  {draft.state.upscaleEnabled ? 'Upscale enabled.' : 'Upscale disabled.'}
                </span>
              </div>
            </label>
            {#if draft.state.upscaleEnabled}
              <div class="shrink-0 space-y-1">
                <label class="field-label block" for="ws-upscale-factor">Factor</label>
                <select
                  id="ws-upscale-factor"
                  class="surface-select w-20 rounded-md focus:border-primary-main focus:ring-4 focus:ring-primary-main"
                  value={String(draft.state.upscaleFactor)}
                  onchange={(e) => draft.update('upscaleFactor', Number((e.currentTarget as HTMLSelectElement).value))}
                >
                  <option value="2">2x</option>
                  <option value="4">4x</option>
                </select>
              </div>
            {/if}
          </div>
          <!-- Submit the upscale factor when enabled; omit the field entirely when disabled -->
          {#if draft.state.upscaleEnabled}
            <input type="hidden" name="upscale" value={String(draft.state.upscaleFactor)}>
          {/if}
        </div>

        <!-- Advanced upscale options (shown when upscale is enabled) -->
        {#if draft.state.upscaleEnabled}
          {#if visibleControls.has('image_upscale_denoise')}
            <div>
              <label class="field-hint-label mb-1 block" for="ws-upscale-denoise">Denoise Strength</label>
              <input
                id="ws-upscale-denoise"
                name="upscale_denoise"
                type="number"
                min="0"
                max="1"
                step="0.01"
                placeholder="auto"
                value={draft.state.upscaleDenoise !== null ? draft.state.upscaleDenoise : ''}
                class="surface-input w-full rounded-md font-mono focus:border-primary-main focus:ring-4 focus:ring-primary-main"
                oninput={(e) => {
                  const v = (e.currentTarget as HTMLInputElement).value;
                  draft.update('upscaleDenoise', v ? Number(v) : null);
                }}
              >
            </div>
          {/if}
          {#if visibleControls.has('image_upscale_steps')}
            <div>
              <label class="field-hint-label mb-1 block" for="ws-upscale-steps">Upscale Steps</label>
              <input
                id="ws-upscale-steps"
                name="upscale_steps"
                type="number"
                min="1"
                step="1"
                placeholder="auto"
                value={draft.state.upscaleSteps !== null ? draft.state.upscaleSteps : ''}
                class="surface-input w-full rounded-md font-mono focus:border-primary-main focus:ring-4 focus:ring-primary-main"
                oninput={(e) => {
                  const v = (e.currentTarget as HTMLInputElement).value;
                  draft.update('upscaleSteps', v ? Number(v) : null);
                }}
              >
            </div>
          {/if}
          {#if visibleControls.has('image_upscale_guidance')}
            <div>
              <label class="field-hint-label mb-1 block" for="ws-upscale-guidance">Upscale Guidance</label>
              <input
                id="ws-upscale-guidance"
                name="upscale_guidance"
                type="number"
                min="0"
                step="0.1"
                placeholder="auto"
                value={draft.state.upscaleGuidance !== null ? draft.state.upscaleGuidance : ''}
                class="surface-input w-full rounded-md font-mono focus:border-primary-main focus:ring-4 focus:ring-primary-main"
                oninput={(e) => {
                  const v = (e.currentTarget as HTMLInputElement).value;
                  draft.update('upscaleGuidance', v ? Number(v) : null);
                }}
              >
            </div>
          {/if}
          {#if visibleControls.has('image_upscale_sharpen')}
            <div class="flex items-center justify-between">
              <span class="field-hint-label">Upscale Sharpen</span>
              <Toggle
                id="ws-upscale-sharpen"
                checked={draft.state.upscaleSharpen}
                disabled={busy}
                onchange={(e) => draft.update('upscaleSharpen', (e.currentTarget as HTMLInputElement).checked)}
              />
              <input type="hidden" name="upscale_sharpen" value={draft.state.upscaleSharpen ? 'true' : 'false'}>
            </div>
          {/if}
        {/if}
      </div>
    {/if}

    <!-- Video-only: Audio + Low Memory -->
    {#if showVideoControls}
      <div class="space-y-3 border-t border-border-subtle pb-4 pt-4">
        <span class="field-label mb-2 block">
          Video Settings
        </span>

        {#if visibleControls.has('audio')}
        <label class="flex items-center gap-3 cursor-pointer group">
          <div class="relative flex items-center">
            <input
              type="checkbox"
              name="audio"
              class="sr-only peer"
              checked={draft.state.audio}
              onchange={(e) => draft.update('audio', (e.currentTarget as HTMLInputElement).checked)}
            >
            <input type="hidden" name="audio" value="false">
            <div class="w-9 h-5 bg-zinc-800 rounded-full peer peer-focus:ring-2 peer-focus:ring-teal-500 peer-checked:after:translate-x-full peer-checked:bg-teal-500 after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:border-zinc-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all"></div>
          </div>
          <div>
            <span class="text-sm font-medium text-zinc-300 group-hover:text-zinc-100 block">Audio (Video output)</span>
          </div>
        </label>
        {/if}

        {#if visibleControls.has('low_memory')}
        <label class="flex items-center gap-3 cursor-pointer group">
          <div class="relative flex items-center">
            <input
              type="checkbox"
              name="low_memory"
              class="sr-only peer"
              checked={draft.state.lowMemory}
              onchange={(e) => draft.update('lowMemory', (e.currentTarget as HTMLInputElement).checked)}
            >
            <input type="hidden" name="low_memory" value="false">
            <div class="w-9 h-5 bg-zinc-800 rounded-full peer peer-focus:ring-2 peer-focus:ring-teal-500 peer-checked:after:translate-x-full peer-checked:bg-teal-500 after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:border-zinc-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all"></div>
          </div>
          <div>
            <span class="text-sm font-medium text-zinc-300 group-hover:text-zinc-100 block">Low Memory Mode</span>
          </div>
        </label>
        {/if}

        <!-- Video upscale -->
        {#if showVideoUpscale}
          <div class="surface-card-muted space-y-3 rounded-md p-3">
            <div class="flex items-center justify-between gap-3">
              <div>
                <span class="text-sm font-medium text-zinc-300 block">Upscale</span>
                <span class="text-xs text-zinc-500 block mt-0.5">
                  {draft.state.videoUpscaleEnabled ? 'Upscale enabled.' : 'Upscale disabled.'}
                </span>
              </div>
              <Toggle
                id="ws-video-upscale"
                checked={draft.state.videoUpscaleEnabled}
                disabled={busy}
                onchange={(e) => draft.update('videoUpscaleEnabled', (e.currentTarget as HTMLInputElement).checked)}
              />
            </div>
            {#if draft.state.videoUpscaleEnabled}
              {#if visibleControls.has('video_upscale_factor')}
                <div class="flex items-center justify-between gap-3">
                  <label class="field-hint-label" for="ws-video-upscale-factor">Factor</label>
                  <Select
                    id="ws-video-upscale-factor"
                    options={[{ value: '2', label: '2x' }, { value: '4', label: '4x' }]}
                    value={String(draft.state.videoUpscaleFactor)}
                    class="w-20"
                    onchange={(e) => draft.update('videoUpscaleFactor', Number((e.currentTarget as HTMLSelectElement).value))}
                  />
                </div>
              {/if}
              <input type="hidden" name="upscale" value={String(draft.state.videoUpscaleFactor)}>
              <input type="hidden" name="video_upscale_factor" value={String(draft.state.videoUpscaleFactor)}>
            {/if}
          </div>
        {/if}
      </div>
    {/if}

  </div><!-- end scroll area -->

  <!-- Sticky submit bar -->
  <div class="panel-footer absolute bottom-0 left-0 right-0 z-10 w-full shrink-0 p-4 backdrop-blur-sm">
    <p
      id="ws-busy-note"
      class="surface-card-muted {busy ? '' : 'hidden'} mb-3 px-3 py-2 text-xs text-zinc-400"
    >
      An exclusive generation job is running. New runs are disabled until it finishes.
    </p>
    {#if promptFileMode && !draft.state.promptFileOptionId}
      <p class="surface-card-muted mb-3 px-3 py-2 text-xs text-amber-400">
        Select one active prompt option from the prompt file before generating.
      </p>
    {/if}
    <button
      id="ws-submit"
      type="submit"
      disabled={submitDisabled}
      class="surface-button surface-button-primary surface-button-glow flex w-full items-center justify-center gap-2 rounded-md py-2.5 font-medium active:scale-[0.98]"
    >
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
      </svg>
      <span>{busy ? 'Generation In Progress' : 'Generate'}</span>
      <span class="surface-shortcut ml-2 px-1.5 py-0.5 text-xs font-mono opacity-80">⌘↵</span>
    </button>
  </div>
  {/if}
</section>
