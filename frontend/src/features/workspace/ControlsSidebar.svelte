<script lang="ts">
  import { draft } from '$lib/state/draft.svelte';
  import type { WorkspaceContext } from '$lib/types';

  interface Props {
    context: WorkspaceContext | null;
    busy: boolean;
    imageFile: File | null;
    onImageFileChange: (file: File | null) => void;
  }

  let { context, busy, imageFile, onImageFileChange }: Props = $props();

  // Workflow-based visibility
  const isImageMode = $derived(
    draft.state.workflow === 'txt2img' || draft.state.workflow === 'img2img'
  );
  const showRefImage = $derived(
    draft.state.workflow === 'img2img' || draft.state.workflow === 'img2vid'
  );
  const showVideoControls = $derived(
    draft.state.workflow === 'txt2vid' || draft.state.workflow === 'img2vid'
  );
  const showGuidance = $derived(
    draft.state.workflow === 'txt2img' || draft.state.workflow === 'img2img'
  );
  const showI2IStrength = $derived(draft.state.workflow === 'img2img');
  const showEnhancements = $derived(
    draft.state.workflow === 'txt2img' || draft.state.workflow === 'img2img'
  );

  // Size/ratio options from context or defaults
  const imageRatios = $derived(context?.image_ratios ?? ['1:1', '2:3', '3:2', '4:5', '5:4', '9:16', '16:9']);
  const videoRatios = $derived(context?.video_ratios ?? ['16:9', '9:16', '4:3', '1:1']);
  const currentRatios = $derived(isImageMode ? imageRatios : videoRatios);

  const imageSizeOptions = $derived(
    context?.image_size_options ?? { '1:1': ['xs', 's', 'm', 'l', 'xl'] }
  );
  const videoSizeOptions = $derived(
    context?.video_size_options ?? { '16:9': ['s', 'm', 'l'] }
  );

  // Image sizes for current ratio
  const currentRatio = $derived(draft.state.workflow === 'txt2img' || draft.state.workflow === 'img2img'
    ? (imageRatios[0] ?? '1:1')
    : (videoRatios[0] ?? '16:9')
  );

  // Dimension mode toggle
  let dimensionMode = $state<'ratio' | 'custom'>('ratio');

  // Upscale toggle
  let upscaleEnabled = $state(false);

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

  function randomizeSeed(): void {
    draft.update('seed', Math.floor(Math.random() * 2 ** 32));
  }
</script>

<section class="w-80 bg-zinc-950 border-r border-zinc-900 flex flex-col relative h-full">
  <div class="p-4 space-y-6 flex-1 overflow-y-auto custom-scrollbar pb-24">

    <!-- Prompts -->
    <div>
      <label class="block text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-2" for="ws-prompt">
        Prompts
      </label>
      <textarea
        id="ws-prompt"
        name="prompt"
        rows="4"
        class="w-full bg-zinc-900 border border-zinc-800 text-sm rounded-md px-3 py-2 text-zinc-50 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 resize-none mb-3 shadow-sm transition placeholder-zinc-600"
        placeholder="Describe the scene..."
        required
        value={draft.state.prompt}
        oninput={(e) => draft.update('prompt', (e.currentTarget as HTMLTextAreaElement).value)}
      ></textarea>

      <div id="ws-negative-shell">
        <label class="block text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-2" for="ws-negative-prompt">
          Negative Prompt
        </label>
        <textarea
          id="ws-negative-prompt"
          name="negative_prompt"
          rows="2"
          class="w-full bg-zinc-900 border border-zinc-800 text-sm rounded-md px-3 py-2 text-zinc-50 focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 resize-none shadow-sm transition placeholder-zinc-600"
          placeholder="What to exclude..."
          value={draft.state.negativePrompt}
          oninput={(e) => draft.update('negativePrompt', (e.currentTarget as HTMLTextAreaElement).value)}
        ></textarea>
      </div>
    </div>

    <!-- Reference Image (i2i, i2v only) -->
    {#if showRefImage}
      <div class="pt-4 border-t border-zinc-900 space-y-3">
        <div class="flex items-center justify-between gap-3">
          <label class="block text-[11px] font-semibold text-zinc-500 uppercase tracking-wider" for="ws-image-path">
            Reference / Starting Image
          </label>
          <button type="button" onclick={clearImage} class="text-[11px] font-medium text-zinc-500 hover:text-zinc-300 transition">
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
          class="w-full rounded-lg border border-dashed {dragOver ? 'border-teal-500 bg-zinc-900' : 'border-zinc-700 bg-zinc-900/60 hover:border-teal-500/60 hover:bg-zinc-900'} px-4 py-5 text-left transition cursor-pointer block"
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
          <span class="mt-3 inline-flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-xs font-medium text-zinc-300">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path>
            </svg>
            Browse Image
          </span>
        </label>
        <div>
          <label class="block text-xs text-zinc-400 mb-1" for="ws-image-path">Or paste an absolute path</label>
          <input
            id="ws-image-path"
            name="image_path"
            type="text"
            class="w-full bg-zinc-900 border border-zinc-800 rounded px-3 py-2 text-sm text-zinc-300 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            placeholder="/path/to/reference.png"
            value={draft.state.referenceImagePath ?? ''}
            oninput={(e) => draft.update('referenceImagePath', (e.currentTarget as HTMLInputElement).value || null)}
          >
        </div>
      </div>
    {/if}

    <!-- Dimensions -->
    <div class="pt-4 border-t border-zinc-900">
      <div class="flex items-center justify-between mb-3">
        <label class="block text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Dimensions</label>
        <div class="flex items-center bg-zinc-900 rounded border border-zinc-800 p-0.5">
          <button
            type="button"
            class="{dimensionMode === 'ratio' ? 'bg-zinc-800 text-zinc-200 shadow-sm' : 'text-zinc-500 hover:text-zinc-300'} px-2 py-0.5 text-[10px] font-medium rounded"
            onclick={() => (dimensionMode = 'ratio')}
          >Ratio</button>
          <button
            type="button"
            class="{dimensionMode === 'custom' ? 'bg-zinc-800 text-zinc-200 shadow-sm' : 'text-zinc-500 hover:text-zinc-300'} px-2 py-0.5 text-[10px] font-medium rounded"
            onclick={() => (dimensionMode = 'custom')}
          >Custom W/H</button>
        </div>
      </div>

      {#if dimensionMode === 'ratio'}
        <div class="space-y-3 mb-4">
          <div>
            <label class="block text-xs text-zinc-400 mb-1">Size / Base Resolution</label>
            <select
              name="size"
              class="w-full bg-zinc-900 border border-zinc-800 rounded px-3 py-1.5 text-sm text-zinc-300 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            >
              {#if isImageMode}
                {#each (imageSizeOptions[currentRatio] ?? ['s', 'm', 'l']) as size}
                  <option value={size}>{size}</option>
                {/each}
              {:else}
                {#each (videoSizeOptions[currentRatio] ?? ['s', 'm', 'l']) as size}
                  <option value={size}>{size}</option>
                {/each}
              {/if}
            </select>
          </div>
          <div>
            <label class="block text-xs text-zinc-400 mb-1">Aspect Ratio</label>
            <select
              name="ratio"
              class="w-full bg-zinc-900 border border-zinc-800 rounded px-3 py-1.5 text-sm text-zinc-300 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            >
              {#each currentRatios as ratio}
                <option value={ratio}>{ratio}</option>
              {/each}
            </select>
          </div>
        </div>
      {:else}
        <div class="grid grid-cols-2 gap-3 mb-4">
          <div>
            <label class="block text-xs text-zinc-400 mb-1" for="ws-width">Width</label>
            <input
              id="ws-width"
              name="width"
              type="number"
              min="16"
              step="16"
              value={draft.state.width}
              class="w-full bg-zinc-900 border border-zinc-800 rounded px-3 py-1.5 text-sm text-zinc-300 font-mono"
              oninput={(e) => draft.update('width', Number((e.currentTarget as HTMLInputElement).value))}
            >
          </div>
          <div>
            <label class="block text-xs text-zinc-400 mb-1" for="ws-height">Height</label>
            <input
              id="ws-height"
              name="height"
              type="number"
              min="16"
              step="16"
              value={draft.state.height}
              class="w-full bg-zinc-900 border border-zinc-800 rounded px-3 py-1.5 text-sm text-zinc-300 font-mono"
              oninput={(e) => draft.update('height', Number((e.currentTarget as HTMLInputElement).value))}
            >
          </div>
        </div>
      {/if}
    </div>

    <!-- Generation Settings -->
    <div class="pt-4 border-t border-zinc-900 space-y-4">
      <label class="block text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-2">
        Generation Settings
      </label>

      <div class="flex gap-3 mb-2">
        <div class="flex-1">
          <label class="block text-xs text-zinc-400 mb-1">Batch Size</label>
          <input
            type="number"
            name="runs"
            value={draft.state.runs}
            min="1"
            max="16"
            step="1"
            class="w-full bg-zinc-900 border border-zinc-800 rounded px-3 py-1.5 text-sm text-zinc-300 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 text-center font-mono"
            oninput={(e) => draft.update('runs', Number((e.currentTarget as HTMLInputElement).value))}
          >
        </div>
        {#if showVideoControls}
          <div class="flex-1">
            <label class="block text-xs text-zinc-400 mb-1" title="For Video">Frames</label>
            <input
              type="number"
              name="frames"
              value={draft.state.frameCount}
              min="1"
              max="256"
              step="1"
              class="w-full bg-zinc-900 border border-zinc-800 rounded px-3 py-1.5 text-sm text-zinc-300 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 text-center font-mono"
              oninput={(e) => draft.update('frameCount', Number((e.currentTarget as HTMLInputElement).value))}
            >
          </div>
        {/if}
      </div>

      <div>
        <div class="flex justify-between mb-1">
          <label class="text-xs text-zinc-400" for="ws-steps">Steps</label>
          <span class="text-xs text-zinc-500 font-mono">{draft.state.steps}</span>
        </div>
        <input
          id="ws-steps"
          name="steps"
          type="range"
          min="1"
          max="60"
          step="1"
          value={draft.state.steps}
          class="w-full accent-teal-500"
          oninput={(e) => draft.update('steps', Number((e.currentTarget as HTMLInputElement).value))}
        >
      </div>

      {#if showGuidance}
        <div>
          <div class="flex justify-between mb-1">
            <label class="text-xs text-zinc-400" for="ws-guidance">Guidance Scale</label>
            <span class="text-xs text-zinc-500 font-mono">{draft.state.guidance}</span>
          </div>
          <input
            id="ws-guidance"
            name="guidance"
            type="range"
            min="0.0"
            max="10.0"
            step="0.1"
            value={draft.state.guidance}
            class="w-full accent-teal-500"
            oninput={(e) => draft.update('guidance', Number((e.currentTarget as HTMLInputElement).value))}
          >
        </div>
      {/if}

      {#if showI2IStrength}
        <div>
          <div class="flex justify-between mb-1">
            <label class="text-xs text-zinc-400" for="ws-image-strength">Image Strength</label>
            <span class="text-xs text-zinc-500 font-mono">{draft.state.referenceImageStrength}</span>
          </div>
          <input
            id="ws-image-strength"
            name="image_strength"
            type="range"
            min="0.0"
            max="1.0"
            step="0.01"
            value={draft.state.referenceImageStrength}
            class="w-full accent-teal-500"
            oninput={(e) => draft.update('referenceImageStrength', Number((e.currentTarget as HTMLInputElement).value))}
          >
        </div>
      {/if}

      <div>
        <label class="block text-xs text-zinc-400 mb-1" for="ws-seed">Seed</label>
        <div class="flex gap-2">
          <input
            id="ws-seed"
            name="seed"
            type="number"
            min="0"
            placeholder="Random"
            value={draft.state.seed ?? ''}
            class="flex-1 bg-zinc-900 border border-zinc-800 rounded px-3 py-1.5 text-sm text-zinc-300 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 font-mono"
            oninput={(e) => {
              const v = (e.currentTarget as HTMLInputElement).value;
              draft.update('seed', v ? Number(v) : null);
            }}
          >
          <button
            type="button"
            title="Randomize"
            class="p-2 border border-zinc-800 bg-zinc-900 rounded hover:bg-zinc-800 text-zinc-400 transition"
            onclick={randomizeSeed}
          >🎲</button>
        </div>
      </div>
    </div>

    <!-- Enhancements & System -->
    {#if showEnhancements}
      <div class="pt-4 border-t border-zinc-900 pb-4 space-y-4">
        <label class="block text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-2">
          Enhancements &amp; System
        </label>

        <div>
          <div class="flex justify-between mb-1">
            <label class="text-xs text-zinc-400" for="ws-contrast">Contrast</label>
            <span class="text-xs text-zinc-500 font-mono">1.0</span>
          </div>
          <input
            id="ws-contrast"
            type="range"
            min="0.5"
            max="1.5"
            step="0.05"
            value="1.0"
            class="w-full accent-teal-500"
          >
        </div>

        <div>
          <div class="flex justify-between mb-1">
            <label class="text-xs text-zinc-400" for="ws-saturation">Saturation</label>
            <span class="text-xs text-zinc-500 font-mono">1.0</span>
          </div>
          <input
            id="ws-saturation"
            type="range"
            min="0.5"
            max="1.5"
            step="0.05"
            value="1.0"
            class="w-full accent-teal-500"
          >
        </div>

        <div>
          <div class="flex justify-between mb-1">
            <label class="text-xs text-zinc-400" for="ws-sharpen">Sharpening (CAS)</label>
            <span class="text-xs text-zinc-500 font-mono">0.8</span>
          </div>
          <input
            id="ws-sharpen"
            type="range"
            min="0"
            max="1"
            step="0.1"
            value="0.8"
            class="w-full accent-teal-500"
          >
        </div>

        <!-- Upscale toggle -->
        <div class="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3 space-y-3">
          <div class="flex items-start justify-between gap-3">
            <label class="flex items-center gap-3 cursor-pointer group">
              <div class="relative flex items-center">
                <input
                  type="checkbox"
                  class="sr-only peer"
                  bind:checked={upscaleEnabled}
                >
                <div class="w-9 h-5 bg-zinc-800 rounded-full peer peer-focus:ring-2 peer-focus:ring-teal-500 peer-checked:after:translate-x-full peer-checked:bg-teal-500 after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-zinc-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all"></div>
              </div>
              <div>
                <span class="text-sm font-medium text-zinc-300 group-hover:text-zinc-100 block">Upscale</span>
                <span class="text-xs text-zinc-500 block mt-0.5">
                  {upscaleEnabled ? 'Upscale enabled.' : 'Upscale disabled.'}
                </span>
              </div>
            </label>
            {#if upscaleEnabled}
              <div class="shrink-0 space-y-1">
                <label class="block text-[11px] font-semibold text-zinc-500 uppercase tracking-wider" for="ws-upscale-factor">Factor</label>
                <select
                  id="ws-upscale-factor"
                  class="w-20 bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-sm text-zinc-300 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
                >
                  <option value="2">2x</option>
                  <option value="4">4x</option>
                </select>
              </div>
            {/if}
          </div>
        </div>
      </div>
    {/if}

    <!-- Video-only: Audio + Low Memory -->
    {#if showVideoControls}
      <div class="pt-4 border-t border-zinc-900 pb-4 space-y-3">
        <label class="flex items-center gap-3 cursor-pointer group">
          <div class="relative flex items-center">
            <input type="checkbox" name="audio" class="sr-only peer" checked>
            <div class="w-9 h-5 bg-zinc-800 rounded-full peer peer-focus:ring-2 peer-focus:ring-teal-500 peer-checked:after:translate-x-full peer-checked:bg-teal-500 after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-zinc-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all"></div>
          </div>
          <div>
            <span class="text-sm font-medium text-zinc-300 group-hover:text-zinc-100 block">Audio (Video output)</span>
          </div>
        </label>

        <label class="flex items-center gap-3 cursor-pointer group">
          <div class="relative flex items-center">
            <input type="checkbox" name="low_memory" class="sr-only peer">
            <div class="w-9 h-5 bg-zinc-800 rounded-full peer peer-focus:ring-2 peer-focus:ring-teal-500 peer-checked:after:translate-x-full peer-checked:bg-teal-500 after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-zinc-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all"></div>
          </div>
          <div>
            <span class="text-sm font-medium text-zinc-300 group-hover:text-zinc-100 block">Low Memory Mode</span>
          </div>
        </label>
      </div>
    {/if}

  </div><!-- end scroll area -->

  <!-- Sticky submit bar -->
  <div class="absolute bottom-0 left-0 right-0 p-4 bg-zinc-950/95 border-t border-zinc-900 shrink-0 backdrop-blur-sm z-10 w-full">
    <p
      id="ws-busy-note"
      class="{busy ? '' : 'hidden'} mb-3 rounded-md border border-zinc-800 bg-zinc-900/80 px-3 py-2 text-xs text-zinc-400"
    >
      An exclusive generation job is running. New runs are disabled until it finishes.
    </p>
    <button
      id="ws-submit"
      type="submit"
      disabled={busy}
      class="surface-button w-full bg-teal-400 hover:bg-teal-500 disabled:bg-zinc-800 disabled:text-zinc-500 disabled:shadow-none text-zinc-950 font-medium py-2.5 rounded-md shadow-[0_0_15px_rgba(20,184,166,0.3)] transition-all active:scale-[0.98] flex items-center justify-center gap-2"
    >
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
      </svg>
      <span>{busy ? 'Generation In Progress' : 'Generate'}</span>
      <span class="text-teal-700 text-xs font-mono ml-2 opacity-80 border border-teal-600/30 px-1.5 py-0.5 rounded">⌘↵</span>
    </button>
  </div>
</section>
