<script lang="ts">
  import { onMount } from 'svelte';
  import { draft } from '$lib/state/draft.svelte';
  import { jobStore } from '$lib/state/job.svelte';
  import { historyStore } from '$lib/state/history.svelte';
  import { router } from '$lib/state/router.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import { getWorkspaceContext, submitGenerate, parseUrlPrefill } from '$lib/api/workspace';
  import { JobCard } from '$lib/components/molecules';
  import ControlsSidebar from './ControlsSidebar.svelte';
  import HistoryPane from './HistoryPane.svelte';
  import type { WorkspaceContext, Workflow } from '$lib/types';

  let context = $state<WorkspaceContext | null>(null);
  let loadError = $state<string | null>(null);
  let busy = $state(false);
  let formEl = $state<HTMLFormElement | undefined>(undefined);
  let imageFile = $state<File | null>(null);

  // Mode derived from workflow
  const isImageMode = $derived(
    draft.state.workflow === 'txt2img' || draft.state.workflow === 'img2img'
  );

  // Map Svelte workflow names → HTMX form values
  const workflowFormMap: Record<Workflow, string> = {
    txt2img: 'image',
    img2img: 'i2i',
    img2vid: 'i2v',
    txt2vid: 'video'
  };

  const imageModels = $derived(context?.image_models ?? []);
  const videoModels = $derived(context?.video_models ?? []);
  const currentModels = $derived(isImageMode ? imageModels : videoModels);
  const loraOptions = $derived(context?.loras ?? []);
  const quantizeOptions = $derived(context?.quantize_options ?? []);

  // Active lora chips parsed from draft.state.loraString
  interface LoraChip {
    name: string;
    weight: number;
  }

  const loraChips = $derived<LoraChip[]>(
    draft.state.loraString
      ? draft.state.loraString.split(',').flatMap((entry) => {
          const [name, w] = entry.trim().split(':');
          if (!name) return [];
          return [{ name, weight: w ? Number(w) : 1.0 }];
        })
      : []
  );

  let loraPopoverOpen = $state(false);

  function addLora(name: string): void {
    const existing = loraChips.find((c) => c.name === name);
    if (existing) { loraPopoverOpen = false; return; }
    const newStr = [...loraChips, { name, weight: 1.0 }]
      .map((c) => `${c.name}:${c.weight}`)
      .join(',');
    draft.update('loraString', newStr);
    loraPopoverOpen = false;
  }

  function removeLora(name: string): void {
    const newStr = loraChips
      .filter((c) => c.name !== name)
      .map((c) => `${c.name}:${c.weight}`)
      .join(',');
    draft.update('loraString', newStr);
  }

  function updateLoraWeight(name: string, weight: number): void {
    const newStr = loraChips
      .map((c) => (c.name === name ? `${c.name}:${weight}` : `${c.name}:${c.weight}`))
      .join(',');
    draft.update('loraString', newStr);
  }

  onMount(() => {
    // Apply URL params BEFORE loading draft (URL params override saved draft)
    const urlParams = parseUrlPrefill();
    if (Object.keys(urlParams).length > 0) {
      draft.loadFromUrl(urlParams);
    } else {
      draft.loadDraft();
    }

    // Load workspace context
    getWorkspaceContext()
      .then((ctx) => {
        context = ctx;
        if (ctx.history_assets.length > 0) {
          historyStore.seedHistory(ctx.history_assets);
        }
        // Set default model from context if draft model is empty
        if (!draft.state.model) {
          const defaultModel = ctx.current_image_model || ctx.image_models[0]?.id || '';
          if (defaultModel) draft.update('model', defaultModel);
        }
      })
      .catch((e: unknown) => {
        loadError = e instanceof Error ? e.message : 'Failed to load workspace context';
      });

    // Keyboard shortcut ⌘↵ / Ctrl↵
    function handleKeydown(e: KeyboardEvent): void {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        formEl?.requestSubmit();
      }
    }
    document.addEventListener('keydown', handleKeydown);
    return () => document.removeEventListener('keydown', handleKeydown);
  });

  async function handleSubmit(e: Event): Promise<void> {
    e.preventDefault();
    if (!formEl || busy) return;
    busy = true;
    draft.saveDraft();

    try {
      const formData = new FormData(formEl);
      // Override hidden fields with correct mapped values
      formData.set('mode', isImageMode ? 'image' : 'video');
      formData.set('workflow', workflowFormMap[draft.state.workflow]);
      // Attach image file if present
      if (imageFile) {
        formData.set('image_file', imageFile);
      }
      // Sync lora string
      formData.set('lora', draft.state.loraString);

      const jobCtx = await submitGenerate(formData);
      jobStore.startJob(
        jobCtx,
        async () => {
          await historyStore.refreshHistory();
          busy = false;
          addToast('Generation complete', 'success');
        },
        () => {
          busy = false;
          addToast('Generation failed', 'error');
        }
      );
    } catch (err) {
      busy = false;
      loadError = err instanceof Error ? err.message : 'Generate failed';
      addToast('Generation failed', 'error');
    }
  }

  function onModelChange(e: Event): void {
    const sel = e.currentTarget as HTMLSelectElement;
    const newModel = sel.value;
    draft.update('model', newModel);
    // Apply per-model defaults when switching models
    if (context) {
      const defaultsMap = isImageMode ? context.image_model_defaults : context.video_model_defaults;
      const modelDefaults = defaultsMap?.[newModel];
      if (modelDefaults) {
        draft.update('steps', modelDefaults.steps);
        if (modelDefaults.guidance) {
          draft.update('guidance', modelDefaults.guidance);
        }
      }
    }
  }
</script>

<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<form
  bind:this={formEl}
  class="flex min-h-0 flex-1 flex-col"
  onsubmit={handleSubmit}
>
  <!-- Hidden fields -->
  <input type="hidden" name="mode" value={isImageMode ? 'image' : 'video'}>
  <input type="hidden" name="workflow" value={workflowFormMap[draft.state.workflow]}>
  <input type="hidden" name="output" value={context?.output_dir ?? context?.config?.output_dir ?? ''}>
  <input type="hidden" name="lora" value={draft.state.loraString}>
  <input type="hidden" name="upscale" value="">
  <input type="hidden" name="sharpen_enabled" value="1">
  <input type="hidden" name="sharpen_amount" value="0.8">
  <input type="hidden" name="contrast_enabled" value="0">
  <input type="hidden" name="contrast_amount" value="1.0">
  <input type="hidden" name="saturation_enabled" value="0">
  <input type="hidden" name="saturation_amount" value="1.0">

  <!-- Toolbar bar: model, quantize, loras -->
  <div class="bg-zinc-900 border-b border-zinc-800 shrink-0 shadow-sm z-10">
    <div class="flex flex-wrap items-center px-4 py-2 gap-4">

      <!-- Model selector -->
      <div class="flex items-center gap-2 shrink-0">
        <label class="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Model</label>
        <div class="relative w-56 group cursor-pointer">
          <div class="bg-zinc-950 border border-zinc-700 hover:border-zinc-600 text-sm rounded px-3 py-1.5 flex items-center justify-between transition text-zinc-200">
            <div class="flex items-center gap-2 truncate">
              <svg class="w-4 h-4 text-teal-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path>
              </svg>
              <span class="truncate">{draft.state.model || (currentModels[0]?.label ?? 'No models')}</span>
            </div>
            <svg class="w-4 h-4 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
            </svg>
          </div>
          <select
            name="model"
            class="absolute inset-0 opacity-0 cursor-pointer"
            value={draft.state.model}
            onchange={onModelChange}
          >
            {#each currentModels as m}
              <option value={m.id}>{m.label}</option>
            {/each}
          </select>
        </div>

        <!-- Quantize selector -->
        <div class="relative w-44 group cursor-pointer">
          <div class="bg-zinc-950 border border-zinc-700 hover:border-zinc-600 text-sm rounded px-3 py-1.5 flex items-center justify-between transition text-zinc-200 font-mono">
            <span class="truncate">Quant: None</span>
            <svg class="w-4 h-4 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
            </svg>
          </div>
          <select name="quantize" class="absolute inset-0 opacity-0 cursor-pointer">
            <option value="">Quant: None</option>
            {#each quantizeOptions as opt}
              <option value={String(opt)}>Quant: q{opt}</option>
            {/each}
          </select>
        </div>
      </div>

      <div class="w-px h-6 bg-zinc-800 shrink-0"></div>

      <!-- LoRA chips area -->
      <div class="flex items-center gap-2 flex-1 min-w-0">
        <label class="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider shrink-0 mt-0.5">LoRAs</label>
        <div class="flex flex-wrap items-center gap-2 flex-1 min-w-0">
          <!-- Chips -->
          {#each loraChips as chip}
            <div class="flex items-center gap-1 rounded-md border border-zinc-700 bg-zinc-900 px-2 py-0.5 text-xs text-zinc-300">
              <span class="truncate max-w-[80px]">{chip.name}</span>
              <input
                type="number"
                step="0.1"
                min="0"
                max="2"
                value={chip.weight}
                class="w-10 bg-transparent text-zinc-400 font-mono text-center focus:outline-none"
                onchange={(e) => updateLoraWeight(chip.name, Number((e.currentTarget as HTMLInputElement).value))}
              >
              <button
                type="button"
                class="text-zinc-500 hover:text-zinc-300 ml-0.5"
                onclick={() => removeLora(chip.name)}
                aria-label="Remove {chip.name}"
              >×</button>
            </div>
          {/each}

          <!-- Add LoRA popover -->
          <div class="relative shrink-0">
            <button
              type="button"
              class="text-xs font-medium text-zinc-400 hover:text-zinc-200 bg-zinc-950 hover:bg-zinc-800 rounded px-2.5 py-1 border border-zinc-700 hover:border-zinc-500 transition flex items-center gap-1 border-dashed disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={loraOptions.length === 0}
              onclick={() => (loraPopoverOpen = !loraPopoverOpen)}
            >
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path>
              </svg>
              Add LoRA
            </button>
            {#if loraPopoverOpen}
              <div class="absolute left-0 top-full mt-2 w-64 rounded-lg border border-zinc-800 bg-zinc-950 shadow-2xl shadow-black/40 p-2 z-20">
                <p class="px-2 pb-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Available LoRAs</p>
                <div class="max-h-56 overflow-y-auto custom-scrollbar flex flex-col gap-1">
                  {#each loraOptions as lora}
                    <button
                      type="button"
                      class="flex items-center justify-between rounded-md px-2 py-2 text-left text-sm text-zinc-300 hover:bg-zinc-900 hover:text-zinc-100 transition"
                      onclick={() => addLora(lora.name)}
                    >
                      <span class="truncate">{lora.name}</span>
                      <span class="text-[10px] uppercase tracking-wide text-zinc-500">Add</span>
                    </button>
                  {/each}
                </div>
              </div>
            {/if}
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Main 3-column layout -->
  <main class="flex-1 flex overflow-hidden">

    <!-- Left: Controls Sidebar -->
    <ControlsSidebar
      {context}
      {busy}
      {imageFile}
      onImageFileChange={(f) => { imageFile = f; }}
    />

    <!-- Center: Canvas -->
    <section class="flex-1 min-w-0 bg-zinc-900 relative z-0 flex flex-col p-6">
      <div
        class="flex-1 bg-zinc-950 border border-zinc-800 rounded-lg shadow-inner flex items-center justify-center relative overflow-hidden"
      >
        {#if loadError}
          <div class="text-center p-8">
            <p class="text-red-400 text-sm font-medium">Error</p>
            <p class="text-zinc-500 text-xs mt-1">{loadError}</p>
          </div>
        {:else if jobStore.current}
          <div class="w-full h-full flex items-center justify-center p-6">
            <div class="w-full max-w-md">
              <JobCard
                job={jobStore.current}
                oncancel={(id) => fetch(`/jobs/${id}/controls/quit`, { method: 'POST' })}
                onpause={(id) => fetch(`/jobs/${id}/controls/pause`, { method: 'POST' })}
                onresume={(id) => fetch(`/jobs/${id}/controls/resume`, { method: 'POST' })}
                onnext={(id) => fetch(`/jobs/${id}/controls/next`, { method: 'POST' })}
                onrepeat={(id) => fetch(`/jobs/${id}/controls/repeat`, { method: 'POST' })}
              />
            </div>
          </div>
        {:else if historyStore.assets.length > 0}
          {@const latest = historyStore.assets[0]}
          <div class="w-full h-full flex items-center justify-center p-4">
            {#if latest.media_type === 'video'}
              <video
                src={latest.url}
                controls
                muted
                preload="metadata"
                class="max-w-full max-h-full object-contain rounded"
              ></video>
            {:else}
              <img
                src={latest.url}
                alt={latest.prompt}
                class="max-w-full max-h-full object-contain rounded"
              >
            {/if}
          </div>
        {:else}
          <div class="flex flex-col items-center justify-center gap-4 text-center p-8">
            <div class="rounded-full bg-zinc-900 p-4 border border-zinc-800">
              <svg class="w-8 h-8 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <p class="text-zinc-400 text-sm font-medium">No generated assets yet</p>
              <p class="text-zinc-600 text-xs mt-1">Write a prompt and press Generate to get started</p>
            </div>
          </div>
        {/if}
      </div>
    </section>

    <!-- Right: History pane -->
    <HistoryPane assets={historyStore.assets} loading={historyStore.loading} />
  </main>
</form>
