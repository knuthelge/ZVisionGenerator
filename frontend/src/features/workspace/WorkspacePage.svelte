<script lang="ts">
  import { onMount, tick, untrack } from 'svelte';
  import { api } from '$lib/api/client';
  import { draft } from '$lib/state/draft.svelte';
  import { enhancedOverrideActive, submittedPrompt } from '$lib/state/promptEnhance';
  import { jobStore } from '$lib/state/job.svelte';
  import { historyStore } from '$lib/state/history.svelte';
  import { router } from '$lib/state/router.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import { getWorkspaceCoreContext, submitGenerate, parseUrlPrefill } from '$lib/api/workspace';
  import { deleteAsset } from '$lib/api/gallery';
  import { referenceParams, reuseParams, type DeleteOptions, type ReferenceTarget } from '$lib/state/assetActions';
  import { Icon, MascotSpot, ToolbarSelectShell } from '$lib/components/atoms';
  import { rememberMascotSpots } from '$lib/components/atoms/MascotSpot.svelte';
  import {
    DROWSY_AFTER_MS,
    GREETING_DURATION_MS,
    REACTION_DURATION_MS,
    TYPING_DURATION_MS,
    mascotMood as pickMascotMood,
    type MascotReaction,
  } from '$lib/state/mascot';
  import { ActionMenu, AssetTile, AssetViewer, JobCard, ModelStatusBadges, requestConfirm, type ActionMenuEntry } from '$lib/components/molecules';
  import { jobSettingsPrefill } from '$lib/state/jobSettings';
  import { formatLoraString, loraLabel, parseLoraString, type LoraChip } from '$lib/state/loras';
  import QueuePanel from './QueuePanel.svelte';
  import { confirmDeleteAsset } from '$lib/state/assetActions';
  import ControlsSidebar from './ControlsSidebar.svelte';
  import HistoryStrip from './HistoryStrip.svelte';
  import { fitOutputGrid } from './outputGrid';
  import { randomSeed } from './seed';
  import { hasOpenModal, isCommandKey } from '$lib/keyboard';
  import { startUpscale } from '$lib/state/upscale';
  import { applyPromptRun, takePromptRun, type PendingPromptRun } from '$lib/state/pendingPromptRun';
  import type { GalleryAsset, JobSnapshot, UpscaleFactor, WorkspaceContext, Workflow } from '$lib/types';

  let context = $state<WorkspaceContext | null>(null);
  let loadError = $state<string | null>(null);
  let busy = $state(false);
  let formEl = $state<HTMLFormElement | undefined>(undefined);
  let imageFile = $state<File | null>(null);
  let lightboxOpen = $state(false);
  let lightboxIndex = $state(0);
  let lightboxMode = $state<'history' | 'completed-output'>('history');
  let completedOutputTrigger = $state<HTMLElement | null>(null);
  let completedOutputAssetId = $state<string | null>(null);
  let completedOutputJobId = $state<string | null>(null);
  let reaction = $state<MascotReaction | null>(null);
  let greeting = $state(true);
  let typing = $state(false);
  let drowsy = $state(false);
  let loadedLatestUrl = $state<string | null>(null);
  // False until the first history fetch settles, so a returning user never sees "no assets" first.
  let historyChecked = $state(false);
  let deletingIds = $state<Set<string>>(new Set());
  let reactionTimer: ReturnType<typeof setTimeout> | undefined;
  let typingTimer: ReturnType<typeof setTimeout> | undefined;

  const OUTPUT_GAP_PX = 12;
  const MIN_OUTPUT_CELL_PX = 160;
  let outputAreaWidth = $state(0);
  let outputAreaHeight = $state(0);
  const jobOutputs = $derived<GalleryAsset[]>(jobStore.current?.outputs ?? []);
  const outputGrid = $derived(fitOutputGrid(
    jobOutputs.length,
    outputAreaWidth,
    outputAreaHeight,
    OUTPUT_GAP_PX,
    MIN_OUTPUT_CELL_PX
  ));
  const hasCompletedOutputs = $derived(jobStore.current?.status === 'completed' && jobOutputs.length > 0);
  const previewView = $derived<'error' | 'outputs' | 'job' | 'latest' | 'empty'>(
    loadError ? 'error'
      : hasCompletedOutputs ? 'outputs'
      : jobStore.current && (jobStore.isRunning || jobOutputs.length > 0) ? 'job'
      : historyStore.assets.length > 0 ? 'latest'
      : 'empty'
  );
  const latestAsset = $derived<GalleryAsset | null>(historyStore.assets[0] ?? null);
  // A reference path that points at a known asset can show that asset as its thumbnail.
  const referencePreviewUrl = $derived(
    draft.state.referenceImagePath
      ? (historyStore.assets.find((asset) => asset.file_path === draft.state.referenceImagePath)?.url ?? null)
      : null
  );
  // Keep the preview occupied until the latest image has painted. Videos show at once:
  // some browsers load nothing until play is pressed, so no load event is guaranteed.
  const latestLoading = $derived(
    previewView === 'latest'
      && latestAsset !== null
      && latestAsset.media_type !== 'video'
      && loadedLatestUrl !== latestAsset.url
  );
  // The latest asset fills the stage, so the mascot docks in a corner there (outputs have their own header spot).
  const dockMascot = $derived(previewView === 'latest' && !latestLoading);
  // Measure the mascot before the preview swaps views, so it hops from where it really was.
  $effect.pre(() => {
    void previewView;
    void latestLoading;
    untrack(rememberMascotSpots);
  });
  const lookingForHistory = $derived(!historyChecked || historyStore.loading);
  const mascotMood = $derived(pickMascotMood({
    job: jobStore.current,
    reaction,
    loadError: loadError !== null,
    loading: latestLoading || (previewView === 'empty' && lookingForHistory),
    greeting,
    typing,
    drowsy,
  }));
  const lightboxAssets = $derived<GalleryAsset[]>(
    lightboxMode === 'completed-output'
      ? jobOutputs
      : historyStore.assets
  );

  // Surface successful batch items in History as soon as their SSE event lands.
  // The terminal refresh remains authoritative and reconciles the full inventory.
  $effect(() => {
    const outputs = jobOutputs;
    if (jobStore.isRunning && outputs.length > 0) {
      untrack(() => historyStore.mergeOutputs(outputs));
    }
  });

  // Mode derived from workflow
  const isImageMode = $derived(
    draft.state.workflow === 'txt2img' || draft.state.workflow === 'img2img'
  );
  const authorityReady = $derived(context !== null && draft.authorityReady);

  const imageModels = $derived(context?.image_models ?? []);
  const videoModels = $derived(context?.video_models ?? []);
  const currentModels = $derived(isImageMode ? imageModels : videoModels);
  // Matches the toolbar label, which falls back to the first model when the draft has none.
  const selectedModel = $derived(currentModels.find((m) => m.id === (draft.state.model || currentModels[0]?.id)) ?? null);
  const loraOptions = $derived(context?.loras ?? []);
  const quantizeOptions = $derived(context?.quantize_options ?? []);
  const visibleControls = $derived(
    new Set(context?.workflow_contract.definitions[draft.state.workflow]?.visible_controls ?? [])
  );
  const supportsQuantize = $derived(
    authorityReady
      && visibleControls.has('quantize')
      && isImageMode
      ? (context?.image_model_defaults?.[draft.state.model]?.supports_quantize ?? context?.defaults?.supports_quantize ?? false)
      : false
  );

  // Active lora chips parsed from draft.state.loraString; reused settings record paths, shown by LoRA name.
  const loraChips = $derived<LoraChip[]>(parseLoraString(draft.state.loraString, loraOptions));

  let loraPopoverOpen = $state(false);
  let loraButtonEl = $state<HTMLButtonElement | null>(null);
  const loraMenuItems = $derived<ActionMenuEntry[]>([
    { kind: 'heading', label: 'Available LoRAs' },
    ...loraOptions.map((lora): ActionMenuEntry => ({ kind: 'item', id: lora.name, label: lora.name, onselect: () => addLora(lora.name) })),
  ]);

  function addLora(name: string): void {
    const existing = loraChips.find((c) => c.name === name);
    if (existing) { loraPopoverOpen = false; return; }
    draft.update('loraString', formatLoraString([...loraChips, { name, weight: 1.0 }]));
    loraPopoverOpen = false;
  }

  function removeLora(name: string): void {
    draft.update('loraString', formatLoraString(loraChips.filter((c) => c.name !== name)));
  }

  function updateLoraWeight(name: string, weight: number): void {
    draft.update('loraString', formatLoraString(loraChips.map((c) => (c.name === name ? { name, weight } : c))));
  }


  function react(next: MascotReaction): void {
    clearTimeout(reactionTimer);
    reaction = next;
    reactionTimer = setTimeout(() => { reaction = null; }, REACTION_DURATION_MS[next]);
  }

  function markTyping(): void {
    clearTimeout(typingTimer);
    typing = true;
    typingTimer = setTimeout(() => { typing = false; }, TYPING_DURATION_MS);
  }

  // Jobs download models on first use, so re-read download and memory status once one ends.
  // Only the latest refresh applies, so a slow older response cannot overwrite newer status.
  let modelStatusRequest = 0;
  function refreshModelStatus(): void {
    const request = ++modelStatusRequest;
    getWorkspaceCoreContext()
      .then((fresh) => {
        if (request !== modelStatusRequest || !context) return;
        context = { ...context, image_models: fresh.image_models, video_models: fresh.video_models };
      })
      .catch(() => undefined);
  }

  async function handleJobComplete(): Promise<void> {
    react('cheerful');
    refreshModelStatus();
    await historyStore.refreshHistory();
    addToast('Generation complete', 'success');
  }

  function handleJobFailed(): void {
    react('sad');
    refreshModelStatus();
    addToast(jobStore.queuedJobs.length > 0 ? 'Generation failed. Starting the next queued job.' : 'Generation failed', 'error');
  }

  async function handleJobLost(): Promise<void> {
    refreshModelStatus();
    addToast('Lost track of the job. Refreshed the gallery with any results.', 'info');
    await historyStore.refreshHistory();
  }

  function handleJobCancelled(): void {
    react('surprised');
    refreshModelStatus();
    addToast('Generation stopped', 'info');
  }

  function openHistoryViewer(asset: GalleryAsset, trigger: HTMLElement): void {
    const index = historyStore.assets.findIndex((item) => item.id === asset.id);
    if (index < 0) return;
    lightboxMode = 'history';
    lightboxIndex = index;
    completedOutputTrigger = trigger;
    lightboxOpen = true;
  }

  function openOutputViewer(asset: GalleryAsset, trigger: HTMLElement): void {
    openCompletedOutputViewer(jobOutputs.findIndex((output) => output.id === asset.id), trigger);
  }

  /** Apply workspace prefill params (workflow first, then model defaults, then the params on top). */
  function applyPrefill(params: Record<string, string>, options: { keepSettings?: boolean } = {}): void {
    if (!context) return;
    if (params.workflow) draft.loadFromUrl({ workflow: params.workflow }, context);
    draft.hydrateFromContext(context, params.model ?? null, options);
    draft.loadFromUrl(params, context);
    // The prefill already re-hydrated for its workflow; the workflow-change effect must not redo it.
    _prevWorkflow = draft.state.workflow;
    draft.saveDraft();
  }

  /** Refill the form from the settings a queued job was submitted with. */
  function loadQueuedSettings(job: JobSnapshot): void {
    const { params, patch } = jobSettingsPrefill(job.settings ?? {});
    applyPrefill(params);
    draft.patch(patch);
    imageFile = null;
    addToast('Loaded the queued job\'s settings', 'success');
  }

  async function removeQueuedJob(job: JobSnapshot): Promise<void> {
    try {
      await jobStore.removeQueued(job.job_id ?? job.id);
    } catch {
      addToast('Could not remove the job from the queue', 'error');
    }
  }

  async function clearQueuedJobs(): Promise<void> {
    const count = jobStore.queuedJobs.length;
    const approved = await requestConfirm({
      question: `Remove ${count} queued job${count === 1 ? '' : 's'}?`,
      info: 'The running job keeps going.',
      confirmLabel: 'Clear queue',
      cancelLabel: 'Keep',
    });
    if (!approved) return;
    try {
      await jobStore.clearQueue();
    } catch {
      addToast('Could not clear the queue', 'error');
    }
    await tick();
    // The panel (and its button) is gone; Generate is the next useful place.
    const generate = document.querySelector<HTMLButtonElement>(draft.state.sidebarCollapsed ? '[data-testid="sidebar-strip"] button[type="submit"]' : '#ws-submit');
    generate?.focus();
  }

  function reuseAsset(asset: GalleryAsset): void {
    applyPrefill(reuseParams(asset));
    imageFile = null;
    closeLightbox();
    addToast(`Loaded settings from ${asset.filename}`, 'success');
  }

  async function upscaleAsset(asset: GalleryAsset, factor: UpscaleFactor): Promise<void> {
    // Like Generate, an upscale joins the queue while another job runs.
    if (await startUpscale(asset, factor)) closeLightbox();
  }

  // Reference targets the model that would run them can't use, with the reason shown in the menu.
  const referenceUnavailable = $derived.by<Partial<Record<ReferenceTarget, string>>>(() => {
    if (!context) return {};
    const imageModel = isImageMode ? draft.state.model : (context.current_image_model ?? '');
    const videoModel = isImageMode ? (context.current_video_model ?? '') : draft.state.model;
    const reasons: Partial<Record<ReferenceTarget, string>> = {};
    if (context.image_model_defaults?.[imageModel]?.supports_img2img === false) reasons.image = `${imageModel} can't use a reference image`;
    if (context.video_model_defaults?.[videoModel]?.supports_i2v === false) reasons.video = `${videoModel} can't start from an image`;
    return reasons;
  });

  function useAsReference(asset: GalleryAsset, target: ReferenceTarget): void {
    if (!asset.file_path || referenceUnavailable[target]) return;
    const params = referenceParams(asset, target);
    if (draft.state.workflow === params.workflow) draft.update('referenceImagePath', params.image_path);
    else applyPrefill(params, { keepSettings: true });
    imageFile = null;
    closeLightbox();
    addToast(`${asset.filename} is now the reference image`, 'success');
  }

  // Tried in order: the visible prompt field first, else any control in the pane.
  const COMPOSE_FOCUS = ['.compose-pane textarea:not([hidden]):not([disabled])', '.compose-pane button:not([disabled])'];
  const SETTINGS_FOCUS = ['.settings-pane :is(input, select, textarea, button):not([disabled]):not([type="hidden"])'];
  // The newest history tile, else the strip's toggle when there is no history yet.
  const HISTORY_FOCUS = ['#ws-history-scroll .asset-tile-media', '#ws-history-toggle'];
  const PANE_FOCUS: Readonly<Record<string, readonly string[]>> = { Digit1: COMPOSE_FOCUS, Digit2: SETTINGS_FOCUS, Digit3: HISTORY_FOCUS };

  function focusFirst(selectors: readonly string[]): void {
    for (const selector of selectors) {
      const el = document.querySelector<HTMLElement>(selector);
      if (el) { el.focus(); return; }
    }
  }

  async function focusHistory(): Promise<void> {
    if (draft.state.historyCollapsed) {
      draft.update('historyCollapsed', false);
      await tick();
    }
    focusFirst(HISTORY_FOCUS);
  }

  async function deleteWorkspaceAsset(asset: GalleryAsset, options: DeleteOptions = {}): Promise<void> {
    if (deletingIds.has(asset.id)) return;
    if (options.confirm !== false && !(await confirmDeleteAsset(asset))) return;
    if (deletingIds.has(asset.id)) return;
    deletingIds = new Set([...deletingIds, asset.id]);
    try {
      await deleteAsset(asset.id);
      jobStore.removeOutputs([asset.id]);
      historyStore.removeAssets([asset.id]);
      // A reference pointing at the deleted file would fail the next run.
      if (asset.file_path && draft.state.referenceImagePath === asset.file_path) draft.update('referenceImagePath', null);
      addToast('Deleted', 'success');
    } catch {
      addToast('Delete failed', 'error');
    } finally {
      deletingIds = new Set([...deletingIds].filter((id) => id !== asset.id));
    }
  }

  function openCompletedOutputViewer(index: number, trigger: HTMLElement): void {
    if (!jobOutputs[index]) return;
    lightboxMode = 'completed-output';
    lightboxIndex = index;
    completedOutputTrigger = trigger;
    completedOutputAssetId = jobOutputs[index].id;
    completedOutputJobId = jobStore.current?.job_id ?? null;
    lightboxOpen = true;
  }

  function closeLightbox(): void {
    if (!lightboxOpen) return;
    lightboxOpen = false;
    const trigger = completedOutputTrigger?.isConnected ? completedOutputTrigger : null;
    lightboxMode = 'history';
    completedOutputTrigger = null;
    completedOutputAssetId = null;
    completedOutputJobId = null;
    queueMicrotask(() => trigger?.focus());
  }

  function navigateLightbox(index: number): void {
    lightboxIndex = index;
    if (lightboxMode === 'completed-output') {
      completedOutputAssetId = jobOutputs[index]?.id ?? null;
    }
  }

  $effect(() => {
    if (lightboxMode !== 'completed-output') return;
    if (jobOutputs.length === 0 || completedOutputJobId !== jobStore.current?.job_id) {
      lightboxOpen = false;
      lightboxIndex = 0;
      completedOutputAssetId = null;
      completedOutputJobId = null;
      return;
    }
    const selectedIndex = jobOutputs.findIndex((output) => output.id === completedOutputAssetId);
    if (selectedIndex >= 0) {
      if (lightboxIndex !== selectedIndex) lightboxIndex = selectedIndex;
    } else {
      // The shown output was deleted: stay on its neighbour, like the history and gallery viewers.
      const nextIndex = Math.min(lightboxIndex, jobOutputs.length - 1);
      lightboxIndex = nextIndex;
      completedOutputAssetId = jobOutputs[nextIndex].id;
    }
  });

  // Track prev workflow to detect user-initiated changes after context loads.
  let _prevWorkflow: Workflow | null = null;

  $effect(() => {
    const currentWorkflow = draft.state.workflow;
    if (_prevWorkflow !== null && context !== null && currentWorkflow !== _prevWorkflow) {
      draft.onWorkflowChange(currentWorkflow, context);
    }
    _prevWorkflow = currentWorkflow;
  });

  onMount(() => {
    const unsubscribeLifecycle = jobStore.subscribeLifecycle({
      onComplete: handleJobComplete,
      onFailed: handleJobFailed,
      onCancelled: handleJobCancelled,
      onLost: handleJobLost,
    });
    const stopJobSync = jobStore.startSync();
    const urlParams = parseUrlPrefill();
    const hasUrlParams = Object.keys(urlParams).length > 0;

    // Load stored draft as the baseline state.
    draft.loadDraft();

    let cancelled = false;
    let historyTimer: ReturnType<typeof setTimeout> | undefined;

    // Load workspace authority first; gallery/history hydration follows after the core page can paint.
    getWorkspaceCoreContext()
      .then((ctx) => {
        if (cancelled) return;
        context = ctx;
        if (ctx.history_assets.length > 0) {
          historyStore.seedHistory(ctx.history_assets);
        }
        if (urlParams.workflow) {
          draft.loadFromUrl({ workflow: urlParams.workflow }, ctx);
        }
        // Hydrate model + defaults from backend for the current workflow.
        // preferredModel = URL-specified model (may be null).
        // A saved draft keeps its settings across visits and restarts unless a URL names a model to reuse.
        draft.hydrateFromContext(ctx, urlParams.model ?? null, { keepSettings: !urlParams.model });

        // Re-apply remaining URL params on top of the backend defaults so that
        // explicit URL values (prompt, steps, ratio, etc.) take precedence.
        if (hasUrlParams) {
          draft.loadFromUrl(urlParams, ctx);
        }

        // After full hydration, sync _prevWorkflow so the workflow-change $effect
        // does not fire for the initial state.
        _prevWorkflow = draft.state.workflow;
        jobStore.seedQueue(ctx.queued_jobs ?? []);
        void jobStore.reconnectActiveJob({ snapshot: ctx.active_job });
        // "Generate this one" on the Prompts page: queue that prompt with these settings once the form exists.
        const run = takePromptRun();
        if (run) void tick().then(() => submitForm(run));

        historyTimer = setTimeout(() => {
          if (cancelled) return;
          void historyStore.refreshHistory().finally(() => {
            if (!cancelled) historyChecked = true;
          });
        }, 0);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        loadError = e instanceof Error ? e.message : 'Failed to load workspace context';
      });

    // ⌘↵ / Ctrl↵ generates; with ⇧ a locked seed is re-rolled first. Alt+1 / 2 / 3 jump to Compose / Settings / History.
    function handleKeydown(e: KeyboardEvent): void {
      // The full-screen viewer covers the form; generating behind it would be a surprise.
      if (e.defaultPrevented || lightboxOpen || hasOpenModal()) return;
      if (isCommandKey(e) && !e.altKey && e.key === 'Enter') {
        e.preventDefault();
        // A held key must not queue the same run again and again.
        if (e.repeat) return;
        // Commit the focused field first, so a typed number settles on a valid step before validation.
        (document.activeElement as HTMLElement | null)?.blur?.();
        if (e.shiftKey && draft.state.seed !== null) draft.update('seed', randomSeed());
        void tick().then(() => formEl?.requestSubmit());
      } else if (e.altKey && !isCommandKey(e) && !e.shiftKey && e.code in PANE_FOCUS) {
        // `code`, not `key`: Alt+1 types "¡" on a Mac keyboard.
        e.preventDefault();
        if (e.code === 'Digit3') void focusHistory();
        else focusFirst(PANE_FOCUS[e.code]);
      }
    }
    document.addEventListener('keydown', handleKeydown);

    // Mascot: wave hello, then doze off after a quiet minute.
    const greetingTimer = setTimeout(() => { greeting = false; }, GREETING_DURATION_MS);
    let lastActivity = Date.now();
    function handleActivity(): void {
      lastActivity = Date.now();
      if (drowsy) drowsy = false;
    }
    const activityEvents = ['pointerdown', 'pointermove', 'keydown', 'wheel'] as const;
    activityEvents.forEach((type) => document.addEventListener(type, handleActivity, { passive: true }));
    const drowsyTimer = setInterval(() => {
      drowsy = Date.now() - lastActivity > DROWSY_AFTER_MS;
    }, 5000);

    return () => {
      clearTimeout(greetingTimer);
      clearTimeout(reactionTimer);
      clearTimeout(typingTimer);
      clearInterval(drowsyTimer);
      activityEvents.forEach((type) => document.removeEventListener(type, handleActivity));
      cancelled = true;
      stopJobSync();
      unsubscribeLifecycle();
      if (historyTimer) clearTimeout(historyTimer);
      document.removeEventListener('keydown', handleKeydown);
    };
  });

  let revealingInvalid = false;

  // The browser cannot report an invalid field inside the collapsed sidebar, so expand it and report once shown.
  function revealInvalidField(event: Event): void {
    const field = event.target as HTMLInputElement;
    if (!draft.state.sidebarCollapsed || !field.closest?.('#ws-controls-sidebar')) return;
    event.preventDefault();
    if (revealingInvalid) return;
    revealingInvalid = true;
    draft.update('sidebarCollapsed', false);
    void tick().then(() => {
      revealingInvalid = false;
      field.reportValidity();
    });
  }

  async function handleSubmit(e: Event): Promise<void> {
    e.preventDefault();
    await submitForm(null);
  }

  /** Submit the form; with `run`, queue that one prompt-file prompt instead of the form's prompt. */
  async function submitForm(run: PendingPromptRun | null): Promise<void> {
    if (!formEl || busy || !authorityReady) return;
    if (!run && draft.state.promptSource === 'file' && (!draft.state.promptFilePath || draft.state.promptFileOptionIds.length === 0)) {
      return;
    }
    loadError = null;
    busy = true;
    draft.saveDraft();

    try {
      const formData = new FormData(formEl);
      // Send canonical workflow directly; backend accepts and normalises it.
      formData.set('mode', isImageMode ? 'image' : 'video');
      formData.set('workflow', draft.state.workflow);
      // Attach image file only for workflows that actually use a reference image
      if (imageFile && (draft.state.workflow === 'img2img' || draft.state.workflow === 'img2vid')) {
        formData.set('image_file', imageFile);
      }
      // Sync lora string
      formData.set('lora', formatLoraString(loraChips));
      // A non-blank Enhanced prompt replaces the inline prompt (never in JSON-caption, file, or auto-enhance mode).
      if (run) {
        applyPromptRun(formData, run);
      } else if (enhancedOverrideActive(draft.state) && formData.has('prompt')) {
        formData.set('prompt', submittedPrompt(draft.state));
      }

      const jobCtx = await submitGenerate(formData);
      jobStore.jobSubmitted(jobCtx);
      if (run) addToast(`Queued ${run.optionId} from the prompt file.`, 'success');
      if (jobCtx.queue_position) addToast(`Added to the queue as #${jobCtx.queue_position}.`, 'info');
    } catch (err) {
      loadError = err instanceof Error ? err.message : 'Generate failed';
      addToast('Generation failed', 'error');
    } finally {
      busy = false;
    }
  }

  function onModelChange(e: Event): void {
    const sel = e.currentTarget as HTMLSelectElement;
    const newModel = sel.value;
    // Re-hydrate defaults for the new model, preserving the current workflow.
    if (context) {
      draft.hydrateFromContext(context, newModel);
    } else {
      draft.update('model', newModel);
    }
  }
</script>

{#snippet assetMeta(asset: GalleryAsset)}
  <p class="latest-meta">
    <span class="truncate" title={asset.prompt}>{asset.prompt || asset.filename}</span>
    {#if asset.model}<b>{asset.model}</b>{/if}
    {#if asset.width && asset.height}<b>{asset.width}×{asset.height}</b>{/if}
    {#if asset.seed != null}<b>seed {asset.seed}</b>{/if}
  </p>
{/snippet}

<form
  bind:this={formEl}
  class="flex min-h-0 flex-1 flex-col"
  onsubmit={handleSubmit}
  oninput={markTyping}
  oninvalidcapture={revealInvalidField}
>
  <!-- Hidden fields -->
  <input type="hidden" name="mode" value={isImageMode ? 'image' : 'video'}>
  <input type="hidden" name="workflow" value={draft.state.workflow}>
  <input type="hidden" name="lora" value={formatLoraString(loraChips)}>

  <!-- Toolbar bar: model, quantize, loras -->
  <div class="ui-pane-toolbar z-10 shrink-0">
    <div class="flex flex-wrap items-center px-3 py-2 gap-3">

      <!-- Model selector -->
      <div class="flex flex-wrap items-center gap-2 min-w-0">
        <label class="ui-label" for="ws-model">Model</label>
        <ToolbarSelectShell
          id="ws-model"
          name="model"
          testId="model-shell"
          class="w-48"
          value={draft.state.model}
          disabled={!authorityReady || currentModels.length === 0}
          onchange={onModelChange}
        >
          {#snippet children()}
            <span class="flex min-w-0 items-center gap-2 truncate pointer-events-none">
              <svg class="text-primary-main h-4 w-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path>
              </svg>
              <span class="truncate">{authorityReady ? (draft.state.model || (currentModels[0]?.label ?? 'No models')) : 'Loading models…'}</span>
            </span>
          {/snippet}
          {#snippet options()}
            {#each currentModels as m}
              <option value={m.id}>{m.downloaded === false ? `${m.label} (not downloaded)` : m.label}</option>
            {/each}
          {/snippet}
        </ToolbarSelectShell>

        <!-- Quantize selector (image workflows only, when model supports it) -->
        {#if supportsQuantize}
          <ToolbarSelectShell
            name="quantize"
            testId="quantize-shell"
            class="w-40"
            value={draft.state.quantize !== null ? String(draft.state.quantize) : ''}
            disabled={!authorityReady}
            onchange={(e) => {
              const v = (e.currentTarget as HTMLSelectElement).value;
              draft.update('quantize', v ? Number(v) : null);
            }}
          >
            {#snippet children()}
              <span class="truncate pointer-events-none">{draft.state.quantize !== null ? `Quant: q${draft.state.quantize}` : 'Quant: None'}</span>
            {/snippet}
            {#snippet options()}
              <option value="">Quant: None</option>
              {#each quantizeOptions as opt}
                <option value={String(opt)}>Quant: q{opt}</option>
              {/each}
            {/snippet}
          </ToolbarSelectShell>
        {/if}

        {#if authorityReady && selectedModel}
          <ModelStatusBadges
            downloaded={selectedModel.downloaded}
            memoryFit={selectedModel.memory_fit}
            quantize={supportsQuantize ? draft.state.quantize : null}
            lowMemory={!visibleControls.has('low_memory') || draft.state.lowMemory}
            size="control"
            tooltipPlacement="bottom"
          />
        {/if}
      </div>

      <div class="bg-border-subtle h-6 w-px shrink-0"></div>

      <!-- LoRA chips area -->
      <div class="flex items-center gap-2 flex-1 min-w-0">
        <span class="ui-label shrink-0">LoRAs</span>
        {#if authorityReady}
          <div class="flex flex-wrap items-center gap-2 flex-1 min-w-0">
            <!-- Chips -->
            {#each loraChips as chip}
              <!-- Neutral chip: coral is reserved for $snippet syntax in prompts. -->
              <div class="ui-chip lora-chip">
                <span class="max-w-28 truncate" title={chip.name}>{loraLabel(chip.name)}</span>
                <input
                  type="number"
                  step="0.1"
                  min="-10"
                  max="10"
                  value={chip.weight}
                  aria-label="{loraLabel(chip.name)} weight"
                  class="lora-weight"
                  onchange={(e) => updateLoraWeight(chip.name, Number((e.currentTarget as HTMLInputElement).value))}
                >
                <button
                  type="button"
                  class="lora-remove"
                  onclick={() => removeLora(chip.name)}
                  aria-label="Remove {loraLabel(chip.name)}"
                ><Icon name="close" size={11} /></button>
              </div>
            {/each}

            <!-- Add LoRA menu -->
            <button
              type="button"
              bind:this={loraButtonEl}
              class="ui-btn ui-btn-add"
              disabled={loraOptions.length === 0}
              aria-haspopup="menu"
              aria-expanded={loraPopoverOpen}
              onclick={() => (loraPopoverOpen = !loraPopoverOpen)}
            >
              <Icon name="plus" size={12} />
              Add LoRA
            </button>
            <ActionMenu
              open={loraPopoverOpen}
              anchor={loraButtonEl}
              align="start"
              label="Available LoRAs"
              items={loraMenuItems}
              onclose={() => (loraPopoverOpen = false)}
            />
          </div>
        {:else}
          <p class="ui-help">Loading models…</p>
        {/if}
      </div>
    </div>
  </div>

  <!-- Left column (Compose + Settings) and the stage with its history filmstrip -->
  <main class="workspace-layout min-h-0 flex-1 overflow-hidden" class:sidebar-collapsed={draft.state.sidebarCollapsed}>
    <ControlsSidebar
      {context}
      {busy}
      jobsActive={jobStore.jobsActive}
      queuedCount={jobStore.queuedJobs.length}
      {imageFile}
      {referencePreviewUrl}
      lastSeed={historyStore.assets[0]?.seed ?? null}
      onImageFileChange={(f) => { imageFile = f; }}
    />

    <section class="workspace-preview relative z-0 flex min-h-0 min-w-0 flex-col bg-bg-base">
      <div class="ui-pane-header flex h-10 shrink-0 items-center justify-between px-3">
        <h2 class="ui-area-label">Preview</h2>
        <span class="text-meta text-text-muted">{jobStore.isRunning ? 'Generating…' : hasCompletedOutputs || historyStore.assets.length ? 'Latest output' : 'Ready'}</span>
      </div>
      <div class="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden">
        {#if loadError}
          <div class="flex flex-col items-center gap-3 text-center p-8">
            <MascotSpot mood={mascotMood} size={112} />
            <p class="font-heading text-content font-extrabold text-error">The Workspace could not load</p>
            <p class="ui-help">{loadError}</p>
          </div>
        {:else if hasCompletedOutputs}
          <div class="completed-output-region flex h-full w-full min-w-0 flex-col p-4">
            <div class="mb-3 flex shrink-0 items-center justify-between gap-3">
              <div class="flex items-center gap-2">
                <MascotSpot mood={mascotMood} size={48} />
                <h3 class="ui-area-label">Completed outputs</h3>
              </div>
              <span class="font-mono text-meta text-text-muted">{jobOutputs.length}</span>
            </div>
            {#if jobOutputs.length === 1}
              {@const output = jobOutputs[0]}
              <div class="latest-region flex min-h-0 flex-1 flex-col px-16">
                <AssetTile
                  asset={output}
                  density="stage"
                  eager
                  deleting={deletingIds.has(output.id)}
                  class="completed-output min-h-0 flex-1"
                  onpreview={openOutputViewer}
                  onreuse={reuseAsset}
                  onreference={useAsReference}
                  onupscale={upscaleAsset}
                  {referenceUnavailable}
                  ondelete={deleteWorkspaceAsset}
                />
                {@render assetMeta(output)}
              </div>
            {:else}
              <!-- Centred while the outputs fit; once they overflow, the auto margins collapse and the grid scrolls from the top. -->
              <div
                class="completed-output-scroll min-h-0 flex-1 overflow-y-auto"
                bind:clientWidth={outputAreaWidth}
                bind:clientHeight={outputAreaHeight}
              >
                <div
                  class="completed-output-grid"
                  style="grid-template-columns: repeat({outputGrid.columns}, {outputGrid.cellSize}px); grid-auto-rows: {outputGrid.cellSize}px; gap: {OUTPUT_GAP_PX}px"
                >
                  {#each jobOutputs as output, index (output.id)}
                    <AssetTile
                      asset={output}
                      density="stage"
                      eager={index === 0}
                      deleting={deletingIds.has(output.id)}
                      class="completed-output"
                      onpreview={openOutputViewer}
                      onreuse={reuseAsset}
                      onreference={useAsReference}
                      onupscale={upscaleAsset}
                  {referenceUnavailable}
                      ondelete={deleteWorkspaceAsset}
                    />
                  {/each}
                </div>
              </div>
            {/if}
          </div>
        {:else if jobStore.current && (jobStore.isRunning || jobOutputs.length > 0)}
          <div class="h-full w-full overflow-y-auto p-6">
            <div class="mx-auto w-full max-w-4xl">
              <MascotSpot mood={mascotMood} size={112} class="mx-auto mb-2 w-fit" />
              <JobCard
                job={jobStore.current!}
                oncancel={(id) => api.post(`/jobs/${encodeURIComponent(id)}/controls/quit`)}
                onpause={(id) => api.post(`/jobs/${encodeURIComponent(id)}/controls/pause`)}
                onresume={(id) => api.post(`/jobs/${encodeURIComponent(id)}/controls/resume`)}
                onnext={(id) => api.post(`/jobs/${encodeURIComponent(id)}/controls/next`)}
                onrepeat={(id) => api.post(`/jobs/${encodeURIComponent(id)}/controls/repeat`)}
              />
              <QueuePanel jobs={jobStore.queuedJobs} onremove={removeQueuedJob} onload={loadQueuedSettings} onclear={clearQueuedJobs} />
            </div>
          </div>
        {:else if latestAsset}
          {@const latest = latestAsset}
          <!-- Side padding keeps wide media clear of the docked mascot. -->
          <div class="latest-region flex h-full w-full flex-col px-20 py-4">
            <AssetTile
              asset={latest}
              density="stage"
              eager
              deleting={deletingIds.has(latest.id)}
              class="latest-media min-h-0 flex-1 {latestLoading ? '' : 'loaded'}"
              onmediaload={() => { loadedLatestUrl = latest.url; }}
              onpreview={openHistoryViewer}
              onreuse={reuseAsset}
              onreference={useAsReference}
              onupscale={upscaleAsset}
              {referenceUnavailable}
              ondelete={deleteWorkspaceAsset}
            />
            {@render assetMeta(latest)}
          </div>
          {#if latestLoading}
            <div class="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-3 text-center p-4" data-testid="latest-loading">
              <MascotSpot mood={mascotMood} size={128} />
              <p class="text-content text-text-secondary">Loading latest output…</p>
            </div>
          {/if}
        {:else}
          <div class="flex flex-col items-center justify-center gap-3 text-center p-4">
            <MascotSpot mood={mascotMood} size={128} />
            {#if lookingForHistory}
              <p class="text-content text-text-secondary">Looking for your latest work…</p>
            {:else}
              <div>
                <p class="ui-empty-title">No generated assets yet</p>
                <p class="mt-1 text-ui text-text-muted">Write a prompt and press Generate to get started</p>
              </div>
            {/if}
          </div>
        {/if}
        {#if dockMascot}
          <div class="mascot-dock pointer-events-none absolute top-3 left-3 z-10" data-testid="mascot-dock">
            <MascotSpot mood={mascotMood} size={64} />
          </div>
        {/if}
      </div>

      <HistoryStrip
        assets={historyStore.assets}
        loading={historyStore.loading}
        {deletingIds}
        onpreview={openHistoryViewer}
        onreuse={reuseAsset}
        onreference={useAsReference}
        onupscale={upscaleAsset}
        {referenceUnavailable}
        ondelete={deleteWorkspaceAsset}
      />
    </section>
  </main>
</form>

<!-- One viewer for history and completed outputs -->
<AssetViewer
  assets={lightboxAssets}
  currentIndex={lightboxIndex}
  open={lightboxOpen}
  setLabel={lightboxMode === 'completed-output' ? 'This run' : 'History'}
  {deletingIds}
  onclose={closeLightbox}
  onnavigate={navigateLightbox}
  onreuse={reuseAsset}
  onreference={useAsReference}
  {referenceUnavailable}
  onupscale={upscaleAsset}
  ondelete={deleteWorkspaceAsset}
/>

<style>
  /* Chips in the toolbar share the height of the selects beside them. */
  .lora-chip { height: var(--spacing-control); padding: 0 8px; border-radius: var(--radius-sm); color: var(--color-text-primary); }
  .lora-weight { field-sizing: content; min-width: 2ch; border-radius: var(--radius-xs); background: transparent; text-align: center; font-family: var(--font-mono); font-weight: 400; color: var(--color-text-muted); appearance: textfield; -moz-appearance: textfield; }
  .lora-weight::-webkit-inner-spin-button, .lora-weight::-webkit-outer-spin-button { margin: 0; -webkit-appearance: none; }
  .lora-weight:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }
  .lora-remove { display: inline-grid; place-items: center; width: 16px; height: 16px; margin-right: -3px; border-radius: var(--radius-xs); color: var(--color-text-muted); }
  .lora-remove:hover { background: var(--color-error-surface); color: var(--color-error); }
  .workspace-layout { display: grid; grid-template-columns: 360px minmax(0, 1fr); }
  .workspace-layout.sidebar-collapsed { grid-template-columns: 56px minmax(0, 1fr); }
  @media (max-width: 639px) {
    .workspace-layout { display: flex; flex-direction: column; overflow-y: auto; }
    .workspace-preview { flex: none; min-height: 420px; }
  }
  .latest-region :global(.latest-media) { opacity: 0; transition: opacity 250ms ease; }
  .latest-region :global(.latest-media.loaded) { opacity: 1; }
  .latest-meta { display: flex; flex: none; align-items: center; justify-content: center; gap: 14px; min-width: 0; padding-top: 8px; font-size: var(--text-ui); color: var(--color-text-muted); }
  .latest-meta span { max-width: 520px; color: var(--color-text-secondary); }
  .latest-meta b { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); font-weight: 500; color: var(--color-text-secondary); }
  .completed-output-scroll { display: flex; flex-direction: column; }
  .completed-output-grid { display: grid; justify-content: center; margin-block: auto; }
</style>
