<script lang="ts">
  import { onMount, tick, untrack } from 'svelte';
  import { api, errorText } from '$lib/api/client';
  import { draft, videoTimingFor } from '$lib/state/draft.svelte';
  import { anyEnhancedUsed, enhancedPartUsed, submittedCaption, submittedPair } from '$lib/state/promptEnhance';
  import { jobStore } from '$lib/state/job.svelte';
  import { historyStore } from '$lib/state/history.svelte';
  import { router } from '$lib/state/router.svelte';
  import { hold } from '$lib/actions/hold';
  import { canScrollToward, OWN_DRAG, pan, rubberBand, type PanEnd, type PanMove, type PanStart } from '$lib/actions/pan';
  import { GALLERY_PULL_PX, galleryPullReady, stageSwipeOutcome, wheelStep, type StageMoves } from './stageSwipe';
  import { addToast } from '$lib/state/toasts.svelte';
  import { getWorkspaceCoreContext, submitGenerate, parseUrlPrefill } from '$lib/api/workspace';
  import { deleteAsset } from '$lib/api/gallery';
  import { canExtend, extendParams, referenceParams, reuseParams, type DeleteOptions, type ReferenceTarget } from '$lib/state/assetActions';
  import { Icon, MascotSpot, ToolbarSelectShell } from '$lib/components/atoms';
  import { rememberMascotSpots } from '$lib/components/atoms/MascotSpot.svelte';
  import {
    DROWSY_AFTER_MS,
    GREETING_DURATION_MS,
    REACTION_DURATION_MS,
    TYPING_DURATION_MS,
    mascotMood as pickMascotMood,
    mascotTool as pickMascotTool,
    type MascotReaction,
  } from '$lib/state/mascot';
  import { ActionMenu, AssetTile, AssetViewer, JobBar, JobCard, JobPreview, ModelStatusBadges, ReviewStage, requestConfirm, type ActionMenuEntry } from '$lib/components/molecules';
  import { compactUi } from '$lib/state/compactUi.svelte';
  import { failureToastText, isReviewing } from '$lib/state/jobView';
  import { jobSettingsPrefill } from '$lib/state/jobSettings';
  import { cliCommand, formSettings, type CliCaveat, type CliCommand } from '$lib/state/cliCommand';
  import { extendGuard, finalizeSceneFields, type ExtendGuard } from '$lib/state/extend';
  import { extendInspection } from '$lib/state/extendInspection.svelte';
  import { clipMemory } from '$lib/state/clipMemory.svelte';
  import { INPUT_DEBOUNCE_MS } from '$lib/state/debounce';
  import { clipMemoryRequest } from '$lib/state/memoryFit';
  import { formatLoraString, loraLabel, parseLoraString, type LoraChip } from '$lib/state/loras';
  import QueuePanel from './QueuePanel.svelte';
  import { confirmDeleteAsset } from '$lib/state/assetActions';
  import ControlsSidebar from './ControlsSidebar.svelte';
  import ComposePane from './ComposePane.svelte';
  import HistoryStrip from './HistoryStrip.svelte';
  import SettingsStrip from './SettingsStrip.svelte';
  import SettingPopover from './SettingPopover.svelte';
  import { settingsChips, type ChipKey } from './settingsChips';
  import { Sheet } from '$lib/components/molecules';
  import { fileName } from '$lib/components/molecules/assetDetails';
  import { settingDefaultsFor } from '$lib/state/draft.svelte';
  import { framesToSeconds, roundSeconds } from '$lib/state/videoDuration';
  import { fitOutputGrid } from './outputGrid';
  import { randomSeed } from './seed';
  import { workspaceCapabilities } from './capabilities';
  import { acceptsPageShortcut, hasOpenModal, isCommandKey } from '$lib/keyboard';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { startUpscale } from '$lib/state/upscale';
  import { hasPendingJobSettings, takeJobSettings } from '$lib/state/pendingJobSettings.svelte';
  import { applyPromptRun, takePromptRun, type PendingPromptRun } from '$lib/state/pendingPromptRun';
  import type { DraftState, GalleryAsset, JobControlAction, JobSettings, JobSnapshot, UpscaleFactor, UpscaleMethod, WorkspaceContext, Workflow } from '$lib/types';

  interface Props {
    /** The workflows the server offers, once its context has loaded; the top bar shows a tab for each. */
    workflows?: Workflow[] | null;
  }

  let { workflows = $bindable(null) }: Props = $props();

  // Portrait tablet: the stage over Compose and Settings side by side, instead of a sidebar beside a narrow stage.
  const tall = $derived(layout.mode === 'tall');
  // Phone: the stage is the screen; settings are a strip of chips that open one control each, the prompt a composer.
  const compact = $derived(layout.mode === 'compact');
  let activeChip = $state<ChipKey | null>(null);
  let sheetOpen = $state(false);
  let composerOpen = $state(false);
  // The running job's full card and the queue, as a sheet on a phone; the strip's job state asks for it too.
  let jobSheetOpen = $state(false);
  let seenJobSheetRequests = compactUi.jobSheetRequests;
  $effect(() => {
    const requests = compactUi.jobSheetRequests;
    if (requests === seenJobSheetRequests) return;
    seenJobSheetRequests = requests;
    if (jobStore.current || jobStore.queuedJobs.length > 0) jobSheetOpen = true;
    else addToast('No job is running and the queue is empty.', 'info');
  });
  // The sheet has nothing to show once the job is gone and the queue is empty.
  $effect(() => {
    if (!jobStore.current && jobStore.queuedJobs.length === 0) jobSheetOpen = false;
  });

  let context = $state<WorkspaceContext | null>(null);
  let loadError = $state<string | null>(null);
  let busy = $state(false);
  // Why the server refused the last Generate, and the draft it refused; any settings change clears it.
  let submitError = $state.raw<{ message: string; draft: DraftState } | null>(null);
  const generateError = $derived(submitError?.draft === draft.state ? submitError.message : null);
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
  // When the user (or a job reaction) last caught the mascot's attention.
  let lastActivity = Date.now();

  const CLI_CAVEAT_NOTES: Record<CliCaveat, string> = {
    negative_prompt: 'The CLI takes negative prompts only from a prompts file, so it was left out.',
    prompt_options: 'The CLI runs every prompt in the file, not only the chosen ones.',
    uploaded_image: 'The uploaded reference image is named by file name; replace it with its path.',
  };
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
  // A finished job's outputs hold the stage until the user steps away (a swipe or an arrow), so the history is one
  // gesture away again. A phone goes straight back to the latest: its stage shows one output at a time.
  let leftOutputsOf = $state<string | null>(null);
  const hasCompletedOutputs = $derived(
    jobStore.current?.status === 'completed'
      && jobOutputs.length > 0
      && !compact
      && leftOutputsOf !== jobStore.current.job_id
  );
  const reviewing = $derived(isReviewing(jobStore.current));
  const previewView = $derived<'error' | 'outputs' | 'job' | 'latest' | 'empty'>(
    loadError ? 'error'
      : hasCompletedOutputs ? 'outputs'
      : jobStore.current && (jobStore.isRunning || (jobOutputs.length > 0 && jobStore.current.status !== 'completed')) ? 'job'
      : historyStore.assets.length > 0 ? 'latest'
      : 'empty'
  );
  // The stage steps through history in place: the viewed asset is tracked by id, so new outputs arriving at the
  // front never move it; null means the latest. A new job or a vanished asset brings it back to the latest.
  let viewedId = $state<string | null>(null);
  const viewedIndex = $derived.by(() => {
    if (viewedId === null) return 0;
    const index = historyStore.assets.findIndex((asset) => asset.id === viewedId);
    return index < 0 ? 0 : index;
  });
  const viewedAsset = $derived<GalleryAsset | null>(historyStore.assets[viewedIndex] ?? null);
  const atLatest = $derived(viewedIndex === 0);
  const historyTotal = $derived(historyStore.totalCount ?? historyStore.assets.length);
  const STEP_NEAR_END = 3;

  /** Show the asset `delta` places older (positive) or newer (negative) than the viewed one, loading more history near the end. */
  function stepView(delta: number): void {
    const assets = historyStore.assets;
    if (assets.length === 0) return;
    const next = Math.min(assets.length - 1, Math.max(0, viewedIndex + delta));
    viewedId = next === 0 ? null : assets[next].id;
    if (next >= assets.length - STEP_NEAR_END && historyStore.hasMore) void historyStore.loadMore();
  }

  function showLatest(): void {
    viewedId = null;
  }

  /** Step through history from wherever the stage is: from a job's outputs, leave them for the history first. */
  function stepStage(delta: number): void {
    if (previewView === 'outputs') {
      leftOutputsOf = jobStore.current?.job_id ?? null;
      viewedId = null;
      if (delta > 0) stepView(delta);
      return;
    }
    stepView(delta);
  }

  /** The stage can step through history: it shows the latest (or an earlier) output, or a finished job's outputs. */
  const canStepStage = $derived(previewView === 'latest' || previewView === 'empty' || previewView === 'outputs');

  // A job takes the stage and its output lands at the front: the stage is back at the latest when it returns.
  $effect(() => {
    if (jobStore.isRunning) viewedId = null;
  });

  // A finger drags the stage: sideways through history (a mouse has ← → and the arrows, a trackpad swipes), and up,
  // on a phone, to the Gallery, whatever the stage shows. It follows the finger, then commits or springs back.
  let stageEl = $state<HTMLDivElement | null>(null);
  let stageDrag = $state<PanMove | null>(null);
  // After the finger lifts, what the stage shows slides back into place for a moment before the drag lets go of it.
  let stageSettling = $state(false);
  let settleTimer: ReturnType<typeof setTimeout> | undefined;
  const stageMoves = $derived<StageMoves>({
    older: canStepStage && previewView !== 'empty' && (viewedIndex < historyStore.assets.length - 1 || historyStore.hasMore || previewView === 'outputs'),
    newer: canStepStage && previewView === 'latest' && !atLatest,
    gallery: compact && previewView !== 'error',
  });
  const stageDragOffset = $derived.by(() => {
    if (!stageDrag) return { x: 0, y: 0 };
    const { axis, offset } = stageDrag;
    if (axis === 'y') return { x: 0, y: rubberBand(offset, 120) };
    const open = offset < 0 ? stageMoves.older : stageMoves.newer;
    return { x: open ? offset : rubberBand(offset, 40), y: 0 };
  });

  function claimStageDrag({ axis, dx, dy, target }: PanStart): boolean {
    if (!stageEl || target.closest(OWN_DRAG)) return false;
    if (axis === 'x') return (stageMoves.older || stageMoves.newer) && !canScrollToward(target, stageEl, 'x', dx > 0 ? 1 : -1);
    return stageMoves.gallery && dy < 0 && !canScrollToward(target, stageEl, 'y', -1);
  }

  function releaseStageDrag(): void {
    stageDrag = null;
    stageSettling = true;
    clearTimeout(settleTimer);
    settleTimer = setTimeout(() => { stageSettling = false; }, STAGE_SETTLE_MS);
  }
  const STAGE_SETTLE_MS = 260;

  function onStageDragEnd(end: PanEnd): void {
    const outcome = stageSwipeOutcome(end, stageMoves);
    if (outcome === 'older' || outcome === 'newer') {
      // The next output comes in from the side the finger pushed toward, then settles.
      stepStage(outcome === 'older' ? 1 : -1);
      stageDrag = { axis: 'x', offset: outcome === 'older' ? 48 : -48 };
      requestAnimationFrame(() => requestAnimationFrame(releaseStageDrag));
      return;
    }
    releaseStageDrag();
    if (outcome === 'gallery') router.navigate('gallery');
  }

  // A trackpad's sideways two-finger swipe steps once per gesture; the gesture ends when the wheel stops for a moment.
  function stageWheel(node: HTMLElement): { destroy: () => void } {
    let total = 0;
    let stepped = false;
    let idle: ReturnType<typeof setTimeout> | undefined;
    function onWheel(event: WheelEvent): void {
      if (Math.abs(event.deltaX) <= Math.abs(event.deltaY) || !(stageMoves.older || stageMoves.newer)) return;
      event.preventDefault();
      clearTimeout(idle);
      idle = setTimeout(() => { total = 0; stepped = false; }, 200);
      if (stepped) return;
      total += event.deltaX;
      const step = wheelStep(total);
      if (step === 0) return;
      stepped = true;
      if (step > 0 ? stageMoves.older : stageMoves.newer) stepStage(step);
    }
    node.addEventListener('wheel', onWheel, { passive: false });
    return { destroy: () => { clearTimeout(idle); node.removeEventListener('wheel', onWheel); } };
  }
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
      && viewedAsset !== null
      && viewedAsset.media_type !== 'video'
      && loadedLatestUrl !== viewedAsset.url
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
  // A running job decides the tool; otherwise the form's workflow does.
  const mascotTool = $derived(pickMascotTool(jobStore.isRunning ? jobStore.current?.workflow : draft.state.workflow));
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

  // The phone's settings strip: one chip per setting, derived from the same state the inspector shows.
  const chips = $derived.by(() => {
    if (!compact || !context || !authorityReady) return [];
    const s = draft.state;
    const caps = workspaceCapabilities(context, s);
    const defaults = settingDefaultsFor(context, { workflow: s.workflow, model: s.model }) as Partial<DraftState>;
    const presets = (caps.isImageMode ? context.image_size_dimensions : (context.video_size_dimensions ?? {}))[s.ratio]?.[s.size] ?? [s.width, s.height];
    const timing = videoTimingFor(context, s.model);
    const sourceFps = draft.extendSource?.fps ?? null;
    const memoryStatus = s.workflow === 'extend' ? draft.extendSource?.memory_fit?.status : clipMemory.fit?.status;
    return settingsChips({
      caps,
      state: s,
      changed: (...keys) => keys.some((key) => key in defaults && s[key] !== defaults[key]),
      width: s.dimensionMode === 'custom' ? s.width : presets[0],
      height: s.dimensionMode === 'custom' ? s.height : presets[1],
      seconds: caps.showFrameCount ? roundSeconds(framesToSeconds(s.frameCount, timing.fps)) : null,
      extendSeconds: sourceFps && s.extendFrames !== null ? roundSeconds(s.extendFrames / sourceFps) : null,
      extendChanged: false,
      loraNames: loraChips.map((chip) => loraLabel(chip.name)),
      modelLabel: selectedModel?.label ?? s.model,
      modelDownloaded: selectedModel?.downloaded !== false,
      referenceName: imageFile?.name ?? (s.referenceImagePath ? fileName(s.referenceImagePath) : null),
      sourceName: s.extendSourcePath ? fileName(s.extendSourcePath) : null,
      memoryWarn: memoryStatus === 'tight' || memoryStatus === 'too_large',
    });
  });
  const changedCount = $derived(chips.filter((chip) => chip.changed).length);
  // What the folded composer reads: the prompt file and its count, the Enhanced text, or the prompt.
  const composerSummary = $derived.by(() => {
    const s = draft.state;
    if (s.promptSource === 'file') return { label: 'Prompt file', text: s.promptFilePath ? `${fileName(s.promptFilePath)} · ${s.promptFileOptionIds.length} ${s.promptFileOptionIds.length === 1 ? 'prompt' : 'prompts'}` : 'Choose a prompt file', empty: !s.promptFilePath };
    const video = s.workflow === 'img2vid' || s.workflow === 'txt2vid' || s.workflow === 'extend';
    const enhanced = anyEnhancedUsed(s);
    const text = enhanced ? s.enhancedPrompt || s.enhancedScene : s.prompt || s.scene;
    return { label: enhanced ? 'Enhanced' : video ? 'Action' : 'Prompt', text: text || (video ? 'What happens in the clip…' : 'Describe what to make…'), empty: !text };
  });
  const compactSubmitDisabled = $derived(busy || !authorityReady || promptChoiceMissing() || currentExtendGuard().blocked);

  /** Open the composer and put the caret in the prompt (the Scene box comes first for a video). */
  async function openComposer(): Promise<void> {
    activeChip = null;
    composerOpen = true;
    await tick();
    document.querySelector<HTMLElement>('.composer-pane #ws-prompt:not([hidden]), .composer-pane textarea:not([hidden]):not([disabled])')?.focus();
  }

  // Press and hold Generate on a phone, or tap the chevron beside it, for the run ⌘⇧↵ gives a keyboard: a new seed first.
  let generateMenuOpen = $state(false);
  let generateButton = $state<HTMLButtonElement | null>(null);
  const generateMenuItems = $derived<ActionMenuEntry[]>([
    { kind: 'item', id: 'generate-new-seed', label: 'Generate with a new seed', onselect: () => { if (draft.state.seed !== null) draft.update('seed', randomSeed()); void tick().then(() => formEl?.requestSubmit()); } },
  ]);

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


  function markActive(): void {
    lastActivity = Date.now();
    if (drowsy) drowsy = false;
  }

  function react(next: MascotReaction): void {
    clearTimeout(reactionTimer);
    // Restart the drowsy clock, so the mascot does not doze off right after reacting to a long job.
    markActive();
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
    addToast(failureToastText(jobStore.current?.message, jobStore.queuedJobs.length > 0), 'error');
  }

  async function handleJobLost(): Promise<void> {
    react('surprised');
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

  /** Refill the form from the settings a job was submitted with. */
  function loadJobSettings(settings: JobSettings, message: string): void {
    const { params, patch } = jobSettingsPrefill(settings, draft.state.enhanceSettingsVideo?.partsByWorkflow);
    applyPrefill(params);
    draft.patch(patch);
    imageFile = null;
    addToast(message, 'success');
  }

  function loadQueuedSettings(job: JobSnapshot): void {
    loadJobSettings(job.settings ?? {}, 'Loaded the queued job\'s settings');
  }

  // Settings the Job log asked for wait until the context has loaded, because loading the draft would overwrite an early patch.
  $effect(() => {
    if (context === null || !hasPendingJobSettings()) return;
    const settings = takeJobSettings();
    if (settings) untrack(() => loadJobSettings(settings, 'Loaded the job\'s settings'));
  });

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

  async function upscaleAsset(asset: GalleryAsset, factor: UpscaleFactor, method: UpscaleMethod): Promise<void> {
    // Like Generate, an upscale joins the queue while another job runs.
    const job = await startUpscale(asset, factor, method);
    if (job?.queue_position) react('nodding');
    if (job) closeLightbox();
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
    if (draft.state.workflow === params.workflow) {
      draft.loadFromUrl(params, context);
      draft.saveDraft();
    } else applyPrefill(params, { keepSettings: true });
    imageFile = null;
    closeLightbox();
    addToast(`${asset.filename} is now the reference image`, 'success');
  }

  /** Make a video the source of an extension; its recorded scene and prompt fill the boxes once the server has described it. */
  function extendAsset(asset: GalleryAsset): void {
    if (!asset.file_path || !canExtend(asset)) return;
    const params = extendParams(asset);
    if (draft.state.workflow === params.workflow) {
      draft.loadFromUrl(params, context);
      draft.saveDraft();
    } else applyPrefill(params, { keepSettings: true });
    imageFile = null;
    closeLightbox();
    addToast(`${asset.filename} is now the source video`, 'success');
    void extendInspection.inspect(context, { replaceText: true });
  }

  // Tried in order: the visible prompt field (the Scene box sits above it), else any text field, else any control in the pane.
  const COMPOSE_FOCUS = [
    '.compose-pane #ws-prompt:not([hidden]), .compose-pane #ws-enhanced-prompt:not([hidden]):not([disabled])',
    '.compose-pane textarea:not([hidden]):not([disabled])',
    '.compose-pane button:not([disabled])',
  ];
  const SETTINGS_FOCUS = ['.settings-pane :is(input, select, textarea, button):not([disabled]):not([type="hidden"])'];
  // The newest history tile, else the strip's toggle when there is no history yet.
  const HISTORY_FOCUS = ['#ws-history-scroll .asset-tile-media', '#ws-history-toggle'];
  // Each Alt+digit pane: the draft flag that collapses it (Compose and Settings share the sidebar) and its focus targets.
  const PANES: Readonly<Record<string, { collapsedKey: 'sidebarCollapsed' | 'historyCollapsed'; focus: readonly string[] }>> = {
    Digit1: { collapsedKey: 'sidebarCollapsed', focus: COMPOSE_FOCUS },
    Digit2: { collapsedKey: 'sidebarCollapsed', focus: SETTINGS_FOCUS },
    Digit3: { collapsedKey: 'historyCollapsed', focus: HISTORY_FOCUS },
  };

  function focusFirst(selectors: readonly string[]): void {
    for (const selector of selectors) {
      const el = document.querySelector<HTMLElement>(selector);
      if (el) { el.focus(); return; }
    }
  }

  async function focusPane(code: string): Promise<void> {
    const { collapsedKey, focus } = PANES[code];
    if (draft.state[collapsedKey]) {
      draft.update(collapsedKey, false);
      await tick();
    }
    focusFirst(focus);
  }

  function togglePane(code: string): void {
    const { collapsedKey } = PANES[code];
    if (draft.state[collapsedKey]) void focusPane(code);
    else draft.update(collapsedKey, true);
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

  // An extension's source is described by the server: on load, and again when its length, model or memory setting change
  // (the memory note depends on them). A source the user just chose is inspected by the Source video section instead.
  $effect(() => {
    const path = draft.state.extendSourcePath;
    const watching = authorityReady && draft.state.workflow === 'extend' && path !== null;
    if (!watching || extendInspection.isCurrent()) return;
    const timer = setTimeout(() => void extendInspection.inspect(context), extendInspection.requestedPath() === path ? INPUT_DEBOUNCE_MS : 0);
    return () => clearTimeout(timer);
  });

  // The memory note of a Text to Video or Image to Video clip follows its model, size, length and memory settings.
  // An extension is checked with its source instead, so it, like an image workflow, has nothing to ask here.
  $effect(() => {
    const state = draft.state;
    if (!authorityReady || context === null) {
      clipMemory.watch(null);
      return;
    }
    const caps = workspaceCapabilities(context, state);
    const { frameAlignment } = videoTimingFor(context, state.model);
    clipMemory.watch(clipMemoryRequest(state, { frameAlignment, lowMemory: caps.showLowMemory, upscale: caps.showVideoUpscale }));
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
    extendInspection.reset();
    clipMemory.reset();

    let cancelled = false;
    let historyTimer: ReturnType<typeof setTimeout> | undefined;

    // Load workspace authority first; gallery/history hydration follows after the core page can paint.
    getWorkspaceCoreContext()
      .then((ctx) => {
        if (cancelled) return;
        context = ctx;
        workflows = ctx.workflow_contract.values;
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
        if (run) {
          void (async () => {
            await tick();
            // A saved source has not been described yet; the run waits for the answer rather than being refused for want of it.
            if (ctx && draft.state.workflow === 'extend' && draft.state.extendSourcePath) await extendInspection.inspect(ctx);
            if (!cancelled) await submitForm(run);
          })();
        }

        historyTimer = setTimeout(() => {
          if (cancelled) return;
          void historyStore.refreshHistory().finally(() => {
            if (!cancelled) historyChecked = true;
          });
        }, 0);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        loadError = errorText(e, 'Failed to load workspace context');
      });

    // ⌘↵ / Ctrl↵ generates; with ⇧ a locked seed is re-rolled first. Alt+1 / 2 / 3 open and focus Compose / Settings / History;
    // with ⇧ they toggle that pane instead, focusing it when it opens.
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
      } else if (e.altKey && !isCommandKey(e) && e.code in PANES) {
        // `code`, not `key`: Alt+1 types "¡" on a Mac keyboard.
        e.preventDefault();
        if (e.shiftKey) togglePane(e.code);
        else void focusPane(e.code);
      } else if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && canStepStage && previewView !== 'empty' && acceptsPageShortcut(e) && !(e.target instanceof HTMLMediaElement)) {
        // ← → step the preview through history; → goes back in time, like the strip reads. A focused video seeks instead.
        e.preventDefault();
        stepStage(e.key === 'ArrowRight' ? 1 : -1);
      }
    }
    document.addEventListener('keydown', handleKeydown);

    // Mascot: wave hello, then doze off after a quiet minute.
    const greetingTimer = setTimeout(() => { greeting = false; }, GREETING_DURATION_MS);
    const activityEvents = ['pointerdown', 'pointermove', 'keydown', 'wheel'] as const;
    activityEvents.forEach((type) => document.addEventListener(type, markActive, { passive: true }));
    const drowsyTimer = setInterval(() => {
      drowsy = Date.now() - lastActivity > DROWSY_AFTER_MS;
    }, 5000);

    return () => {
      clearTimeout(greetingTimer);
      clearTimeout(reactionTimer);
      clearTimeout(typingTimer);
      clearInterval(drowsyTimer);
      activityEvents.forEach((type) => document.removeEventListener(type, markActive));
      cancelled = true;
      clipMemory.reset();
      stopJobSync();
      unsubscribeLifecycle();
      if (historyTimer) clearTimeout(historyTimer);
      document.removeEventListener('keydown', handleKeydown);
    };
  });

  let revealingInvalid = false;

  // The browser cannot report an invalid field it cannot show: inside the collapsed sidebar, the folded composer or the
  // closed All sheet. Open the pane that holds it and report once shown.
  function revealInvalidField(event: Event): void {
    const field = event.target as HTMLInputElement;
    let reveal: (() => void) | null = null;
    if (field.closest?.('.composer-pane') && !composerOpen) reveal = () => { composerOpen = true; };
    else if (field.closest?.('[data-testid="sheet"]') && !sheetOpen) reveal = () => { sheetOpen = true; };
    else if (draft.state.sidebarCollapsed && field.closest?.('#ws-controls-sidebar')) reveal = () => draft.update('sidebarCollapsed', false);
    if (!reveal) return;
    event.preventDefault();
    if (revealingInvalid) return;
    revealingInvalid = true;
    reveal();
    void tick().then(() => {
      revealingInvalid = false;
      field.reportValidity();
    });
  }

  /** Collect the fields Generate submits; with `run`, that one prompt-file prompt replaces the form's prompt. */
  function buildFormData(form: HTMLFormElement, run: PendingPromptRun | null): FormData {
    const formData = new FormData(form);
    // Send canonical workflow directly; backend accepts and normalises it.
    formData.set('mode', isImageMode ? 'image' : 'video');
    formData.set('workflow', draft.state.workflow);
    // Attach image file only for workflows that actually use a reference image
    if (imageFile && (draft.state.workflow === 'img2img' || draft.state.workflow === 'img2vid')) {
      formData.set('image_file', imageFile);
    }
    // Sync lora string
    formData.set('lora', formatLoraString(loraChips));
    // Each part with Enhanced text is sent enhanced and the other as written (never in JSON-caption, file, or auto-enhance mode).
    const caption = run ? null : submittedCaption(draft.state);
    if (run) {
      applyPromptRun(formData, run);
    } else if (caption !== null && formData.has('prompt')) {
      // An Enhanced JSON caption runs as the structured caption, like one written in the { } JSON box.
      formData.delete('prompt');
      formData.delete('negative_prompt');
      formData.set('json_prompt', caption);
      recordEnhancedText(formData);
    } else if (anyEnhancedUsed(draft.state) && formData.has('prompt')) {
      // The Enhanced parts run; the user's own text is recorded beside them so Reuse can offer either.
      const pair = submittedPair(draft.state);
      formData.set('prompt', pair.prompt);
      if (formData.has('scene')) formData.set('scene', pair.scene);
      recordEnhancedText(formData);
    }
    // Outside an extension a blank scene is no scene: today's form data, without the field. An extension always sends it.
    finalizeSceneFields(formData, draft.state);
    return formData;
  }

  /** Record the user's own prompt and how the Enhanced text was made, so the asset keeps both. */
  function recordEnhancedText(formData: FormData): void {
    // A part that runs enhanced records the text its rewrite was made from, even when Your text changed since (out of date).
    const from = draft.state.enhancedFrom;
    const input = (part: 'scene' | 'action', current: string, source: string | undefined): string => (enhancedPartUsed(draft.state, part) && source !== undefined ? source : current);
    formData.set('prompt_input', input('action', draft.state.prompt, from?.prompt));
    if (formData.has('scene')) formData.set('scene_input', input('scene', draft.state.scene, from?.scene));
    if (from?.record) formData.set('enhance_record', JSON.stringify(from.record));
  }

  /** Whether an extension may start now; any other workflow always may. */
  function currentExtendGuard(): ExtendGuard {
    return extendGuard(draft.state, draft.extendSource, extendInspection.statusFor(draft.state.extendSourcePath));
  }

  /** Tell the user why an extension cannot start and bring them to the source's path field; false when nothing stands in the way. */
  function extendBlocked(): boolean {
    const guard = currentExtendGuard();
    if (!guard.blocked) return false;
    // Still being checked: nothing is wrong yet, so there is nothing to tell and nowhere to send the user.
    if (guard.kind === 'checking') return true;
    if (guard.note) addToast(guard.note, 'error');
    // The path field sits in the Source video section; the collapsed sidebar has to open for it to take focus.
    void (async () => {
      if (draft.state.sidebarCollapsed) {
        draft.update('sidebarCollapsed', false);
        await tick();
      }
      document.getElementById('ws-source-path')?.focus();
    })();
    return true;
  }

  /** Wait for the answer about the source when Generate or Copy CLI is pressed while it is still being checked. */
  async function settleExtendSource(): Promise<void> {
    if (currentExtendGuard().kind !== 'checking') return;
    busy = true;
    try {
      if (extendInspection.isCurrent()) await extendInspection.settled();
      else await extendInspection.inspect(context);
    } finally {
      busy = false;
    }
  }

  /** Whether prompt-file mode lacks a file or a chosen prompt, so there is nothing to run. */
  function promptChoiceMissing(): boolean {
    return draft.state.promptSource === 'file' && (!draft.state.promptFilePath || draft.state.promptFileOptionIds.length === 0);
  }

  /** Copy the CLI command that would run what Generate submits now. */
  async function copyCliCommand(): Promise<void> {
    // Report empty or invalid fields the way Generate does, rather than copying a command for them.
    if (!formEl || !authorityReady || !formEl.reportValidity()) return;
    await settleExtendSource();
    if (extendBlocked()) return;
    if (promptChoiceMissing()) {
      addToast(context?.prompt_file.help.option_required ?? 'Choose a prompt first.', 'error');
      return;
    }
    const settings = formSettings(buildFormData(formEl, null));
    // Only a count from the file the command names is trusted; an unknown count reports the caveat.
    const loaded = draft.state.promptFileLoaded;
    const promptFileOptionCount = loaded && loaded.path === settings.prompts_file ? loaded.count : null;
    let built: CliCommand;
    try {
      const extendSource = draft.extendSource ? { scene: draft.extendSource.scene, legacy: draft.extendSource.legacy } : null;
      built = cliCommand(settings, { outputDir: context?.output_dir, promptFileOptionCount, extendSource });
    } catch (err) {
      addToast(errorText(err, 'Could not build the CLI command'), 'error');
      return;
    }
    const { command, caveats } = built;
    try {
      await navigator.clipboard.writeText(command);
    } catch {
      addToast('Could not copy the CLI command', 'error');
      return;
    }
    if (caveats.length === 0) {
      addToast('CLI command copied', 'success');
      return;
    }
    // Caveat notes take longer to read than the plain confirmation.
    addToast(`CLI command copied. ${caveats.map((caveat) => CLI_CAVEAT_NOTES[caveat]).join(' ')}`, 'info', { timeout: 10000 });
  }

  /** Send a job control; the job's events show what it did. */
  function sendControl(action: JobControlAction): (jobId: string) => Promise<unknown> {
    return (jobId) => api.post(`/jobs/${encodeURIComponent(jobId)}/controls/${action}`);
  }

  async function handleSubmit(e: Event): Promise<void> {
    e.preventDefault();
    await submitForm(null);
  }

  /** Submit the form; with `run`, queue that one prompt-file prompt instead of the form's prompt. */
  async function submitForm(run: PendingPromptRun | null): Promise<void> {
    if (!formEl || busy || !authorityReady) return;
    if (!run && promptChoiceMissing()) {
      return;
    }
    await settleExtendSource();
    if (extendBlocked()) return;
    submitError = null;
    busy = true;
    draft.saveDraft();

    try {
      const formData = buildFormData(formEl, run);
      const jobCtx = await submitGenerate(formData);
      // The response omits the settings; the review line needs the upscale factor.
      jobStore.jobSubmitted({ ...jobCtx, settings: formSettings(formData) });
      // The phone shows the job where the output lands: fold the composer and close any open control.
      composerOpen = false;
      activeChip = null;
      sheetOpen = false;
      if (jobCtx.queue_position) {
        react('nodding');
        addToast(`${run ? `Queued ${run.optionId}` : 'Added to the queue'} as #${jobCtx.queue_position}.`, 'info');
      } else if (run) addToast(`Queued ${run.optionId} from the prompt file.`, 'success');
    } catch (err) {
      const message = errorText(err, 'Generation failed.');
      submitError = { message, draft: draft.state };
      addToast(message, 'error');
    } finally {
      busy = false;
    }
  }

  /** Switch the model: settings the user changed stay, the rest take the new model's defaults. */
  function switchModel(model: string): void {
    if (context) {
      draft.hydrateFromContext(context, model, { keepChangedSettings: true });
    } else {
      draft.update('model', model);
    }
  }

  function onModelChange(e: Event): void {
    switchModel((e.currentTarget as HTMLSelectElement).value);
  }
</script>

{#snippet assetMeta(asset: GalleryAsset)}
  <p class="latest-meta">
    <span class="truncate" title={asset.prompt}>{asset.prompt || asset.filename}</span>
    {#if asset.model}<b>{asset.model}</b>{/if}
    {#if asset.width && asset.height}<b>{asset.width}×{asset.height}</b>{/if}
    {#if asset.seed != null}<b class="latest-seed">seed {asset.seed}</b>{/if}
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

  {#snippet toolbar(stacked: boolean)}
  <!-- Toolbar bar: model, quantize, loras. Stacked, it heads the phone's All sheet. -->
  <div class="ui-pane-toolbar z-10 shrink-0" class:toolbar-stacked={stacked}>
    <div class="flex items-center px-3 py-2 gap-3 {stacked ? 'flex-col items-stretch' : 'flex-wrap'}">

      <!-- Model selector -->
      <div class="flex items-center gap-2 min-w-0 {stacked ? 'flex-nowrap' : 'flex-wrap'}">
        <label class="ui-label {stacked ? 'sr-only' : ''}" for="ws-model">Model</label>
        <ToolbarSelectShell
          id="ws-model"
          name="model"
          testId="model-shell"
          class={stacked ? 'min-w-0 flex-1' : 'w-48'}
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
            class={stacked ? 'w-auto shrink-0' : 'w-40'}
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

      <div class="bg-border-subtle shrink-0 {stacked ? 'h-px w-full' : 'h-6 w-px'}"></div>

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

  {/snippet}

  {#snippet stage()}
    <section class="workspace-preview relative z-0 flex min-h-0 min-w-0 flex-col bg-bg-base">
      <!-- On a phone the stage starts under the top strip; stepped back in history, a pill over it says where. -->
      <div class="ui-pane-header flex h-10 shrink-0 items-center justify-between px-3" hidden={compact}>
        <h2 class="ui-area-label">Preview</h2>
        {#if !compact && previewView === 'latest' && !atLatest}
          <span class="flex items-center gap-2 text-meta text-text-muted" data-testid="stage-position">
            Earlier output · <b class="font-mono font-medium text-text-secondary">{viewedIndex + 1}</b> of {historyTotal}
            <button type="button" class="ui-btn ui-btn-sm" onclick={showLatest}>Latest</button>
          </span>
        {:else}
          <span class="text-meta text-text-muted">{reviewing ? 'Review' : jobStore.isRunning ? 'Generating…' : hasCompletedOutputs || historyStore.assets.length ? 'Latest output' : 'Ready'}</span>
        {/if}
      </div>
      {#if compact && previewView === 'latest' && !atLatest}
        <div class="stage-position-pill" data-testid="stage-position">
          <span><b>{viewedIndex + 1}</b> of {historyTotal}</span>
          <button type="button" onclick={showLatest}>Latest<Icon name="chevright" size={12} /></button>
        </div>
      {/if}
      <!-- The drags and the trackpad swipe are shortcuts for the arrows, the keys and the Gallery button beside them. -->
      <div
        bind:this={stageEl}
        class="stage-body relative flex min-h-0 flex-1 items-center justify-center overflow-hidden"
        class:dragging={stageDrag !== null}
        class:moved={stageDrag !== null || stageSettling}
        style:--stage-dx="{stageDragOffset.x}px"
        style:--stage-dy="{stageDragOffset.y}px"
        use:pan={{ axes: ['x', 'y'], claim: claimStageDrag, onmove: (move) => { stageDrag = move; }, onend: onStageDragEnd, oncancel: releaseStageDrag }}
        use:stageWheel
      >
        {#if stageDrag?.axis === 'y'}
          <span class="gallery-pull" class:ready={galleryPullReady(stageDrag.offset)} style:opacity={Math.min(1, -stageDrag.offset / GALLERY_PULL_PX)} aria-hidden="true">
            <Icon name="grid" size={14} />{galleryPullReady(stageDrag.offset) ? 'Release for the Gallery' : 'Gallery'}
          </span>
        {:else if stageDrag?.axis === 'x' && historyStore.assets.length > 0}
          <span class="gallery-pull" aria-hidden="true">{viewedIndex + 1} of {historyTotal}</span>
        {/if}
        {#if loadError}
          <div class="flex flex-col items-center gap-3 text-center p-8">
            <MascotSpot mood={mascotMood} tool={mascotTool} size={112} />
            <p class="font-heading text-content font-extrabold text-error">The Workspace could not load</p>
            <p class="ui-help">{loadError}</p>
          </div>
        {:else if hasCompletedOutputs}
          <div class="completed-output-region flex h-full w-full min-w-0 flex-col p-4">
            <div class="mb-3 flex shrink-0 items-center justify-between gap-3">
              <div class="flex items-center gap-2">
                <MascotSpot mood={mascotMood} tool={mascotTool} size={48} />
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
                  onextend={extendAsset}
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
                      onextend={extendAsset}
                      onupscale={upscaleAsset}
                  {referenceUnavailable}
                      ondelete={deleteWorkspaceAsset}
                    />
                  {/each}
                </div>
              </div>
            {/if}
          </div>
        {:else if previewView === 'job' && jobStore.current}
          <!-- The forming image takes the stage; the card with the job's phases and controls is a column beside it.
               On a phone the compact bar lies over the preview's foot, and the full card is a sheet under ⋯. -->
          <div class="job-view" class:job-view-compact={compact} data-testid="job-view">
            <div class="job-view-preview">
              <JobPreview job={jobStore.current!} />
              {#if !compact}<MascotSpot mood={mascotMood} tool={mascotTool} size={64} class="job-view-mascot" />{/if}
            </div>
            {#if compact}
              <JobBar
                job={jobStore.current!}
                queuedCount={jobStore.queuedJobs.length}
                onpause={sendControl('pause')}
                onresume={sendControl('resume')}
                ondetails={() => { jobSheetOpen = true; }}
              />
            {:else}
            <div class="job-view-card">
              <JobCard
                job={jobStore.current!}
                preview={false}
                oncancel={sendControl('quit')}
                onpause={sendControl('pause')}
                onresume={sendControl('resume')}
                onnext={sendControl('next')}
                onrepeat={sendControl('repeat')}
              />
              <QueuePanel jobs={jobStore.queuedJobs} onremove={removeQueuedJob} onload={loadQueuedSettings} onclear={clearQueuedJobs} />
            </div>
            {/if}
          </div>
        {:else if viewedAsset}
          {@const latest = viewedAsset}
          <!-- Newer and older, for a mouse; they appear on hover and never on a touch screen. -->
          <button type="button" class="stage-nav prev ui-btn ui-btn-icon" aria-label="Newer output" disabled={atLatest} onclick={() => stepView(-1)}>
            <Icon name="chevleft" size={16} />
          </button>
          <button type="button" class="stage-nav next ui-btn ui-btn-icon" aria-label="Older output" disabled={viewedIndex >= historyStore.assets.length - 1 && !historyStore.hasMore} onclick={() => stepView(1)}>
            <Icon name="chevright" size={16} />
          </button>
          <!-- Side padding keeps wide media clear of the docked mascot. -->
          <div class="latest-region flex h-full w-full flex-col px-20 py-4" class:fit={latest.width && latest.height} style:--ar={latest.width && latest.height ? latest.width / latest.height : null} style:--natural-w={latest.width ? `${latest.width}px` : null}>
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
              onextend={extendAsset}
              onupscale={upscaleAsset}
              {referenceUnavailable}
              ondelete={deleteWorkspaceAsset}
            />
            {@render assetMeta(latest)}
          </div>
          {#if latestLoading}
            <div class="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-3 text-center p-4" data-testid="latest-loading">
              <MascotSpot mood={mascotMood} tool={mascotTool} size={128} />
              <p class="text-content text-text-secondary">Loading latest output…</p>
            </div>
          {/if}
        {:else}
          <div class="flex flex-col items-center justify-center gap-3 text-center p-4">
            <MascotSpot mood={mascotMood} tool={mascotTool} size={128} />
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
        {#if reviewing && jobStore.current}
          {@const job = jobStore.current}
          <!-- Scoped to the preview area: settings, history, navigation and menus stay usable. The job card stays under the backdrop. -->
          <div class="review-overlay absolute inset-0 z-20 flex items-center justify-center bg-scrim p-3 sm:p-6" data-testid="review-overlay">
            <div class="ui-overlay flex h-full w-full max-w-4xl min-h-0 min-w-0 flex-col">
              <ReviewStage {job} onapprove={sendControl('approve')} ondiscard={sendControl('discard')} />
            </div>
          </div>
        {/if}
        {#if dockMascot}
          <div class="mascot-dock pointer-events-none absolute top-3 left-3 z-10" data-testid="mascot-dock">
            <MascotSpot mood={mascotMood} tool={mascotTool} size={64} />
          </div>
        {/if}
      </div>

      {#if !compact}
        <HistoryStrip
          assets={historyStore.assets}
          loading={historyStore.loading}
          totalCount={historyStore.totalCount}
          hasMore={historyStore.hasMore}
          loadingMore={historyStore.loadingMore}
          onloadmore={() => void historyStore.loadMore()}
          {deletingIds}
          onpreview={openHistoryViewer}
          onreuse={reuseAsset}
          onreference={useAsReference}
          onextend={extendAsset}
          onupscale={upscaleAsset}
          {referenceUnavailable}
          ondelete={deleteWorkspaceAsset}
        />
      {/if}
    </section>
  {/snippet}

  {#if compact}
    <!-- The phone: the stage is the screen; under it the settings strip, then the composer with Generate. Compose and
         Settings stay inside the form (the composer and the All sheet host them) so every field still submits. -->
    <div class="compact-workspace" data-testid="compact-workspace">
      {@render stage()}
      {#if authorityReady && context}
        <div class="compact-controls">
          {#if activeChip}
            <button type="button" class="pop-scrim" aria-label="Close setting" tabindex="-1" onclick={() => { activeChip = null; }}></button>
            <SettingPopover
              key={activeChip}
              {context}
              {busy}
              models={currentModels}
              quantizeOptions={supportsQuantize ? quantizeOptions : null}
              {imageFile}
              {referencePreviewUrl}
              lastSeed={historyStore.assets[0]?.seed ?? null}
              onImageFileChange={(f) => { imageFile = f; submitError = null; }}
              onModelChange={switchModel}
              onUseLatestAsReference={() => { const latest = historyStore.assets[0]; if (latest) useAsReference(latest, draft.state.workflow === 'img2vid' ? 'video' : 'image'); activeChip = null; }}
              onclose={() => { activeChip = null; }}
            />
          {/if}
          <SettingsStrip {chips} activeKey={activeChip} {changedCount} onopen={(key) => { activeChip = activeChip === key ? null : key; }} onall={() => { activeChip = null; sheetOpen = true; }} />
          <div class="composer" class:expanded={composerOpen} data-testid="composer">
            {#if generateError}<p class="ui-alert ui-alert-error composer-error" role="alert"><Icon name="alert" size={14} /><span class="min-w-0 break-words">{generateError}</span></p>{/if}
            <div class="composer-row" hidden={composerOpen}>
              <button type="button" class="composer-summary" aria-label="Edit the prompt" onclick={() => void openComposer()}>
                <span class="ui-area-label composer-label">{composerSummary.label}</span>
                <span class="composer-line" class:placeholder={composerSummary.empty}>{composerSummary.text}</span>
              </button>
              <span class="composer-split">
                <button
                  id="ws-submit-compact"
                  type="submit"
                  class="composer-generate ui-btn ui-btn-primary ui-holdable"
                  disabled={compactSubmitDisabled}
                  title="Hold for more"
                  use:hold={{ onhold: () => { generateMenuOpen = true; } }}
                >
                  <Icon name="bolt" size={16} />{busy ? 'Submitting…' : jobStore.jobsActive ? 'Queue' : 'Generate'}
                  {#if jobStore.jobsActive && jobStore.queuedJobs.length > 0}<span class="composer-q">+{jobStore.queuedJobs.length}</span>{/if}
                </button>
                <button
                  bind:this={generateButton}
                  type="button"
                  class="composer-more ui-btn ui-btn-primary"
                  aria-label="More ways to generate"
                  aria-haspopup="menu"
                  aria-expanded={generateMenuOpen}
                  disabled={compactSubmitDisabled}
                  onclick={() => { generateMenuOpen = true; }}
                ><Icon name="chevdown" size={14} /></button>
              </span>
              <ActionMenu open={generateMenuOpen} anchor={generateButton} label="Generate" items={generateMenuItems} onclose={() => { generateMenuOpen = false; }} />
            </div>
            <div class="composer-pane" hidden={!composerOpen}>
              <ComposePane {context} {busy} jobsActive={jobStore.jobsActive} />
              <div class="composer-foot">
                <button type="button" class="ui-btn" onclick={() => { composerOpen = false; }}>Done</button>
                <button type="submit" class="ui-btn ui-btn-primary ui-btn-main" disabled={compactSubmitDisabled}><Icon name="bolt" size={16} />{jobStore.jobsActive ? 'Queue' : 'Generate'}</button>
              </div>
            </div>
          </div>
        </div>
      {/if}
      <Sheet bind:open={jobSheetOpen} title="Job">
        <div class="job-sheet">
          {#if jobStore.current}
            <JobCard
              job={jobStore.current}
              preview={false}
              oncancel={sendControl('quit')}
              onpause={sendControl('pause')}
              onresume={sendControl('resume')}
              onnext={sendControl('next')}
              onrepeat={sendControl('repeat')}
            />
          {/if}
          <QueuePanel jobs={jobStore.queuedJobs} onremove={removeQueuedJob} onload={loadQueuedSettings} onclear={clearQueuedJobs} />
        </div>
      </Sheet>
      <Sheet bind:open={sheetOpen} title="All settings" meta={changedCount ? `${changedCount} changed` : undefined} keepMounted>
        {#snippet headActions()}
          <button type="button" class="ui-btn ui-btn-sm ui-btn-icon" title="Copy these settings as a ziv CLI command" aria-label="Copy CLI command" onclick={copyCliCommand}><Icon name="terminal" size={14} /></button>
        {/snippet}
        <div class="sheet-model">{@render toolbar(true)}</div>
        <ControlsSidebar
          {context}
          {busy}
          panes="settings"
          settingsSubmit={false}
          onsettingsdone={() => { sheetOpen = false; }}
          jobsActive={jobStore.jobsActive}
          queuedCount={jobStore.queuedJobs.length}
          submitError={generateError}
          {imageFile}
          {referencePreviewUrl}
          lastSeed={historyStore.assets[0]?.seed ?? null}
          onImageFileChange={(f) => { imageFile = f; submitError = null; }}
        />
      </Sheet>
    </div>
  {:else}
    {@render toolbar(false)}
    <!-- Left column (Compose + Settings) and the stage with its history filmstrip -->
    <main class="workspace-layout min-h-0 flex-1 overflow-hidden" class:tall class:sidebar-collapsed={!tall && draft.state.sidebarCollapsed}>
      <ControlsSidebar
        {context}
        {busy}
        {tall}
        jobsActive={jobStore.jobsActive}
        queuedCount={jobStore.queuedJobs.length}
        submitError={generateError}
        {imageFile}
        {referencePreviewUrl}
        lastSeed={historyStore.assets[0]?.seed ?? null}
        onImageFileChange={(f) => { imageFile = f; submitError = null; }}
        oncopycli={copyCliCommand}
      />
      {@render stage()}
    </main>
  {/if}
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
  onnearend={() => { if (lightboxMode === 'history') void historyStore.loadMore(); }}
  onreuse={reuseAsset}
  onreference={useAsReference}
  onextend={extendAsset}
  {referenceUnavailable}
  onupscale={upscaleAsset}
  ondelete={deleteWorkspaceAsset}
/>

<style>
  /* Chips in the toolbar share the height of the selects beside them. */
  .lora-chip { height: var(--spacing-control); padding: 0 8px; border-radius: var(--radius-sm); color: var(--color-text-primary); }
  .lora-weight { field-sizing: content; min-width: 2ch; border-radius: var(--radius-xs); background: transparent; text-align: center; font-family: var(--font-mono); font-weight: 400; color: var(--color-text-muted); appearance: textfield; -moz-appearance: textfield; }
  /* Browsers without field-sizing would give the number its default width of 150px or more. */
  @supports not (field-sizing: content) { .lora-weight { width: 4.5ch; } }
  .lora-weight::-webkit-inner-spin-button, .lora-weight::-webkit-outer-spin-button { margin: 0; -webkit-appearance: none; }
  .lora-weight:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }
  .lora-remove { display: inline-grid; place-items: center; width: 16px; height: 16px; margin-right: -3px; border-radius: var(--radius-xs); color: var(--color-text-muted); }
  @media (hover: hover) { .lora-remove:hover { background: var(--color-error-surface); color: var(--color-error); } }
  .workspace-layout { display: grid; grid-template-columns: 360px minmax(0, 1fr); }
  .workspace-layout.sidebar-collapsed { grid-template-columns: 56px minmax(0, 1fr); }
  /* Portrait tablet: one column, the stage on top with the strip, the two panes under it at 46% of the height. */
  .workspace-layout.tall { grid-template-columns: minmax(0, 1fr); grid-template-rows: minmax(0, 1fr) 46%; }
  .workspace-layout.tall > :global(.workspace-left) { order: 2; min-height: 0; border-top: 1px solid var(--color-border-subtle); }
  .workspace-layout.tall > .workspace-preview { order: 1; }
  /* Keep toasts centred on the preview, clear of the sidebar's Generate button. */
  @media (min-width: 640px) {
    :global(:root:has(.workspace-layout)) { --toast-inset-left: 360px; }
    :global(:root:has(.workspace-layout.sidebar-collapsed)) { --toast-inset-left: 56px; }
    :global(:root:has(.workspace-layout.tall)) { --toast-inset-left: 0px; }
  }
  /* Phone: toasts dock above the composer and the strip. */
  :global(:root:has(.compact-workspace)) { --toast-bottom: calc(132px + env(safe-area-inset-bottom, 0px)); }
  :global(:root:has(.compact-workspace .job-bar)) { --toast-bottom: calc(224px + env(safe-area-inset-bottom, 0px)); }
  /* The phone composition. */
  .compact-workspace { display: flex; flex: 1; min-height: 0; flex-direction: column; }
  .compact-workspace > :global(.workspace-preview) { flex: 1; min-height: 0; }
  .compact-workspace :global(.latest-region) { padding: 8px 12px; }
  .compact-workspace :global(.mascot-dock) { display: none; }
  .compact-workspace :global(.latest-meta span) { max-width: none; flex: 1 1 auto; }
  .compact-workspace :global(.latest-meta) { justify-content: flex-start; gap: 10px; }
  .compact-workspace :global(.latest-seed) { display: none; }
  /* The image and its caption are one block, centred: the tile takes the image's shape, as large as fits above the caption. */
  .compact-workspace :global(.latest-region.fit) { container-type: size; align-items: center; justify-content: center; }
  .compact-workspace :global(.latest-region.fit > *) { --fit-w: min(100cqw, calc((100cqh - 36px) * var(--ar)), var(--natural-w, 100cqw)); width: var(--fit-w); }
  .compact-workspace :global(.latest-region.fit .latest-media) { flex: none; aspect-ratio: var(--ar); }
  .compact-workspace :global(.latest-region.fit .latest-media .asset-tile-img) { object-fit: contain; }
  .stage-position-pill { position: absolute; top: 10px; left: 50%; z-index: 4; display: flex; align-items: center; gap: 2px; padding: 3px 3px 3px 12px; translate: -50% 0; border-radius: 999px; background: color-mix(in srgb, var(--color-bg-overlay) 88%, transparent); box-shadow: 0 2px 10px var(--color-scrim); font-size: var(--text-meta); color: var(--color-text-muted); white-space: nowrap; backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); }
  .stage-position-pill b { font-family: var(--font-mono); font-weight: 500; color: var(--color-text-primary); }
  .stage-position-pill button { display: inline-flex; align-items: center; gap: 2px; height: 30px; margin-left: 8px; padding: 0 8px 0 12px; border-radius: 999px; background: var(--color-bg-surface); font-weight: 600; color: var(--color-text-primary); }
  @media (hover: none) { .stage-position-pill button:active { background: var(--color-bg-surface-hover); } }
  /* Phone: the preview takes the stage with the bar over its foot. */
  .job-view-compact { position: relative; padding: 12px 12px 12px; }
  .job-view-compact .job-view-preview { padding-bottom: 104px; }
  .job-view-compact > :global(.job-bar) { position: absolute; left: 12px; right: 12px; bottom: 12px; }
  .job-sheet { display: flex; flex-direction: column; gap: 12px; padding: 12px; }
  .compact-controls { position: relative; flex: none; }
  /* The popover rises from the strip; a clear scrim behind it catches the tap that dismisses it, and the strip stays
     above the scrim so another chip switches the popover in one tap. */
  .compact-controls > .pop-scrim { position: fixed; inset: 0; z-index: 44; background: transparent; cursor: default; }
  .compact-controls > :global(.setting-popover) { position: absolute; left: 0; right: 0; bottom: 100%; z-index: 45; }
  .compact-controls > :global(.settings-strip) { position: relative; z-index: 46; }
  .composer { display: flex; flex-direction: column; gap: 8px; padding: 8px max(12px, env(safe-area-inset-right, 0px)) calc(8px + env(safe-area-inset-bottom, 0px)) max(12px, env(safe-area-inset-left, 0px)); border-top: 1px solid var(--color-border-subtle); background: var(--color-bg-overlay); }
  .composer-error { margin: 0; }
  .composer-row { display: flex; align-items: center; gap: 8px; }
  .composer-row[hidden] { display: none; }
  .composer-summary { display: flex; flex: 1; min-width: 0; flex-direction: column; justify-content: center; gap: 1px; height: 44px; padding: 0 12px; border-radius: var(--radius-md); background: var(--color-bg-fill); text-align: left; color: var(--color-text-primary); }
  .composer-label { font-size: 10px; line-height: 12px; }
  .composer-line { overflow: hidden; font-size: 15px; line-height: 20px; white-space: nowrap; text-overflow: ellipsis; }
  .composer-line.placeholder { color: var(--color-zinc-600); }
  .composer-split { display: flex; flex: none; gap: 1px; }
  .composer-generate { flex: none; height: 44px; padding: 0 14px 0 16px; border-radius: var(--radius-md) 0 0 var(--radius-md); font-size: 14px; }
  .composer-more { flex: none; width: 32px; height: 44px; padding: 0; border-radius: 0 var(--radius-md) var(--radius-md) 0; }
  .composer-q { font-family: var(--font-mono); font-size: var(--text-ui); font-weight: 500; opacity: 0.85; }
  .composer-pane[hidden] { display: none; }
  .composer-pane :global(.compose-pane) { max-height: 56dvh; padding: 4px 0 0; border-bottom: 0; }
  .composer-foot { display: flex; gap: 8px; padding-top: 4px; }
  .composer-foot .ui-btn { height: 44px; font-size: 14px; }
  .composer-foot .ui-btn-main { flex: 1; }
  .sheet-model { flex: none; }
  .sheet-model :global(.toolbar-stacked) { background: transparent; }
  .sheet-model :global(.toolbar-stacked .ui-select-shell) { height: 40px; font-size: 14px; }
  /* A running job: the preview fills the height on the left, the card is a column that scrolls on its own. */
  /* The view may shrink below its content's width: a centred flex item otherwise grows to the preview and spills both
     edges (and the phone's job bar) off the screen. The preview is a size container that the live image fits. */
  .job-view { display: flex; width: 100%; min-width: 0; height: 100%; align-items: stretch; gap: 16px; padding: 16px; }
  .job-view-preview { position: relative; display: flex; flex: 1; min-width: 0; align-items: center; justify-content: center; container-type: size; }
  .job-view :global(.job-view-mascot) { position: absolute; top: 0; left: 0; pointer-events: none; }
  .job-view-card { display: flex; flex: none; width: 400px; max-width: 46%; flex-direction: column; gap: 12px; overflow-y: auto; }
  /* A drag moves what the stage shows with the finger, and lets it settle back or away when it lifts. */
  .stage-body > :global(:not(.gallery-pull)) { transition: translate 260ms cubic-bezier(0.32, 0.72, 0, 1); }
  .stage-body.moved > :global(:not(.gallery-pull)) { translate: var(--stage-dx, 0px) var(--stage-dy, 0px); }
  .stage-body.dragging > :global(:not(.gallery-pull)) { transition: none; }
  .gallery-pull { position: absolute; bottom: 16px; left: 50%; z-index: 4; display: inline-flex; align-items: center; gap: 6px; height: 32px; padding: 0 14px; border-radius: 16px; translate: -50% 0; background: color-mix(in srgb, var(--color-bg-overlay) 85%, transparent); font-size: var(--text-ui); font-weight: 600; white-space: nowrap; color: var(--color-text-secondary); pointer-events: none; }
  .gallery-pull.ready { background: var(--color-primary-main); color: var(--color-primary-ink); }
  /* The in-place history arrows: visible while the pointer is over the stage, never for a finger (it swipes). */
  .stage-nav { position: absolute; top: 50%; z-index: 3; transform: translateY(-50%); opacity: 0; transition: opacity 0.12s ease; }
  .stage-nav.prev { left: 12px; }
  .stage-nav.next { right: 12px; }
  .stage-nav:focus-visible { opacity: 1; }
  @media (hover: hover) { .stage-body:hover .stage-nav:not(:disabled) { opacity: 1; } }
  @media (pointer: coarse) { .stage-nav { display: none; } }
  .latest-region :global(.latest-media) { opacity: 0; transition: opacity 250ms ease; }
  .latest-region :global(.latest-media.loaded) { opacity: 1; }
  .latest-meta { display: flex; flex: none; align-items: center; justify-content: center; gap: 14px; min-width: 0; padding-top: 8px; font-size: var(--text-ui); color: var(--color-text-muted); }
  .latest-meta span { max-width: 520px; color: var(--color-text-secondary); }
  .latest-meta b { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); font-weight: 500; color: var(--color-text-secondary); }
  .completed-output-scroll { display: flex; flex-direction: column; }
  .completed-output-grid { display: grid; justify-content: center; margin-block: auto; }
</style>
