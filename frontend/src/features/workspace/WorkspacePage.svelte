<script lang="ts">
  import { onMount, tick, untrack } from 'svelte';
  import { errorText } from '$lib/api/client';
  import { draft, extendStepFor, slotPaths, videoTimingFor } from '$lib/state/draft.svelte';
  import { hasPath } from '$lib/state/imageSlots';
  import { anyEnhancedUsed, enhancedPartUsed, submittedCaption, submittedPair } from '$lib/state/promptEnhance';
  import { jobStore } from '$lib/state/job.svelte';
  import { downloadStore, type DownloadRequestResult } from '$lib/state/downloads.svelte';
  import { downloadStatusText } from '$lib/state/downloadProgress';
  import { galleryStore } from '$lib/state/gallery.svelte';
  import { inProject, projectPick } from '$lib/state/projectPick.svelte';
  import { projectsStore } from '$lib/state/projects.svelte';
  import { router } from '$lib/state/router.svelte';
  import { hold } from '$lib/actions/hold';
  import { canScrollToward, OWN_DRAG, pan, rubberBand, type PanEnd, type PanMove, type PanStart } from '$lib/actions/pan';
  import { GALLERY_PULL_PX, galleryPullReady, stageSwipeOutcome, wheelStep, type StageMoves } from './stageSwipe';
  import { addToast } from '$lib/state/toasts.svelte';
  import { firstValues } from '$lib/prefillParams';
  import { checkReferencePaths, getWorkspaceCoreContext, planGenerate, sendJobControl, skipFirstRun, submitGenerate, parseUrlPrefill } from '$lib/api/workspace';
  import { confirmDownloads } from '$lib/state/confirmDownloads';
  import { updateConfig } from '$lib/api/config';
  import { workspaceMemory } from '$lib/state/pageMemory';
  import { emptySlotsHolding } from '$lib/state/assetRemoval';
  import { deleteAsset } from '$lib/api/gallery';
  import { completionToast } from '$lib/state/completionToast';
  import { moveAndReconcile } from '$lib/state/moveAndReconcile';
  import { audioTrackParams, canEdit, canExtend, editParams, extendParams, frameParams, referenceParams, reuseParams, frameSetNotice, useAsState, voiceParams, type AudioUseTarget, type DeleteOptions, type FrameTarget, type ReferenceTarget } from '$lib/state/assetActions';
  import { arrivalFor, flashArrival } from '$lib/arrivalFlash';
  import type { FramePoint } from '$lib/state/framePoint';
  import { runFrameGrab } from '$lib/state/frameGrab.svelte';
  import { audioModelLabel, cutNote } from '$lib/state/audioAsset';
  import { formatClock } from '$lib/state/clock';
  import { Badge, Icon, MascotSpot, Tooltip } from '$lib/components/atoms';
  import { rememberMascotSpots } from '$lib/components/atoms/MascotSpot.svelte';
  import { mascotHome } from '$lib/state/mascotHome.svelte';
  import {
    DROWSY_AFTER_MS,
    GREETING_DURATION_MS,
    REACTION_DURATION_MS,
    TYPING_DURATION_MS,
    mascotMood as pickMascotMood,
    mascotProgress as pickMascotProgress,
    mascotSpot as pickMascotSpot,
    mascotTool as pickMascotTool,
    type MascotReaction,
  } from '$lib/state/mascot';
  import { ActionMenu, AssetTile, AssetViewer, JobBar, JobCard, JobPreview, LicenceMark, ModelStatusBadges, ReviewStage, SaveVoiceDialog, Segmented, Select, requestConfirm, type ActionMenuEntry } from '$lib/components/molecules';
  import { compactUi } from '$lib/state/compactUi.svelte';
  import { failureToastText, isReviewing, isVoiceDesignJob } from '$lib/state/jobView';
  import { first, jobSettingsPrefill } from '$lib/state/jobSettings';
  import { cliCommand, cliOutputDir, formSettings, type CliCaveat, type CliCommand } from '$lib/state/cliCommand';
  import { appliedPreset, presetDiffersFrom, presetSettingsFromDraft } from '$lib/state/presets';
  import { presetsStore } from '$lib/state/presets.svelte';
  import { extendGuard, finalizeSceneFields, type ExtendGuard } from '$lib/state/extend';
  import { extendInspection } from '$lib/state/extendInspection.svelte';
  import { clipMemory } from '$lib/state/clipMemory.svelte';
  import { INPUT_DEBOUNCE_MS } from '$lib/state/debounce';
  import { clipMemoryRequest } from '$lib/state/memoryFit';
  import { familyLabel, formatLoraString, groupLorasForModel, loraLabel, modelLoraFamily, parseLoraString, recentLoraNames, type LoraChip } from '$lib/state/loras';
  import { chipDownload, failureText, waitText } from '$lib/state/loraDownloads';
  import { waitForListing } from '$lib/state/loraListing';
  import { addLoraChip, removeLoraChip } from '$lib/state/loraActions';
  import { generateBlock } from '$lib/state/generateBlock.svelte';
  import { textWriting } from '$lib/state/textWriting.svelte';
  import { dockHeight } from '$lib/actions/dockHeight';
  import { pendingTriggers } from '$lib/state/loraTriggers.svelte';
  import DiscoverDrawer, { type LoraUse } from './discover/DiscoverDrawer.svelte';
  import { discoverTarget } from './discover/discoverView';
  import WaitingLabel from './WaitingLabel.svelte';
  import QueuePanel from './QueuePanel.svelte';
  import FirstRunDialog from './FirstRunDialog.svelte';
  import { firstRun, shouldOfferFirstRun } from './firstRun.svelte';
  import { confirmDeleteAsset } from '$lib/state/assetActions';
  import { trackLengths } from '$lib/state/audioTrackLength.svelte';
  import ControlsSidebar from './ControlsSidebar.svelte';
  import ComposePane from './ComposePane.svelte';
  import GalleryStrip from './GalleryStrip.svelte';
  import SettingsStrip from './SettingsStrip.svelte';
  import SettingPopover from './SettingPopover.svelte';
  import { imagesSummary } from './imagesSummary';
  import { currentAudioTrackView } from './audioTrackView';
  import { audioFormData, currentAudioView } from './audioView';
  import { AUDIO_TEXT_KEYS, audioModelsOf, availableKinds, musicSecondsShown, replacesVoice } from '$lib/state/audioDraft';
  import { AUDIO_BOX_OF_KIND, rewriteUsed } from '$lib/state/audioRewrites';
  import { keyNameOf } from '$lib/state/musicSettings';
  import { autoSoundSeconds, soundTypeLabel } from '$lib/state/soundSettings';
  import { deliverySummary } from '$lib/state/speechDelivery';
  import { voiceConsent } from '$lib/state/voiceConsent.svelte';
  import { voiceWindow } from '$lib/state/voiceWindow.svelte';
  import { voicesStore } from '$lib/state/voices.svelte';
  import { AUDIO_KIND_INFO, IMAGE_KIND_INFO, VIDEO_KIND_INFO, type KindChoice } from './kindInfo';
  import { currentImagesView } from './imagesView';
  import Painter from './edit/Painter.svelte';
  import ImageToEditCard from './edit/ImageToEditCard.svelte';
  import { followDraft } from './edit/editActions';
  import { currentEditFacts } from './edit/currentEdit';
  import { editFormData } from './edit/editForm';
  import { modelsOf } from '$lib/state/workflowModels';
  import { painterLayer } from './edit/editStore.svelte';
  import { editPlaceholder } from './edit/editView';
  import PresetMenu from './PresetMenu.svelte';
  import VoicesWindow from './VoicesWindow.svelte';
  import { audioChips, pendingChips, settingsChips, type ChipKey } from './settingsChips';
  import { Sheet } from '$lib/components/molecules';
  import { fileName } from '$lib/components/molecules/assetDetails';
  import { defaultsFor, settingDefaultsFor, stepOptionsFor } from '$lib/state/draft.svelte';
  import { changedRowCount, chipChanged } from './settingRows';
  import { framesToSeconds, roundSeconds } from '$lib/state/videoDuration';
  import { fitOutputGrid } from './outputGrid';
  import { randomSeed } from './seed';
  import { workspaceCapabilities } from './capabilities';
  import { referenceEditor } from './referenceEditor';
  import { acceptsPageShortcut, hasOpenModal, isCommandKey } from '$lib/keyboard';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { startUpscale } from '$lib/state/upscale';
  import { hasPendingJobSettings, takeJobSettings } from '$lib/state/pendingJobSettings.svelte';
  import { hasPendingReference, takeReference } from '$lib/state/pendingReference.svelte';
  import { heldUploads } from '$lib/state/heldUploads.svelte';
  import { IMAGE_KINDS, isAudioWorkflow, videoKinds, workspaceTab } from '$lib/state/workspaceTab';
  import { applyPromptRun, takePromptRun, type PendingPromptRun } from '$lib/state/pendingPromptRun';
  import type { DraftState, GalleryAsset, JobControlAction, JobSettings, JobSnapshot, ModelOption, PrefillParams, UpscaleFactor, UpscaleMethod, WorkspaceContext, WorkspaceTab, Workflow, WorkspaceWorkflow, AudioWorkflow, ImageKind, VideoKind } from '$lib/types';

  interface Props {
    /** The workflows the server offers, once its context has loaded; the top bar shows a tab for each. */
    workflows?: WorkspaceWorkflow[] | null;
  }

  let { workflows = $bindable(null) }: Props = $props();

  // Portrait tablet: the stage over Compose and Settings side by side, instead of a sidebar beside a narrow stage.
  const tall = $derived(layout.mode === 'tall');
  // Phone: the stage is the screen; settings are a strip of chips that open one control each, the prompt a composer.
  const compact = $derived(layout.mode === 'compact');
  let activeChip = $state<ChipKey | null>(null);
  // The phone's Preset chip opens the preset menu (an action sheet), not a popover.
  let presetMenuOpen = $state(false);
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
  let lightboxOpen = $state(false);
  // The viewer follows its asset by id, not by place: a new output lands at the front of History while it is open.
  let viewerAssetId = $state<string | null>(null);
  let lightboxMode = $state<'history' | 'completed-output'>('history');
  let completedOutputTrigger = $state<HTMLElement | null>(null);
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
    uploaded_image: 'An uploaded image is named by file name; replace it with its path.',
    uploaded_audio: 'An uploaded audio track is named by file name; replace it with its path.',
    uploaded_voice: 'An uploaded voice recording is named by file name; replace it with its path.',
    edit_mask: 'The painted area is not a file: save it as a PNG with the shape of the image, painted where it should change, and name it after --mask.',
  };
  const OUTPUT_GAP_PX = 12;
  const MIN_OUTPUT_CELL_PX = 160;
  let outputAreaWidth = $state(0);
  let outputAreaHeight = $state(0);
  // The stage and the lightbox show the pick's outputs only; the job card and queue stay global.
  const jobOutputs = $derived<GalleryAsset[]>((jobStore.current?.outputs ?? []).filter((output) => inProject(output, projectPick.current)));
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
      : galleryStore.assets.length > 0 ? 'latest'
      : 'empty'
  );
  // The stage steps through history in place: the viewed asset is tracked by id, so new outputs arriving at the
  // front never move it; null means the latest. A new job or a vanished asset brings it back to the latest.
  let viewedId = $state<string | null>(null);
  const viewedIndex = $derived.by(() => {
    if (viewedId === null) return 0;
    const index = galleryStore.assets.findIndex((asset) => asset.id === viewedId);
    return index < 0 ? 0 : index;
  });
  const viewedAsset = $derived<GalleryAsset | null>(galleryStore.assets[viewedIndex] ?? null);
  const atLatest = $derived(viewedIndex === 0);
  const historyTotal = $derived(galleryStore.totalCount ?? galleryStore.assets.length);
  const STEP_NEAR_END = 3;

  /** Show the asset `delta` places older (positive) or newer (negative) than the viewed one, loading more history near the end. */
  function stepView(delta: number): void {
    const assets = galleryStore.assets;
    if (assets.length === 0) return;
    const next = Math.min(assets.length - 1, Math.max(0, viewedIndex + delta));
    viewedId = next === 0 ? null : assets[next].id;
    if (next >= assets.length - STEP_NEAR_END && galleryStore.hasMore) void galleryStore.loadMore();
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
    older: canStepStage && previewView !== 'empty' && (viewedIndex < galleryStore.assets.length - 1 || galleryStore.hasMore || previewView === 'outputs'),
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
  // Which images the open tab uses (uploads picked from this device are held in `heldUploads`, not in the draft), what a
  // submit sends and why Generate is off when one of them is gone: dropping a reference would renumber the prompt's "image 2".
  const images = $derived(context === null ? null : currentImagesView(context));
  const imageBlock = $derived(images?.block ?? null);
  // Whether the open tab follows an audio track, what a submit sends for it, and why Generate is off when an uploaded track is gone.
  const audioTrack = $derived(context === null ? null : currentAudioTrackView(context));
  const trackBlock = $derived(audioTrack?.block ?? null);
  // What the Audio tab sends and why Generate is off on it: the kind's text, the voice and the speaker's tick.
  const audio = $derived(context === null ? null : currentAudioView(context));
  const audioReason = $derived(audio?.reason ?? null);
  // A voice recording's length is read once it is chosen: it must be within what the model follows.
  $effect(() => {
    if (audio?.voice.recording && audio.voice.set) trackLengths.ensure(audio.voice.path, audio.voice.file);
  });
  // The track's length shows beside its name and sets the note under Start at: read it once the track is in use.
  $effect(() => {
    if (audioTrack?.visible && audioTrack.set) trackLengths.ensure(audioTrack.path, audioTrack.file);
  });
  // Keep the preview occupied until the latest image has painted. Videos show at once:
  // some browsers load nothing until play is pressed, so no load event is guaranteed.
  const latestLoading = $derived(
    previewView === 'latest'
      && viewedAsset !== null
      && viewedAsset.media_type !== 'video'
      && loadedLatestUrl !== viewedAsset.url
  );
  const lookingForHistory = $derived(!historyChecked || galleryStore.loading);
  const mascotMood = $derived(pickMascotMood({
    job: jobStore.current,
    reaction,
    loadError: loadError !== null,
    loading: latestLoading || (previewView === 'empty' && lookingForHistory),
    greeting,
    typing,
    drowsy,
  }));
  // The creating mood's canvas follows the running job's real steps.
  const mascotProgress = $derived(pickMascotProgress(jobStore.current));
  // A running job decides the tool; otherwise the form's workflow does.
  const mascotTool = $derived(pickMascotTool(jobStore.isRunning ? jobStore.current?.workflow : draft.state.workflow));
  const lightboxAssets = $derived<GalleryAsset[]>(
    lightboxMode === 'completed-output'
      ? jobOutputs
      : galleryStore.assets
  );

  // Surface successful batch items in History as soon as their SSE event lands.
  // The terminal refresh remains authoritative and reconciles the full inventory.
  $effect(() => {
    // The store keeps the pick's outputs only, so it gets them all.
    const outputs = jobStore.current?.outputs ?? [];
    if (jobStore.isRunning && outputs.length > 0) {
      untrack(() => galleryStore.mergeOutputs(outputs));
    }
  });

  // Mode derived from workflow
  const isImageMode = $derived(
    draft.state.workflow === 'txt2img' || draft.state.workflow === 'img2img' || draft.state.workflow === 'edit'
  );
  // The Image tab's Edit kind: the image to edit, its paint and the prompt that says what to change.
  const isEdit = $derived(draft.state.workflow === 'edit');
  // What the open tab and model offer (the controls the workflow shows, and the prompt file when the tab takes one).
  const pageCaps = $derived(workspaceCapabilities(context, draft.state));
  // The one reason on the phone's composer: the Audio tab's own, else a run that is writing the text Generate sends, else a gone or unpaired image.
  const composerReason = $derived(audio?.active ? audioReason : (textWriting.reason ?? imageBlock));
  // Generate waits for the stored uploads too: until they are read back a lost upload cannot be told from one still coming.
  const authorityReady = $derived(context !== null && draft.authorityReady && heldUploads.restored);

  const imageModels = $derived(context?.image_models ?? []);
  // The Audio tab is open: the kind's models, and the kinds this machine runs, in place of the image or video models.
  const isAudio = $derived(isAudioWorkflow(draft.state.workflow));
  const audioKinds = $derived(context ? availableKinds(context) : []);
  // The Video tab's kinds (New, and Extend where the machine can extend). With one kind there is nothing to choose, so no Kind control.
  const videoKindOptions = $derived(videoKinds(context?.workflow_contract.values ?? null));
  const showVideoKind = $derived(!isAudio && !isImageMode && videoKindOptions.length > 1);
  const kindChoice = $derived<KindChoice | null>(
    isAudio
      ? { kinds: audioKinds, info: AUDIO_KIND_INFO, value: draft.state.workflow, pick: (kind) => draft.setAudioKind(kind as AudioWorkflow, context) }
      : isImageMode
        ? { kinds: IMAGE_KINDS, info: IMAGE_KIND_INFO, value: isEdit ? 'edit' : 'new', pick: (kind) => draft.setKind('image', kind as ImageKind, context) }
      : showVideoKind
        ? { kinds: videoKindOptions, info: VIDEO_KIND_INFO, value: draft.state.workflow === 'extend' ? 'extend' : 'new', pick: (kind) => draft.setKind('video', kind as VideoKind, context) }
        : null,
  );
  // The models the open tab offers: Edit's are the ones that can edit.
  const currentModels = $derived<ModelOption[]>(!context ? [] : isAudio ? audioModelsOf(context, draft.state.workflow as AudioWorkflow) : modelsOf(context, draft.state.workflow));
  const modelOptions = $derived(currentModels.map((m) => ({ value: m.id, label: modelOptionText(m) })));
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

  // The settings preset the form started from, once the list knows it, and whether the form has moved off it.
  const activePreset = $derived(appliedPreset(presetsStore.list, draft.state));
  // The form's preset settings as one string: it only changes when a preset setting does, so typing the prompt leaves `presetChanged` alone.
  const presetSettingsJson = $derived(activePreset !== null && context !== null ? JSON.stringify(presetSettingsFromDraft(draft.state, context)) : null);
  // The applied preset is always of the form's mode, so its own mode is the one compared in.
  const presetChanged = $derived(activePreset !== null && context !== null && presetSettingsJson !== null && presetDiffersFrom(activePreset, activePreset.mode, JSON.parse(presetSettingsJson), context));

  // Active lora chips parsed from draft.state.loraString; reused settings record paths, shown by LoRA name.
  const loraChips = $derived<LoraChip[]>(parseLoraString(draft.state.loraString, loraOptions));
  // What the chips hold Generate back for: a download on its way, or one that failed.
  const block = $derived(generateBlock(draft.state, loraOptions));
  const waitFull = $derived(block.waiting ? waitText(block.waiting, false).full : undefined);
  // The error slot names a failed LoRA download; while it is on screen, its toast would say the same thing again.
  const failureSlotShown = $derived(compact || tall || !draft.state.sidebarCollapsed);
  $effect(() => {
    downloadStore.showFailuresInline(failureSlotShown ? loraChips.map((chip) => chip.name) : []);
    return () => downloadStore.showFailuresInline([]);
  });

  // The phone's settings strip: one chip per setting, derived from the same state the inspector shows.
  // Before the context answers, placeholder chips give the strip its size, so the stage never has to shrink later.
  const chips = $derived.by(() => {
    if (!compact) return [];
    if (!context || !authorityReady) return pendingChips(draft.state.workflow);
    const s = draft.state;
    const caps = workspaceCapabilities(context, s);
    const defaults = defaultsFor(context, s, draft.extendSource);
    if (caps.isAudio && caps.audioKind && audio) {
      const soundVocabulary = context.audio_settings;
      // A sound on Auto shows the length its type will use; a song shows its own, and a take that sets none has no length.
      const lengthAuto = caps.audioKind === 'sound' && soundVocabulary !== undefined && s.audioSeconds === 0;
      const seconds = caps.audioKind === 'music' ? musicSecondsShown(s.musicSeconds, audio.model) : lengthAuto ? autoSoundSeconds(s.soundType, soundVocabulary!) : s.audioSeconds;
      return audioChips({
        kind: caps.audioKind,
        state: s,
        modelLabel: selectedModel?.label ?? s.model,
        modelDownloaded: selectedModel?.downloaded !== false,
        voice: caps.showVoice ? audio.voice : null,
        voiceBlocks: audio.reasonFromVoice,
        changed: (chip) => chipChanged(chip, caps, s, defaults),
        delivery: { shown: caps.showDelivery && context.audio_settings !== undefined, summary: context.audio_settings ? deliverySummary(s.speechDelivery, context.audio_settings) : '' },
        showLength: caps.showLength,
        seconds,
        lengthAuto,
        soundType: { shown: caps.showSoundType && soundVocabulary !== undefined, label: soundVocabulary ? soundTypeLabel(s.soundType, soundVocabulary) : '' },
        tempo: { shown: caps.showTempo },
        key: { shown: caps.showKey, name: s.audioKey === null ? null : keyNameOf(context.audio_settings, s.audioKey) },
        showSeed: caps.showSeed,
        showRuns: caps.showRuns,
      });
    }
    const presets = (caps.isImageMode ? context.image_size_dimensions : (context.video_size_dimensions ?? {}))[s.ratio]?.[s.size] ?? [s.width, s.height];
    const timing = videoTimingFor(context, s.model);
    const sourceFps = draft.extendSource?.fps ?? null;
    const memoryStatus = s.workflow === 'extend' ? draft.extendSource?.memory_fit?.status : clipMemory.fit?.status;
    return settingsChips({
      caps,
      state: s,
      videoKind: showVideoKind ? (s.workflow === 'extend' ? 'extend' : 'new') : null,
      changed: (chip) => chipChanged(chip, caps, s, defaults),
      width: s.dimensionMode === 'custom' ? s.width : presets[0],
      height: s.dimensionMode === 'custom' ? s.height : presets[1],
      seconds: caps.showFrameCount ? roundSeconds(framesToSeconds(s.frameCount, timing.fps)) : null,
      extendSeconds: sourceFps && s.extendFrames !== null ? roundSeconds(s.extendFrames / sourceFps) : null,
      loraNames: loraChips.map((chip) => loraLabel(chip.name)),
      loraDownload: block.failures.length > 0 ? { kind: 'failed' } : block.waiting ? { kind: 'downloading', percent: block.waiting.percent } : undefined,
      modelLabel: selectedModel?.label ?? s.model,
      modelDownloaded: selectedModel?.downloaded !== false,
      sourceName: s.extendSourcePath ? fileName(s.extendSourcePath) : null,
      images: images?.visible ? imagesSummary(images) : null,
      track: audioTrack?.visible ? audioTrack : null,
      memoryWarn: memoryStatus === 'tight' || memoryStatus === 'too_large',
      presetName: activePreset?.name ?? null,
      presetSlug: activePreset?.slug ?? null,
      presetChanged,
      openKey: activeChip,
    });
  });
  // The same count as the Reset badge: the rows of the Settings pane that differ from their defaults. The preset counts none of them.
  const changedCount = $derived(context && authorityReady ? changedRowCount(workspaceCapabilities(context, draft.state), draft.state, defaultsFor(context, draft.state, draft.extendSource)) : 0);
  // What the folded composer reads: the prompt file and its count, the Enhanced text, or the prompt.
  const composerSummary = $derived.by(() => {
    const s = draft.state;
    // The Audio tab's composer reads the open kind's own text.
    if (isAudioWorkflow(s.workflow)) {
      const kind = { music: { label: 'Style · Lyrics', empty: 'Describe the style…' }, speech: { label: 'Text', empty: 'What to say…' }, sound: { label: 'Prompt', empty: 'Describe the sound…' } }[s.workflow];
      // Generate sends the Enhanced text when the box has one, so the dock reads it too.
      const rewrite = s.audioRewrites[AUDIO_BOX_OF_KIND[s.workflow]];
      if (rewriteUsed(rewrite)) return { label: 'Enhanced', text: rewrite.text, empty: false };
      const text = s[AUDIO_TEXT_KEYS[s.workflow]];
      return { label: kind.label, text: text || kind.empty, empty: !text };
    }
    if (pageCaps.promptFileMode) return { label: 'Prompt file', text: s.promptFilePath ? `${fileName(s.promptFilePath)} · ${s.promptFileOptionIds.length} ${s.promptFileOptionIds.length === 1 ? 'prompt' : 'prompts'}` : 'Choose a prompt file', empty: !s.promptFilePath };
    const video = s.workflow === 'img2vid' || s.workflow === 'txt2vid' || s.workflow === 'extend';
    const enhanced = anyEnhancedUsed(s);
    const text = enhanced ? s.enhancedPrompt || s.enhancedScene : s.prompt || (isEdit ? '' : s.scene);
    const empty = video ? 'What happens in the clip…' : isEdit ? editPlaceholder(currentEditFacts()) : 'Describe what to make…';
    return { label: enhanced ? 'Enhanced' : video ? 'Action' : 'Prompt', text: text || empty, empty: !text };
  });
  const compactSubmitDisabled = $derived(busy || !authorityReady || composerReason !== null || promptChoiceMissing() || currentExtendGuard().blocked || imageBlock !== null || trackBlock !== null || block.blocked);
  // The phone's strip and composer are drawn before the context loads (the frame first); they wake once it has.
  const controlsPending = $derived(!context || !authorityReady);

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
  // Discover LoRAs: the drawer, and the red chip's menu.
  let discoverOpen = $state(false);
  let openDiscoverWhenReady = $state(false);
  let failedChip = $state<{ name: string; anchor: HTMLElement } | null>(null);
  const discover = $derived(context && draft.state.model ? discoverTarget(context, draft.state.model, selectedModel?.label ?? draft.state.model) : null);
  function openDiscover(): void {
    loraPopoverOpen = false;
    if (sheetOpen) {
      // On a phone the toolbar sits inside All settings: close it first, so the drawer does not open over it.
      sheetOpen = false;
      void tick().then(() => { discoverOpen = true; });
      return;
    }
    discoverOpen = true;
  }
  $effect(() => {
    if (!openDiscoverWhenReady || !context || !authorityReady) return;
    untrack(() => {
      openDiscoverWhenReady = false;
      discoverOpen = true;
    });
  });
  // The picker leads with the LoRAs that fit the selected model; the rest fold away under "Other models".
  // Reading the model through its own derived keeps the groups from being rebuilt on every other draft edit.
  const draftModel = $derived(draft.state.model);
  const loraGroups = $derived.by(() => {
    const model = context ? modelLoraFamily(context, selectedModel?.id ?? draftModel) : null;
    return groupLorasForModel(loraOptions, model, recentLoraNames(galleryStore.allRecent, model?.family ?? null, loraOptions));
  });
  const loraMenuItems = $derived.by<ActionMenuEntry[]>(() => {
    const { main, other } = loraGroups;
    const entries: ActionMenuEntry[] = [
      { kind: 'heading', label: 'Available LoRAs' },
      ...main.map((lora): ActionMenuEntry => ({ kind: 'item', id: lora.name, label: lora.name, onselect: () => addLora(lora.name) })),
    ];
    if (other.length > 0) {
      entries.push({
        kind: 'group',
        id: 'other-models',
        label: `Other models (${other.length})`,
        defaultOpen: main.length === 0,
        items: other.map((lora) => ({
          kind: 'item',
          id: lora.name,
          label: lora.name,
          hint: lora.family ? familyLabel(lora.family, lora.size_variant) : undefined,
          onselect: () => addLora(lora.name),
        })),
      });
    }
    entries.push({ kind: 'separator' }, { kind: 'item', id: 'discover-loras', label: 'Discover LoRAs…', icon: 'search', onselect: openDiscover });
    return entries;
  });

  /** Open a phone setting chip's popover, or close it when it is already open. */
  function toggleChip(key: ChipKey): void {
    if (key === 'preset') {
      activeChip = null;
      presetMenuOpen = true;
      return;
    }
    activeChip = activeChip === key ? null : key;
  }

  function addLora(name: string): void {
    const existing = loraChips.find((c) => c.name === name);
    if (existing) { loraPopoverOpen = false; return; }
    addLoraChip(loraChips, name, 1.0);
    loraPopoverOpen = false;
  }

  /** Add LoRA: with no LoRA file yet there is no list to choose from, so it opens Discover. */
  function toggleLoraMenu(): void {
    if (loraOptions.length === 0) openDiscover();
    else loraPopoverOpen = !loraPopoverOpen;
  }

  const removeLora = (name: string): void => removeLoraChip(loraChips, name);

  /** What turning on Use in LoRAs brought back: a chip at its weight, and the trigger of a file that is still downloading. The drawer stays open. */
  function useDiscoveredLora(use: LoraUse): void {
    addLoraChip(loraChips, use.name, use.weight);
    if (use.downloading) pendingTriggers.set(use.name, use.trigger);
  }

  /** A deleted file's chips go, on this model and on every other the draft remembers. */
  function dropDeletedLora(name: string): void {
    draft.removeLora(name, loraOptions);
    pendingTriggers.forget(name);
    if (context) context = { ...context, loras: context.loras.filter((lora) => lora.name !== name) };
  }

  /** Read the LoRA rows again: a typed trigger, a download that finished, a file that was deleted. */
  function refreshLoras(): void {
    getWorkspaceCoreContext()
      .then((fresh) => {
        if (!context) return;
        context = { ...context, loras: fresh.loras };
        workspaceMemory.set(context);
      })
      .catch(() => undefined);
  }

  // A LoRA whose download is done but whose file the page has not listed yet: read the list again, pausing longer each
  // time, until it appears. If it never does, its chip counts as ready so Generate is not held back for good (the
  // server refuses a missing file when Generate is pressed).
  const watchedListings = new Set<string>();
  let listingStopped = false;
  function watchListing(name: string, itemId: string): void {
    if (watchedListings.has(itemId)) return;
    watchedListings.add(itemId);
    const read = async () => {
      const fresh = await getWorkspaceCoreContext();
      if (context) {
        context = { ...context, loras: fresh.loras };
        workspaceMemory.set(context);
      }
      return fresh.loras;
    };
    void waitForListing(name, read, { stopped: () => listingStopped }).then((result) => {
      if (result === 'gave-up') downloadStore.markUnlisted(itemId);
    });
  }
  $effect(() => {
    if (!context) return;
    for (const chip of loraChips) {
      const status = chipDownload(chip.name, loraOptions, downloadStore.items, downloadStore.unlisted);
      if (status.state === 'downloading' && status.item?.state === 'done') watchListing(chip.name, status.item.id);
    }
  });

  const failedChipItems = $derived.by<ActionMenuEntry[]>(() => {
    const name = failedChip?.name;
    if (!name) return [];
    const status = chipDownload(name, loraOptions, downloadStore.items, downloadStore.unlisted);
    const item = status.state === 'failed' ? status.item : null;
    return [
      { kind: 'note', text: status.state === 'failed' ? status.reason : `${name} didn't download.`, tone: 'error' },
      { kind: 'item', id: 'retry-lora', label: 'Retry download', icon: 'reset', disabled: item === null, onselect: () => { if (item) void downloadStore.retryLora(item); } },
      { kind: 'item', id: 'remove-lora', label: 'Remove', icon: 'close', onselect: () => removeLora(name) },
    ];
  });

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
        // The core fetch leaves the history out, so it is the one field to keep.
        context = { ...fresh, history_assets: context.history_assets };
        workspaceMemory.set(context);
      })
      .catch(() => undefined);
  }

  /** A picker option's text: the model's label, plus its download state when its files are not all there. */
  function modelOptionText(model: ModelOption): string {
    if (model.downloaded !== false) return model.label;
    const status = downloadStatusText(downloadStore.activeModel(model));
    return `${model.label} (${status ? status.toLowerCase() : 'not downloaded'})`;
  }

  async function handleJobComplete(outputs: GalleryAsset[]): Promise<void> {
    react('cheerful');
    refreshModelStatus();
    // A design job's voices reach no Gallery; the app announces them, on this page and every other.
    if (jobStore.current && isVoiceDesignJob(jobStore.current)) return;
    await galleryStore.refresh();
    completionToast(outputs, projectPick.current);
  }

  // A new pick empties the stores' lists (`projectsStore.choose`); the strip and the recent list load again here, since the
  // Gallery page loads its own and no other page shows them. The first run only records the pick.
  let _shownPick = projectPick.current;
  $effect(() => {
    const pick = projectPick.current;
    if (pick === _shownPick) return;
    _shownPick = pick;
    untrack(() => void galleryStore.refresh());
  });

  function handleJobFailed(): void {
    react('sad');
    refreshModelStatus();
    addToast(failureToastText(jobStore.current?.message, jobStore.queuedJobs.length > 0), 'error');
  }

  async function handleJobLost(): Promise<void> {
    react('surprised');
    refreshModelStatus();
    addToast('Lost track of the job. Refreshed the gallery with any results.', 'info');
    await galleryStore.refresh();
  }

  function handleJobCancelled(): void {
    react('surprised');
    refreshModelStatus();
    addToast('Generation stopped', 'info');
  }

  function openHistoryViewer(asset: GalleryAsset, trigger: HTMLElement): void {
    const index = galleryStore.assets.findIndex((item) => item.id === asset.id);
    if (index < 0) return;
    lightboxMode = 'history';
    viewerAssetId = asset.id;
    completedOutputTrigger = trigger;
    lightboxOpen = true;
  }

  function openOutputViewer(asset: GalleryAsset, trigger: HTMLElement): void {
    openCompletedOutputViewer(jobOutputs.findIndex((output) => output.id === asset.id), trigger);
  }

  /**
   * Drop the uploads a prefill makes stale. The server prefers an uploaded file over a path, so a file held for a slot would
   * win over what the prefill chose, or stay when the prefill names nothing (a reused clip without a last frame or a track).
   * A Use as (marked by `use_as`) sets one slot and drops only that slot's upload, leaving the other slot, the track and the
   * references. Any other prefill replaces the images and the audio track of the tab it opens, and only that tab's uploads go.
   */
  function dropHeldUploads(params: PrefillParams): void {
    const tab = workspaceTab(draft.state.workflow);
    // The Audio tab holds the voice recording, which a Reuse or a Use as → Voice replaces; the other tabs' uploads stay.
    if (tab === 'audio') {
      if (replacesVoice(firstValues(params), draft.state.workflow as AudioWorkflow)) heldUploads.setVoice(null);
      return;
    }
    // The Extend kind holds only an audio track, which a prefill replaces; it leaves the other tabs' uploads alone.
    if (tab === 'extend') {
      if (params.use_as === undefined) heldUploads.setTrack('extend', null);
      return;
    }
    // The Edit kind holds only the image to edit and its paint; every prefill that opens it (a Reuse, a Retry, the Edit action)
    // replaces both, and no other prefill reaches them.
    if (tab === 'edit') {
      heldUploads.clearTab('edit');
      return;
    }
    if (params.use_as === 'start') heldUploads.setStart(tab, null);
    else if (params.use_as === 'last') heldUploads.setLastFrame(null);
    else if (params.use_as === 'audio_track') heldUploads.setTrack('video', null);
    else heldUploads.clearTab(tab);
  }

  /**
   * Apply workspace prefill params (workflow first, then model defaults, then the params on top). The image paths are checked
   * afterwards, unless `recheck` is off for a caller that checks them itself.
   */
  function applyPrefill(params: PrefillParams, options: { keepSettings?: boolean; recheck?: boolean } = {}): void {
    if (!context) return;
    if (params.workflow) draft.loadFromUrl({ workflow: params.workflow }, context);
    draft.hydrateFromContext(context, typeof params.model === 'string' ? params.model : null, options);
    // A prefill that is not a Use as sets every slot of its tab, an unnamed one emptied; the workflow follows from the images.
    // A Reuse or a Load settings (`reuse`) also takes the default of every setting it does not name.
    draft.loadFromUrl(params, context, { replaceSlots: !params.use_as, exact: params.reuse === 'true' });
    dropHeldUploads(params);
    // Recorded image paths may have moved since; flag those slots, as a page load does.
    if (options.recheck !== false) void recheckImages();
    // The prefill already re-hydrated for its tab; the tab-change effect must not redo it.
    _prevTab = workspaceTab(draft.state.workflow);
    // The source brings its own settings, so the form no longer starts from a preset. A Use as brings none (an image or a take
    // only): the settings stay, or were reset with the model by the hydrate above, and so does the preset they came from.
    if (params.use_as === undefined) draft.setAppliedPreset(null);
  }

  /** Refill the form from the settings a job was submitted with. */
  function loadJobSettings(settings: JobSettings, message: string): void {
    const step = context ? extendStepFor(context, first(settings, 'model') ?? draft.state.model) : undefined;
    const { params, patch } = jobSettingsPrefill(settings, draft.state.enhanceSettingsVideo?.partsByWorkflow, step);
    applyPrefill({ ...params, reuse: 'true' });
    draft.patch(patch);
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
    closeLightbox();
    addToast(`Loaded settings from ${asset.filename}`, 'success');
  }

  async function upscaleAsset(asset: GalleryAsset, factor: UpscaleFactor, method: UpscaleMethod): Promise<void> {
    // Like Generate, an upscale joins the queue while another job runs.
    const job = await startUpscale(asset, factor, method);
    if (job?.queue_position) react('nodding');
    if (job) closeLightbox();
  }

  // What the Use as menus show: each target's hint, or why it can't be used, from the model its tab opens on.
  const useAs = $derived(useAsState(context, draft.state));

  // Reference rows are added to the Image tab's list whichever tab is open.
  const useAsEditor = referenceEditor(() => images?.references?.max ?? 0);

  /** Add an image as the next reference on the Image tab, keeping the prompt; a model that can't take one says why. */
  function addReference(path: string): void {
    const state = useAs.reference;
    if (!context || !state) return;
    if (state.disabled) {
      if (state.hint) addToast(state.hint, 'info');
      return;
    }
    if (workspaceTab(draft.state.workflow) !== 'image') {
      draft.setTab('image');
      // References belong to New: the Image tab opens on it, also when it was left on Edit.
      draft.setKind('image', 'new', context);
      draft.onTabChange(context);
      _prevTab = workspaceTab(draft.state.workflow);
    }
    useAsEditor.append([{ path, file: null }]);
    addToast(`Added as image ${draft.state.referenceImagePaths.length}`, 'success');
  }

  // An image the Gallery sent as a reference waits until the draft and the stored uploads are back, then joins the list.
  $effect(() => {
    if (!authorityReady || !hasPendingReference()) return;
    const path = takeReference();
    if (path) untrack(() => addReference(path));
  });

  // Pick one, from the ready notice on any page, opens the Voices window on New voice. It waits until the draft is back.
  $effect(() => {
    if (!authorityReady || !voiceWindow.requested) return;
    voiceWindow.takeRequest();
    untrack(openNewVoice);
  });

  /** Open the Speech tab and the Voices window on New voice. */
  function openNewVoice(): void {
    if (!context) return;
    if (draft.state.workflow !== 'speech') {
      draft.setTab('audio');
      draft.setAudioKind('speech', context);
      _prevTab = workspaceTab(draft.state.workflow);
    }
    voiceWindow.show('new');
  }

  function useAsReference(asset: GalleryAsset, target: ReferenceTarget): void {
    if (!asset.file_path) return;
    if (target === 'reference') {
      addReference(asset.file_path);
      closeLightbox();
      return;
    }
    if (useAs[target]?.disabled) return;
    const params = referenceParams(asset, target);
    // The slot sits on a tab: within it the image just goes in place, and the opened tab shows it without a toast.
    if (workspaceTab(draft.state.workflow) === workspaceTab(params.workflow as Workflow)) {
      draft.loadFromUrl(params, context);
      dropHeldUploads(params);
      draft.saveDraft();
      void recheckImages();
    } else applyPrefill(params, { keepSettings: true });
    closeLightbox();
  }

  // The speech take being saved as a voice.
  let voiceTake = $state<GalleryAsset | null>(null);

  /**
   * Make an audio take the Video tab's audio track (from 0 s) or the Speech tab's voice. Both go through the same prefill as
   * a Reuse, marked by `use_as` (`audio_track`, `voice`) so only that one slot changes; see `audioTrackParams` and `voiceParams`.
   */
  function useAudio(asset: GalleryAsset, target: AudioUseTarget): void {
    if (!asset.file_path || useAs[target]?.disabled) return;
    // Saved as a voice, a take opens its dialog over the page (and the viewer) and nothing else changes.
    if (target === 'save_voice') {
      voiceTake = asset;
      return;
    }
    const params = target === 'audio_track' ? audioTrackParams(asset) : voiceParams(asset);
    // The track sits on the Video tab: within it the take just goes in place, and the opened tab shows it without a toast.
    if (target === 'audio_track' && workspaceTab(draft.state.workflow) === 'video') {
      draft.loadFromUrl(params, context);
      dropHeldUploads(params);
      draft.saveDraft();
    } else applyPrefill(params, { keepSettings: true });
    closeLightbox();
  }

  /**
   * Make a video the source of an extension; its recorded scene and prompt fill the boxes once the server has described it.
   * With a *point*, the handle starts there; a point too early to cut is refused (the surface that offered it said why).
   */
  function extendAsset(asset: GalleryAsset, point?: FramePoint): void {
    if (!asset.file_path || !canExtend(asset) || point?.tooShort) return;
    const params = extendParams(asset, point);
    if (draft.state.workflow === params.workflow) {
      // The new source brings its own track (or none), so the held one goes with the draft's.
      draft.loadFromUrl(params, context, { replaceSlots: true });
      dropHeldUploads(params);
      draft.setAppliedPreset(null);
    } else applyPrefill(params, { keepSettings: true });
    closeLightbox();
    addToast(`${asset.filename} is now the source video`, 'success');
    void extendInspection.inspect(context, { replaceText: true });
    void tick().then(() => flashArrival('source-video'));
  }

  /**
   * Make the paused frame of a video the Video tab's first or last frame: the frame is extracted as a PNG, then fills the slot
   * like any Use as. The other frame stays. A first frame says so when a last frame stays, and a frame set with no other frame
   * says where it came from. The viewer closes only if it is still showing that video when the extraction ends.
   */
  function useFrame(asset: GalleryAsset, point: FramePoint, target: FrameTarget): void {
    if (!asset.file_path || useAs[target]?.disabled) return;
    void runFrameGrab(asset, point, target, async (saved) => {
      const notice = frameSetNotice(target, point, draft.state, heldUploads);
      const params = frameParams(asset, saved.path, target);
      // The slot sits on the Video tab: within it the frame just goes in place, and the opened tab shows it without a toast.
      if (workspaceTab(draft.state.workflow) === workspaceTab(params.workflow as Workflow)) {
        draft.loadFromUrl(params, context);
        dropHeldUploads(params);
        draft.saveDraft();
        void recheckImages();
      } else applyPrefill(params, { keepSettings: true });
      if (lightboxOpen && viewerAssetId === asset.id) closeLightbox();
      if (notice) addToast(notice.message, notice.type);
      await tick();
      flashArrival(target === 'last' ? 'last-frame' : 'first-frame');
    });
  }

  /**
   * Edit on an image (the viewer's button and M, the ⋯ menus): open the Image tab's Edit kind with that image to edit and no
   * paint, on its recorded model and LoRAs where that model can edit. The prompt starts empty and any Enhanced text goes.
   */
  function editAsset(asset: GalleryAsset): void {
    if (!context || !canEdit(asset)) return;
    applyPrefill(editParams(asset), { keepSettings: true });
    closeLightbox();
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
      galleryStore.removeAssets([asset]);
      // A slot pointing at the deleted file would fail the next run: it empties, and the workflow follows from what is left.
      // A reference row stays, flagged missing, so the prompt's "image 2" keeps meaning the same row.
      if (emptySlotsHolding([asset.file_path], context)) void recheckImages();
      addToast('Deleted', 'success');
    } catch {
      addToast('Delete failed', 'error');
    } finally {
      deletingIds = new Set([...deletingIds].filter((id) => id !== asset.id));
    }
  }

  async function moveWorkspaceAsset(asset: GalleryAsset, project: string | null): Promise<void> {
    if (deletingIds.has(asset.id)) return;
    const moved = (await moveAndReconcile([asset], project, context))?.moved.find((move) => move.from === asset.id);
    // The stage and the viewer follow a file that stays in the list under its new ID; one that left is stepped past.
    if (moved && viewedId === asset.id) viewedId = moved.asset.id;
    if (moved && viewerAssetId === asset.id) viewerAssetId = moved.asset.id;
  }

  function openCompletedOutputViewer(index: number, trigger: HTMLElement): void {
    if (!jobOutputs[index]) return;
    lightboxMode = 'completed-output';
    viewerAssetId = jobOutputs[index].id;
    completedOutputTrigger = trigger;
    completedOutputJobId = jobStore.current?.job_id ?? null;
    lightboxOpen = true;
  }

  function closeLightbox(): void {
    if (!lightboxOpen) return;
    lightboxOpen = false;
    const trigger = completedOutputTrigger?.isConnected ? completedOutputTrigger : null;
    lightboxMode = 'history';
    viewerAssetId = null;
    completedOutputTrigger = null;
    completedOutputJobId = null;
    queueMicrotask(() => trigger?.focus());
  }

  function navigateLightbox(index: number): void {
    viewerAssetId = lightboxAssets[index]?.id ?? null;
  }

  // The viewer closes with nothing left to show. When its asset is deleted it moves to the neighbour itself.
  $effect(() => {
    if (!lightboxOpen) return;
    if (lightboxMode === 'history') {
      if (galleryStore.assets.length === 0) closeLightbox();
    } else if (jobOutputs.length === 0 || completedOutputJobId !== jobStore.current?.job_id) {
      closeLightbox();
    }
  });

  // Track the previous tab to detect user-initiated changes after context loads. A start image or first frame changes the
  // workflow within its tab and needs no re-hydration.
  let _prevTab: WorkspaceTab | null = null;

  $effect(() => {
    const currentTab = workspaceTab(draft.state.workflow);
    if (_prevTab !== null && context !== null && currentTab !== _prevTab) {
      draft.onTabChange(context);
    }
    _prevTab = currentTab;
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
    const { frameRule } = videoTimingFor(context, state.model);
    clipMemory.watch(clipMemoryRequest(state, { frameRule, lowMemory: caps.showLowMemory, upscale: caps.showVideoUpscale, steps: stepOptionsFor(context, state.model) !== null, review: caps.showVideoUpscaleReview }));
  });

  /**
   * Ask the server which image files are gone and flag those slots: the start image, the first and last frame and the
   * reference rows. The flags saved with the draft may be stale (a file can return or leave while the page was away), so the
   * server's answer replaces them. A failed check leaves them.
   */
  async function recheckImages(): Promise<void> {
    const paths = slotPaths(draft.state).filter(hasPath);
    if (paths.length === 0) return;
    try {
      draft.patch({ missingReferencePaths: await checkReferencePaths(paths) });
    } catch {
      // The flags stay as they were; Generate refuses a file that is not there.
    }
  }

  onMount(() => {
    const unsubscribeLifecycle = jobStore.subscribeLifecycle({
      onComplete: handleJobComplete,
      onFailed: handleJobFailed,
      onCancelled: handleJobCancelled,
      onLost: handleJobLost,
    });
    const stopDownloadSync = downloadStore.startSync();
    const stopDownloaded = downloadStore.onDownloaded(refreshModelStatus);
    // Read the uploads held in the browser back; Generate waits for them.
    void heldUploads.restore();
    const urlParams = parseUrlPrefill();
    // The Models page links here to open Discover LoRAs; the flag is not a setting and leaves the address at once.
    const wantsDiscover = urlParams.discover !== undefined;
    delete urlParams.discover;
    const hasUrlParams = Object.keys(urlParams).length > 0;
    if (wantsDiscover) {
      openDiscoverWhenReady = true;
      if (!hasUrlParams) router.replace('workspace');
    }

    // Load stored draft as the baseline state.
    draft.loadDraft();
    void presetsStore.load();
    void voicesStore.load();
    void voicesStore.loadDesign();
    extendInspection.reset();
    clipMemory.reset();

    let cancelled = false;
    let historyTimer: ReturnType<typeof setTimeout> | undefined;
    // The URL's prefill goes into the draft once; the server's context, arriving after the remembered one, must not redo it over edits.
    let prefillApplied = false;

    /** Take a context as the page's authority: the draft hydrated from it, with the URL's prefill on top the first time. */
    function adoptContext(ctx: WorkspaceContext): void {
      context = ctx;
      workflows = ctx.workflow_contract.values;
      if (ctx.history_assets.length > 0) {
        galleryStore.seed(ctx.history_assets);
      }
      if (hasUrlParams && !prefillApplied) {
        prefillApplied = true;
        // A link into the workspace loads a source's settings; a saved draft keeps its own unless the link names a model, and a Use as (the Gallery's Edit included) keeps them unless its model differs.
        // The server's context check follows right after, so the prefill does not ask for the paths too.
        applyPrefill(urlParams, { keepSettings: !urlParams.model || urlParams.use_as !== undefined, recheck: false });
        // A frame or an Extend the Gallery sent: the slot or section it filled flashes once the page has drawn it.
        const arrival = arrivalFor(urlParams);
        if (arrival) void tick().then(() => flashArrival(arrival));
        // The prefill is in the draft now; drop it from the address so a reload or Back keeps the user's edits.
        router.replace('workspace');
      } else {
        // A saved draft keeps its settings across visits and restarts.
        draft.hydrateFromContext(ctx, null, { keepSettings: true });
        // The tab-change $effect must not fire for the initial state.
        _prevTab = workspaceTab(draft.state.workflow);
      }
    }

    // Pages remember: the context a previous visit loaded answers the first frame; the server's fresh one replaces it.
    const remembered = workspaceMemory.get();
    if (remembered) adoptContext(remembered);

    // Load workspace authority first; gallery/history hydration follows after the core page can paint.
    getWorkspaceCoreContext()
      .then((ctx) => {
        if (cancelled) return;
        workspaceMemory.set(ctx);
        adoptContext(ctx);
        void recheckImages();
        jobStore.seedQueue(ctx.queued_jobs ?? []);
        void jobStore.reconnectActiveJob({ snapshot: ctx.active_job });
        // "Generate this one" on the Prompts page: queue that prompt with these settings once the form exists.
        const run = takePromptRun();
        if (run) {
          void (async () => {
            // A prompt of a file is New's: Edit has no prompt file, so an Image tab left on Edit moves to New (Edit's image and paint stay).
            draft.setKind('image', 'new', ctx);
            await tick();
            // Generate waits for the held uploads, so the run does too.
            await heldUploads.restore();
            // A saved source has not been described yet; the run waits for the answer rather than being refused for want of it.
            if (ctx && draft.state.workflow === 'extend' && draft.state.extendSourcePath) await extendInspection.inspect(ctx);
            if (!cancelled) await submitForm(run);
          })();
        }

        historyTimer = setTimeout(() => {
          if (cancelled) return;
          void galleryStore.refresh().finally(() => {
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
      if (e.defaultPrevented || hasOpenModal()) return;
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
      } else if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && canStepStage && previewView !== 'empty' && acceptsPageShortcut(e)) {
        // ← → step the preview through history; → goes back in time, like the strip reads. With focus in the player they step or seek it instead, which marks the key handled before it reaches here.
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
      stopDownloadSync();
      stopDownloaded();
      listingStopped = true;
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

  /** Name the picked project as the job's destination; All files sends no `project`. */
  function withProject(formData: FormData): FormData {
    const pick = projectPick.current;
    if (pick !== null) formData.set('project', pick);
    return formData;
  }

  /** Collect the fields Generate submits; with `run`, that one prompt-file prompt replaces the form's prompt. */
  function buildFormData(form: HTMLFormElement, run: PendingPromptRun | null): FormData {
    // The Audio tab sends the fields of its open kind, built from the draft: the same ones Copy CLI reads.
    if (audio?.active) return withProject(audioFormData(audio, draft.state));
    // The Edit kind sends the fields of the `edit` workflow, built from the draft: the form carries New's.
    if (isEdit) {
      return withProject(editFormData(draft.state, {
        sourceFile: heldUploads.editSource,
        maskFile: heldUploads.editMask,
        prompt: submittedPair(draft.state).prompt,
        lora: formatLoraString(loraChips),
        quantize: supportsQuantize,
      }));
    }
    const formData = new FormData(form);
    // Send canonical workflow directly; backend accepts and normalises it.
    formData.set('mode', isImageMode ? 'image' : 'video');
    formData.set('workflow', draft.state.workflow);
    // The images come from the images view, not the form's inputs: an upload wins over a path.
    const sent = images?.sent ?? { image: null, strength: null, end: null };
    if (typeof sent.image === 'string') formData.set('image_path', sent.image);
    else if (sent.image !== null) formData.set('image_file', sent.image);
    if (sent.strength !== null) formData.set('image_strength', String(sent.strength));
    if (typeof sent.end === 'string') formData.set('end_image_path', sent.end);
    else if (sent.end !== null) formData.set('end_image_file', sent.end);
    // The audio track is the clip's sound (an upload wins over a path), so no Audio switch is sent beside it.
    const track = audioTrack?.sent ?? null;
    if (track !== null) {
      if (track.file !== null) formData.set('audio_track_file', track.file);
      else if (track.path !== null) formData.set('audio_track_path', track.path);
      formData.set('audio_track_start', String(track.start));
    }
    // References go as one repeated field, a file or a path per row in prompt order, so the rows keep their numbers.
    for (const row of images?.references?.rows ?? []) {
      if (row.file) formData.append('reference', row.file);
      else if (row.path) formData.append('reference', row.path);
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
    return withProject(formData);
  }

  /** The settings a job is shown with: the form's fields, and never the speaker's tick (a job's settings carry none, as the server's record). */
  function submittedSettings(formData: FormData): JobSettings {
    const settings = formSettings(formData);
    delete settings.voice_consent;
    return settings;
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
    // A prompt file means nothing on the Audio tab or in Edit, whatever the other tabs last chose.
    return pageCaps.promptFileMode && (!draft.state.promptFilePath || draft.state.promptFileOptionIds.length === 0);
  }

  /** Copy the CLI command that would run what Generate submits now. */
  async function copyCliCommand(): Promise<void> {
    // Report empty or invalid fields the way Generate does, rather than copying a command for them.
    if (!formEl || !authorityReady || !formEl.reportValidity()) return;
    await settleExtendSource();
    if (extendBlocked()) return;
    const refusal = imageBlock ?? trackBlock;
    if (refusal) {
      addToast(refusal, 'error');
      return;
    }
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
      const upscaleSharpenDefault = context ? settingDefaultsFor(context, draft.state).upscaleSharpen : undefined;
      // `draft.extendSource` outlives the Extend tab, so only that workflow's own source names the folder.
      const outputDir = cliOutputDir({
        root: context?.output_dir,
        projectPath: projectsStore.currentPath,
        extendFolder: draft.state.workflow === 'extend' ? draft.extendSource?.output_folder : null,
      });
      built = cliCommand(settings, { outputDir, promptFileOptionCount, extendSource, upscaleSharpenDefault });
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
    return (jobId) => sendJobControl(jobId, action);
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
    // A LoRA chip whose file is downloading (or failed) holds every run back, the keyboard's included.
    if (block.blocked) return;
    // The Audio tab's one reason (a missing text, a recording that is not ready, a tick not given) holds every run back, the keyboard's included.
    if (audioReason !== null || textWriting.reason !== null) return;
    await settleExtendSource();
    if (extendBlocked()) return;
    const refusal = imageBlock ?? trackBlock;
    if (refusal) {
      addToast(refusal, 'error');
      return;
    }
    submitError = null;
    busy = true;
    draft.saveDraft();

    try {
      const formData = buildFormData(formEl, run);
      // A job that downloads a model asks first; Cancel queues nothing.
      if (!(await confirmDownloads(await planGenerate(formData)))) return;
      const jobCtx = await submitGenerate(formData);
      // The response omits the settings; the review line needs the upscale factor.
      jobStore.jobSubmitted({ ...jobCtx, settings: submittedSettings(formData) });
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

  // First run: nothing is downloaded and the user has not skipped, so offer the image models once the lane is known.
  const offerFirstRun = $derived(
    shouldOfferFirstRun(context, {
      downloadsSeeded: downloadStore.seeded,
      activeImageDownload: downloadStore.items.some((item) => item.kind === 'model' && imageModels.some((model) => model.id === item.name) && (item.state === 'queued' || item.state === 'running')),
      dismissed: firstRun.dismissed,
    }),
  );
  // The mascot lives in the logo and is called out to one spot here while there is work for him.
  const mascotAt = $derived(pickMascotSpot({
    view: previewView,
    running: jobStore.isRunning,
    reviewing,
    reacting: reaction !== null,
    lookingForHistory,
    welcoming: offerFirstRun && context !== null,
  }));
  // Measure the mascot before the preview swaps spots, so it hops from where it really was.
  $effect.pre(() => {
    void previewView;
    void mascotAt;
    untrack(rememberMascotSpots);
  });
  $effect.pre(() => {
    mascotHome.publish({ away: mascotAt !== null, mood: mascotMood, tool: mascotTool });
  });
  $effect(() => () => mascotHome.clear());
  // The painter is a layer of this page; it goes with it.
  $effect(() => () => painterLayer.hide());
  // The image the Edit kind shows follows the draft's: after a reload, a Reuse, an Edit from the Gallery, a Reset or a delete.
  $effect(() => {
    if (!isEdit) return;
    void [draft.state.editSourcePath, draft.state.editSourceUpload, heldUploads.editSource];
    untrack(followDraft);
  });

  /** Download the model chosen in the first-run dialog; once it is queued, make it the default and select it. */
  async function downloadFirstRunModel(model: ModelOption): Promise<DownloadRequestResult> {
    const result = await downloadStore.request('model', model.id, model.label);
    if (result !== 'queued' && result !== 'present') return result;
    firstRun.dismiss();
    try {
      await updateConfig({ 'ui.default_models.image': model.id });
    } catch (e) {
      addToast(errorText(e, 'Could not save the default model.'), 'error');
    }
    if (isImageMode) switchModel(model.id);
    return result;
  }

  async function skipFirstRunDialog(): Promise<void> {
    firstRun.dismiss();
    try {
      await skipFirstRun();
    } catch (e) {
      addToast(errorText(e, 'Could not remember that you skipped.'), 'error');
    }
  }
</script>

<!-- Generate's label; on a phone with a project picked, the project's name is a second line, since Generate is where a wrong save costs. -->
{#snippet generateText(label: string)}
  {#if projectPick.current !== null}
    <span class="generate-stack"><span>{label}</span><span class="generate-project" title={projectPick.current}>{projectPick.current}</span></span>
  {:else}
    {label}
  {/if}
{/snippet}

{#snippet assetMeta(asset: GalleryAsset)}
  {#if asset.media_type === 'audio'}
    {@const cut = cutNote(asset, context?.audio_models)}
    <!-- A take's caption: its words, then the model, the length and the seed; a take that hit its model's cap says so. -->
    <p class="latest-meta latest-meta-audio" data-testid="audio-caption">
      <span class="latest-words truncate" title={asset.prompt}>{asset.kind == null ? asset.filename : (asset.prompt || asset.filename)}</span>
      {#if asset.kind != null && asset.model}<b>{audioModelLabel(asset, context?.audio_models)}</b>{/if}
      {#if asset.duration != null}<b>{formatClock(asset.duration)}</b>{/if}
      {#if asset.seed != null}<b>seed {asset.seed}</b>{/if}
      {#if cut}
        <Tooltip text={cut.tip} testId="audio-cut-tip" class="cut-note"><Badge variant="warning" class="whitespace-nowrap">{cut.text}</Badge></Tooltip>
      {/if}
    </p>
  {:else}
  <p class="latest-meta">
    <span class="truncate" title={asset.prompt}>{asset.prompt || asset.filename}</span>
    {#if asset.model}<b>{asset.model}</b>{/if}
    {#if asset.width && asset.height}<b>{asset.width}×{asset.height}</b>{/if}
    {#if asset.seed != null}<b class="latest-seed">seed {asset.seed}</b>{/if}
  </p>
  {/if}
{/snippet}

<form
  bind:this={formEl}
  class="flex min-h-0 flex-1 flex-col"
  onsubmit={handleSubmit}
  oninput={markTyping}
  oninvalidcapture={revealInvalidField}
>
  <!-- Hidden fields -->
  <input type="hidden" name="mode" value={isAudio ? 'audio' : isImageMode ? 'image' : 'video'}>
  <input type="hidden" name="workflow" value={draft.state.workflow}>
  <input type="hidden" name="lora" value={formatLoraString(loraChips)}>

  {#snippet toolbar(stacked: boolean)}
  <!-- Toolbar bar: model, quantize, loras. Stacked, it heads the phone's All sheet. -->
  <div class="ui-pane-toolbar z-10 shrink-0" class:toolbar-stacked={stacked}>
    <div class="flex items-center px-3 py-2 gap-3 {stacked ? 'flex-col items-stretch' : 'flex-wrap'}">

      {#if kindChoice}
        <!-- The Kind first: Music, Speech or Sound on Audio; New or Extend on Video. On a tablet the words Kind and Model give way to Video's row. -->
        <div class="flex items-center gap-2 min-w-0 {stacked ? 'justify-between' : ''}">
          <span class="ui-label {!isAudio && tall && !stacked ? 'sr-only' : ''}" id="ws-kind-label">Kind</span>
          <Segmented label="Kind" testId="kind-control" value={kindChoice.value} disabled={!authorityReady} options={kindChoice.kinds.map((kind) => ({ value: kind, label: kindChoice.info[kind].label, icon: kindChoice.info[kind].icon, title: kindChoice.info[kind].title }))} onchange={kindChoice.pick} />
        </div>
      {/if}
      {#if context === null || pageCaps.showPreset}
      <!-- Settings presets, listed by the workflow's contract (Audio and Edit have none); a disabled placeholder until the contract has loaded -->
      <PresetMenu {context} {stacked} applied={activePreset} changed={presetChanged} />
      {/if}

      <!-- Model selector -->
      <div class="flex items-center gap-2 min-w-0 {stacked ? 'flex-nowrap' : 'flex-wrap'}">
        <label class="ui-label {stacked || (tall && (showVideoKind || isImageMode)) ? 'sr-only' : ''}" for="ws-model">Model</label>
        <Select
          id="ws-model"
          name="model"
          testId="model-shell"
          variant="shell"
          class={stacked ? 'min-w-0 flex-1' : 'w-48'}
          value={draft.state.model}
          options={modelOptions}
          disabled={!authorityReady || currentModels.length === 0}
          onchange={switchModel}
        >
          {#snippet trigger()}
            <span class="flex min-w-0 items-center gap-2 truncate pointer-events-none">
              <svg class="text-primary-main h-4 w-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path>
              </svg>
              <span class="truncate">{authorityReady ? (selectedModel?.label ?? (draft.state.model || 'No models')) : 'Loading models…'}</span>
            </span>
          {/snippet}
        </Select>

        <!-- The terms of the selected model: a sibling of the Select (a button cannot sit inside its trigger), gone for a model with none. -->
        {#if authorityReady && selectedModel?.licences?.length}
          <LicenceMark licences={selectedModel.licences} testId="model-licence" />
        {/if}

        <!-- Quantize selector (image workflows only, when model supports it) -->
        {#if supportsQuantize}
          <Select
            name="quantize"
            testId="quantize-shell"
            variant="shell"
            ariaLabel="Quantize"
            class={stacked ? 'w-auto shrink-0' : 'w-40'}
            value={draft.state.quantize !== null ? String(draft.state.quantize) : ''}
            options={[{ value: '', label: 'Quant: None' }, ...quantizeOptions.map((opt) => ({ value: String(opt), label: `Quant: q${opt}` }))]}
            disabled={!authorityReady}
            onchange={(v) => draft.update('quantize', v ? Number(v) : null)}
          />
        {/if}

        {#if authorityReady && selectedModel}
          <ModelStatusBadges
            downloaded={selectedModel.downloaded}
            downloadStatus={downloadStatusText(downloadStore.activeModel(selectedModel))}
            memoryFit={selectedModel.memory_fit}
            quantize={supportsQuantize ? draft.state.quantize : null}
            lowMemory={isAudio || !visibleControls.has('low_memory') || draft.state.lowMemory}
            size="control"
            tooltipPlacement="bottom"
          />
        {/if}
      </div>

      {#if !isAudio}
      <div class="bg-border-subtle shrink-0 {stacked ? 'h-px w-full' : 'h-6 w-px'}"></div>

      <!-- LoRA chips area. On a tablet the word LoRAs gives way to the row, and the area keeps the width of its content so it drops to a second line instead of being squeezed out of view. -->
      <div class="flex items-center gap-2 flex-1 {tall && !stacked ? 'min-w-fit' : 'min-w-0'}">
        <span class="ui-label shrink-0 {tall && !stacked ? 'sr-only' : ''}">LoRAs</span>
        {#if authorityReady}
          <div class="flex flex-wrap items-center gap-2 flex-1 min-w-0">
            <!-- Chips -->
            {#each loraChips as chip (chip.name)}
              {@const file = chipDownload(chip.name, loraOptions, downloadStore.items, downloadStore.unlisted)}
              <!-- Neutral chip: coral is reserved for $snippet syntax in prompts. Its file may be downloading (a line along its foot) or have failed (red). -->
              <div
                class="ui-chip lora-chip"
                class:ui-chip-dl={file.state === 'downloading'}
                class:ui-chip-err={file.state === 'failed'}
                data-download={file.state === 'ready' ? undefined : file.state}
                style:--p={file.state === 'downloading' ? `${file.percent ?? 0}%` : undefined}
              >
                {#if file.state === 'downloading'}<Icon name="download" size={11} />{/if}
                {#if file.state === 'failed'}
                  <button type="button" class="lora-failed" aria-haspopup="menu" aria-expanded={failedChip?.name === chip.name} title={file.reason} onclick={(e) => { failedChip = { name: chip.name, anchor: e.currentTarget }; }}>
                    <Icon name="alert" size={11} /><span class="max-w-28 truncate">{loraLabel(chip.name)}</span>
                  </button>
                {:else}
                  <span class="max-w-28 truncate" title={file.state === 'downloading' ? `Downloading ${file.percent ?? 0} %. Generate waits for it.` : chip.name}>{loraLabel(chip.name)}</span>
                {/if}
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
              disabled={!discover && loraOptions.length === 0}
              aria-haspopup={loraOptions.length === 0 ? 'dialog' : 'menu'}
              aria-expanded={loraOptions.length === 0 ? discoverOpen : loraPopoverOpen}
              onclick={toggleLoraMenu}
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
            <ActionMenu
              open={failedChip !== null}
              anchor={failedChip?.anchor ?? null}
              align="start"
              label="Download failed"
              items={failedChipItems}
              onclose={() => { failedChip = null; }}
            />
          </div>
        {:else}
          <p class="ui-help">Loading models…</p>
        {/if}
      </div>
      {/if}
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
          <span class="text-meta text-text-muted">{reviewing ? 'Review' : jobStore.isRunning ? 'Generating…' : hasCompletedOutputs || galleryStore.assets.length ? 'Latest output' : 'Ready'}</span>
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
        {:else if stageDrag?.axis === 'x' && galleryStore.assets.length > 0}
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
            <div class="mb-3 flex min-h-12 shrink-0 items-center justify-between gap-3">
              <div class="flex items-center gap-2">
                {#if mascotAt === 'outputs'}
                  <MascotSpot mood={mascotMood} tool={mascotTool} size={48} />
                {/if}
                <h3 class="ui-area-label">Completed outputs</h3>
              </div>
              <span class="font-mono text-meta text-text-muted">{jobOutputs.length}</span>
            </div>
            {#if jobOutputs.length === 1}
              {@const output = jobOutputs[0]}
              <div class="latest-region flex min-h-0 flex-1 flex-col px-16" class:audio={output.media_type === 'audio'}>
                <AssetTile
                  asset={output}
                  density="stage"
                  eager
                  deleting={deletingIds.has(output.id)}
                  class="completed-output min-h-0 flex-1"
                  onpreview={openOutputViewer}
                  onreuse={reuseAsset}
                  onreference={useAsReference}
                  onuseaudio={useAudio}
                  onextend={extendAsset}
                  onuseframe={useFrame}
                  onedit={editAsset}
                  onupscale={upscaleAsset}
                  {useAs}
                  ondelete={deleteWorkspaceAsset}
                  onmove={moveWorkspaceAsset}
                />
                {@render assetMeta(output)}
              </div>
            {:else if jobOutputs[0].media_type === 'audio'}
              <!-- Takes are players, not pictures: one panel under another, scrolling from the top. -->
              <div class="completed-output-scroll min-h-0 flex-1 overflow-y-auto" data-testid="audio-outputs">
                <div class="audio-output-list">
                  {#each jobOutputs as output (output.id)}
                    <div class="audio-output latest-region">
                      <AssetTile
                        asset={output}
                        density="stage"
                        deleting={deletingIds.has(output.id)}
                        class="completed-output"
                        onreuse={reuseAsset}
                        onuseaudio={useAudio}
                        {useAs}
                        ondelete={deleteWorkspaceAsset}
                        onmove={moveWorkspaceAsset}
                      />
                      {@render assetMeta(output)}
                    </div>
                  {/each}
                </div>
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
                      onuseaudio={useAudio}
                      onextend={extendAsset}
                      onuseframe={useFrame}
                      onedit={editAsset}
                      onupscale={upscaleAsset}
                      {useAs}
                      ondelete={deleteWorkspaceAsset}
                      onmove={moveWorkspaceAsset}
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
            </div>
            {#if compact}
              <div class="job-bar-dock">
                <!-- Blob rides the bar, peeking over its top edge. -->
                <div class="job-bar-mascot">
                  {#if mascotAt === 'job'}
                    <MascotSpot mood={mascotMood} tool={mascotTool} progress={mascotProgress} size={56} />
                  {/if}
                </div>
                <JobBar
                  job={jobStore.current!}
                  queuedCount={jobStore.queuedJobs.length}
                  onpause={sendControl('pause')}
                  onresume={sendControl('resume')}
                  ondetails={() => { jobSheetOpen = true; }}
                />
              </div>
            {:else}
            <div class="job-view-card">
              <!-- Blob hosts the job: it stands on the card, clear of the forming image. His place stays when he goes home. -->
              <div class="job-view-mascot">
                {#if mascotAt === 'job'}
                  <MascotSpot mood={mascotMood} tool={mascotTool} progress={mascotProgress} size={112} />
                {/if}
              </div>
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
          <button type="button" class="stage-nav next ui-btn ui-btn-icon" aria-label="Older output" disabled={viewedIndex >= galleryStore.assets.length - 1 && !galleryStore.hasMore} onclick={() => stepView(1)}>
            <Icon name="chevright" size={16} />
          </button>
          <!-- Side padding keeps wide media clear of the newer and older arrows. -->
          <div class="latest-region flex h-full w-full flex-col px-20 py-4" class:fit={latest.width && latest.height} class:audio={latest.media_type === 'audio'} style:--ar={latest.width && latest.height ? latest.width / latest.height : null} style:--natural-w={latest.width ? `${latest.width}px` : null}>
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
              onuseaudio={useAudio}
              onextend={extendAsset}
              onuseframe={useFrame}
              onedit={editAsset}
              onupscale={upscaleAsset}
              {useAs}
              ondelete={deleteWorkspaceAsset}
              onmove={moveWorkspaceAsset}
            />
            {@render assetMeta(latest)}
          </div>
          {#if latestLoading}
            <div class="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-3 text-center p-4" data-testid="latest-loading">
              <p class="text-content text-text-secondary">Loading latest output…</p>
            </div>
          {/if}
        {:else}
          <div class="flex flex-col items-center justify-center gap-3 text-center p-4">
            {#if mascotAt === 'empty'}
              <MascotSpot mood={mascotMood} tool={mascotTool} size={128} />
            {/if}
            {#if lookingForHistory}
              <p class="text-content text-text-secondary">Looking for your latest work…</p>
            {:else}
              <div>
                <p class="ui-empty-title">{projectPick.current === null ? 'No generated assets yet' : `Nothing in ${projectPick.current} yet`}</p>
                <p class="mt-1 text-ui text-text-muted">Write a prompt and press Generate to get started</p>
              </div>
            {/if}
          </div>
        {/if}
        {#if reviewing && jobStore.current}
          {@const job = jobStore.current}
          <!-- Scoped to the preview area: settings, history, navigation and menus stay usable. The job card stays under the backdrop. -->
          <div class="review-overlay absolute inset-0 z-20 flex items-center justify-center bg-scrim p-3 sm:p-6" data-testid="review-overlay">
            <div class="ui-overlay relative flex h-full w-full max-w-4xl min-h-0 min-w-0 flex-col">
              <ReviewStage {job} onapprove={sendControl('approve')} ondiscard={sendControl('discard')} />
              {#if mascotAt === 'review'}
                <!-- Blob waits beside the decision. -->
                <MascotSpot mood={mascotMood} tool={mascotTool} size={compact ? 48 : 72} class="review-mascot" />
              {/if}
            </div>
          </div>
        {/if}
      </div>

      {#if !compact}
        <GalleryStrip
          assets={galleryStore.assets}
          loading={galleryStore.loading}
          totalCount={galleryStore.totalCount}
          hasMore={galleryStore.hasMore}
          loadingMore={galleryStore.loadingMore}
          onloadmore={() => void galleryStore.loadMore()}
          {deletingIds}
          onpreview={openHistoryViewer}
          onreuse={reuseAsset}
          onreference={useAsReference}
          onuseaudio={useAudio}
          onextend={extendAsset}
          onedit={editAsset}
          onupscale={upscaleAsset}
          {useAs}
          ondelete={deleteWorkspaceAsset}
          onmove={moveWorkspaceAsset}
        />
      {/if}
    </section>
  {/snippet}

  {#if compact}
    <!-- The phone: the stage is the screen; under it the settings strip, then the composer with Generate. Compose and
         Settings stay inside the form (the composer and the All sheet host them) so every field still submits. -->
    <div class="compact-workspace" data-testid="compact-workspace">
      {@render stage()}
      <!-- Always in the frame, even before the context answers: a loading state never changes the stage's height. -->
      <div class="compact-controls" use:dockHeight={'--toast-dock'}>
        {#if activeChip && context && images && audioTrack && audio}
          <button type="button" class="pop-scrim" aria-label="Close setting" tabindex="-1" onclick={() => { activeChip = null; }}></button>
          <SettingPopover
            key={activeChip}
            {context}
            {busy}
            {images}
            {audioTrack}
            {audio}
            models={currentModels}
            quantizeOptions={supportsQuantize ? quantizeOptions : null}
            lastSeed={galleryStore.allRecent[0]?.seed ?? null}
            onModelChange={switchModel}
            onclose={() => { activeChip = null; }}
            ondiscover={() => { activeChip = null; openDiscover(); }}
          />
        {/if}
        {#if context}<PresetMenu {context} applied={activePreset} changed={presetChanged} button={false} bind:open={presetMenuOpen} />{/if}
        {#if isEdit}<ImageToEditCard />{/if}
        <SettingsStrip {chips} tab={workspaceTab(draft.state.workflow)} activeKey={activeChip} {changedCount} pending={controlsPending} onopen={toggleChip} onall={() => { activeChip = null; sheetOpen = true; }} />
        <div class="composer" class:expanded={composerOpen} data-testid="composer">
          {#if block.failures.length > 0}
            {@const failedItem = chipDownload(block.failures[0].name, loraOptions, downloadStore.items, downloadStore.unlisted)}
            <p class="ui-alert ui-alert-error composer-error" role="alert" data-testid="lora-failure">
              <Icon name="alert" size={14} /><span class="min-w-0 flex-1 break-words">{failureText(block.failures[0])}</span>
              {#if failedItem.state === 'failed'}<button type="button" class="ui-btn ui-btn-sm" onclick={() => void downloadStore.retryLora(failedItem.item)}>Retry</button>{/if}
            </p>
          {/if}
          {#if generateError}<p class="ui-alert ui-alert-error composer-error" role="alert"><Icon name="alert" size={14} /><span class="min-w-0 break-words">{generateError}</span></p>{/if}
          <!-- The one reason (the Audio tab's, or a run writing the text), above the row, so a disabled Generate says why. -->
          {#if composerReason}
            <p id="ws-generate-reason-compact" class="ui-alert ui-alert-warning composer-error" data-testid={audio?.active ? 'audio-reason' : 'generate-reason'}>
              <span class="min-w-0 flex-1 break-words">{composerReason}</span>
              {#if audio?.reasonFromVoice}<button type="button" class="ui-btn ui-btn-sm" data-testid="audio-reason-voice" onclick={() => { activeChip = 'voice'; }}>Voice</button>{/if}
            </p>
          {/if}
          <div class="composer-row" hidden={composerOpen}>
            <button type="button" class="composer-summary" aria-label="Edit the prompt" disabled={controlsPending} onclick={() => void openComposer()}>
              <span class="ui-area-label composer-label">{composerSummary.label}</span>
              <span class="composer-line" class:placeholder={composerSummary.empty}>{composerSummary.text}</span>
            </button>
            <span class="composer-split">
              <button
                id="ws-submit-compact"
                type="submit"
                class="composer-generate ui-btn ui-btn-primary ui-holdable"
                class:ui-btn-wait={block.waiting !== null}
                style:--p={block.waiting ? `${block.waiting.percent ?? 0}%` : undefined}
                disabled={compactSubmitDisabled}
                aria-describedby={composerReason ? 'ws-generate-reason-compact' : undefined}
                title={block.waiting ? waitFull : 'Hold for more'}
                aria-label={block.waiting ? waitFull : undefined}
                use:hold={{ onhold: () => { generateMenuOpen = true; } }}
              >
                {#if block.waiting}
                  <WaitingLabel waiting={block.waiting} compact />
                {:else}
                  <Icon name="bolt" size={16} />{@render generateText(busy ? 'Submitting…' : jobStore.jobsActive ? 'Queue' : 'Generate')}
                  {#if jobStore.jobsActive && jobStore.queuedJobs.length > 0}<span class="composer-q">+{jobStore.queuedJobs.length}</span>{/if}
                {/if}
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
            {#if context && images && audio}<ComposePane {context} {busy} {images} {audio} jobsActive={jobStore.jobsActive} />{/if}
            <div class="composer-foot">
              <button type="button" class="ui-btn" onclick={() => { composerOpen = false; }}>Done</button>
              <button
                type="submit"
                class="ui-btn ui-btn-primary ui-btn-main"
                class:ui-btn-wait={block.waiting !== null}
                style:--p={block.waiting ? `${block.waiting.percent ?? 0}%` : undefined}
                disabled={compactSubmitDisabled}
                aria-describedby={composerReason ? 'ws-generate-reason-compact' : undefined}
                aria-label={block.waiting ? waitFull : undefined}
                title={block.waiting ? waitFull : undefined}
              >{#if block.waiting}<WaitingLabel waiting={block.waiting} compact />{:else}<Icon name="bolt" size={16} />{@render generateText(jobStore.jobsActive ? 'Queue' : 'Generate')}{/if}</button>
            </div>
          </div>
        </div>
      </div>
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
          lastSeed={galleryStore.allRecent[0]?.seed ?? null}
          {images}
          {audioTrack}
          {audio}
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
        lastSeed={galleryStore.allRecent[0]?.seed ?? null}
        {images}
        {audioTrack}
        {audio}
        oncopycli={copyCliCommand}
      />
      {@render stage()}
    </main>
  {/if}
</form>

{#if offerFirstRun && context}
  <FirstRunDialog
    models={imageModels}
    recommended={context.first_run?.recommended_model ?? null}
    ondownload={downloadFirstRunModel}
    onskip={() => void skipFirstRunDialog()}
    onclose={() => firstRun.dismiss()}
  />
{/if}

{#if audio?.active && audio.kind === 'speech'}
  <VoicesWindow voice={audio.voice} model={audio.model} vocabulary={context?.audio_settings} />
{/if}

{#if context}
  <DiscoverDrawer
    bind:open={discoverOpen}
    {context}
    target={discover}
    chips={loraChips}
    onuse={useDiscoveredLora}
    onstop={removeLora}
    ondeleted={dropDeletedLora}
    onloraschange={refreshLoras}
  />
{/if}

<!-- The painter of Edit: a layer of its own over the page, closed by Done or Cancel. -->
{#if painterLayer.open}
  <Painter onclose={() => painterLayer.hide()} />
{/if}

<!-- One viewer for history and completed outputs -->
<AssetViewer
  assets={lightboxAssets}
  origin={completedOutputTrigger}
  currentId={viewerAssetId}
  open={lightboxOpen}
  setLabel={lightboxMode === 'completed-output' ? 'This run' : 'Gallery'}
  {deletingIds}
  onclose={closeLightbox}
  onnavigate={navigateLightbox}
  onnearend={() => { if (lightboxMode === 'history') void galleryStore.loadMore(); }}
  onreuse={reuseAsset}
  onreference={useAsReference}
  onuseaudio={useAudio}
  onextend={extendAsset}
  onuseframe={useFrame}
  onedit={editAsset}
  {useAs}
  audioModels={context?.audio_models}
  audioVocabulary={context?.audio_settings}
  onupscale={upscaleAsset}
  ondelete={deleteWorkspaceAsset}
  onmove={moveWorkspaceAsset}
/>

<SaveVoiceDialog asset={voiceTake} model={useAs.voiceModel ?? null} vocabulary={context?.audio_settings} onclose={() => { voiceTake = null; }} />

<style>
  /* Chips in the toolbar share the height of the selects beside them. */
  .lora-chip { height: var(--spacing-control); padding: 0 8px; border-radius: var(--radius-sm); color: var(--color-text-primary); }
  .lora-chip.ui-chip-err, .lora-chip.ui-chip-err .lora-weight, .lora-chip.ui-chip-err .lora-remove { color: var(--color-error); }
  .lora-failed { display: inline-flex; align-items: center; gap: 4px; min-width: 0; color: inherit; }
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
  /* Phone: toasts sit above the strip, the composer and any message over Generate, whatever height they have (`dockHeight` measures them). */
  :global(:root:has(.compact-workspace)) { --toast-bottom: calc(var(--toast-dock, 120px) + 12px); }
  :global(:root:has(.compact-workspace .job-bar)) { --toast-bottom: calc(var(--toast-dock, 120px) + 104px); }
  /* The phone composition. */
  .compact-workspace { display: flex; flex: 1; min-height: 0; flex-direction: column; }
  .compact-workspace > :global(.workspace-preview) { flex: 1; min-height: 0; }
  .compact-workspace :global(.latest-region) { padding: 8px 12px; }
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
  .job-view-compact .job-view-preview { padding-bottom: 140px; }
  .job-bar-dock { position: absolute; left: 12px; right: 12px; bottom: 12px; display: flex; flex-direction: column; }
  /* The mascot's place on the bar; his feet overlap its top edge. */
  .job-bar-mascot { position: relative; z-index: 1; height: 56px; margin: 0 0 -14px 8px; pointer-events: none; }
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
  .generate-stack { display: flex; flex-direction: column; min-width: 64px; max-width: 96px; line-height: 16px; text-align: left; }
  .generate-project { overflow: hidden; font-family: var(--font-sans); font-size: var(--text-meta); font-weight: 700; line-height: 13px; text-overflow: ellipsis; white-space: nowrap; }
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
  /* Blob peeks over the card's top edge; its foot overlaps the card by a little. */
  .job-view-mascot { position: relative; z-index: 1; flex: none; align-self: center; height: 112px; margin: 4px 0 -26px; pointer-events: none; }
  .review-overlay :global(.review-mascot) { position: absolute; left: 12px; bottom: 12px; pointer-events: none; }
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
  /* An audio take: its panel and caption sit together in the middle of the stage. */
  .latest-region.audio { justify-content: center; }
  .latest-meta.latest-meta-audio { flex-wrap: wrap; row-gap: 4px; }
  .latest-meta.latest-meta-audio span.latest-words { flex: 0 1 auto; max-width: 420px; }
  .latest-meta.latest-meta-audio :global(.cut-note) { flex: none; max-width: none; }
  /* A phone: the words take a line of their own, and the facts and the cut badge sit under them. */
  .compact-workspace :global(.latest-meta.latest-meta-audio span.latest-words) { flex: 1 1 100%; max-width: none; text-align: center; }
  .audio-output-list { display: flex; flex-direction: column; gap: 20px; width: 100%; margin: auto 0; padding: 4px 0; }
  .audio-output { display: flex; flex-direction: column; align-items: center; }
  .latest-meta { display: flex; flex: none; align-items: center; justify-content: center; gap: 14px; min-width: 0; padding-top: 8px; font-size: var(--text-ui); color: var(--color-text-muted); }
  .latest-meta span { max-width: 520px; color: var(--color-text-secondary); }
  .latest-meta b { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); font-weight: 500; color: var(--color-text-secondary); }
  .completed-output-scroll { display: flex; flex-direction: column; }
  .completed-output-grid { display: grid; justify-content: center; margin-block: auto; }
</style>
