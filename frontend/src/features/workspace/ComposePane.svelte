<script lang="ts">
  import { tick } from 'svelte';
  import { currentEditFacts } from './edit/currentEdit';
  import { editPlaceholder } from './edit/editView';
  import { draft } from '$lib/state/draft.svelte';
  import { anyEnhancedUsed, canSubmitText, effectiveFormat, enhancedPartUsed, enhancedStaleReason, isJsonCaption, runParts, workflowMode } from '$lib/state/promptEnhance';
  import { isOriginalLegacyPrompt, normalizePromptText } from '$lib/state/extend';
  import { autogrow } from '$lib/actions/autogrow';
  import { hasOpenModal, isCommandKey } from '$lib/keyboard';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { Icon } from '$lib/components/atoms';
  import { HelpLine, PromptFileField, PromptTools, Segmented, TextBoxHeader } from '$lib/components/molecules';
  import { clearRewritesPatch, editEnhancedPatch, rewriteStepper, scratchRewritePatch, stepRewritePatch } from '$lib/state/rewriteHistory';
  import { ClearUndo } from '$lib/state/clearUndo.svelte';
  import { parseLoraString } from '$lib/state/loras';
  import { pendingTriggers } from '$lib/state/loraTriggers.svelte';
  import type { DraftState, PromptSource, RecentPrompt, WorkspaceContext } from '$lib/types';
  import PromptEnhancer from './PromptEnhancer.svelte';
  import RecentPromptsMenu from './RecentPromptsMenu.svelte';
  import SavePromptMenu from './SavePromptMenu.svelte';
  import { promptToSave } from './savePrompt';
  import { workspaceCapabilities } from './capabilities';
  import type { ImagesView } from './imagesView';
  import AudioCompose from './AudioCompose.svelte';
  import type { AudioView } from './audioView';
  import TriggerMenu from './TriggerMenu.svelte';
  import { insertWordPatch, triggerMenu, triggerModes } from './triggerMenu';

  interface Props {
    context: WorkspaceContext;
    busy: boolean;
    /** The images view of the open tab, computed once by the Workspace; the enhancer counts the last frame and references it sends. */
    images: ImagesView;
    /** The Audio tab's view of the open kind, computed once by the Workspace; its boxes replace the prompt's on that tab. */
    audio: AudioView;
    /** A job is running or queued. */
    jobsActive?: boolean;
    /** Fold the sidebar into its strip; no button is shown when absent. */
    oncollapse?: () => void;
  }

  let { context, busy, images, audio, jobsActive = false, oncollapse }: Props = $props();

  /** Remember the loaded prompts file; an unchanged file is not written again. */
  function setPromptFileLoaded(loaded: DraftState['promptFileLoaded']): void {
    const current = draft.state.promptFileLoaded;
    if (current?.path === loaded?.path && current?.count === loaded?.count) return;
    draft.update('promptFileLoaded', loaded);
  }

  const s = $derived(draft.state);
  const caps = $derived(workspaceCapabilities(context, s));
  // Edit has no prompt file: its prompt is typed.
  const promptFileMode = $derived(caps.promptFileMode);
  const jsonMode = $derived(caps.showPromptInline && caps.supportsJsonPrompt && s.jsonPromptEnabled);
  /** A box with one view only: JSON on, or Enhance each image on. It has a "Prompt" heading and no tabs or marks. */
  const singleView = $derived(caps.showPromptInline && caps.showEnhance && (jsonMode || s.enhanceAuto));
  const showTabs = $derived(caps.showPromptInline && caps.showEnhance && !singleView);
  const enhancedUsed = $derived(anyEnhancedUsed(s));
  const staleReason = $derived(enhancedStaleReason(s));
  const stale = $derived(staleReason !== null);

  let activeTab = $state<'mine' | 'enhanced'>('mine');
  // Image and Video share this choice, but Enhanced text belongs to the mode it was written in: a change between them opens on your own text.
  // A run in progress carries on, so the pane is not remounted.
  let lastMode = workflowMode(draft.state.workflow);
  $effect(() => {
    const mode = workflowMode(s.workflow);
    if (mode === lastMode) return;
    lastMode = mode;
    activeTab = 'mine';
    cleared.forget();
  });
  let enhancer = $state<ReturnType<typeof PromptEnhancer> | null>(null);
  let enhancing = $state(false);
  let streamingText = $state('');
  let streamingScene = $state('');
  /** An error or the last result's note: one help line under the box, cleared on the next edit or run. */
  let errorText = $state<string | null>(null);
  let noteText = $state<string | null>(null);
  let statusText = $state<string | null>(null);
  let promptEl = $state<HTMLTextAreaElement | null>(null);
  let recentsButton = $state<HTMLButtonElement | null>(null);
  let recentsOpen = $state(false);
  /** The prompt box when the menu was opened with Alt+↑ from it; Escape returns focus there instead of to the button. */
  let recentsReturn = $state<HTMLElement | null>(null);

  /** Video workflows write a Scene beside the Prompt; the tabs then switch the pair. Images keep one prompt box. */
  const pairMode = $derived(caps.showScene && !promptFileMode && caps.showPromptInline);
  const showingEnhanced = $derived(showTabs && activeTab === 'enhanced');
  const enhancedValue = $derived(enhancing ? streamingText : s.enhancedPrompt);
  const enhancedSceneValue = $derived(enhancing ? streamingScene : s.enhancedScene);
  const saveSelection = $derived(
    promptToSave(s, { enhancedView: showingEnhanced, enhancing, jsonMode, pairMode, showNegative: caps.showNegativePrompt })
  );
  /** Where the shown version sits among the last five, or "– of N" after Scratch; only in the Enhanced view and while nothing is enhancing. */
  const stepper = $derived(rewriteStepper(s, enhancing));
  // A video needs a scene or an action; Enhanced text in either box stands in for both.
  const textMissing = $derived(!enhancedUsed && !canSubmitText(s));

  const SCENE_PLACEHOLDER = 'Who and where, look, style, camera. Kept for extensions.';
  const IMAGE_SCENE_PLACEHOLDER = 'What the image shows: who and where, look, style, camera.';
  const LEGACY_SCENE_PLACEHOLDER = 'Move the scene description here from the prompt (who, where, look). Then change the prompt to what happens next.';
  const extending = $derived(s.workflow === 'extend');
  const fromImage = $derived(s.workflow === 'img2vid');
  // A track is the clip's sound, so the box asks for who speaks or sings instead of what is heard.
  const heard = $derived(caps.audioTrackInEffect ? 'who speaks or sings' : 'sound, dialogue');
  const promptPlaceholder = $derived(
    extending ? 'What happens next…'
      : fromImage ? `What happens from the image: motion, camera moves, ${heard}.`
      : caps.showScene ? `What happens: action, camera moves, ${heard}.`
      : caps.isEdit ? editPlaceholder(currentEditFacts(context))
      : 'Describe the scene...'
  );
  // A source that recorded one prompt and no scene: the old prompt is pre-filled and the scene is left for the user to write.
  const legacySource = $derived(extending && draft.extendSource?.legacy === true);
  const scenePlaceholder = $derived(legacySource ? LEGACY_SCENE_PLACEHOLDER : fromImage ? IMAGE_SCENE_PLACEHOLDER : SCENE_PLACEHOLDER);
  // Shown from the prefill until the Prompt is edited; it never blocks Generate.
  const originalPromptShown = $derived(extending && !promptFileMode && isOriginalLegacyPrompt(s, draft.extendSource));
  // The parts whose Enhanced text Generate sends: in your view such a part's text is grey, and one the rewrite skipped stays bright.
  const sceneUsed = $derived(enhancedPartUsed(s, 'scene'));
  const actionUsed = $derived(enhancedPartUsed(s, 'action'));
  const rewritesScene = $derived(runParts(s) === 'both');
  const enhancedScenePlaceholder = $derived(enhancing && rewritesScene ? statusText || 'Enhancing…' : 'Not rewritten. Generate sends your scene.');
  const enhancedPromptPlaceholder = $derived(
    enhancing ? statusText || 'Enhancing…' : pairMode ? 'Not rewritten. Generate sends your action.' : 'No enhanced text. Generate sends your Prompt.'
  );
  // An extension's clip link: put back what the clip recorded. A legacy clip recorded one prompt and no scene.
  const clipScene = $derived(extending && pairMode ? (draft.extendSource?.scene ?? '').trim() : '');
  const showClipScene = $derived(clipScene !== '' && s.scene.trim() !== clipScene);
  const clipPrompt = $derived(legacySource && pairMode ? (draft.extendSource?.prompt ?? '') : '');
  const showClipPrompt = $derived(clipPrompt.trim() !== '' && normalizePromptText(s.prompt) !== normalizePromptText(clipPrompt));
  const DEFAULT_SCENE_HELP = 'Used for prompts in the file that have no scene: of their own.';

  // The trigger words of the LoRAs in use, marked where the text Generate sends already holds them; null when there is nothing to offer.
  const triggers = $derived(triggerMenu({ state: s, modes: triggerModes(caps, s), chips: parseLoraString(s.loraString, context.loras), installed: context.loras, pending: pendingTriggers.all }));
  /** The box each draft field is shown in, and the tab that holds it. */
  const FIELD_BOXES = { scene: ['ws-scene', 'mine'], enhancedScene: ['ws-enhanced-scene', 'enhanced'], prompt: ['ws-prompt', 'mine'], enhancedPrompt: ['ws-enhanced-prompt', 'enhanced'] } as const;

  /** Put a trigger word at the start of the text Generate sends, and show that text (the Enhanced tab when that is what is sent). */
  function insertTrigger(word: string): void {
    const patch = insertWordPatch(s, triggerModes(caps, s), word);
    const keys = Object.keys(patch) as (keyof typeof FIELD_BOXES)[];
    // Enhanced text changes through the one path every writer of it takes: a version edited in place, its record dropped.
    draft.patch(keys.some((key) => key.startsWith('enhanced')) ? editEnhancedPatch(s, { prompt: patch.enhancedPrompt ?? s.enhancedPrompt, scene: patch.enhancedScene ?? s.enhancedScene }) : patch);
    clearMessages();
    const [id, tab] = FIELD_BOXES[keys[0]];
    // A finger would raise the keyboard over the result: the tab still changes, the focus stays.
    if (!layout.coarse) void focusField(id, tab);
    else if (showTabs) activeTab = tab;
  }

  /** The help line says only the last result: an edit or a run clears it, and so goes the Undo of a Clear. */
  function clearMessages(): void {
    errorText = null;
    noteText = null;
    cleared.forget();
  }

  /** The Clear of the prompt box, and the tab it showed. */
  const cleared = new ClearUndo<'mine' | 'enhanced'>();
  /** The id of the Undo on the help line, which a finger is given the focus on. */
  const UNDO_ID = 'ws-prompt-clear-undo';
  /** The words the Clear names: the one box, or a video's scene and action together. */
  const clearWhat = $derived(jsonMode ? 'JSON prompt' : pairMode ? 'scene and action' : 'prompt');
  /** The box holds something to clear: your text, an Enhanced text or its versions; with JSON on, the JSON prompt. */
  const holdsText = $derived(
    jsonMode
      ? s.jsonPrompt !== ''
      : s.prompt !== '' || (pairMode && s.scene !== '') || s.enhancedPrompt !== '' || s.enhancedScene !== '' || s.enhancedHistory.length > 0
  );

  /** Empty the box: your text, the Enhanced text and every version (the JSON prompt, with JSON on). Undo puts them back. */
  function clearPrompt(): void {
    const patch: Partial<DraftState> = jsonMode ? { jsonPrompt: '' } : { prompt: '', ...(pairMode ? { scene: '' } : {}), ...clearRewritesPatch() };
    const view = activeTab;
    clearMessages();
    cleared.clear(patch, view);
    activeTab = 'mine';
    // A finger would raise the keyboard over the result: it gets the Undo, a mouse the empty text.
    void tick().then(() => document.getElementById(layout.coarse ? UNDO_ID : jsonMode ? 'ws-json-prompt' : 'ws-prompt')?.focus());
  }

  /** Put back what Clear took, with the tab it showed. */
  function undoClear(): void {
    const undone = cleared.undo();
    if (undone === null) return;
    activeTab = undone.view;
    if (!layout.coarse) void focusField(jsonMode ? 'ws-json-prompt' : undone.view === 'enhanced' ? 'ws-enhanced-prompt' : 'ws-prompt', undone.view);
  }

  // A video accepts a scene in place of a prompt, so its message names both. Declarative, so it clears itself once either has text.
  $effect(() => {
    promptEl?.setCustomValidity(pairMode && textMissing ? 'Write an action or a scene.' : '');
  });

  const SOURCE_LABELS: Record<PromptSource, string> = { inline: 'Inline', file: 'Prompt file' };

  function onPromptInput(value: string): void {
    clearMessages();
    draft.update('prompt', value);
  }

  function onSceneInput(value: string): void {
    clearMessages();
    draft.update('scene', value);
  }

  function onEnhancedInput(value: string): void {
    clearMessages();
    draft.patch(editEnhancedPatch(s, { prompt: value, scene: s.enhancedScene }));
  }

  function onEnhancedSceneInput(value: string): void {
    clearMessages();
    draft.patch(editEnhancedPatch(s, { prompt: s.enhancedPrompt, scene: value }));
  }

  // An empty prompt can fail validation while the Enhanced tab hides it; bring it forward so the message shows.
  function showPromptForValidation(event: Event): void {
    if (!showingEnhanced) return;
    activeTab = 'mine';
    const field = event.currentTarget as HTMLTextAreaElement;
    void tick().then(() => field.reportValidity());
  }

  /** Empty the Enhanced text, so Generate sends your text; the header shows your tab. */
  function scratch(): void {
    clearMessages();
    draft.patch(scratchRewritePatch(s));
  }

  /** Show version *index* (1-based); Generate then sends it. */
  function showRewrite(index: number): void {
    clearMessages();
    draft.patch(stepRewritePatch(s, index));
  }

  /** Bring a recent prompt back: its text into the boxes this workflow has, and no Enhanced text or versions left over from the previous one. */
  function pickRecent(entry: RecentPrompt): void {
    const patch: Partial<DraftState> = clearRewritesPatch();
    // Where there is no JSON prompt (an edit), a JSON caption is the text like any other.
    if (entry.format === 'json' && caps.supportsJsonPrompt) {
      patch.jsonPromptEnabled = true;
      patch.jsonPrompt = entry.prompt;
    } else {
      patch.jsonPromptEnabled = false;
      patch.prompt = entry.prompt;
      if (caps.showScene) patch.scene = entry.scene;
    }
    if (caps.showNegativePrompt) patch.negativePrompt = entry.negative_prompt;
    draft.patch(patch);
    activeTab = 'mine';
    clearMessages();
    recentsOpen = false;
    // A finger would raise the keyboard over the result.
    if (!layout.coarse) void focusField(entry.format === 'json' && caps.supportsJsonPrompt ? 'ws-json-prompt' : 'ws-prompt', 'mine');
  }

  // Alt+↑ in an empty prompt (and, for a video, an empty scene) opens the recent prompts; with text in it the browser keeps its caret movement.
  function onPromptKeydown(event: KeyboardEvent): void {
    if (event.key !== 'ArrowUp' || !event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    if ((event.currentTarget as HTMLTextAreaElement).value !== '' || (pairMode && s.scene !== '')) return;
    event.preventDefault();
    recentsReturn = event.currentTarget as HTMLElement;
    recentsOpen = true;
  }

  /** Bring a field forward on the tab that shows it, then focus it. */
  async function focusField(id: string, tab: 'mine' | 'enhanced'): Promise<void> {
    if (showTabs) activeTab = tab;
    await tick();
    document.getElementById(id)?.focus();
  }

  /** Replace the Scene or Prompt with the text the clip recorded. */
  function restoreClip(part: 'scene' | 'prompt'): void {
    const recorded = draft.extendSource?.[part];
    if (!recorded) return;
    clearMessages();
    draft.update(part, recorded);
    void focusField(`ws-${part}`, 'mine');
  }

  // ⌘E / Ctrl+E enhances the prompt, even while typing in it.
  $effect(() => {
    if (!enhancer) return;
    const target = enhancer;
    function handleKeydown(event: KeyboardEvent): void {
      if (event.defaultPrevented || !isCommandKey(event) || event.altKey || event.shiftKey || event.key.toLowerCase() !== 'e') return;
      if (hasOpenModal()) return;
      event.preventDefault();
      if (!enhancing) void target.runEnhance();
    }
    document.addEventListener('keydown', handleKeydown);
    return () => document.removeEventListener('keydown', handleKeydown);
  });
</script>

{#snippet header()}
  {#if singleView}
    <TextBoxHeader heading="Prompt" bare={pairMode} mineId="ws-tab-prompt" enhancedId="ws-tab-enhanced" mineControls="ws-prompt" enhancedControls="ws-enhanced-prompt" />
  {:else if showTabs}
    <TextBoxHeader
      bind:view={activeTab}
      label="Prompt"
      tablistLabel={pairMode ? 'Text version' : 'Prompt version'}
      mineId="ws-tab-prompt"
      enhancedId="ws-tab-enhanced"
      mineControls={pairMode ? 'ws-text-panel' : 'ws-prompt'}
      enhancedControls={pairMode ? 'ws-text-panel' : 'ws-enhanced-prompt'}
      bare={pairMode}
      used={enhancedUsed}
      {stale}
      running={enhancing}
      {stepper}
      onstep={showRewrite}
      onscratch={scratch}
      onstop={() => enhancer?.stop()}
    />
  {/if}
{/snippet}

{#snippet partLabel(part: 'scene' | 'action')}
  {@const name = part === 'scene' ? 'Scene' : 'Action'}
  {@const mine = part === 'scene' ? 'ws-scene' : 'ws-prompt'}
  <div class="label-row">
    <label class="ui-label" for={mine} hidden={showingEnhanced}>{name}</label>
    {#if showTabs}<label class="ui-label" for={`ws-enhanced-${part === 'scene' ? 'scene' : 'prompt'}`} hidden={!showingEnhanced}>{name}</label>{/if}
    {#if part === 'scene' ? showClipScene : showClipPrompt}
      <span class="label-extras">
        {#if part === 'scene'}
          <button type="button" class="prompt-link" title="Replace your scene with the one recorded in the clip" onclick={() => restoreClip('scene')}>Use the clip's scene</button>
        {:else}
          <button type="button" class="prompt-link" title="Replace your action with the clip's recorded prompt" onclick={() => restoreClip('prompt')}>Use the clip's prompt</button>
        {/if}
      </span>
    {/if}
  </div>
{/snippet}

{#snippet promptFields()}
  {#if jsonMode}
    <label for="ws-json-prompt" class="sr-only">Prompt (JSON)</label>
    <textarea
      id="ws-json-prompt"
      name="json_prompt"
      class="prompt-text font-mono"
      placeholder={'{"high_level_description": "..."}'}
      value={s.jsonPrompt}
      use:autogrow={s.jsonPrompt}
      oninput={(e) => draft.update('jsonPrompt', e.currentTarget.value)}
    ></textarea>
  {:else}
    {#if !pairMode}<label for="ws-prompt" class="sr-only">Prompt</label>{/if}
    <textarea
      id="ws-prompt"
      name="prompt"
      class="prompt-text"
      bind:this={promptEl}
      role={showTabs && !pairMode ? 'tabpanel' : undefined}
      aria-labelledby={showTabs && !pairMode ? 'ws-tab-prompt' : undefined}
      placeholder={promptPlaceholder}
      data-unsent={actionUsed && !showingEnhanced ? '' : undefined}
      aria-describedby={originalPromptShown ? 'ws-prompt-original' : undefined}
      required={textMissing}
      hidden={showingEnhanced}
      oninvalid={showPromptForValidation}
      onkeydown={onPromptKeydown}
      value={s.prompt}
      use:autogrow={[s.prompt, showingEnhanced]}
      oninput={(e) => onPromptInput(e.currentTarget.value)}
    ></textarea>
    {#if showTabs}
      <textarea
        id="ws-enhanced-prompt"
        class="prompt-text"
        class:font-mono={!enhancing && isJsonCaption(enhancedValue)}
        role={pairMode ? undefined : 'tabpanel'}
        aria-labelledby={pairMode ? undefined : 'ws-tab-enhanced'}
        hidden={!showingEnhanced}
        placeholder={enhancedPromptPlaceholder}
        readonly={enhancing}
        aria-busy={enhancing}
        value={enhancedValue}
        use:autogrow={[enhancedValue, showingEnhanced]}
        oninput={(e) => onEnhancedInput(e.currentTarget.value)}
      ></textarea>
    {/if}
  {/if}
{/snippet}

{#snippet toolsRow()}
  <PromptTools onclear={holdsText && !enhancing ? clearPrompt : undefined} {clearWhat}>
    {#if caps.showEnhance && caps.enhancer && !jsonMode}
      <PromptEnhancer
        bind:this={enhancer}
        bind:enhancing
        bind:streamingText
        bind:streamingScene
        bind:errorText
        bind:noteText
        bind:statusText
        contract={caps.enhancer}
        variant="inline"
        {busy}
        {jobsActive}
        {stale}
        {staleReason}
        endImage={images.sent.end !== null}
        references={images.references?.rows.length ?? 0}
        track={caps.audioTrackInEffect}
        jsonCapable={caps.supportsJsonPrompt}
        maxWords={caps.enhanceMaxWords}
        onstart={() => { activeTab = 'enhanced'; clearMessages(); }}
        onfail={() => { activeTab = 'mine'; }}
      />
    {/if}
    {#if caps.supportsJsonPrompt}
      <button
        type="button"
        id="ws-json-prompt-toggle"
        class="ui-btn ui-btn-sm ui-btn-icon"
        aria-pressed={s.jsonPromptEnabled}
        disabled={busy}
        aria-label="JSON caption"
        title="JSON caption: replaces the normal prompt"
        onclick={() => draft.update('jsonPromptEnabled', !s.jsonPromptEnabled)}
      ><Icon name="braces" size={14} /></button>
    {/if}
    {#snippet end()}
      {#if triggers}<TriggerMenu menu={triggers} onpick={insertTrigger} />{/if}
      <button
        bind:this={recentsButton}
        type="button"
        id="ws-recent-prompts"
        class="ui-btn ui-btn-sm ui-btn-icon"
        aria-label="Recent prompts"
        aria-haspopup="dialog"
        aria-expanded={recentsOpen}
        title="Recent prompts (Alt+↑ in an empty prompt)"
        onclick={() => { recentsReturn = null; recentsOpen = !recentsOpen; }}
      ><Icon name="history" size={14} /></button>
      <RecentPromptsMenu open={recentsOpen} anchor={recentsButton} returnFocus={recentsReturn ?? recentsButton} mode={caps.isImageMode ? 'image' : 'video'} jsonCapable={caps.supportsJsonPrompt} onpick={pickRecent} onclose={() => { recentsOpen = false; }} />
      {#if !jsonMode}
        <SavePromptMenu selection={saveSelection} workspaceFile={s.promptFilePath} enhancedView={showingEnhanced} />
      {/if}
    {/snippet}
  </PromptTools>
{/snippet}

<section class="compose-pane" class:audio-compose={caps.isAudio} aria-labelledby="ws-compose-title">
  <div class="compose-head">
    <div class="compose-title">
      {#if oncollapse}
        <button type="button" class="collapse-toggle" aria-label="Collapse sidebar" title="Collapse sidebar" aria-controls="ws-controls-sidebar" aria-expanded="true" onclick={oncollapse}>
          <Icon name="collapse" size={14} />
        </button>
      {/if}
      <h2 id="ws-compose-title" class="ui-area-label">Compose</h2>
    </div>
    {#if caps.showPromptSource}
      <Segmented
        size="sm"
        label="Prompt source"
        value={s.promptSource}
        options={context.prompt_sources.map((source) => ({ value: source, label: SOURCE_LABELS[source] }))}
        onchange={(source: PromptSource) => draft.update('promptSource', source)}
      />
      <input type="hidden" name="prompt_source" value={s.promptSource}>
    {/if}
  </div>

  {#if caps.isAudio}
    <AudioCompose view={audio} enhancer={context.prompt_enhancer ?? null} vocabulary={context.audio_settings} {busy} {jobsActive} />
  {/if}

  {#if !promptFileMode && caps.showPromptInline}
    {#if pairMode}
      {@render header()}
      <div
        id="ws-text-panel"
        class="text-pair"
        role={showTabs ? 'tabpanel' : undefined}
        aria-labelledby={showTabs ? (showingEnhanced ? 'ws-tab-enhanced' : 'ws-tab-prompt') : undefined}
      >
        <div class="field">
          {@render partLabel('scene')}
          <div class="ui-prompt-box" hidden={showingEnhanced}>
            <textarea
              id="ws-scene"
              name="scene"
              class="prompt-text scene-text"
              class:scene-legacy={legacySource && s.scene === ''}
              data-unsent={sceneUsed && !showingEnhanced ? '' : undefined}
              placeholder={scenePlaceholder}
              value={s.scene}
              use:autogrow={[s.scene, showingEnhanced]}
              oninput={(e) => onSceneInput(e.currentTarget.value)}
            ></textarea>
          </div>
          {#if showTabs}
            <div class="ui-prompt-box" hidden={!showingEnhanced}>
              <textarea
                id="ws-enhanced-scene"
                class="prompt-text scene-text"
                placeholder={enhancedScenePlaceholder}
                readonly={enhancing}
                aria-busy={enhancing}
                value={enhancedSceneValue}
                use:autogrow={[enhancedSceneValue, showingEnhanced]}
                oninput={(e) => onEnhancedSceneInput(e.currentTarget.value)}
              ></textarea>
            </div>
          {/if}
        </div>
        <div class="field">
          {@render partLabel('action')}
          <div class="ui-prompt-box">
            {@render promptFields()}
            {@render toolsRow()}
          </div>
        </div>
      </div>
    {:else}
      <div class="ui-prompt-box">
        {@render header()}
        {@render promptFields()}
        {@render toolsRow()}
      </div>
    {/if}

    <!-- The region stays in the page so assistive technology announces the warning when it appears. -->
    <div aria-live="polite">
      {#if originalPromptShown}
        <p id="ws-prompt-original" class="ui-help ui-help-warning original-warning" data-testid="original-prompt-warning">
          <Icon name="alert" size={12} />Still the original prompt. Change it to what happens next.
        </p>
      {/if}
    </div>
    <HelpLine error={errorText} note={noteText} undo={cleared.offered ? { what: clearWhat, id: UNDO_ID, onundo: undoClear } : null} />
  {/if}

  {#if promptFileMode && caps.showScene && caps.showPromptFileControls}
    <div class="field">
      <label class="ui-label mb-1" for="ws-scene" title={DEFAULT_SCENE_HELP}>Default scene</label>
      <div class="ui-prompt-box">
        <textarea
          id="ws-scene"
          name="scene"
          class="prompt-text scene-text"
          class:scene-legacy={legacySource && s.scene === ''}
          placeholder={scenePlaceholder}
          aria-describedby="ws-scene-help"
          value={s.scene}
          use:autogrow={[s.scene, showingEnhanced]}
          oninput={(e) => onSceneInput(e.currentTarget.value)}
        ></textarea>
      </div>
      <p id="ws-scene-help" class="sr-only">{DEFAULT_SCENE_HELP}</p>
    </div>
  {/if}

  {#if promptFileMode && context.prompt_file && caps.showPromptFileControls}
    <PromptFileField
      contract={context.prompt_file}
      promptSource={s.promptSource}
      path={s.promptFilePath}
      selectedOptionIds={s.promptFileOptionIds}
      workflowMode={caps.isImageMode ? 'image' : 'video'}
      captionRun={Boolean(caps.showEnhanceAuto && caps.enhancer) && effectiveFormat(s, caps.supportsJsonPrompt) === 'json'}
      negativePromptSupported={caps.supportsNegativePrompt}
      disabled={busy}
      onPathChange={(path) => draft.update('promptFilePath', path)}
      onOptionChange={(optionIds) => draft.update('promptFileOptionIds', optionIds)}
      onOptionsLoaded={setPromptFileLoaded}
    >
      {#snippet tools()}
        {#if caps.showEnhanceAuto && caps.enhancer}
          <PromptEnhancer contract={caps.enhancer} variant="file" {busy} {jobsActive} track={caps.audioTrackInEffect} jsonCapable={caps.supportsJsonPrompt} maxWords={caps.enhanceMaxWords} />
        {/if}
      {/snippet}
    </PromptFileField>
  {/if}

  {#if !promptFileMode && caps.showNegativePrompt}
    <div id="ws-negative-shell">
      <label class="ui-label mb-1" for="ws-negative-prompt">Negative prompt</label>
      <textarea
        id="ws-negative-prompt"
        name="negative_prompt"
        rows="1"
        class="ui-field negative-text"
        placeholder="What to exclude..."
        value={s.negativePrompt}
        use:autogrow={s.negativePrompt}
        oninput={(e) => draft.update('negativePrompt', e.currentTarget.value)}
      ></textarea>
    </div>
  {/if}
</section>

<style>
  .compose-pane { display: flex; flex: none; flex-direction: column; gap: 8px; max-height: 44%; overflow-y: auto; padding: 12px; border-bottom: 1px solid var(--color-border-subtle); }
  /* The Audio tab's Settings are short, so its boxes and status line get more of the column before Compose scrolls. */
  @media (min-width: 640px) { .compose-pane.audio-compose { max-height: 60%; } }
  .compose-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .compose-title { display: flex; align-items: center; gap: 6px; }
  .collapse-toggle { display: grid; place-items: center; width: 24px; height: 24px; margin-left: -4px; border-radius: 4px; color: var(--color-text-muted); }
  @media (hover: hover) { .collapse-toggle:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); } }
  /* Below the phone breakpoint the sidebar stacks above the preview and always shows. */
  @media (max-width: 639px) {
    .collapse-toggle { display: none; }
  }
  .text-pair { display: flex; flex-direction: column; gap: 8px; }
  .field label[hidden] { display: none; }
  /* The label sits left; the clip link is pushed right and wraps rather than being cut. */
  .label-row { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; min-height: 22px; margin-bottom: 4px; }
  .label-extras { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 4px 6px; margin-left: auto; }
  .prompt-text { display: block; width: 100%; min-height: 88px; max-height: 180px; overflow-y: auto; resize: none; padding: 9px 10px; border: 0; background: transparent; font-size: var(--text-content); line-height: 1.5; color: var(--color-text-primary); }
  .prompt-text[hidden] { display: none; }
  /* The focus ring belongs to the prompt box, not the textarea. */
  .prompt-text:focus, .prompt-text:focus-visible { outline: none; }
  .prompt-text::placeholder { color: var(--color-zinc-600); }
  /* Your text steps back to secondary while the Enhanced text is what Generate sends. */
  .prompt-text[data-unsent] { color: var(--color-text-secondary); }
  /* The Scene box is the smaller of the two: two lines, growing to four, then it scrolls. */
  .scene-text { min-height: 58px; max-height: 98px; }
  /* The long legacy placeholder takes three lines while the box is empty, so it is never cut off. */
  .scene-text.scene-legacy { min-height: 78px; }
  .original-warning { display: flex; align-items: flex-start; gap: 4px; }
  .original-warning :global(svg) { margin-top: 2px; }
  .prompt-link { font-size: var(--text-meta); font-weight: 600; color: var(--color-text-muted); }
  @media (hover: hover) { .prompt-link:hover { color: var(--color-text-primary); } }
  .prompt-link:disabled { opacity: 0.5; }
  .negative-text { min-height: 34px; max-height: 96px; resize: none; }
  /* Touch: 16px text, below which iOS zooms into a field as it focuses. */
  @media (pointer: coarse) {
    .prompt-text, .negative-text { font-size: 16px; }
  }
</style>
