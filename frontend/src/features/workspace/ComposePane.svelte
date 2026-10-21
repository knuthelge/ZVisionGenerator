<script lang="ts">
  import { tick } from 'svelte';
  import { draft } from '$lib/state/draft.svelte';
  import { anyEnhancedUsed, canSubmitText, effectiveFormat, enhancedPartUsed, isEnhancedStale, isJsonCaption, runParts } from '$lib/state/promptEnhance';
  import { isOriginalLegacyPrompt, normalizePromptText } from '$lib/state/extend';
  import { autogrow } from '$lib/actions/autogrow';
  import { hasOpenModal, isCommandKey } from '$lib/keyboard';
  import { Icon } from '$lib/components/atoms';
  import { PromptFileField, Segmented } from '$lib/components/molecules';
  import type { DraftState, PromptSource, WorkspaceContext } from '$lib/types';
  import PromptEnhancer from './PromptEnhancer.svelte';
  import SavePromptMenu from './SavePromptMenu.svelte';
  import { promptToSave } from './savePrompt';
  import { workspaceCapabilities } from './capabilities';

  interface Props {
    context: WorkspaceContext;
    busy: boolean;
    /** A job is running or queued. */
    jobsActive?: boolean;
    /** Fold the sidebar into its strip; no button is shown when absent. */
    oncollapse?: () => void;
  }

  let { context, busy, jobsActive = false, oncollapse }: Props = $props();

  /** Remember the loaded prompts file; an unchanged file is not written again. */
  function setPromptFileLoaded(loaded: DraftState['promptFileLoaded']): void {
    const current = draft.state.promptFileLoaded;
    if (current?.path === loaded?.path && current?.count === loaded?.count) return;
    draft.update('promptFileLoaded', loaded);
  }

  const s = $derived(draft.state);
  const caps = $derived(workspaceCapabilities(context, s));
  const promptFileMode = $derived(s.promptSource === 'file');
  const jsonMode = $derived(caps.showPromptInline && caps.supportsJsonPrompt && s.jsonPromptEnabled);
  const showTabs = $derived(caps.showPromptInline && caps.showEnhance && !jsonMode);
  const enhancedUsed = $derived(anyEnhancedUsed(s));
  const stale = $derived(isEnhancedStale(s));

  let activeTab = $state<'prompt' | 'enhanced'>('prompt');
  let enhancer = $state<ReturnType<typeof PromptEnhancer> | null>(null);
  let enhancing = $state(false);
  let streamingText = $state('');
  let streamingScene = $state('');
  let errorText = $state<string | null>(null);
  let noteText = $state<string | null>(null);
  let statusText = $state<string | null>(null);
  /** A result announced in the status line until the next edit or enhancement: a part switched to your text, a clip text restored. */
  let noticeText = $state<string | null>(null);
  let promptEl = $state<HTMLTextAreaElement | null>(null);

  /** Video workflows write a Scene beside the Prompt; the tabs then switch the pair. Images keep one prompt box. */
  const pairMode = $derived(caps.showScene && !promptFileMode && caps.showPromptInline);
  const showingEnhanced = $derived(showTabs && activeTab === 'enhanced');
  const enhancedValue = $derived(enhancing ? streamingText : (s.enhanceAuto ? '' : s.enhancedPrompt));
  const enhancedSceneValue = $derived(enhancing ? streamingScene : (s.enhanceAuto ? '' : s.enhancedScene));
  const visibleText = $derived(jsonMode ? s.jsonPrompt : showingEnhanced ? enhancedValue : s.prompt);
  const visibleScene = $derived(pairMode ? (showingEnhanced ? enhancedSceneValue : s.scene) : '');
  const wordCount = $derived(countWords(`${visibleScene} ${visibleText}`));
  const saveSelection = $derived(
    promptToSave(s, { enhancedView: showingEnhanced, enhancing, jsonMode, pairMode, showNegative: caps.showNegativePrompt })
  );
  const hasEnhanced = $derived(s.enhancedPrompt !== '' || (pairMode && s.enhancedScene !== ''));
  // A video needs a scene or an action; Enhanced text in either box stands in for both.
  const textMissing = $derived(!enhancedUsed && !canSubmitText(s));

  const SCENE_PLACEHOLDER = 'Who and where, look, style, camera. Kept for extensions.';
  const IMAGE_SCENE_PLACEHOLDER = 'What the image shows: who and where, look, style, camera.';
  const LEGACY_SCENE_PLACEHOLDER = 'Move the scene description here from the prompt (who, where, look). Then change the prompt to what happens next.';
  const extending = $derived(s.workflow === 'extend');
  const fromImage = $derived(s.workflow === 'img2vid');
  const promptPlaceholder = $derived(
    extending ? 'What happens next…'
      : fromImage ? 'What happens from the image: motion, camera moves, sound, dialogue.'
      : caps.showScene ? 'What happens: action, camera moves, sound, dialogue.'
      : 'Describe the scene...'
  );
  // A source that recorded one prompt and no scene: the old prompt is pre-filled and the scene is left for the user to write.
  const legacySource = $derived(extending && draft.extendSource?.legacy === true);
  const scenePlaceholder = $derived(legacySource ? LEGACY_SCENE_PLACEHOLDER : fromImage ? IMAGE_SCENE_PLACEHOLDER : SCENE_PLACEHOLDER);
  // Shown from the prefill until the Prompt is edited; it never blocks Generate.
  const originalPromptShown = $derived(extending && !promptFileMode && isOriginalLegacyPrompt(s, draft.extendSource));
  // The parts a part's Enhanced box stands for, so its placeholder tells what Generate does with it.
  const sceneUsed = $derived(pairMode && enhancedPartUsed(s, 'scene'));
  const actionUsed = $derived(pairMode && enhancedPartUsed(s, 'action'));
  const rewritesScene = $derived(runParts(s) === 'both');
  const enhancedScenePlaceholder = $derived(
    (enhancing || s.enhanceAuto) && rewritesScene ? (enhancing ? 'Enhancing…' : 'Enhanced when the job runs') : 'Not enhanced. Generate uses your scene.'
  );
  const enhancedPromptPlaceholder = $derived(
    pairMode
      ? (enhancing ? 'Enhancing…' : s.enhanceAuto ? 'Enhanced when the job runs' : 'Not enhanced. Generate uses your action.')
      : (s.enhanceAuto ? 'Generated per image when the job runs.' : s.prompt || 'Enhance the prompt, or type a variant here. When this has text, it is what gets generated.')
  );
  // An extension's clip link: put back what the clip recorded. A legacy clip recorded one prompt and no scene.
  const clipScene = $derived(extending && pairMode ? (draft.extendSource?.scene ?? '').trim() : '');
  const showClipScene = $derived(clipScene !== '' && s.scene.trim() !== clipScene);
  const clipPrompt = $derived(legacySource && pairMode ? (draft.extendSource?.prompt ?? '') : '');
  const showClipPrompt = $derived(clipPrompt.trim() !== '' && normalizePromptText(s.prompt) !== normalizePromptText(clipPrompt));
  const OUT_OF_DATE_TITLE = 'Your text changed after this was enhanced. It is still used until you choose Use mine or re-enhance.';
  const DEFAULT_SCENE_HELP = 'Used for prompts in the file that have no scene: of their own.';

  function countWords(text: string): number {
    return text.trim() ? text.trim().split(/\s+/).length : 0;
  }

  // A video accepts a scene in place of a prompt, so its message names both. Declarative, so it clears itself once either has text.
  $effect(() => {
    promptEl?.setCustomValidity(pairMode && textMissing ? 'Write an action or a scene.' : '');
  });

  const SOURCE_LABELS: Record<PromptSource, string> = { inline: 'Inline', file: 'Prompt file' };

  function onPromptInput(value: string): void {
    noticeText = null;
    draft.update('prompt', value);
  }

  function onSceneInput(value: string): void {
    noticeText = null;
    draft.update('scene', value);
  }

  function onEnhancedInput(value: string): void {
    draft.update('enhancedPrompt', value);
    // Hand-typed text (no source) is never "out of date"; edits to a generated rewrite keep its source.
    if (value.trim() === '' && (!pairMode || s.enhancedScene.trim() === '')) draft.update('enhancedFrom', null);
    else forgetEnhanceRecord();
  }

  function onEnhancedSceneInput(value: string): void {
    draft.update('enhancedScene', value);
    if (value.trim() === '' && s.enhancedPrompt.trim() === '') draft.update('enhancedFrom', null);
    else forgetEnhanceRecord();
  }

  /** Text edited by hand is no longer what the enhancer wrote: the asset must not record the enhancer's options and model for it. */
  function forgetEnhanceRecord(): void {
    if (!s.enhancedFrom?.record) return;
    const { record: _record, ...source } = s.enhancedFrom;
    draft.update('enhancedFrom', source);
  }

  // An empty prompt can fail validation while the Enhanced tab hides it; bring it forward so the message shows.
  function showPromptForValidation(event: Event): void {
    if (!showingEnhanced) return;
    activeTab = 'prompt';
    const field = event.currentTarget as HTMLTextAreaElement;
    void tick().then(() => field.reportValidity());
  }

  function clearEnhanced(): void {
    draft.patch({ enhancedPrompt: '', enhancedScene: '', enhancedFrom: null });
    noteText = null;
    noticeText = null;
    activeTab = 'prompt';
  }

  /** Bring a field forward on the tab that shows it, then focus it. */
  async function focusField(id: string, tab: 'prompt' | 'enhanced'): Promise<void> {
    if (showTabs) activeTab = tab;
    await tick();
    document.getElementById(id)?.focus();
  }

  /** Empty one part's Enhanced box, so Generate sends your text for it. */
  function useMine(part: 'scene' | 'action'): void {
    const scene = part === 'scene' ? '' : s.enhancedScene;
    const action = part === 'action' ? '' : s.enhancedPrompt;
    draft.patch({ enhancedScene: scene, enhancedPrompt: action, ...(scene.trim() === '' && action.trim() === '' ? { enhancedFrom: null } : {}) });
    noticeText = part === 'scene' ? 'Scene now uses your text.' : 'Action now uses your text.';
    void focusField(showingEnhanced ? `ws-enhanced-${part === 'scene' ? 'scene' : 'prompt'}` : part === 'scene' ? 'ws-scene' : 'ws-prompt', showingEnhanced ? 'enhanced' : 'prompt');
  }

  /** Replace the Scene or Prompt with the text the clip recorded. */
  function restoreClip(part: 'scene' | 'prompt'): void {
    const recorded = draft.extendSource?.[part];
    if (!recorded) return;
    draft.update(part, recorded);
    noticeText = `${part === 'scene' ? 'Scene' : 'Prompt'} restored from the clip.`;
    void focusField(`ws-${part}`, 'prompt');
  }

  function selectTab(tab: 'prompt' | 'enhanced', event: KeyboardEvent | MouseEvent): void {
    activeTab = tab;
    if (event instanceof KeyboardEvent) {
      queueMicrotask(() => document.getElementById(tab === 'prompt' ? 'ws-tab-prompt' : 'ws-tab-enhanced')?.focus());
    }
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

  function onTabKeydown(event: KeyboardEvent): void {
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault();
      selectTab(activeTab === 'prompt' ? 'enhanced' : 'prompt', event);
    }
  }
</script>

{#snippet tabRow()}
  <div class="prompt-tabs" class:pair-tabs={pairMode} role="tablist" aria-label={pairMode ? 'Text version' : 'Prompt version'}>
    <button
      type="button"
      role="tab"
      id="ws-tab-prompt"
      class="prompt-tab"
      aria-selected={activeTab === 'prompt'}
      aria-controls={pairMode ? 'ws-text-panel' : 'ws-prompt'}
      tabindex={activeTab === 'prompt' ? 0 : -1}
      onclick={(e) => selectTab('prompt', e)}
      onkeydown={onTabKeydown}
    >{pairMode ? 'Your text' : 'Prompt'}{#if !pairMode && !enhancedUsed}<span class="ui-badge ui-badge-info used-pill">used</span>{/if}</button>
    <button
      type="button"
      role="tab"
      id="ws-tab-enhanced"
      class="prompt-tab"
      aria-selected={activeTab === 'enhanced'}
      aria-controls={pairMode ? 'ws-text-panel' : 'ws-enhanced-prompt'}
      tabindex={activeTab === 'enhanced' ? 0 : -1}
      onclick={(e) => selectTab('enhanced', e)}
      onkeydown={onTabKeydown}
    >Enhanced{#if enhancedUsed}<span class="ui-badge ui-badge-info used-pill">used</span>{:else if pairMode && s.enhanceAuto}<span class="ui-badge ui-badge-info used-pill">auto</span>{/if}{#if stale && !s.enhanceAuto}<span class="ui-badge ui-badge-warning used-pill" title={pairMode ? 'Your text or the settings changed after this was enhanced. It is still used until you clear it.' : 'The prompt or workflow changed after this was enhanced. It is still used until you clear it.'}>Out of date</span>{/if}</button>
  </div>
{/snippet}

{#snippet partLabel(part: 'scene' | 'action')}
  {@const used = part === 'scene' ? sceneUsed : actionUsed}
  {@const name = part === 'scene' ? 'Scene' : 'Action'}
  {@const mine = part === 'scene' ? 'ws-scene' : 'ws-prompt'}
  <div class="label-row">
    <label class="ui-label" for={mine} hidden={showingEnhanced}>{name}</label>
    {#if showTabs}<label class="ui-label" for={`ws-enhanced-${part === 'scene' ? 'scene' : 'prompt'}`} hidden={!showingEnhanced}>{name}</label>{/if}
    {#if (part === 'scene' ? showClipScene : showClipPrompt) || used}
      <span class="label-extras">
        {#if part === 'scene' && showClipScene}
          <button type="button" class="prompt-link" title="Replace your scene with the one recorded in the clip" onclick={() => restoreClip('scene')}>Use the clip's scene</button>
        {:else if part === 'action' && showClipPrompt}
          <button type="button" class="prompt-link" title="Replace your action with the clip's recorded prompt" onclick={() => restoreClip('prompt')}>Use the clip's prompt</button>
        {/if}
        {#if used}
          <span id={`ws-${part}-marker`} class="label-marker">
            <span class="ui-badge ui-badge-info" title={`Generate uses the enhanced ${part}`}>Uses enhanced</span>
            {#if stale}<span class="ui-badge ui-badge-warning" title={OUT_OF_DATE_TITLE}>Out of date</span>{/if}
          </span>
          <button type="button" class="prompt-link" title={`Clear the enhanced ${part}; Generate uses yours`} onclick={() => useMine(part)}>Use mine</button>
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
      aria-describedby={[originalPromptShown ? 'ws-prompt-original' : '', actionUsed ? 'ws-action-marker' : ''].filter(Boolean).join(' ') || undefined}
      required={textMissing}
      hidden={showingEnhanced}
      oninvalid={showPromptForValidation}
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
        aria-describedby={actionUsed ? 'ws-action-marker' : undefined}
        disabled={s.enhanceAuto}
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
  <div class="ui-prompt-tools">
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
        jsonCapable={caps.supportsJsonPrompt}
        maxWords={caps.enhanceMaxWords}
        onstart={() => { activeTab = 'enhanced'; noticeText = null; }}
      />
    {/if}
    {#if caps.supportsJsonPrompt}
      <button
        type="button"
        id="ws-json-prompt-toggle"
        class="ui-btn ui-btn-sm"
        aria-pressed={s.jsonPromptEnabled}
        disabled={busy}
        title="Structured JSON caption: replaces the normal prompt"
        onclick={() => draft.update('jsonPromptEnabled', !s.jsonPromptEnabled)}
      >{'{ }'} JSON</button>
    {/if}
    {#if !jsonMode}
      <SavePromptMenu selection={saveSelection} workspaceFile={s.promptFilePath} enhancedView={showingEnhanced} />
    {/if}
    {#if showingEnhanced && !s.enhanceAuto}
      {#if stale}
        <button type="button" class="prompt-link" disabled={busy || jobsActive || enhancing} onclick={() => enhancer?.runEnhance()}>Re-enhance</button>
      {/if}
      {#if hasEnhanced}
        <button type="button" class="prompt-link" onclick={clearEnhanced}>Clear</button>
      {/if}
    {/if}
    <!-- The region stays in the page, hidden visually while empty, so assistive technology announces what fills it. -->
    <span class="prompt-status" class:sr-only={!(statusText ?? noticeText)} role="status">{statusText ?? noticeText ?? ''}</span>
    <span class="word-count">{wordCount} words</span>
  </div>
{/snippet}

<section class="compose-pane" aria-labelledby="ws-compose-title">
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

  {#if !promptFileMode && caps.showPromptInline}
    {#if pairMode}
      {#if showTabs}{@render tabRow()}{/if}
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
              placeholder={scenePlaceholder}
              aria-describedby={sceneUsed ? 'ws-scene-marker' : undefined}
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
                aria-describedby={sceneUsed ? 'ws-scene-marker' : undefined}
                disabled={s.enhanceAuto}
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
        {#if showTabs}{@render tabRow()}{/if}
        {@render promptFields()}
        {@render toolsRow()}
      </div>
    {/if}

    {#if jsonMode}
      <p class="ui-help">Must be a JSON object. Replaces the normal prompt.</p>
    {/if}
    <!-- The region stays in the page so assistive technology announces the warning when it appears. -->
    <div aria-live="polite">
      {#if originalPromptShown}
        <p id="ws-prompt-original" class="ui-help ui-help-warning original-warning" data-testid="original-prompt-warning">
          <Icon name="alert" size={12} />Still the original prompt. Change it to what happens next.
        </p>
      {/if}
    </div>
    {#if errorText}
      <p class="ui-help ui-help-error" role="alert">{errorText}</p>
    {/if}
    {#if noteText}
      <p class="ui-help">{noteText}</p>
    {:else if enhancedUsed && !pairMode}
      <p class="ui-help">The Enhanced text is generated instead of the prompt.</p>
    {/if}
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
          <PromptEnhancer contract={caps.enhancer} variant="file" {busy} {jobsActive} jsonCapable={caps.supportsJsonPrompt} maxWords={caps.enhanceMaxWords} />
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
  .compose-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .compose-title { display: flex; align-items: center; gap: 6px; }
  .collapse-toggle { display: grid; place-items: center; width: 24px; height: 24px; margin-left: -4px; border-radius: 4px; color: var(--color-text-muted); }
  @media (hover: hover) { .collapse-toggle:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); } }
  /* Below the phone breakpoint the sidebar stacks above the preview and always shows. */
  @media (max-width: 639px) {
    .collapse-toggle { display: none; }
  }
  .prompt-tabs { display: flex; align-items: center; gap: 2px; padding: 4px 4px 0; border-bottom: 1px solid var(--color-border-subtle); }
  .prompt-tab { position: relative; display: inline-flex; align-items: center; gap: 6px; padding: 5px 10px 6px; border-radius: 6px 6px 0 0; font-size: var(--text-ui); font-weight: 600; color: var(--color-text-muted); }
  @media (hover: hover) { .prompt-tab:hover { color: var(--color-text-primary); } }
  .prompt-tab:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: -2px; }
  .prompt-tab[aria-selected='true'] { color: var(--color-text-primary); }
  .prompt-tab[aria-selected='true']::after { content: ''; position: absolute; right: 8px; bottom: -1px; left: 8px; height: 2px; border-radius: 2px; background: var(--color-primary-main); }
  .used-pill { height: 16px; padding: 0 4px; }
  /* Video: the tabs sit above the Scene and Prompt boxes and line up with them. */
  .pair-tabs { margin: 0 -4px; }
  .text-pair { display: flex; flex-direction: column; gap: 8px; }
  .field label[hidden] { display: none; }
  /* The label sits left; clip links and the enhanced marker are pushed right and wrap rather than being cut. */
  .label-row { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; min-height: 22px; margin-bottom: 4px; }
  .label-extras { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 4px 6px; margin-left: auto; }
  .label-marker { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; }
  .prompt-text { display: block; width: 100%; min-height: 88px; max-height: 180px; overflow-y: auto; resize: none; padding: 9px 10px; border: 0; background: transparent; font-size: var(--text-content); line-height: 1.5; color: var(--color-text-primary); }
  .prompt-text[hidden] { display: none; }
  /* The focus ring belongs to the prompt box, not the textarea. */
  .prompt-text:focus, .prompt-text:focus-visible { outline: none; }
  .prompt-text::placeholder { color: var(--color-zinc-600); }
  .prompt-text:disabled { opacity: 0.6; }
  /* The Scene box is the smaller of the two: two lines, growing to four, then it scrolls. */
  .scene-text { min-height: 58px; max-height: 98px; }
  /* The long legacy placeholder takes three lines while the box is empty, so it is never cut off. */
  .scene-text.scene-legacy { min-height: 78px; }
  .original-warning { display: flex; align-items: flex-start; gap: 4px; }
  .original-warning :global(svg) { margin-top: 2px; }
  .prompt-link { font-size: var(--text-meta); font-weight: 600; color: var(--color-text-muted); }
  @media (hover: hover) { .prompt-link:hover { color: var(--color-text-primary); } }
  .prompt-link:disabled { opacity: 0.5; }
  .prompt-status { font-size: var(--text-meta); color: var(--color-text-muted); }
  .word-count { margin-left: auto; padding-right: 4px; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  .negative-text { min-height: 34px; max-height: 96px; resize: none; }
  /* Touch: 16px text, below which iOS zooms into a field as it focuses. */
  @media (pointer: coarse) {
    .prompt-text, .negative-text { font-size: 16px; }
  }
</style>
