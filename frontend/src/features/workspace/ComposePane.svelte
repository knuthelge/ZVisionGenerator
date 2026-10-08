<script lang="ts">
  import { tick } from 'svelte';
  import { draft } from '$lib/state/draft.svelte';
  import { enhancedOverrideActive, isEnhancedStale } from '$lib/state/promptEnhance';
  import { autogrow } from '$lib/actions/autogrow';
  import { hasOpenModal, isCommandKey } from '$lib/keyboard';
  import { Icon } from '$lib/components/atoms';
  import { PromptFileField, Segmented } from '$lib/components/molecules';
  import type { PromptSource, WorkspaceContext } from '$lib/types';
  import PromptEnhancer from './PromptEnhancer.svelte';
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

  const s = $derived(draft.state);
  const caps = $derived(workspaceCapabilities(context, s));
  const promptFileMode = $derived(s.promptSource === 'file');
  const jsonMode = $derived(caps.showPromptInline && caps.supportsJsonPrompt && s.jsonPromptEnabled);
  const showTabs = $derived(caps.showPromptInline && caps.showEnhance && !jsonMode);
  const enhancedUsed = $derived(enhancedOverrideActive(s));
  const stale = $derived(isEnhancedStale(s));

  let activeTab = $state<'prompt' | 'enhanced'>('prompt');
  let enhancer = $state<ReturnType<typeof PromptEnhancer> | null>(null);
  let enhancing = $state(false);
  let streamingText = $state('');
  let errorText = $state<string | null>(null);
  let noteText = $state<string | null>(null);
  let statusText = $state<string | null>(null);

  const showingEnhanced = $derived(showTabs && activeTab === 'enhanced');
  const enhancedValue = $derived(enhancing ? streamingText : (s.enhanceAuto ? '' : s.enhancedPrompt));
  const visibleText = $derived(jsonMode ? s.jsonPrompt : showingEnhanced ? enhancedValue : s.prompt);
  const wordCount = $derived(visibleText.trim() ? visibleText.trim().split(/\s+/).length : 0);

  const SOURCE_LABELS: Record<PromptSource, string> = { inline: 'Inline', file: 'Prompt file' };

  function onEnhancedInput(value: string): void {
    draft.update('enhancedPrompt', value);
    // Hand-typed text (no source) is never "out of date"; edits to a generated rewrite keep its source.
    if (value.trim() === '') draft.update('enhancedFrom', null);
  }

  // An empty prompt can fail validation while the Enhanced tab hides it; bring it forward so the message shows.
  function showPromptForValidation(event: Event): void {
    if (!showingEnhanced) return;
    activeTab = 'prompt';
    const field = event.currentTarget as HTMLTextAreaElement;
    void tick().then(() => field.reportValidity());
  }

  function clearEnhanced(): void {
    draft.patch({ enhancedPrompt: '', enhancedFrom: null });
    noteText = null;
    activeTab = 'prompt';
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

<section class="compose-pane custom-scrollbar" aria-labelledby="ws-compose-title">
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
    <div class="prompt-box">
      {#if showTabs}
        <div class="prompt-tabs" role="tablist" aria-label="Prompt version">
          <button
            type="button"
            role="tab"
            id="ws-tab-prompt"
            class="prompt-tab"
            aria-selected={activeTab === 'prompt'}
            aria-controls="ws-prompt"
            tabindex={activeTab === 'prompt' ? 0 : -1}
            onclick={(e) => selectTab('prompt', e)}
            onkeydown={onTabKeydown}
          >Prompt{#if !enhancedUsed}<span class="ui-badge ui-badge-info used-pill">used</span>{/if}</button>
          <button
            type="button"
            role="tab"
            id="ws-tab-enhanced"
            class="prompt-tab"
            aria-selected={activeTab === 'enhanced'}
            aria-controls="ws-enhanced-prompt"
            tabindex={activeTab === 'enhanced' ? 0 : -1}
            onclick={(e) => selectTab('enhanced', e)}
            onkeydown={onTabKeydown}
          >Enhanced{#if enhancedUsed}<span class="ui-badge ui-badge-info used-pill">used</span>{/if}{#if stale && !s.enhanceAuto}<span class="ui-badge ui-badge-warning used-pill" title="The prompt or workflow changed after this was enhanced. It is still used until you clear it.">Out of date</span>{/if}</button>
        </div>
      {/if}

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
        <label for="ws-prompt" class="sr-only">Prompt</label>
        <textarea
          id="ws-prompt"
          name="prompt"
          class="prompt-text"
          role={showTabs ? 'tabpanel' : undefined}
          aria-labelledby={showTabs ? 'ws-tab-prompt' : undefined}
          placeholder="Describe the scene..."
          required={!enhancedUsed}
          hidden={showingEnhanced}
          oninvalid={showPromptForValidation}
          value={s.prompt}
          use:autogrow={s.prompt}
          oninput={(e) => draft.update('prompt', e.currentTarget.value)}
        ></textarea>
        {#if showTabs}
          <textarea
            id="ws-enhanced-prompt"
            class="prompt-text"
            role="tabpanel"
            aria-labelledby="ws-tab-enhanced"
            hidden={!showingEnhanced}
            placeholder={s.enhanceAuto ? 'Generated per image when the job runs.' : s.prompt || 'Enhance the prompt, or type a variant here. When this has text, it is what gets generated.'}
            disabled={s.enhanceAuto}
            readonly={enhancing}
            aria-busy={enhancing}
            value={enhancedValue}
            use:autogrow={[enhancedValue, showingEnhanced]}
            oninput={(e) => onEnhancedInput(e.currentTarget.value)}
          ></textarea>
        {/if}
      {/if}

      <div class="prompt-tools">
        {#if caps.showEnhance && caps.enhancer && !jsonMode}
          <PromptEnhancer
            bind:this={enhancer}
            bind:enhancing
            bind:streamingText
            bind:errorText
            bind:noteText
            bind:statusText
            contract={caps.enhancer}
            variant="inline"
            {busy}
            {jobsActive}
            maxWords={caps.enhanceMaxWords}
            onstart={() => { activeTab = 'enhanced'; }}
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
        {#if showingEnhanced && !s.enhanceAuto}
          {#if stale}
            <button type="button" class="prompt-link" disabled={busy || jobsActive || enhancing} onclick={() => enhancer?.runEnhance()}>Re-enhance</button>
          {/if}
          {#if s.enhancedPrompt}
            <button type="button" class="prompt-link" onclick={clearEnhanced}>Clear</button>
          {/if}
        {/if}
        {#if statusText}
          <span class="prompt-status" role="status">{statusText}</span>
        {/if}
        <span class="word-count">{wordCount} words</span>
      </div>
    </div>

    {#if jsonMode}
      <p class="ui-help">Must be a JSON object. Replaces the normal prompt.</p>
    {/if}
    {#if errorText}
      <p class="ui-help ui-help-error" role="alert">{errorText}</p>
    {/if}
    {#if noteText}
      <p class="ui-help">{noteText}</p>
    {:else if enhancedUsed}
      <p class="ui-help">The Enhanced text is generated instead of the prompt.</p>
    {/if}
  {/if}

  {#if promptFileMode && context.prompt_file && caps.showPromptFileControls}
    <PromptFileField
      contract={context.prompt_file}
      promptSource={s.promptSource}
      path={s.promptFilePath}
      selectedOptionIds={s.promptFileOptionIds}
      workflowMode={caps.isImageMode ? 'image' : 'video'}
      negativePromptSupported={caps.supportsNegativePrompt}
      disabled={busy}
      onPathChange={(path) => draft.update('promptFilePath', path)}
      onOptionChange={(optionIds) => draft.update('promptFileOptionIds', optionIds)}
    >
      {#snippet tools()}
        {#if caps.showEnhanceAuto && caps.enhancer}
          <PromptEnhancer contract={caps.enhancer} variant="file" {busy} {jobsActive} maxWords={caps.enhanceMaxWords} />
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
  .collapse-toggle:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); }
  /* Below the phone breakpoint the sidebar stacks above the preview and always shows. */
  @media (max-width: 639px) {
    .collapse-toggle { display: none; }
  }
  .prompt-tabs { display: flex; align-items: center; gap: 2px; padding: 4px 4px 0; border-bottom: 1px solid var(--color-border-subtle); }
  .prompt-tab { position: relative; display: inline-flex; align-items: center; gap: 6px; padding: 5px 10px 6px; border-radius: 6px 6px 0 0; font-size: var(--text-ui); font-weight: 600; color: var(--color-text-muted); }
  .prompt-tab:hover { color: var(--color-text-primary); }
  .prompt-tab:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: -2px; }
  .prompt-tab[aria-selected='true'] { color: var(--color-text-primary); }
  .prompt-tab[aria-selected='true']::after { content: ''; position: absolute; right: 8px; bottom: -1px; left: 8px; height: 2px; border-radius: 2px; background: var(--color-primary-main); }
  .used-pill { height: 16px; padding: 0 4px; }
  .prompt-text { display: block; width: 100%; min-height: 88px; max-height: 180px; overflow-y: auto; resize: none; padding: 9px 10px; border: 0; background: transparent; font-size: var(--text-content); line-height: 1.5; color: var(--color-text-primary); }
  .prompt-text[hidden] { display: none; }
  /* The focus ring belongs to the prompt box, not the textarea. */
  .prompt-text:focus, .prompt-text:focus-visible { outline: none; }
  .prompt-text::placeholder { color: var(--color-zinc-600); }
  .prompt-text:disabled { opacity: 0.6; }
  .prompt-link { font-size: var(--text-meta); font-weight: 600; color: var(--color-text-muted); }
  .prompt-link:hover { color: var(--color-text-primary); }
  .prompt-link:disabled { opacity: 0.5; }
  .prompt-status { font-size: var(--text-meta); color: var(--color-text-muted); }
  .word-count { margin-left: auto; padding-right: 4px; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  .negative-text { min-height: 34px; max-height: 96px; resize: none; }
</style>
