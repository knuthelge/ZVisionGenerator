<script lang="ts">
  import { tick } from 'svelte';
  import { Icon } from '$lib/components/atoms';
  import { HelpLine } from '$lib/components/molecules';
  import { draft } from '$lib/state/draft.svelte';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { AUDIO_TEXT_KEYS } from '$lib/state/audioDraft';
  import { hasDescribe } from '$lib/state/audioFill';
  import { EMPTY_REWRITES, clearRewrites, pushRewrite, rewriteUsed } from '$lib/state/audioRewrites';
  import { ClearUndo } from '$lib/state/clearUndo.svelte';
  import { hasOpenModal, isCommandKey } from '$lib/keyboard';
  import type { LyricsTarget } from '$lib/state/lyricsHelper';
  import { editLyrics, pushVersion, showVersion, versionStepper, writtenByHelper } from '$lib/state/lyricsVersions';
  import { musicStatus } from '$lib/state/musicMode';
  import { hasWords, parseSong, tagName } from '$lib/state/songBlocks';
  import { scriptRecord, type ScriptResult } from '$lib/state/scriptHelper';
  import { formatReadingTime } from '$lib/state/speechDelivery';
  import type { AudioRewrites, AudioSettingsVocabulary, DraftState, PromptEnhancerContract, RecentPrompt } from '$lib/types';
  import AudioEnhance from './AudioEnhance.svelte';
  import AudioTextBox from './AudioTextBox.svelte';
  import DescribeIt from './DescribeIt.svelte';
  import InsertMenu from './InsertMenu.svelte';
  import LyricsHelper from './LyricsHelper.svelte';
  import type { AudioView } from './audioView';
  import RecentPromptsMenu from './RecentPromptsMenu.svelte';
  import ScriptHelper from './ScriptHelper.svelte';
  import SongBlocks from './SongBlocks.svelte';

  interface Props {
    /** The Audio tab's view, computed once by the Workspace: the open kind, its text and the counts the status line reads. */
    view: AudioView;
    /** The prompt enhancer, or null where none is served; the Style and Sound boxes then have no Enhance. */
    enhancer?: PromptEnhancerContract | null;
    /** The vocabulary of the take's settings; a Sound's type and length go to the enhancer as context. */
    vocabulary?: AudioSettingsVocabulary;
    /** A job is being submitted. */
    busy?: boolean;
    /** A take is running or queued; the enhancer would need its memory. */
    jobsActive?: boolean;
  }

  let { view, enhancer = null, vocabulary, busy = false, jobsActive = false }: Props = $props();

  const s = $derived(draft.state);
  const kind = $derived(view.kind ?? 'music');
  /** The open Music model sings lyrics: the Lyrics box and the song's status line. A model without them takes the style alone. */
  const lyrics = $derived(view.shown.showLyrics);
  let recentsButton = $state<HTMLButtonElement | null>(null);
  let recentsOpen = $state(false);
  // The box when the menu was opened with Alt+↑ from it; Escape returns focus there instead of to the button.
  let recentsReturn = $state<HTMLElement | null>(null);

  let tab = $state<'mine' | 'enhanced'>('mine');
  // The Speech text, which the Insert menu edits at its caret.
  let speechBox = $state<HTMLTextAreaElement | null>(null);
  // The Written text's box, which the Insert menu edits instead while that tab is shown.
  let writtenBox = $state<HTMLTextAreaElement | null>(null);
  // The script helper: whether its panel is open, the element it opens beside, and the enhancer's notice about the last result.
  let scriptOpen = $state(false);
  let scriptButton = $state<HTMLButtonElement | null>(null);
  let scriptNote = $state<string | null>(null);
  let enhance = $state<ReturnType<typeof AudioEnhance> | null>(null);
  let describeBox = $state<ReturnType<typeof DescribeIt> | null>(null);
  let enhancing = $state(false);
  let streamingText = $state('');
  let errorText = $state<string | null>(null);
  let noteText = $state<string | null>(null);
  let statusText = $state<string | null>(null);

  // The lyrics helper: what it is open on and the element its popover opens beside, and the enhancer's notice about the last result.
  let helperTarget = $state<LyricsTarget | null>(null);
  let helperAnchor = $state<HTMLElement | null>(null);
  let helperButton = $state<HTMLButtonElement | null>(null);
  let lyricsNote = $state<string | null>(null);
  // The helpers are writing: the box each works on cannot be cleared meanwhile.
  let lyricsRunning = $state(false);
  let scriptRunning = $state(false);
  // The Clear of the open kind's text box (with the tab it showed) and of the Lyrics, each with its Undo.
  const boxCleared = new ClearUndo<'mine' | 'enhanced'>();
  const lyricsCleared = new ClearUndo();
  const BOX_UNDO_ID = 'ws-box-clear-undo';
  const LYRICS_UNDO_ID = 'ws-lyrics-clear-undo';

  const STYLE_PLACEHOLDER = 'Genre, mood, instruments, singer.';
  const STYLE_PLACEHOLDER_INSTRUMENTAL = 'Genre, mood, instruments.';
  const LYRICS_PLACEHOLDER = 'Paste or write the lyrics, or add sections below. Sections without words make an instrumental; more sections, longer piece.';
  const PLACEHOLDERS = { speech: 'What to say.', sound: 'What you hear: rain on a window, a crowd in a hall, a door slam…' };

  const status = $derived(kind === 'music' ? musicStatus(view.music) : '');
  // The characters that are read (marks excluded) against the model's cap, then the time they take at the open Pace.
  const characters = $derived(
    [`${view.characters.toLocaleString('en-US')} / ${(view.maxCharacters ?? 0).toLocaleString('en-US')} characters`, view.readingSeconds === null || view.characters === 0 ? null : `about ${formatReadingTime(view.readingSeconds)}`]
      .filter((part) => part !== null)
      .join(' · '),
  );

  /** The Describe it box (Music and Speech, once an enhancer is served to fill them). */
  const describable = $derived(enhancer !== null && hasDescribe(kind));

  /** Fill in from a helper's empty state: run it when the idea is written, else open the Describe it box to write one. */
  function fillFromHelper(): void {
    if (!hasDescribe(kind)) return;
    if (s.audioDescribe[kind].text.trim() === '') void describeBox?.open();
    else void describeBox?.fill();
  }

  const song = $derived(parseSong(s.lyrics));
  const stepper = $derived(versionStepper(s));
  const lyricsWritten = $derived(writtenByHelper(s));

  /** Speech has a Written tab once an enhancer is served to write it, or a version came back with a take. */
  const written = $derived(enhancer !== null || rewriteUsed(view.rewrites));
  /** The box the Insert menu edits: the one that is shown. */
  const insertBox = $derived(written && tab === 'enhanced' ? writtenBox : speechBox);

  /** Open the lyrics helper on *target* beside *anchor*; the opener it was already open from closes it. While it writes only its Stop ends it. */
  function toggleHelper(target: LyricsTarget, anchor: HTMLElement | null): void {
    if (lyricsRunning) return;
    const same = helperTarget !== null && helperTarget.section === target.section && helperTarget.write === target.write && helperAnchor === anchor;
    helperTarget = same ? null : target;
    helperAnchor = anchor;
  }

  /** Keep the helper's whole-song result as the lyrics and as the newest version. */
  function onHelperResult(text: string, warning: string | null): void {
    draft.patch({ lyrics: text, ...pushVersion(s, text) });
    lyricsCleared.forget();
    lyricsNote = warning;
  }

  /** Type into the lyrics: the shown version follows what is typed. */
  function onLyricsInput(text: string): void {
    draft.patch(editLyrics(s, text));
    lyricsCleared.forget();
    lyricsNote = null;
  }

  /** Show version *index* (1-based) of the lyrics, or the user's own. */
  function showLyricsVersion(index: number): void {
    const patch = showVersion(s, index);
    if (patch) draft.patch(patch);
    lyricsCleared.forget();
    lyricsNote = null;
  }

  /** Keep the script helper's result as the Written text and as its newest version, and show it. */
  function onScriptResult(result: ScriptResult): void {
    const record = result.model ? scriptRecord(result.model, result.settings) : undefined;
    const from = { input: s.speechText, model: s.model, task: 'script' as const, ...(record ? { record } : {}) };
    setRewrites(pushRewrite(view.rewrites, { text: result.text, from }));
    clearMessages();
    scriptNote = result.warning;
    tab = 'enhanced';
  }

  /** The help lines say only the last result: an edit or a run clears them, and so goes the Undo of a Clear. */
  function clearMessages(): void {
    errorText = null;
    noteText = null;
    scriptNote = null;
    boxCleared.forget();
  }

  /** The box holds something to clear: your text, an Enhanced text or its versions. */
  const boxHolds = $derived(view.text !== '' || rewriteUsed(view.rewrites) || view.rewrites.history.length > 0);
  const lyricsHold = $derived(s.lyrics !== '' || s.lyricsVersions.length > 0);
  /** What the open box's Clear names. */
  const boxName = $derived(kind === 'music' ? 'style' : kind === 'sound' ? 'prompt' : 'text');

  /** Empty the open kind's box: your text, the Enhanced text and every version. Undo puts them back. */
  function clearBox(): void {
    if (view.box === null) return;
    const patch = { [AUDIO_TEXT_KEYS[kind]]: '', audioRewrites: { ...s.audioRewrites, [view.box]: EMPTY_REWRITES } } as Partial<DraftState>;
    const shown = tab;
    clearMessages();
    boxCleared.clear(patch, shown);
    tab = 'mine';
    // A finger would raise the keyboard over the result: it gets the Undo, a mouse the empty text.
    void tick().then(() => document.getElementById(layout.coarse ? BOX_UNDO_ID : kind === 'music' ? 'ws-audio-style' : 'ws-prompt')?.focus());
  }

  /** Put back what the box's Clear took, with the tab it showed. */
  function undoBox(): void {
    const undone = boxCleared.undo();
    if (undone === null) return;
    tab = undone.view;
    if (!layout.coarse) void tick().then(() => document.getElementById(undone.view === 'enhanced' ? (kind === 'music' ? 'ws-enhanced-audio-style' : 'ws-enhanced-prompt') : kind === 'music' ? 'ws-audio-style' : 'ws-prompt')?.focus());
  }

  /** Empty the song and its versions. Undo puts them back. */
  function clearLyrics(): void {
    lyricsNote = null;
    lyricsCleared.clear({ lyrics: '', lyricsVersions: [], lyricsCurrent: 0 }, null);
    void tick().then(() => document.getElementById(layout.coarse ? LYRICS_UNDO_ID : 'ws-lyrics')?.focus());
  }

  function undoLyrics(): void {
    if (lyricsCleared.undo() !== null && !layout.coarse) void tick().then(() => document.getElementById('ws-lyrics')?.focus());
  }

  /** Step through, edit or scratch the Enhanced text: what the last result said was about the version before. */
  function onRewrites(next: AudioRewrites): void {
    clearMessages();
    setRewrites(next);
  }

  /** Type into the own text of the open kind's box. */
  function onOwnInput(key: 'audioStyle' | 'soundPrompt' | 'speechText', text: string): void {
    clearMessages();
    draft.update(key, text);
  }

  /** Bring a recent prompt back into the open kind's text box; a song's lyrics come back with its style. */
  function pickRecent(entry: RecentPrompt): void {
    draft.update(AUDIO_TEXT_KEYS[kind], entry.prompt);
    if (kind === 'music' && lyrics) {
      draft.patch({ lyrics: entry.scene, lyricsVersions: [], lyricsCurrent: 0 });
      lyricsCleared.forget();
    }
    // The versions of the text that was there do not belong to the one picked.
    setRewrites(clearRewrites());
    clearMessages();
    tab = 'mine';
    recentsOpen = false;
    // A finger would raise the keyboard over the result.
    if (!layout.coarse) document.getElementById(kind === 'music' ? 'ws-audio-style' : 'ws-prompt')?.focus();
  }

  /** Keep *next* as the Enhanced text of the open kind's box. */
  function setRewrites(next: AudioRewrites): void {
    if (view.box === null) return;
    draft.update('audioRewrites', { ...s.audioRewrites, [view.box]: next });
  }

  // Another kind has its own box: the Enhanced tab, a rewrite's progress and its messages belong to the box left.
  $effect(() => {
    void kind;
    tab = 'mine';
    errorText = null;
    noteText = null;
    helperTarget = null;
    lyricsNote = null;
    scriptOpen = false;
    scriptNote = null;
    boxCleared.forget();
    lyricsCleared.forget();
  });

  // A model without lyrics has no Lyrics box to help with.
  $effect(() => {
    if (!lyrics) helperTarget = null;
  });

  // ⌘E / Ctrl+E enhances the box's text, even while typing in it.
  $effect(() => {
    if (!enhance) return;
    const target = enhance;
    function handleKeydown(event: KeyboardEvent): void {
      if (event.defaultPrevented || !isCommandKey(event) || event.altKey || event.shiftKey || event.key.toLowerCase() !== 'e') return;
      if (hasOpenModal()) return;
      event.preventDefault();
      if (!enhancing) void target.runEnhance();
    }
    document.addEventListener('keydown', handleKeydown);
    return () => document.removeEventListener('keydown', handleKeydown);
  });

  // Alt+↑ in an empty box opens the recent prompts; with text in it the browser keeps its caret movement.
  function onKeydown(event: KeyboardEvent): void {
    if (event.key !== 'ArrowUp' || !event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    if (view.text !== '') return;
    event.preventDefault();
    recentsReturn = event.currentTarget as HTMLElement;
    recentsOpen = true;
  }
</script>

{#snippet recents()}
    <button
      bind:this={recentsButton}
      type="button"
      id="ws-recent-prompts"
      class="ui-btn ui-btn-sm ui-btn-icon"
      aria-label="Recent prompts"
      aria-haspopup="dialog"
      aria-expanded={recentsOpen}
      title="Recent prompts (Alt+↑ in an empty box)"
      onclick={() => { recentsReturn = null; recentsOpen = !recentsOpen; }}
    ><Icon name="history" size={14} /></button>
    <RecentPromptsMenu open={recentsOpen} anchor={recentsButton} returnFocus={recentsReturn ?? recentsButton} mode={kind} jsonCapable={false} onpick={pickRecent} onclose={() => { recentsOpen = false; }} />
{/snippet}

{#snippet lyricsTools()}
  {#if enhancer}
    <button
      bind:this={helperButton}
      type="button"
      id="ws-lyrics-helper-toggle"
      class="ui-btn ui-btn-sm"
      aria-haspopup="dialog"
      aria-expanded={helperTarget !== null && helperTarget.section === null}
      aria-controls="ws-lyrics-helper-panel"
      onclick={() => toggleHelper({ section: null, write: false }, helperButton)}
    >
      <Icon name="edit" size={13} />Improve lyrics<Icon name="chevdown" size={12} />
    </button>
  {/if}
{/snippet}

{#snippet blockHelp(index: number)}
  {#if enhancer && song.blocks[index]}
    {@const block = song.blocks[index]}
    {@const writes = !hasWords(block)}
    <button
      type="button"
      id={`ws-lyrics-help-${index}`}
      class="ui-btn ui-btn-sm"
      aria-haspopup="dialog"
      aria-expanded={helperTarget !== null && helperTarget.section === index}
      aria-label={`${writes ? 'Write' : 'Improve'} section ${index + 1}: ${tagName(block.tag) || 'Section'}`}
      onclick={(e) => toggleHelper({ section: index, write: writes }, e.currentTarget)}
    >
      <Icon name="edit" size={13} />{writes ? 'Write' : 'Improve'}
    </button>
  {/if}
{/snippet}

{#snippet insertTools()}
  <InsertMenu textarea={insertBox} {vocabulary} delivery={view.shown.showDelivery && !view.voice.recording} />
  {#if enhancer}
    <button
      bind:this={scriptButton}
      type="button"
      id="ws-script-helper-toggle"
      class="ui-btn ui-btn-sm"
      aria-haspopup="dialog"
      aria-expanded={scriptOpen}
      aria-controls="ws-script-helper-panel"
      aria-label={view.stale ? 'Script helper (out of date)' : undefined}
      onclick={() => { if (!scriptRunning) scriptOpen = !scriptOpen; }}
    >
      {#if view.stale}<Icon name="alert" size={13} class="ui-icon-warn" />{:else}<Icon name="edit" size={13} />{/if}Script helper<Icon name="chevdown" size={12} />
    </button>
  {/if}
{/snippet}

{#snippet enhanceTools()}
  {#if enhancer}
    <AudioEnhance
      bind:this={enhance}
      bind:enhancing
      bind:streamingText
      bind:errorText
      bind:noteText
      bind:statusText
      contract={enhancer}
      {view}
      {vocabulary}
      {busy}
      {jobsActive}
      stale={view.stale}
      staleReason={view.staleReason}
      onstart={() => { clearMessages(); tab = 'enhanced'; }}
      onfail={() => { tab = 'mine'; }}
    />
  {/if}
{/snippet}

{#if describable && enhancer && hasDescribe(kind)}
  <DescribeIt bind:this={describeBox} {view} {kind} contract={enhancer} {vocabulary} {busy} {jobsActive} onfilled={(wrote) => { if (wrote.includes('style')) tab = 'mine'; }} />
{/if}

{#if kind === 'music'}
  <div class="text-pair" data-testid="audio-compose" data-kind="music">
    <div class="field">
      <AudioTextBox
        bind:tab
        value={s.audioStyle}
        rewrites={view.rewrites}
        stale={view.stale}
        id="ws-audio-style"
        enhancedId="ws-enhanced-audio-style"
        label="Style"
        placeholder={lyrics ? STYLE_PLACEHOLDER : STYLE_PLACEHOLDER_INSTRUMENTAL}
        compact
        {enhancing}
        {streamingText}
        {statusText}
        tools={enhanceTools}
        aside={lyrics ? undefined : recents}
        onclear={boxHolds && !enhancing ? clearBox : undefined}
        clearWhat={boxName}
        oninput={(text) => onOwnInput('audioStyle', text)}
        onrewrites={onRewrites}
        onstop={() => enhance?.stop()}
        onkeydown={onKeydown}
      />
      <HelpLine error={errorText} note={noteText} undo={boxCleared.offered ? { what: boxName, id: BOX_UNDO_ID, onundo: undoBox } : null} />
    </div>
    {#if lyrics}
      <div class="field">
        <SongBlocks
          value={s.lyrics}
          placeholder={LYRICS_PLACEHOLDER}
          describedby="ws-music-status"
          tools={recents}
          lead={enhancer ? lyricsTools : undefined}
          written={lyricsWritten}
          {stepper}
          onstep={showLyricsVersion}
          onclear={lyricsHold && !lyricsRunning ? clearLyrics : undefined}
          blockTools={enhancer ? blockHelp : undefined}
          onhelp={enhancer ? (index, write, anchor) => toggleHelper({ section: index, write }, anchor) : undefined}
          oninput={onLyricsInput}
        />
        <HelpLine note={lyricsNote} testid="lyrics-note" undo={lyricsCleared.offered ? { what: 'lyrics', id: LYRICS_UNDO_ID, onundo: undoLyrics } : null} />
        {#if enhancer}
          <LyricsHelper
            contract={enhancer}
            target={helperTarget}
            anchor={helperAnchor}
            lyrics={s.lyrics}
            style={view.sent.text}
            model={s.model}
            {busy}
            {jobsActive}
            bind:running={lyricsRunning}
            onclose={() => { helperTarget = null; }}
            onfill={describable ? fillFromHelper : undefined}
            onresult={onHelperResult}
          />
        {/if}
      </div>
    {/if}
  </div>
  {#if lyrics}
    <div class="prompt-statusline" data-testid="prompt-statusline" data-state={view.music?.kind ?? 'none'}>
      <span id="ws-music-status" class="statusline-msg" role="status">{status}</span>
    </div>
  {/if}
{:else if kind === 'sound'}
  <div class="field" data-testid="audio-compose" data-kind="sound">
    <AudioTextBox
      bind:tab
      value={s.soundPrompt}
      rewrites={view.rewrites}
      stale={view.stale}
      id="ws-prompt"
      enhancedId="ws-enhanced-prompt"
      label="Prompt"
      placeholder={PLACEHOLDERS.sound}
      {enhancing}
      {streamingText}
      {statusText}
      tools={enhanceTools}
      aside={recents}
      onclear={boxHolds && !enhancing ? clearBox : undefined}
      clearWhat={boxName}
      oninput={(text) => onOwnInput('soundPrompt', text)}
      onrewrites={onRewrites}
      onstop={() => enhance?.stop()}
      onkeydown={onKeydown}
    />
    <HelpLine error={errorText} note={noteText} undo={boxCleared.offered ? { what: boxName, id: BOX_UNDO_ID, onundo: undoBox } : null} />
  </div>
{:else}
  <div class="field" data-testid="audio-compose" data-kind="speech">
    <AudioTextBox
      bind:tab
      bind:textarea={speechBox}
      bind:enhancedTextarea={writtenBox}
      enhanceable={written}
      value={s.speechText}
      rewrites={view.rewrites}
      stale={view.stale}
      id="ws-prompt"
      enhancedId="ws-enhanced-prompt"
      label="Text"
      enhancedLabel="Written"
      placeholder={PLACEHOLDERS.speech}
      required
      invalid={view.overLimit}
      describedby="ws-speech-count"
      tools={insertTools}
      aside={recents}
      onclear={boxHolds && !scriptRunning ? clearBox : undefined}
      clearWhat={boxName}
      oninput={(text) => onOwnInput('speechText', text)}
      onrewrites={onRewrites}
      onkeydown={onKeydown}
    />
    <!-- Under the box: the last result's note on a line of its own, then the character limit at the right of the line below. -->
    <HelpLine note={scriptNote} testid="script-note" />
    <div class="limit-row">
      <div class="limit-undo"><HelpLine undo={boxCleared.offered ? { what: boxName, id: BOX_UNDO_ID, onundo: undoBox } : null} /></div>
      <span id="ws-speech-count" class="limit" class:over={view.overLimit} data-testid="text-count">{characters}</span>
    </div>
    {#if enhancer}
      <ScriptHelper
        contract={enhancer}
        open={scriptOpen}
        anchor={scriptButton}
        text={s.speechText}
        staleReason={view.staleReason}
        pace={view.pace}
        recording={view.voice.recording}
        model={s.model}
        {vocabulary}
        {busy}
        {jobsActive}
        bind:running={scriptRunning}
        onclose={() => { scriptOpen = false; }}
        onfill={describable ? fillFromHelper : undefined}
        onresult={onScriptResult}
      />
    {/if}
  </div>
{/if}

<style>
  .text-pair { display: flex; flex-direction: column; gap: 8px; }
  /* The status line under the boxes: a fixed-height row with the message on the left. */
  .prompt-statusline { display: flex; align-items: center; gap: 8px; min-height: var(--spacing-control-sm); margin-top: -2px; padding: 0 4px; font-size: var(--text-meta); color: var(--color-text-muted); }
  .statusline-msg { flex: 1; min-width: 0; color: var(--color-text-secondary); }
  /* Under Speech's box: the note takes the full width of its own line; the characters read against the limit sit at the right of the line below. */
  .limit-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; min-height: var(--spacing-control-sm); margin-top: -2px; }
  /* Speech's Undo takes the left of the line; the limit keeps the right. */
  .limit-undo { min-width: 0; }
  .limit-undo :global(.help-line) { margin: 0; }
  .limit { margin-left: auto; padding: 0 4px; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  .limit.over { font-weight: 600; color: var(--color-warning); }
</style>
