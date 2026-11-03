<script lang="ts">
  import { autogrow } from '$lib/actions/autogrow';
  import { Icon } from '$lib/components/atoms';
  import { draft } from '$lib/state/draft.svelte';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { AUDIO_TEXT_KEYS } from '$lib/state/audioDraft';
  import { musicStatus } from '$lib/state/musicMode';
  import type { RecentPrompt } from '$lib/types';
  import type { AudioView } from './audioView';
  import RecentPromptsMenu from './RecentPromptsMenu.svelte';

  interface Props {
    /** The Audio tab's view, computed once by the Workspace: the open kind, its text and the counts the status line reads. */
    view: AudioView;
  }

  let { view }: Props = $props();

  const s = $derived(draft.state);
  const kind = $derived(view.kind ?? 'music');
  let recentsButton = $state<HTMLButtonElement | null>(null);
  let recentsOpen = $state(false);
  // The box when the menu was opened with Alt+↑ from it; Escape returns focus there instead of to the button.
  let recentsReturn = $state<HTMLElement | null>(null);

  const STYLE_PLACEHOLDER = 'Genre, mood, instruments, singer.';
  const LYRICS_PLACEHOLDER = '[verse]\nThe words of the first verse…\n[chorus]\n… — or only tags such as [Intro] [Instrumental] [Solo] [Outro] for an instrumental; more sections, longer piece';
  const PLACEHOLDERS = { speech: 'What to say. Tags such as <|sfx:laughter|> work here.', sound: 'What you hear: rain on a window, a crowd in a hall, a door slam…' };
  const PARTS_HELP = 'Long text is read in parts of about 500 characters and joined into one take.';

  const countWords = (text: string): number => (text.trim() ? text.trim().split(/\s+/).length : 0);
  const wordCount = $derived(kind === 'music' ? countWords(s.lyrics) : kind === 'sound' ? countWords(s.soundPrompt) : 0);
  const status = $derived(kind === 'music' ? musicStatus(view.music) : '');
  const characters = $derived(`${view.characters.toLocaleString('en-US')} / ${(view.maxCharacters ?? 0).toLocaleString('en-US')} characters`);

  /** Bring a recent prompt back into the open kind's text box; a song's lyrics come back with its style (when the model takes lyrics). */
  function pickRecent(entry: RecentPrompt): void {
    draft.update(AUDIO_TEXT_KEYS[kind], entry.prompt);
    if (kind === 'music' && view.model?.capabilities.lyrics !== false) draft.update('lyrics', entry.scene);
    recentsOpen = false;
    // A finger would raise the keyboard over the result.
    if (!layout.coarse) document.getElementById(kind === 'music' ? 'ws-audio-style' : 'ws-prompt')?.focus();
  }

  // Alt+↑ in an empty box opens the recent prompts; with text in it the browser keeps its caret movement.
  function onKeydown(event: KeyboardEvent): void {
    if (event.key !== 'ArrowUp' || !event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    if (view.text !== '') return;
    event.preventDefault();
    recentsReturn = event.currentTarget as HTMLElement;
    recentsOpen = true;
  }
</script>

{#snippet tools()}
  <div class="ui-prompt-tools">
    <span class="tools-spacer"></span>
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
  </div>
{/snippet}

{#if kind === 'music'}
  <div class="text-pair" data-testid="audio-compose" data-kind="music">
    <div class="field">
      <div class="label-row">
        <label class="ui-label" for="ws-audio-style">Style</label>
        <span class="label-extras">required</span>
      </div>
      <div class="ui-prompt-box">
        <textarea
          id="ws-audio-style"
          class="prompt-text scene-text"
          placeholder={STYLE_PLACEHOLDER}
          aria-required="true"
          value={s.audioStyle}
          use:autogrow={s.audioStyle}
          onkeydown={onKeydown}
          oninput={(e) => draft.update('audioStyle', e.currentTarget.value)}
        ></textarea>
      </div>
    </div>
    {#if view.model?.capabilities.lyrics !== false}
      <div class="field">
        <div class="label-row">
          <label class="ui-label" for="ws-lyrics">Lyrics</label>
          <span class="label-extras">optional · leave empty or use only [section] tags for an instrumental</span>
        </div>
        <div class="ui-prompt-box">
          <textarea
            id="ws-lyrics"
            class="prompt-text"
            placeholder={LYRICS_PLACEHOLDER}
            aria-describedby="ws-music-status"
            value={s.lyrics}
            use:autogrow={s.lyrics}
            oninput={(e) => draft.update('lyrics', e.currentTarget.value)}
          ></textarea>
          {@render tools()}
        </div>
      </div>
    {/if}
  </div>
  <div class="prompt-statusline" data-testid="prompt-statusline" data-state={view.music?.kind ?? 'none'}>
    <span id="ws-music-status" class="statusline-msg" role="status">{status}</span>
    <span class="word-count">{wordCount} words</span>
  </div>
{:else}
  {@const speech = kind === 'speech'}
  {@const key = AUDIO_TEXT_KEYS[kind]}
  <div class="field" data-testid="audio-compose" data-kind={kind}>
    <div class="label-row">
      <label class="ui-label" for="ws-prompt">{speech ? 'Text' : 'Prompt'}</label>
      <span class="label-extras">required</span>
    </div>
    <div class="ui-prompt-box">
      <textarea
        id="ws-prompt"
        class="prompt-text"
        placeholder={PLACEHOLDERS[kind]}
        aria-required="true"
        aria-describedby={speech ? 'ws-speech-help' : undefined}
        aria-invalid={speech && view.overLimit ? 'true' : undefined}
        value={s[key]}
        use:autogrow={s[key]}
        onkeydown={onKeydown}
        oninput={(e) => draft.update(key, e.currentTarget.value)}
      ></textarea>
      {@render tools()}
    </div>
  </div>
  <div class="prompt-statusline" data-testid="prompt-statusline">
    <span class="statusline-msg"></span>
    {#if speech}
      <span class="word-count" class:over={view.overLimit} data-testid="speech-count">{characters}</span>
    {:else}
      <span class="word-count">{wordCount} words</span>
    {/if}
  </div>
  {#if speech}<p id="ws-speech-help" class="ui-help">{PARTS_HELP}</p>{/if}
{/if}

<style>
  .text-pair { display: flex; flex-direction: column; gap: 8px; }
  /* The label sits left; the note about the box is pushed right and wraps rather than being cut. */
  .label-row { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; min-height: 22px; margin-bottom: 4px; }
  .label-extras { margin-left: auto; font-size: var(--text-meta); color: var(--color-text-muted); text-align: right; }
  .prompt-text { display: block; width: 100%; min-height: 88px; max-height: 180px; overflow-y: auto; resize: none; padding: 9px 10px; border: 0; background: transparent; font-size: var(--text-content); line-height: 1.5; color: var(--color-text-primary); }
  /* The focus ring belongs to the prompt box, not the textarea. */
  .prompt-text:focus, .prompt-text:focus-visible { outline: none; }
  .prompt-text::placeholder { color: var(--color-zinc-600); }
  /* The Style box is the smaller of the two: two lines, growing to four, then it scrolls. */
  .scene-text { min-height: 58px; max-height: 98px; }
  .tools-spacer { flex: 1; }
  /* The status line under the boxes: a fixed-height row, the message on the left and the count on the right. */
  .prompt-statusline { display: flex; align-items: center; gap: 8px; min-height: var(--spacing-control-sm); margin-top: -2px; padding: 0 4px; font-size: var(--text-meta); color: var(--color-text-muted); }
  .statusline-msg { flex: 1; min-width: 0; color: var(--color-text-secondary); }
  .word-count { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  .word-count.over { font-weight: 600; color: var(--color-warning); }
  /* Touch: 16px text, below which iOS zooms into a field as it focuses. */
  @media (pointer: coarse) {
    .prompt-text { font-size: 16px; }
  }
</style>
