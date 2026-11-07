<script lang="ts">
  import type { Snippet } from 'svelte';
  import { autogrow } from '$lib/actions/autogrow';
  import { PromptTools, TextBoxHeader } from '$lib/components/molecules';
  import { EMPTY_REWRITES, editRewrite, rewriteStepper, rewriteUsed, scratchRewrite, stepRewrite } from '$lib/state/audioRewrites';
  import type { AudioRewrites } from '$lib/types';

  interface Props {
    /** The box's own text. */
    value: string;
    /** The box has an Enhanced tab. Off (Speech without an enhancer), the box is the own text alone and the props about the Enhanced text are not read. */
    enhanceable?: boolean;
    /** The box's Enhanced text and versions (`view.rewrites`), and whether it is out of date. */
    rewrites?: AudioRewrites;
    stale?: boolean;
    /** The own text's textarea id; the Enhanced textarea and the tabs take theirs from `enhancedId`. */
    id: string;
    enhancedId?: string;
    /** The own tab's name ("Prompt", "Style", "Text"). */
    label: string;
    /** The second tab's name ("Enhanced"; Speech's is "Written"). */
    enhancedLabel?: string;
    /** The Enhanced textarea, for a tool that inserts at its caret. */
    enhancedTextarea?: HTMLTextAreaElement | null;
    placeholder: string;
    /** The box is the smaller of a pair (Music's Style over its Lyrics): two lines, growing to four. */
    compact?: boolean;
    /** The own text's textarea, for a tool that inserts at its caret. */
    textarea?: HTMLTextAreaElement | null;
    /** The text must be filled in, and is over its limit. */
    required?: boolean;
    invalid?: boolean;
    /** The tab shown. */
    tab?: 'mine' | 'enhanced';
    /** Enhance is running; its text streams into the Enhanced tab. */
    enhancing?: boolean;
    streamingText?: string;
    /** Progress of the run ("Loading enhancer…"), shown as the placeholder of the Enhanced text until the first words stream in. */
    statusText?: string | null;
    describedby?: string;
    /** The tools at the left of the row (Enhance, Insert, Script helper). */
    tools: Snippet;
    /** The tools at the right of the row (Recent prompts). */
    aside?: Snippet;
    /** Clear the box; given only while it holds something and no helper is running in it. */
    onclear?: () => void;
    /** What Clear empties, in the words of its name ("style": "Clear the style"). */
    clearWhat?: string;
    oninput: (text: string) => void;
    onrewrites?: (rewrites: AudioRewrites) => void;
    /** Stop the running Enhance. */
    onstop?: () => void;
    onkeydown?: (event: KeyboardEvent) => void;
  }

  let {
    value,
    enhanceable = true,
    rewrites = EMPTY_REWRITES,
    stale = false,
    id,
    enhancedId = `${id}-enhanced`,
    label,
    enhancedLabel = 'Enhanced',
    enhancedTextarea = $bindable(null),
    placeholder,
    compact = false,
    textarea = $bindable(null),
    required = false,
    invalid = false,
    tab = $bindable('mine'),
    enhancing = false,
    streamingText = '',
    statusText = null,
    describedby,
    tools,
    aside,
    onclear,
    clearWhat = label.toLowerCase(),
    oninput,
    onrewrites = () => {},
    onstop,
    onkeydown,
  }: Props = $props();

  const showingEnhanced = $derived(enhanceable && tab === 'enhanced');
  const enhancedValue = $derived(enhancing ? streamingText : rewrites.text);
  const used = $derived(enhanceable && rewriteUsed(rewrites));
  const stepper = $derived(enhanceable ? rewriteStepper(rewrites, enhancing) : null);
  const tabIds = $derived({ mine: `${id}-tab`, enhanced: `${enhancedId}-tab` });
</script>

<div class="ui-prompt-box">
  {#if enhanceable}
    <TextBoxHeader
      bind:view={tab}
      {label}
      {enhancedLabel}
      mineId={tabIds.mine}
      enhancedId={tabIds.enhanced}
      mineControls={id}
      enhancedControls={enhancedId}
      {used}
      {stale}
      running={enhancing}
      {stepper}
      onstep={(index) => onrewrites(stepRewrite(rewrites, index))}
      onscratch={() => onrewrites(scratchRewrite(rewrites))}
      {onstop}
    />
  {:else}
    <!-- A box with one text has a heading where the tabs would be; the textarea carries the name. -->
    <TextBoxHeader heading={label} mineId={tabIds.mine} enhancedId={tabIds.enhanced} mineControls={id} enhancedControls={enhancedId} />
  {/if}
  <label for={id} class="sr-only">{label}</label>
  <textarea
    {id}
    class="prompt-text"
    class:compact
    data-unsent={used && !showingEnhanced ? '' : undefined}
    role={enhanceable ? 'tabpanel' : undefined}
    aria-labelledby={enhanceable ? tabIds.mine : undefined}
    {placeholder}
    aria-required={required ? 'true' : undefined}
    aria-invalid={invalid ? 'true' : undefined}
    aria-describedby={describedby}
    hidden={showingEnhanced}
    bind:this={textarea}
    {value}
    use:autogrow={[value, showingEnhanced]}
    {onkeydown}
    oninput={(e) => oninput(e.currentTarget.value)}
  ></textarea>
  {#if enhanceable}
    <textarea
      id={enhancedId}
      class="prompt-text"
      class:compact
      role="tabpanel"
      aria-labelledby={tabIds.enhanced}
      hidden={!showingEnhanced}
      placeholder={enhancing ? (statusText || 'Enhancing…') : `No ${enhancedLabel.toLowerCase()} text. Generate sends your ${label}.`}
      bind:this={enhancedTextarea}
      readonly={enhancing}
      aria-busy={enhancing}
      value={enhancedValue}
      use:autogrow={[enhancedValue, showingEnhanced]}
      oninput={(e) => onrewrites(editRewrite(rewrites, e.currentTarget.value))}
    ></textarea>
  {/if}
  <PromptTools end={aside} {onclear} {clearWhat}>{@render tools()}</PromptTools>
</div>

<style>
  .prompt-text { display: block; width: 100%; min-height: 88px; max-height: 180px; overflow-y: auto; resize: none; padding: 9px 10px; border: 0; background: transparent; font-size: var(--text-content); line-height: 1.5; color: var(--color-text-primary); }
  .prompt-text.compact { min-height: 58px; max-height: 98px; }
  .prompt-text[hidden] { display: none; }
  /* Your text steps back to secondary while the Enhanced text is what Generate sends. */
  .prompt-text[data-unsent] { color: var(--color-text-secondary); }
  /* The focus ring belongs to the prompt box, not the textarea. */
  .prompt-text:focus, .prompt-text:focus-visible { outline: none; }
  .prompt-text::placeholder { color: var(--color-zinc-600); }
  /* Touch: 16px text, below which iOS zooms into a field as it focuses. */
  @media (pointer: coarse) {
    .prompt-text { font-size: 16px; }
  }
</style>
