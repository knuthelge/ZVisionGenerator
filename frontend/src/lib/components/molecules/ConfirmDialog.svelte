<script lang="ts" module>
  /** One block of a confirmation that lists what it asks about: a name, its size and the terms that come with it. */
  export interface ConfirmPart {
    label: string;
    /** The size to show on the right; `null` reads "size unknown". */
    size: string | null;
    /** One line per licence record under the part: text followed by its links. */
    lines: { text: string; links: { label: string; href: string; ariaLabel: string }[] }[];
  }
</script>

<script lang="ts">
  import type { Snippet } from 'svelte';
  import Button from '../atoms/Button.svelte';
  import Modal from './Modal.svelte';
  import { layout } from '$lib/state/layoutMode.svelte';

  interface Props {
    open?: boolean;
    /** The question the user answers, e.g. "Remove 3 queued jobs?". */
    question: string;
    /** One line of extra information under the question (under the parts, when there are any). */
    info?: string;
    /** `warning` colours the information line as a caution. */
    infoTone?: 'warning';
    /** What the question is about, one block each (also for a single one), between the question and the information. */
    parts?: ConfirmPart[];
    /** Name of the action, e.g. "Clear queue" or "Delete". */
    confirmLabel: string;
    cancelLabel?: string;
    /** Style the confirm button as destructive. */
    danger?: boolean;
    /** Disable both buttons while the confirmed action runs. */
    pending?: boolean;
    onconfirm: () => void;
    oncancel?: () => void;
    /** Optional details shown under the information line. */
    children?: Snippet;
  }

  let {
    open = $bindable(false),
    question,
    info,
    infoTone,
    parts,
    confirmLabel,
    cancelLabel = 'Cancel',
    danger = true,
    pending = false,
    onconfirm,
    oncancel,
    children,
  }: Props = $props();

  const uid = $props.id();
  const questionId = `${uid}-question`;
  const infoId = `${uid}-info`;
  // A list of parts is a sheet with full-width buttons on a phone.
  const fullWidth = $derived(layout.compact && Boolean(parts?.length));
  const hasTerms = $derived(Boolean(parts?.some((part) => part.lines.length > 0)));

  function cancel(): void {
    if (pending) return;
    open = false;
    oncancel?.();
  }

  function confirm(): void {
    if (pending) return;
    onconfirm();
  }

  // Keys belong to the dialog while it is open: Enter confirms, Esc cancels, and nothing reaches the page or a
  // viewer underneath. Tab passes through to Modal's focus trap.
  $effect(() => {
    if (!open) return;
    function handleKeydown(e: KeyboardEvent): void {
      if (e.key === 'Tab') return;
      e.stopPropagation();
      if (e.key === 'Escape') {
        e.preventDefault();
        if (!e.repeat) cancel();
      } else if (e.key === 'Enter') {
        // A held key must not approve the action; Enter on a focused button or link activates that control natively.
        // A download list is approved with plain Enter: ⌘↵ or Ctrl+↵ (Generate's chord, pressed twice) never confirms it.
        if (e.repeat || (parts && parts.length > 0 && (e.metaKey || e.ctrlKey))) e.preventDefault();
        else if (!(e.target instanceof HTMLButtonElement) && !(e.target instanceof HTMLAnchorElement)) {
          e.preventDefault();
          confirm();
        }
      }
    }
    window.addEventListener('keydown', handleKeydown, true);
    return () => window.removeEventListener('keydown', handleKeydown, true);
  });
</script>

<Modal
  bind:open
  size="sm"
  role="alertdialog"
  labelledby={questionId}
  describedby={info ? infoId : undefined}
  elevated
  onclose={() => oncancel?.()}
>
  <div data-testid="confirm-dialog">
    <p id={questionId} class="font-heading text-content font-extrabold text-text-primary">{question}</p>
    {#if parts && parts.length > 0}
      {#if hasTerms}
        <p class="mt-1.5 text-content text-text-secondary">By downloading you agree to the terms shown under each model.</p>
      {/if}
      <ul class="mt-1.5" data-testid="confirm-parts">
        {#each parts as part, index (index)}
          <li class="border-b border-border-subtle py-2" data-testid="confirm-part">
            <div class="flex justify-between gap-3 text-content">
              <span class="min-w-0 font-semibold text-text-primary [overflow-wrap:anywhere]">{part.label}</span>
              <span class="shrink-0 font-mono {part.size === null ? 'text-text-muted' : 'text-text-primary'}">{part.size ?? 'size unknown'}</span>
            </div>
            {#each part.lines as line, lineIndex (lineIndex)}
              <p class="licence-line mt-1 border-l-2 border-border-strong pl-2.5 text-ui leading-normal text-text-secondary [overflow-wrap:anywhere]">
                {line.text}{#each line.links as link, linkIndex (linkIndex)}{" · "}<a class="licence-link" href={link.href} target="_blank" rel="noopener" aria-label={link.ariaLabel}>{link.label}</a>{/each}
              </p>
            {/each}
          </li>
        {/each}
      </ul>
    {/if}
    {#if info}
      <p id={infoId} class="{parts && parts.length > 0 ? 'mt-3' : 'mt-1.5'} text-content [overflow-wrap:anywhere] {infoTone === 'warning' ? 'text-warning' : 'text-text-secondary'}">
        {#if infoTone === 'warning'}<span aria-hidden="true">⚠&nbsp;</span>{/if}{info}
      </p>
    {/if}
    {#if children}
      <div class="mt-3 space-y-2 text-ui text-text-secondary">{@render children()}</div>
    {/if}
  </div>
  {#snippet footer()}
    <Button class={fullWidth ? 'min-h-11 flex-1' : ''} data-action="cancel" disabled={pending} onclick={cancel}>{cancelLabel}</Button>
    <Button class={fullWidth ? 'min-h-11 flex-1' : ''} variant={danger ? 'danger' : 'primary'} data-action="confirm" loading={pending} onclick={confirm}>{confirmLabel}</Button>
  {/snippet}
</Modal>

<style>
  .licence-link {
    color: var(--color-primary-main);
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  @media (pointer: coarse) {
    .licence-link {
      display: inline-block;
      padding-block: 6px;
    }
  }
</style>
