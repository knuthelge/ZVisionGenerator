<script lang="ts">
  import type { Snippet } from 'svelte';
  import Button from '../atoms/Button.svelte';
  import Modal from './Modal.svelte';

  interface Props {
    open?: boolean;
    /** The question the user answers, e.g. "Remove 3 queued jobs?". */
    question: string;
    /** One line of extra information under the question. */
    info?: string;
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
        // A held key must not approve the action; Enter on a focused button activates that button natively.
        if (e.repeat) e.preventDefault();
        else if (!(e.target instanceof HTMLButtonElement)) {
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
    <p id={questionId} class="text-[15px] font-semibold text-zinc-100">{question}</p>
    {#if info}
      <p id={infoId} class="mt-1.5 text-sm text-zinc-400">{info}</p>
    {/if}
    {#if children}
      <div class="mt-3 space-y-2 text-sm text-zinc-300">{@render children()}</div>
    {/if}
  </div>
  {#snippet footer()}
    <Button variant="ghost" data-action="cancel" disabled={pending} onclick={cancel}>{cancelLabel}</Button>
    <Button variant={danger ? 'danger' : 'primary'} data-action="confirm" loading={pending} onclick={confirm}>{confirmLabel}</Button>
  {/snippet}
</Modal>
