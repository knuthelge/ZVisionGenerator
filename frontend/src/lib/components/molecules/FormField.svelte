<script lang="ts">
  import type { Snippet } from 'svelte';
  import Icon from '../atoms/Icon.svelte';
  import Label from '../atoms/Label.svelte';

  interface Props {
    label?: string;
    for?: string;
    required?: boolean;
    helper?: string;
    error?: string | null;
    status?: string | null;
    statusTone?: 'muted' | 'success' | 'warning' | 'error';
    feedbackId?: string;
    announceFeedback?: boolean;
    /** The value differs from the default: shows a dot after the label. */
    changed?: boolean;
    /** Return to the default; shown as a reset button while `changed`. */
    onreset?: () => void;
    class?: string;
    children?: Snippet;
  }

  let {
    label,
    for: htmlFor,
    required = false,
    helper,
    error = null,
    status = null,
    statusTone = 'muted',
    feedbackId,
    announceFeedback = false,
    changed = false,
    onreset,
    class: extraClass = '',
    children
  }: Props = $props();

  const statusClass = $derived(
    statusTone === 'success'
      ? 'ui-help-success'
      : statusTone === 'warning'
        ? 'ui-help-warning'
        : statusTone === 'error'
          ? 'ui-help-error'
          : ''
  );
  const feedbackText = $derived(error || status || helper || null);
  const feedbackClass = $derived(error ? 'ui-help-error' : status ? statusClass : '');
</script>

<div class="flex min-w-0 flex-col gap-1.5 {extraClass}">
  {#if label}
    <div class="field-head">
      <Label for={htmlFor} {required}>{label}</Label>
      {#if changed}<span class="field-changed" title="Changed from the default"><span class="sr-only">(changed from the default)</span></span>{/if}
      {#if onreset}
        <!-- The slot is always reserved so the row never shifts when the button appears. -->
        <button type="button" class="field-reset" class:field-reset-shown={changed} aria-label="Reset {label} to the default" title="Use the default" tabindex={changed ? 0 : -1} aria-hidden={changed ? undefined : 'true'} onclick={onreset}>
          <Icon name="reset" size={12} />
        </button>
      {/if}
    </div>
  {/if}
  {@render children?.()}
  {#if feedbackText}
    <p
      id={feedbackId}
      class="ui-help {feedbackClass}"
      role={announceFeedback ? (error ? 'alert' : 'status') : undefined}
      aria-live={announceFeedback ? (error ? 'assertive' : 'polite') : undefined}
      aria-atomic={announceFeedback ? 'true' : undefined}
    >{feedbackText}</p>
  {/if}
</div>

<style>
  .field-head { display: flex; align-items: center; gap: 6px; min-height: 20px; }
  .field-changed { width: 6px; height: 6px; flex-shrink: 0; border-radius: 50%; background: var(--color-primary-main); }
  .field-reset { display: inline-grid; place-items: center; width: 20px; height: 20px; margin-left: auto; border-radius: var(--radius-xs); color: var(--color-text-muted); visibility: hidden; }
  .field-reset-shown { visibility: visible; }
  .field-reset:hover { background: var(--color-bg-surface-hover); color: var(--color-primary-main); }
</style>
