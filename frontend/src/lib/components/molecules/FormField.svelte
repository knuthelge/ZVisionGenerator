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
    /** `row` puts the label in a left column and the control, reset and help text beside it. */
    layout?: 'stack' | 'row';
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
    layout = 'stack',
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

{#snippet labelText()}
  <Label for={htmlFor} {required}>{label}</Label>
  {#if changed}<span class="field-changed" title="Changed from the default"><span class="sr-only">(changed from the default)</span></span>{/if}
{/snippet}

{#snippet resetButton()}
  {#if onreset}
    <!-- The slot is always reserved so nothing shifts when the button appears. -->
    <button type="button" class="field-reset" class:field-reset-shown={changed} aria-label="Reset {label} to the default" title="Use the default" tabindex={changed ? 0 : -1} aria-hidden={changed ? undefined : 'true'} onclick={onreset}>
      <Icon name="reset" size={12} />
    </button>
  {/if}
{/snippet}

{#snippet feedback()}
  {#if feedbackText}
    <p
      id={feedbackId}
      class="ui-help {feedbackClass} {layout === 'row' ? 'field-row-help' : ''}"
      role={announceFeedback ? (error ? 'alert' : 'status') : undefined}
      aria-live={announceFeedback ? (error ? 'assertive' : 'polite') : undefined}
      aria-atomic={announceFeedback ? 'true' : undefined}
    >{feedbackText}</p>
  {/if}
{/snippet}

{#if layout === 'row'}
  <div class="field-row {extraClass}">
    <div class="field-row-label">{#if label}{@render labelText()}{/if}</div>
    <div class="field-row-control">
      <div class="min-w-0 flex-1">{@render children?.()}</div>
      {@render resetButton()}
    </div>
    {@render feedback()}
  </div>
{:else}
  <div class="flex min-w-0 flex-col gap-1.5 {extraClass}">
    {#if label}
      <div class="field-head">
        {@render labelText()}
        <span class="ml-auto">{@render resetButton()}</span>
      </div>
    {/if}
    {@render children?.()}
    {@render feedback()}
  </div>
{/if}

<style>
  .field-head { display: flex; align-items: center; gap: 6px; min-height: 20px; }
  .field-changed { width: 6px; height: 6px; flex-shrink: 0; border-radius: 50%; background: var(--color-primary-main); }
  .field-reset { display: inline-grid; flex-shrink: 0; place-items: center; width: 20px; height: 20px; border-radius: var(--radius-xs); color: var(--color-text-muted); visibility: hidden; }
  .field-reset-shown { visibility: visible; }
  .field-reset:hover { background: var(--color-bg-surface-hover); color: var(--color-primary-main); }

  /* Row layout: label column, then the control (up to 480px) with its reset, then help under the control. */
  .field-row { display: grid; grid-template-columns: 200px minmax(0, 480px); column-gap: 24px; row-gap: 4px; align-items: start; padding: 12px 0; border-top: 1px solid var(--color-border-subtle); }
  .field-row:first-child { border-top: 0; padding-top: 0; }
  .field-row:last-child { padding-bottom: 0; }
  .field-row-label { display: flex; align-items: center; gap: 6px; min-height: var(--spacing-control); }
  .field-row-control { display: flex; align-items: flex-start; gap: 6px; min-width: 0; }
  .field-row-control .field-reset { margin-top: 4px; }
  .field-row-help { grid-column: 2; padding-right: 26px; }
  @media (max-width: 639px) {
    .field-row { grid-template-columns: minmax(0, 1fr); }
    .field-row-help { grid-column: 1; }
  }
</style>
