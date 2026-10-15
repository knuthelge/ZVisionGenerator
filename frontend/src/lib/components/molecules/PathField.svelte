<script lang="ts">
  import { errorText } from '$lib/api/client';
  import { Button, Icon } from '$lib/components/atoms';
  import type { BrowsePurposeName } from '$lib/types';
  import FormField from './FormField.svelte';
  import PathBrowser from './PathBrowser.svelte';
  import PathInput from './PathInput.svelte';

  interface Props {
    id: string;
    name?: string;
    label: string;
    value?: string | null;
    placeholder?: string;
    helper?: string;
    disabled?: boolean;
    required?: boolean;
    browseLabel?: string;
    clearLabel?: string;
    /** The value differs from the default (see FormField). */
    changed?: boolean;
    /** Clearing returns to the default: a reset button by the label replaces the Clear button. */
    resettable?: boolean;
    /** Field layout; see FormField. */
    layout?: 'stack' | 'row';
    /** What Browse and completion choose; the server knows the folder or file rules. */
    purpose: BrowsePurposeName;
    onresolve: (candidate: string) => Promise<string>;
    onvaluechange?: (value: string) => void;
    onclear?: () => void;
  }

  let {
    id,
    name,
    label,
    value = $bindable(null),
    placeholder,
    helper,
    disabled = false,
    required = false,
    browseLabel = 'Browse',
    clearLabel = 'Clear',
    changed = false,
    resettable = false,
    layout = 'stack',
    purpose,
    onresolve,
    onvaluechange,
    onclear,
  }: Props = $props();

  let inputValue = $state(value ?? '');
  let syncedValue = $state(value ?? '');
  let editing = $state(false);
  let pending = $state(false);
  let browsing = $state(false);
  let error = $state<string | null>(null);
  let requiredInvalidMessage = $state<string | null>(null);
  let status = $state<string | null>(null);
  let statusTone = $state<'muted' | 'success' | 'warning' | 'error'>('muted');
  let pendingExternalValue = $state<string | null>(null);
  let hiddenInput = $state<HTMLInputElement | null>(null);
  let lastObservedValue = value ?? '';
  let requestVersion = 0;

  function resetLocalState(): void {
    requestVersion += 1;
    editing = false;
    pending = false;
    pendingExternalValue = null;
    error = null;
    requiredInvalidMessage = null;
    status = null;
    statusTone = 'muted';
    syncedValue = '';
    inputValue = '';
    if (hiddenInput) hiddenInput.value = '';
  }

  function applyExternalValue(nextValue: string): void {
    error = null;
    requiredInvalidMessage = null;
    status = null;
    statusTone = 'muted';
    syncedValue = nextValue;
    inputValue = nextValue;
    if (hiddenInput) hiddenInput.value = nextValue;
  }

  function publishValue(nextValue: string): void {
    lastObservedValue = nextValue;
    pendingExternalValue = null;
    value = nextValue;
    if (hiddenInput) hiddenInput.value = nextValue;
    onvaluechange?.(nextValue);
  }

  $effect(() => {
    const normalized = value ?? '';
    if (normalized !== lastObservedValue) {
      lastObservedValue = normalized;
      if (normalized === '') {
        resetLocalState();
      } else if (editing) {
        pendingExternalValue = normalized;
      } else {
        applyExternalValue(normalized);
      }
    }

    if (!editing && pendingExternalValue !== null) {
      const nextValue = pendingExternalValue;
      pendingExternalValue = null;
      applyExternalValue(nextValue);
    }
  });

  // Native form.reset() restores an input's default value after the reset event.
  // Re-apply our controlled value once that native reset has completed so the
  // named field cannot diverge from the visible field or its parent state.
  $effect(() => {
    const form = hiddenInput?.form;
    if (!form) return;

    const reconcileAfterReset = () => {
      queueMicrotask(() => {
        if (hiddenInput) hiddenInput.value = inputValue;
      });
    };

    form.addEventListener('reset', reconcileAfterReset);
    return () => form.removeEventListener('reset', reconcileAfterReset);
  });

  async function resolveCandidate(candidate: string, existingRequestVersion?: number): Promise<void> {
    const trimmed = candidate.trim();
    if (!trimmed) {
      error = 'Enter a path or browse for one first.';
      status = null;
      return;
    }

    const currentRequestVersion = existingRequestVersion ?? ++requestVersion;
    pending = true;
    error = null;
    requiredInvalidMessage = null;
    status = null;
    inputValue = candidate;
    try {
      const normalized = await onresolve(trimmed);
      if (currentRequestVersion !== requestVersion) return;
      syncedValue = normalized;
      inputValue = normalized;
      requiredInvalidMessage = null;
      publishValue(normalized);
      status = 'Path loaded from the host machine.';
      statusTone = 'success';
    } catch (err) {
      if (currentRequestVersion === requestVersion) {
        error = errorText(err, 'Failed to resolve path.');
      }
    } finally {
      if (currentRequestVersion === requestVersion) pending = false;
    }
  }

  function browse(): void {
    error = null;
    requiredInvalidMessage = null;
    browsing = true;
  }

  function clear(): void {
    resetLocalState();
    publishValue('');
    onclear?.();
  }

  const feedbackId = $derived(`${id}-feedback`);
  const fieldError = $derived(error || requiredInvalidMessage);
  const hasFeedback = $derived(Boolean(fieldError || status || helper));
</script>

<FormField
  {label}
  for={id}
  {helper}
  error={fieldError}
  {status}
  {statusTone}
  {required}
  {feedbackId}
  announceFeedback
  {changed}
  onreset={resettable ? clear : undefined}
  {layout}
>
  <div class="flex flex-col gap-2">
    {#if name}
      <input bind:this={hiddenInput} type="hidden" {name} value={value ?? ''}>
    {/if}
    <div class="flex gap-1.5" aria-busy={pending ? 'true' : undefined}>
      <PathInput
        {id}
        {purpose}
        value={inputValue}
        {placeholder}
        {disabled}
        {required}
        ariaDescribedby={hasFeedback ? feedbackId : undefined}
        ariaInvalid={Boolean(fieldError)}
        ariaBusy={pending}
        class="min-w-0 flex-1"
        oninput={(text) => {
          editing = true;
          inputValue = text;
          error = null;
          requiredInvalidMessage = null;
          status = null;
          statusTone = 'muted';
          publishValue(inputValue);
        }}
        onpick={(path) => void resolveCandidate(path)}
        onblur={() => {
          editing = false;
        }}
        onkeydown={(event: KeyboardEvent) => {
          if (event.key === 'Enter') {
            event.preventDefault();
            void resolveCandidate(inputValue);
          }
        }}
        oninvalid={(event) => {
          const input = event.currentTarget as HTMLInputElement;
          if (required && !input.value) {
            requiredInvalidMessage = input.validationMessage || 'Enter or select a path.';
          }
        }}
      />
      <div class="contents">
        <Button type="button" icon aria-label={browseLabel} title={browseLabel} disabled={disabled || pending} loading={pending} onclick={browse}>
          <Icon name="folder" size={14} />
        </Button>
        {#if !resettable}
          <Button type="button" variant="quiet" icon aria-label={clearLabel} title={clearLabel} disabled={disabled || pending || (!syncedValue && !inputValue)} onclick={clear}>
            <Icon name="close" size={14} />
          </Button>
        {/if}
      </div>
    </div>
  </div>
</FormField>

<PathBrowser bind:open={browsing} {purpose} initialPath={inputValue || syncedValue || null} onchoose={(path) => void resolveCandidate(path)} />
