<script lang="ts">
  import { openPathPicker } from '$lib/api/promptFiles';
  import { Button, Input } from '$lib/components/atoms';
  import FormField from './FormField.svelte';

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
    pickerKind?: 'existing_file' | 'directory';
    pickerPurpose?: string;
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
    pickerKind = 'existing_file',
    pickerPurpose = 'path',
    onresolve,
    onvaluechange,
    onclear,
  }: Props = $props();

  let inputValue = $state(value ?? '');
  let syncedValue = $state(value ?? '');
  let editing = $state(false);
  let pending = $state(false);
  let pendingLabel = $state('Browsing for path');
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
    pendingLabel = 'Browsing for path';
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
    pendingLabel = 'Resolving path';
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
        error = err instanceof Error ? err.message : 'Failed to resolve path.';
      }
    } finally {
      if (currentRequestVersion === requestVersion) pending = false;
    }
  }

  async function browse(): Promise<void> {
    const currentRequestVersion = ++requestVersion;
    pending = true;
    pendingLabel = 'Browsing for path';
    error = null;
    requiredInvalidMessage = null;
    try {
      const result = await openPathPicker({
        kind: pickerKind,
        purpose: pickerPurpose,
        initial_path: syncedValue || null,
      });

      if (currentRequestVersion !== requestVersion) return;

      if (result.status === 'selected' && result.path) {
        await resolveCandidate(result.path, currentRequestVersion);
        return;
      }

      if (result.status === 'cancelled') {
        return;
      }

      if (result.status === 'unsupported') {
        status = result.message ?? 'Path picking is not supported on this host.';
        statusTone = 'warning';
      } else {
        error = result.message ?? 'Path picker failed.';
        status = null;
        statusTone = 'muted';
      }
    } catch (err) {
      if (currentRequestVersion === requestVersion) {
        error = err instanceof Error ? err.message : 'Failed to open the path picker.';
      }
    } finally {
      if (currentRequestVersion === requestVersion) pending = false;
    }
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
>
  <div class="flex flex-col gap-2">
    {#if name}
      <input bind:this={hiddenInput} type="hidden" {name} value={value ?? ''}>
    {/if}
    <div class="flex flex-col gap-2 sm:flex-row" aria-busy={pending ? 'true' : undefined}>
      <Input
        {id}
        value={inputValue}
        {placeholder}
        {disabled}
        {required}
        ariaDescribedby={hasFeedback ? feedbackId : undefined}
        ariaInvalid={Boolean(fieldError)}
        ariaBusy={pending}
        class="min-w-0 rounded-md focus:border-primary-main focus:ring-4 focus:ring-primary-main"
        oninput={(event) => {
          editing = true;
          inputValue = (event.currentTarget as HTMLInputElement).value;
          error = null;
          requiredInvalidMessage = null;
          status = null;
          statusTone = 'muted';
          publishValue(inputValue);
        }}
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
      <div class="flex flex-wrap gap-2 sm:contents">
        <Button type="button" size="sm" disabled={disabled || pending} loading={pending} onclick={() => void browse()}>
          {pending ? pendingLabel : browseLabel}
        </Button>
        <Button type="button" size="sm" variant="ghost" disabled={disabled || pending || (!syncedValue && !inputValue)} onclick={clear}>
          {clearLabel}
        </Button>
      </div>
    </div>
  </div>
</FormField>
