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
    value = null,
    placeholder,
    helper,
    disabled = false,
    browseLabel = 'Browse',
    clearLabel = 'Clear',
    pickerKind = 'existing_file',
    pickerPurpose = 'path',
    onresolve,
    onvaluechange,
    onclear,
  }: Props = $props();

  let inputValue = $state('');
  let syncedValue = $state('');
  let editing = $state(false);
  let pending = $state(false);
  let error = $state<string | null>(null);
  let status = $state<string | null>(null);
  let statusTone = $state<'muted' | 'success' | 'warning' | 'error'>('muted');

  $effect(() => {
    const normalized = value ?? '';
    if (!editing && normalized !== inputValue) {
      syncedValue = normalized;
      inputValue = normalized;
    }
  });

  async function resolveCandidate(candidate: string): Promise<void> {
    const trimmed = candidate.trim();
    if (!trimmed) {
      error = 'Enter a path or browse for one first.';
      status = null;
      return;
    }

    pending = true;
    error = null;
    status = null;
    inputValue = candidate;
    try {
      const normalized = await onresolve(trimmed);
      syncedValue = normalized;
      inputValue = normalized;
      onvaluechange?.(normalized);
      status = 'Path loaded from the host machine.';
      statusTone = 'success';
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to resolve path.';
    } finally {
      pending = false;
    }
  }

  async function browse(): Promise<void> {
    pending = true;
    error = null;
    status = null;
    try {
      const result = await openPathPicker({
        kind: pickerKind,
        purpose: pickerPurpose,
        initial_path: inputValue || syncedValue || null,
      });

      if (result.status === 'selected' && result.path) {
        await resolveCandidate(result.path);
        return;
      }

      if (result.status === 'cancelled') {
        return;
      }

      status = result.message ?? (result.status === 'unsupported' ? 'Path picking is not supported on this host.' : 'Path picker failed.');
      statusTone = result.status === 'unsupported' ? 'warning' : 'error';
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to open the path picker.';
    } finally {
      pending = false;
    }
  }

  function clear(): void {
    editing = false;
    pending = false;
    error = null;
    status = null;
    syncedValue = '';
    inputValue = '';
    onvaluechange?.('');
    onclear?.();
  }
</script>

<FormField
  {label}
  for={id}
  {helper}
  {error}
  {status}
  {statusTone}
>
  <div class="flex flex-col gap-2">
    {#if name}
      <input type="hidden" {name} value={inputValue}>
    {/if}
    <div class="flex gap-2">
      <Input
        {id}
        value={inputValue}
        {placeholder}
        {disabled}
        class="rounded-md focus:border-primary-main focus:ring-4 focus:ring-primary-main"
        oninput={(event) => {
          editing = true;
          inputValue = (event.currentTarget as HTMLInputElement).value;
          status = null;
          onvaluechange?.(inputValue);
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
      />
      <Button type="button" size="sm" disabled={disabled || pending} onclick={() => void browse()}>
        {browseLabel}
      </Button>
      <Button type="button" size="sm" variant="ghost" disabled={disabled || (!syncedValue && !inputValue)} onclick={clear}>
        {clearLabel}
      </Button>
    </div>
  </div>
</FormField>
