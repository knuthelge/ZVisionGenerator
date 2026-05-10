<script lang="ts">
  import { onMount } from 'svelte';
  import { getConfig, updateConfig } from '$lib/api/config';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { AppConfig, WritableConfigField, WritableConfigValue } from '$lib/types';
  import { Button, Input, Select } from '$lib/components/atoms';
  import { FormField, PathField } from '$lib/components/molecules';
  import { AdminPageShell } from '$lib/components/organisms';

  type SelectOption = { value: string; label: string; disabled?: boolean };

  const FIELD_LABELS: Record<string, string> = {
    'ui.default_models.image': 'Default Image Model',
    'ui.default_models.video': 'Default Video Model',
    'generation.default_size': 'Base Resolution',
    'ui.output_dir': 'Output Directory',
  };

  let config = $state<AppConfig | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let saveStatus = $state<{ tone: 'success' | 'error'; message: string } | null>(null);
  let saving = $state(false);
  let formValues = $state<Record<string, string>>({});

  onMount(async () => {
    try {
      applyConfig(await getConfig());
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load configuration';
    } finally {
      loading = false;
    }
  });

  function applyConfig(nextConfig: AppConfig): void {
    config = nextConfig;
    formValues = valuesFromConfig(nextConfig);
  }

  function valuesFromConfig(nextConfig: AppConfig): Record<string, string> {
    return Object.fromEntries(nextConfig.writable_config.fields.map((field) => [field.key, fieldValue(field)]));
  }

  function fieldId(field: WritableConfigField): string {
    return `config-${field.key.replace(/[^a-z0-9]+/gi, '-')}`;
  }

  function fieldLabel(field: WritableConfigField): string {
    return FIELD_LABELS[field.key] ?? humanizeFieldKey(field.key);
  }

  function fieldHelper(field: WritableConfigField): string {
    const effective = readableEffectiveValue(field);
    if (field.key === 'ui.default_models.image') {
      return `Leave empty to use the first available image model. Current model: ${effective}.`;
    }
    if (field.key === 'ui.default_models.video') {
      return `Leave empty to use the first available video model. Current model: ${effective}.`;
    }
    if (field.key === 'generation.default_size') {
      return `Leave empty to use the app default. Current base resolution: ${effective}.`;
    }
    if (field.key === 'ui.output_dir') {
      return `Leave empty to use the default output folder. Current folder: ${effective}.`;
    }
    if (field.clearable) {
      return `Leave empty to use the default. Current value: ${effective}.`;
    }
    return `Current value: ${effective}.`;
  }

  function humanizeFieldKey(key: string): string {
    const leaf = key.split('.').at(-1) ?? key;
    return leaf
      .split(/[_-]+/)
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(' ');
  }

  function readableEffectiveValue(field: WritableConfigField): string {
    const value = field.effective_value;
    if (value === null || value === undefined || value === '') return 'not set';
    if (field.key === 'generation.default_size' && config) {
      const option = config.ui.image_size_labels?.find((candidate) => candidate.value === String(value));
      if (option) return option.label;
    }
    return String(value);
  }

  function fieldValue(field: WritableConfigField): string {
    return field.value === null || field.value === undefined ? '' : String(field.value);
  }

  function formValue(field: WritableConfigField): string {
    return formValues[field.key] ?? fieldValue(field);
  }

  function selectOptionsForField(field: WritableConfigField): SelectOption[] | null {
    if (!config) return null;
    const emptyLabel = field.clearable ? 'Use default' : 'Select a value';
    if (field.key === 'ui.default_models.image') {
      return [{ value: '', label: emptyLabel }, ...(config.ui.image_model_options ?? []).map((value) => ({ value, label: value }))];
    }
    if (field.key === 'ui.default_models.video') {
      return [{ value: '', label: emptyLabel }, ...(config.ui.video_model_options ?? []).map((value) => ({ value, label: value }))];
    }
    if (field.key === 'generation.default_size') {
      return [{ value: '', label: emptyLabel }, ...(config.ui.image_size_labels ?? []).map((option) => ({ value: option.value, label: option.label }))];
    }
    if (field.type === 'boolean') {
      return [{ value: '', label: emptyLabel }, { value: 'true', label: 'Enabled' }, { value: 'false', label: 'Disabled' }];
    }
    return null;
  }

  function coercePatchValue(field: WritableConfigField, text: string): WritableConfigValue | undefined {
    const trimmed = text.trim();
    if (trimmed === '') {
      if (field.clearable && field.empty_string === 'clear') return null;
      if (field.empty_string === 'reject' || field.empty_string === 'coerce') return '';
      return undefined;
    }
    if (field.type === 'number') {
      const value = Number(trimmed);
      if (!Number.isFinite(value)) throw new Error(`${fieldLabel(field)} must be a number.`);
      return value;
    }
    if (field.type === 'boolean') {
      if (trimmed === 'true') return true;
      if (trimmed === 'false') return false;
      throw new Error(`${fieldLabel(field)} must be true or false.`);
    }
    return trimmed;
  }

  async function handleSave(e: Event): Promise<void> {
    e.preventDefault();
    if (!config) return;
    saving = true;
    saveStatus = null;
    try {
      const patch: Record<string, WritableConfigValue> = {};
      for (const field of config.writable_config.fields) {
        const key = field.key;
        const value = coercePatchValue(field, formValue(field));
        if (value !== undefined) patch[key] = value;
      }
      const nextConfig = await updateConfig(patch);
      saveStatus = { tone: 'success', message: 'Configuration saved successfully.' };
      applyConfig(nextConfig);
      addToast('Settings saved', 'success');
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to save configuration';
      saveStatus = {
        tone: 'error',
        message: msg
      };
      addToast('Save failed: ' + msg, 'error');
    } finally {
      saving = false;
    }
  }

  function handleReset(): void {
    if (!config) return;
    formValues = valuesFromConfig(config);
    saveStatus = null;
  }
</script>

<AdminPageShell
  title="System Configuration"
  description="Manage persistent settings, paths, and default models to be applied across the application."
  {loading}
  {error}
>
  {#if config}
    <form class="space-y-8" onsubmit={handleSave} onreset={handleReset}>
      {#if saveStatus}
        <div
          class="rounded-lg border px-4 py-3 text-sm
            {saveStatus.tone === 'success'
              ? 'border-teal-500/30 bg-teal-500/10 text-teal-100'
              : 'border-red-500/30 bg-red-500/10 text-red-100'}"
        >
          {saveStatus.message}
        </div>
      {/if}

      <!-- Writable settings are rendered from the backend schema inventory. -->
      <section class="admin-section">
        <div class="admin-section-header">
          <svg class="w-5 h-5 text-teal-500 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
          </svg>
          <h2 class="admin-section-title">Writable Settings</h2>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          {#each config.writable_config.fields as field (field.key)}
            {@const options = selectOptionsForField(field)}
            {#if field.key === 'ui.output_dir'}
              <PathField
                id={fieldId(field)}
                name={field.key}
                label={fieldLabel(field)}
                value={formValue(field)}
                helper={fieldHelper(field)}
                pickerKind="directory"
                pickerPurpose="output_directory"
                onresolve={async (candidate) => candidate}
                onvaluechange={(value) => {
                  formValues[field.key] = value;
                }}
                onclear={() => {
                  formValues[field.key] = '';
                }}
              />
            {:else if options}
              <FormField label={fieldLabel(field)} for={fieldId(field)} helper={fieldHelper(field)}>
                <Select
                  id={fieldId(field)}
                  name={field.key}
                  bind:value={formValues[field.key]}
                  {options}
                />
              </FormField>
            {:else}
              <FormField label={fieldLabel(field)} for={fieldId(field)} helper={fieldHelper(field)}>
                <Input
                  id={fieldId(field)}
                  name={field.key}
                  type={field.type === 'number' ? 'number' : 'text'}
                  bind:value={formValues[field.key]}
                />
              </FormField>
            {/if}
          {/each}
        </div>
      </section>

      <!-- Directories & Storage -->
      <section class="admin-section">
        <div class="admin-section-header">
          <svg class="w-5 h-5 text-teal-500 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"></path>
          </svg>
          <h2 class="admin-section-title">Directories &amp; Storage</h2>
        </div>

        <div class="space-y-6">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="space-y-2">
              <div class="block text-sm font-medium text-zinc-300">Models Cache Directory</div>
              <div class="rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 font-mono break-all">
                {config.ui.model_cache_dir ?? '(runtime-only)'}
              </div>
              <p class="text-xs text-zinc-500">Runtime-only. Not writable from the Web UI.</p>
            </div>
            <div class="space-y-2">
              <div class="block text-sm font-medium text-zinc-300">LoRAs Directory</div>
              <div class="rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 font-mono break-all">
                {config.ui.loras_dir ?? '(runtime-only)'}
              </div>
              <p class="text-xs text-zinc-500">Derived from the current data directory.</p>
            </div>
          </div>
        </div>
      </section>

      <!-- API Keys & Authentication -->
      <section class="admin-section">
        <div class="admin-section-header">
          <svg class="w-5 h-5 text-teal-500 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"></path>
          </svg>
          <h2 class="admin-section-title">API Keys &amp; Authentication</h2>
        </div>

        <div class="space-y-2">
          <div class="block text-sm font-medium text-zinc-300">HuggingFace Token</div>
          <div class="rounded-md border border-zinc-800 bg-zinc-900 px-3 py-3 text-sm text-zinc-200">
            <p class="font-medium text-zinc-100">
              {config.ui.huggingface_token_configured ? 'Available at runtime' : 'Not configured for this process'}
            </p>
            <p class="mt-1 text-xs text-zinc-500">
              {#if config.ui.huggingface_token_configured}
                Read from <span class="font-mono text-zinc-300">{config.ui.huggingface_token_env_var ?? 'HF_TOKEN'}</span>.
              {:else}
                Set <span class="font-mono text-zinc-300">HF_TOKEN</span> before starting the app for gated model downloads.
              {/if}
            </p>
          </div>
        </div>
      </section>

      <!-- Actions -->
      <div class="flex items-center justify-end gap-4 py-8">
        <Button variant="secondary" type="reset">Discard Changes</Button>
        <Button variant="primary" type="submit" disabled={saving} loading={saving}>
          {#if !saving}
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
            </svg>
          {/if}
          {saving ? 'Saving...' : 'Save Configuration'}
        </Button>
      </div>
    </form>
  {/if}
</AdminPageShell>
