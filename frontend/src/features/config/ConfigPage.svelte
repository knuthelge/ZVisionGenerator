<script lang="ts">
  import { onMount } from 'svelte';
  import { getConfig, updateConfig } from '$lib/api/config';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { AppConfig, WritableConfigField, WritableConfigValue } from '$lib/types';
  import { Button, Icon, Input, Select } from '$lib/components/atoms';
  import { Alert, FormField, KeyValueList, Panel, PathField, type KeyValueItem } from '$lib/components/molecules';
  import { AdminPageShell } from '$lib/components/organisms';

  type SelectOption = { value: string; label: string; disabled?: boolean };

  const FIELD_LABELS: Record<string, string> = {
    'ui.default_models.image': 'Default image model',
    'ui.default_models.video': 'Default video model',
    'generation.default_size': 'Base resolution',
    'ui.output_dir': 'Output folder',
    'prompt_enhancer.user_model': 'Prompt enhancer model',
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
    if (field.key === 'prompt_enhancer.user_model') {
      return `Hugging Face repo (owner/name, optionally @revision) or local folder of a chat LLM. Leave empty for the built-in default. Current model: ${effective}.`;
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

  /** Read-only folders and the Hugging Face token, which only the server's environment can change. */
  function locationItems(c: AppConfig): KeyValueItem[] {
    const tokenVar = c.ui.huggingface_token_env_var ?? 'HF_TOKEN';
    return [
      { label: 'Model cache', value: c.ui.model_cache_dir ?? '(runtime-only)', mono: true, hint: 'Set when the app starts; not editable here.' },
      { label: 'LoRAs folder', value: c.ui.loras_dir ?? '(runtime-only)', mono: true, hint: 'Inside the current data folder.' },
      c.ui.huggingface_token_configured
        ? { label: 'Hugging Face token', value: 'Available', tone: 'success', hint: `Read from ${tokenVar}.` }
        : { label: 'Hugging Face token', value: 'Not set', tone: 'muted', hint: 'Set HF_TOKEN before starting the app for gated model downloads.' },
    ];
  }

  function handleReset(): void {
    if (!config) return;
    formValues = valuesFromConfig(config);
    saveStatus = null;
  }
</script>

<AdminPageShell
  title="Config"
  description="Defaults and folders used across the app. Saved settings apply from the next generation."
  {loading}
  {error}
>
  {#snippet actions()}
    <Button type="reset" form="config-form">Discard changes</Button>
    <Button variant="primary" type="submit" form="config-form" loading={saving}>
      {#if !saving}<Icon name="check" size={14} />{/if}
      {saving ? 'Saving…' : 'Save'}
    </Button>
  {/snippet}
  {#if config}
    <form id="config-form" class="flex flex-col gap-4" onsubmit={handleSave} onreset={handleReset}>
      {#if saveStatus}
        <Alert tone={saveStatus.tone} live>{saveStatus.message}</Alert>
      {/if}

      <!-- Writable settings are rendered from the backend schema inventory. -->
      <Panel title="Settings" icon="list">
        <div class="grid grid-cols-1 gap-x-6 gap-y-4 md:grid-cols-2">
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
      </Panel>

      <Panel title="Folders and access" icon="folder">
        <KeyValueList items={locationItems(config)} />
      </Panel>
    </form>
  {/if}
</AdminPageShell>
