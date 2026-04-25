<script lang="ts">
  import { onMount } from 'svelte';
  import { getConfig } from '$lib/api/config';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { AppConfig } from '$lib/types';
  import { Button, Input, Select } from '$lib/components/atoms';
  import { FormField, PathField } from '$lib/components/molecules';
  import { AdminPageShell } from '$lib/components/organisms';

  let config = $state<AppConfig | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let saveStatus = $state<{ tone: 'success' | 'error'; message: string } | null>(null);
  let saving = $state(false);

  onMount(async () => {
    try {
      config = await getConfig();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load configuration';
    } finally {
      loading = false;
    }
  });

  async function handleSave(e: Event): Promise<void> {
    e.preventDefault();
    if (!config) return;
    saving = true;
    saveStatus = null;
    try {
      const form = e.currentTarget as HTMLFormElement;
      const fd = new FormData(form);
      // Build flat dot-notation payload matching what _persist_web_config expects
      const patch: Record<string, string> = {};
      for (const key of ['ui.default_models.image', 'ui.default_models.video', 'generation.default_size', 'ui.output_dir']) {
        const val = fd.get(key);
        if (val !== null && val !== '') {
          patch[key] = val as string;
        }
      }
      const resp = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch)
      });
      if (!resp.ok) {
        const detail = await resp.json().then((d) => d.detail ?? resp.statusText).catch(() => resp.statusText);
        throw new Error(typeof detail === 'string' ? detail : `Save failed: ${resp.statusText}`);
      }
      saveStatus = { tone: 'success', message: 'Configuration saved successfully.' };
      config = await resp.json() as AppConfig;
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
</script>

<AdminPageShell
  title="System Configuration"
  description="Manage persistent settings, paths, and default models to be applied across the application."
  {loading}
  {error}
>
  {#if config}
    <form class="space-y-8" onsubmit={handleSave}>
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

      <!-- Model Defaults -->
      <section class="admin-section">
        <div class="admin-section-header">
          <svg class="w-5 h-5 text-teal-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
          </svg>
          <h2 class="admin-section-title">Model Defaults</h2>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <FormField 
            label="Default Image Model" 
            for="config-default-image-model"
            helper="The model pre-selected when starting a new workspace."
          >
            <Select
              id="config-default-image-model"
              name="ui.default_models.image"
              value={config.ui.default_models?.image ?? ''}
              options={
                (config.ui.image_model_options ?? []).map(opt => ({ 
                  value: opt, 
                  label: opt 
                })) || [{ value: '', label: 'No image models available' }]
              }
            />
          </FormField>

          <FormField 
            label="Base Resolution" 
            for="config-default-size"
            helper="Default base sizing."
          >
            <Select
              id="config-default-size"
              name="generation.default_size"
              value={config.ui.default_image_size ?? ''}
              options={
                (config.ui.image_size_labels ?? []).map(opt => ({ 
                  value: opt.value, 
                  label: opt.label 
                })) || [{ value: 'm', label: 'm (default)' }]
              }
            />
          </FormField>

          <FormField 
            label="Default Video Model" 
            for="config-default-video-model"
            helper="The model pre-selected for Text-to-Video &amp; Image-to-Video."
          >
            <Select
              id="config-default-video-model"
              name="ui.default_models.video"
              value={config.ui.default_models?.video ?? ''}
              options={
                (config.ui.video_model_options ?? []).map(opt => ({ 
                  value: opt, 
                  label: opt 
                })) || [{ value: '', label: 'No video models available' }]
              }
            />
          </FormField>

          <div class="space-y-2">
            <div class="block text-sm font-medium text-zinc-300">Video Runtime Note</div>
            <div class="rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-300">
              Video low-memory mode is currently chosen per run in the workspace.
            </div>
          </div>
        </div>
      </section>

      <!-- Directories & Storage -->
      <section class="admin-section">
        <div class="admin-section-header">
          <svg class="w-5 h-5 text-teal-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"></path>
          </svg>
          <h2 class="admin-section-title">Directories &amp; Storage</h2>
        </div>

        <div class="space-y-6">
          <PathField
            id="config-output-dir"
            name="ui.output_dir"
            label="Output Directory"
            value={config.output_dir ?? config.ui.output_dir ?? ''}
            helper="Location where generated media and JSON metadata files are saved."
            pickerKind="directory"
            pickerPurpose="output_directory"
            onresolve={async (candidate) => candidate}
          />

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
          <svg class="w-5 h-5 text-teal-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
