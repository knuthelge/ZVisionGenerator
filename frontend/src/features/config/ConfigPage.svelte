<script lang="ts">
  import { onMount } from 'svelte';
  import { getConfig } from '$lib/api/config';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { AppConfig } from '$lib/types';
  import Button from '$lib/components/atoms/Button.svelte';

  let config = $state<AppConfig | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let saveStatus = $state<{ tone: 'success' | 'error'; message: string } | null>(null);
  let saving = $state(false);
  let pickingDir = $state(false);

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

  async function pickDirectory(inputId: string): Promise<void> {
    if (!config) return;
    pickingDir = true;
    try {
      const input = document.getElementById(inputId) as HTMLInputElement | null;
      const resp = await fetch('/api/pick-directory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ initial_dir: input?.value?.trim() || null })
      });
      if (!resp.ok) throw new Error('Directory picker failed');
      const data = await resp.json() as { path: string | null };
      if (data.path && input) {
        input.value = data.path;
      }
    } catch {
      // ignore picker errors
    } finally {
      pickingDir = false;
    }
  }
</script>

<main class="flex-1 bg-zinc-900 overflow-y-auto p-6 custom-scrollbar">
  <div class="max-w-4xl mx-auto space-y-8">
    <header class="mb-8">
      <h1 class="text-2xl font-bold text-white tracking-tight">System Configuration</h1>
      <p class="text-zinc-400 mt-2 text-sm">
        Manage persistent settings, paths, and default models to be applied across the application.
      </p>
    </header>

    {#if loading}
      <div class="flex items-center justify-center py-24">
        <div class="animate-spin rounded-full h-8 w-8 border-t-2 border-teal-500"></div>
      </div>
    {:else if error}
      <div class="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-100">
        {error}
      </div>
    {:else if config}
      <form class="space-y-10" onsubmit={handleSave}>
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
        <section class="bg-zinc-950 border border-zinc-800 rounded-lg p-6 shadow-sm">
          <div class="flex items-center gap-3 mb-6 border-b border-zinc-900 pb-4">
            <svg class="w-5 h-5 text-teal-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
            </svg>
            <h2 class="text-lg font-semibold text-zinc-100 mt-0.5">Model Defaults</h2>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="space-y-2">
              <label class="block text-sm font-medium text-zinc-300">Default Image Model</label>
              <select
                name="ui.default_models.image"
                class="w-full bg-zinc-900 border border-zinc-700 hover:border-zinc-600 rounded-md px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition"
              >
                {#each (config.ui.image_model_options ?? []) as opt}
                  <option value={opt} selected={opt === config.ui.default_models?.image}>{opt}</option>
                {/each}
                {#if !(config.ui.image_model_options?.length)}
                  <option value="">No image models available</option>
                {/if}
              </select>
              <p class="text-xs text-zinc-500">The model pre-selected when starting a new workspace.</p>
            </div>
            <div class="space-y-2">
              <label class="block text-sm font-medium text-zinc-300">Base Resolution</label>
              <select
                name="generation.default_size"
                class="w-full bg-zinc-900 border border-zinc-700 hover:border-zinc-600 rounded-md px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition font-mono"
              >
                {#each (config.ui.image_size_labels ?? []) as opt}
                  <option value={opt.value} selected={opt.value === config.ui.default_image_size}>{opt.label}</option>
                {/each}
                {#if !(config.ui.image_size_labels?.length)}
                  <option value="m">m (default)</option>
                {/if}
              </select>
              <p class="text-xs text-zinc-500">Default base sizing.</p>
            </div>
            <div class="space-y-2">
              <label class="block text-sm font-medium text-zinc-300">Default Video Model</label>
              <select
                name="ui.default_models.video"
                class="w-full bg-zinc-900 border border-zinc-700 hover:border-zinc-600 rounded-md px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition"
              >
                {#each (config.ui.video_model_options ?? []) as opt}
                  <option value={opt} selected={opt === config.ui.default_models?.video}>{opt}</option>
                {/each}
                {#if !(config.ui.video_model_options?.length)}
                  <option value="">No video models available</option>
                {/if}
              </select>
              <p class="text-xs text-zinc-500">The model pre-selected for Text-to-Video &amp; Image-to-Video.</p>
            </div>
            <div class="space-y-2">
              <label class="block text-sm font-medium text-zinc-300">Video Runtime Note</label>
              <div class="rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-300">
                Video low-memory mode is currently chosen per run in the workspace.
              </div>
            </div>
          </div>
        </section>

        <!-- Directories & Storage -->
        <section class="bg-zinc-950 border border-zinc-800 rounded-lg p-6 shadow-sm">
          <div class="flex items-center justify-between mb-6 border-b border-zinc-900 pb-4">
            <div class="flex items-center gap-3">
              <svg class="w-5 h-5 text-teal-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"></path>
              </svg>
              <h2 class="text-lg font-semibold text-zinc-100 mt-0.5">Directories &amp; Storage</h2>
            </div>
          </div>

          <div class="space-y-6">
            <div class="space-y-2">
              <label class="block text-sm font-medium text-zinc-300">Output Directory</label>
              <div class="flex gap-2">
                <input
                  id="config-output-dir"
                  type="text"
                  name="ui.output_dir"
                  value={config.output_dir ?? config.ui.output_dir ?? ''}
                  class="flex-1 bg-zinc-900 border border-zinc-700 hover:border-zinc-600 rounded-md px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition font-mono"
                >
                <button
                  type="button"
                  disabled={pickingDir}
                  class="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-zinc-300 rounded-md text-sm font-medium transition shadow-sm disabled:opacity-50"
                  onclick={() => pickDirectory('config-output-dir')}
                >Browse...</button>
              </div>
              <p class="text-xs text-zinc-500">Location where generated media and JSON metadata files are saved.</p>
            </div>

            <div class="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div class="space-y-2">
                <label class="block text-sm font-medium text-zinc-300">Models Cache Directory</label>
                <div class="rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 font-mono break-all">
                  {config.ui.model_cache_dir ?? '(runtime-only)'}
                </div>
                <p class="text-xs text-zinc-500">Runtime-only. Not writable from the Web UI.</p>
              </div>
              <div class="space-y-2">
                <label class="block text-sm font-medium text-zinc-300">LoRAs Directory</label>
                <div class="rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 font-mono break-all">
                  {config.ui.loras_dir ?? '(runtime-only)'}
                </div>
                <p class="text-xs text-zinc-500">Derived from the current data directory.</p>
              </div>
            </div>
          </div>
        </section>

        <!-- API Keys & Authentication -->
        <section class="bg-zinc-950 border border-zinc-800 rounded-lg p-6 shadow-sm">
          <div class="flex items-center gap-3 mb-6 border-b border-zinc-900 pb-4">
            <svg class="w-5 h-5 text-teal-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"></path>
            </svg>
            <h2 class="text-lg font-semibold text-zinc-100 mt-0.5">API Keys &amp; Authentication</h2>
          </div>
          <div class="space-y-6">
            <div class="space-y-2">
              <label class="block text-sm font-medium text-zinc-300">HuggingFace Token</label>
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
          </div>
        </section>

        <!-- Actions -->
        <div class="flex items-center justify-end gap-4 py-8">
          <button
            type="reset"
            class="px-5 py-2.5 border border-zinc-700 hover:border-zinc-600 bg-zinc-900 text-zinc-300 font-medium rounded-md transition shadow-sm"
          >Discard Changes</button>
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
  </div>
</main>
