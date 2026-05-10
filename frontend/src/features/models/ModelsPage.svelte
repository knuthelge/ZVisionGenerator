<script lang="ts">
  import { onMount } from 'svelte';
  import { getModelInventory, convertCheckpoint, importLoraLocal, importLoraHF } from '$lib/api/models';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { ModelInventory } from '$lib/types';
  import { Button, Input, Select } from '$lib/components/atoms';
  import { FormField, PathField } from '$lib/components/molecules';
  import { AdminPageShell } from '$lib/components/organisms';

  let inventory = $state<ModelInventory | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let notice = $state<{ tone: 'success' | 'error'; message: string } | null>(null);
  let formsBusy = $state(false);

  onMount(async () => {
    await loadInventory();
  });

  async function loadInventory(): Promise<void> {
    loading = true;
    error = null;
    try {
      inventory = await getModelInventory();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load model inventory';
    } finally {
      loading = false;
    }
  }

  async function handleConvertCheckpoint(e: Event): Promise<void> {
    e.preventDefault();
    const form = e.currentTarget as HTMLFormElement;
    const fd = new FormData(form);
    const data = {
      input_path: (fd.get('input_path') as string) ?? '',
      name: (fd.get('name') as string) ?? '',
      model_type: (fd.get('model_type') as string) ?? '',
      base_model: (fd.get('base_model') as string) ?? '',
      copy: fd.get('copy') === 'on'
    };
    formsBusy = true;
    notice = null;
    try {
      const result = await convertCheckpoint(data);
      if (result.tone === 'success') {
        notice = { tone: 'success', message: result.message || 'Checkpoint converted successfully.' };
        form.reset();
        await loadInventory();
        addToast('Operation started', 'success');
      } else {
        notice = { tone: 'error', message: result.message || 'Conversion failed.' };
        addToast('Operation failed', 'error');
      }
    } catch (err) {
      notice = { tone: 'error', message: err instanceof Error ? err.message : 'Conversion failed.' };
      addToast('Operation failed', 'error');
    } finally {
      formsBusy = false;
    }
  }

  async function handleImportLoraLocal(e: Event): Promise<void> {
    e.preventDefault();
    const form = e.currentTarget as HTMLFormElement;
    const fd = new FormData(form);
    const data = {
      source_path: (fd.get('source_path') as string) ?? '',
      name: (fd.get('name') as string) ?? ''
    };
    formsBusy = true;
    notice = null;
    try {
      const result = await importLoraLocal(data);
      if (result.tone === 'success') {
        notice = { tone: 'success', message: result.message || 'LoRA imported successfully.' };
        form.reset();
        await loadInventory();
        addToast('Operation started', 'success');
      } else {
        notice = { tone: 'error', message: result.message || 'Import failed.' };
        addToast('Operation failed', 'error');
      }
    } catch (err) {
      notice = { tone: 'error', message: err instanceof Error ? err.message : 'Import failed.' };
      addToast('Operation failed', 'error');
    } finally {
      formsBusy = false;
    }
  }

  async function handleImportLoraHF(e: Event): Promise<void> {
    e.preventDefault();
    const form = e.currentTarget as HTMLFormElement;
    const fd = new FormData(form);
    const data = {
      repo_id: (fd.get('repo_id') as string) ?? '',
      filename: (fd.get('filename') as string) ?? '',
      name: (fd.get('name') as string) ?? ''
    };
    formsBusy = true;
    notice = null;
    try {
      const result = await importLoraHF(data);
      if (result.tone === 'success') {
        notice = { tone: 'success', message: result.message || 'LoRA downloaded successfully.' };
        form.reset();
        await loadInventory();
        addToast('Operation started', 'success');
      } else {
        notice = { tone: 'error', message: result.message || 'Download failed.' };
        addToast('Operation failed', 'error');
      }
    } catch (err) {
      notice = { tone: 'error', message: err instanceof Error ? err.message : 'Download failed.' };
      addToast('Operation failed', 'error');
    } finally {
      formsBusy = false;
    }
  }
</script>

<AdminPageShell
  title="Models &amp; LoRAs"
  description="Manage installed models, convert checkpoints, and import LoRA adapters."
  {loading}
  {error}
>
  {#if inventory}
    <!-- Directory paths -->
    <div class="flex flex-wrap gap-4 mb-8">
      <div class="rounded-md border border-zinc-800 bg-zinc-950 px-4 py-2">
        <span class="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Models Dir</span>
        <p class="text-sm text-zinc-300 font-mono mt-0.5 break-all">{inventory.models_dir || '—'}</p>
      </div>
      <div class="rounded-md border border-zinc-800 bg-zinc-950 px-4 py-2">
        <span class="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">LoRAs Dir</span>
        <p class="text-sm text-zinc-300 font-mono mt-0.5 break-all">{inventory.loras_dir || '—'}</p>
      </div>
    </div>

    {#if notice}
      <div
        class="rounded-lg border px-4 py-3 text-sm mb-8
          {notice.tone === 'success'
            ? 'border-teal-500/30 bg-teal-500/10 text-teal-100'
            : 'border-red-500/30 bg-red-500/10 text-red-100'}"
      >
        {notice.message}
      </div>
    {/if}

    <!-- Runtime access card -->
    <div class="admin-section mb-8">
      <div class="admin-section-header">
        <svg class="w-5 h-5 text-teal-500 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"></path>
        </svg>
        <h3 class="admin-section-title">HuggingFace Access</h3>
      </div>
      <p class="text-sm text-zinc-300">
        {#if inventory.huggingface_configured}
          <span class="text-teal-400 font-medium">Configured.</span>
          Token read from <span class="font-mono">{inventory.huggingface_token_env_var ?? 'HF_TOKEN'}</span>.
        {:else}
          <span class="text-zinc-400">Not configured.</span>
          Set <span class="font-mono text-zinc-300">HF_TOKEN</span> for gated model downloads.
        {/if}
      </p>
    </div>

    <!-- Inventory tables row -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      <!-- Image Models -->
      <div class="admin-section">
        <div class="flex items-center justify-between border-b border-zinc-900 pb-4 mb-4">
          <h3 class="text-sm font-semibold text-zinc-100">Image Models</h3>
          <span class="rounded-full bg-teal-500/10 border border-teal-500/20 px-2 py-0.5 text-xs font-semibold text-teal-400">{inventory.image_models.length}</span>
        </div>
        {#if inventory.image_models.length === 0}
          <p class="text-xs text-zinc-500 text-center">None discovered</p>
        {:else}
          <table class="w-full text-xs border-collapse">
            <thead>
              <tr class="text-zinc-500 uppercase text-[10px] tracking-wider border-b border-zinc-900">
                <th class="px-2 py-2 text-left">Name</th>
                <th class="px-2 py-2 text-left">Family</th>
                <th class="px-2 py-2 text-left">Size</th>
              </tr>
            </thead>
            <tbody>
              {#each inventory.image_models as m}
                <tr class="border-b border-zinc-900 hover:bg-zinc-900/50 transition">
                  <td class="px-2 py-2 text-zinc-200 truncate max-w-20" title={m.name}>{m.name}</td>
                  <td class="px-2 py-2 text-zinc-400 font-mono">{m.family}</td>
                  <td class="px-2 py-2 text-zinc-400 font-mono">{m.size_label ?? '—'}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        {/if}
      </div>

      <!-- Video Models -->
      <div class="admin-section">
        <div class="flex items-center justify-between border-b border-zinc-900 pb-4 mb-4">
          <h3 class="text-sm font-semibold text-zinc-100">Video Models</h3>
          <span class="rounded-full bg-teal-500/10 border border-teal-500/20 px-2 py-0.5 text-xs font-semibold text-teal-400">{inventory.video_models.length}</span>
        </div>
        {#if inventory.video_models.length === 0}
          <p class="text-xs text-zinc-500 text-center">None discovered</p>
        {:else}
          <table class="w-full text-xs border-collapse">
            <thead>
              <tr class="text-zinc-500 uppercase text-[10px] tracking-wider border-b border-zinc-900">
                <th class="px-2 py-2 text-left">Name</th>
                <th class="px-2 py-2 text-left">Family</th>
                <th class="px-2 py-2 text-left">I2V</th>
              </tr>
            </thead>
            <tbody>
              {#each inventory.video_models as m}
                <tr class="border-b border-zinc-900 hover:bg-zinc-900/50 transition">
                  <td class="px-2 py-2 text-zinc-200 truncate max-w-20" title={m.name}>{m.name}</td>
                  <td class="px-2 py-2 text-zinc-400 font-mono">{m.family}</td>
                  <td class="px-2 py-2 text-zinc-400">{m.supports_i2v ? '✓' : '—'}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        {/if}
      </div>

      <!-- LoRAs -->
      <div class="admin-section">
        <div class="flex items-center justify-between border-b border-zinc-900 pb-4 mb-4">
          <h3 class="text-sm font-semibold text-zinc-100">Discovered LoRAs</h3>
          <span class="rounded-full bg-teal-500/10 border border-teal-500/20 px-2 py-0.5 text-xs font-semibold text-teal-400">{inventory.loras.length}</span>
        </div>
        {#if inventory.loras.length === 0}
          <p class="text-xs text-zinc-500 text-center">None discovered</p>
        {:else}
          <table class="w-full text-xs border-collapse">
            <thead>
              <tr class="text-zinc-500 uppercase text-[10px] tracking-wider border-b border-zinc-900">
                <th class="px-2 py-2 text-left">Name</th>
                <th class="px-2 py-2 text-left">Size</th>
              </tr>
            </thead>
            <tbody>
              {#each inventory.loras as l}
                <tr class="border-b border-zinc-900 hover:bg-zinc-900/50 transition">
                  <td class="px-2 py-2 text-zinc-200 truncate max-w-25" title={l.name}>{l.name}</td>
                  <td class="px-2 py-2 text-zinc-400 font-mono">{l.size_label ?? '—'}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        {/if}
      </div>
    </div>

    <!-- Operation Forms -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">

      <!-- Convert Checkpoint -->
      <form class="admin-section flex flex-col gap-5" onsubmit={handleConvertCheckpoint}>
        <h3 class="text-sm font-semibold text-zinc-100 border-b border-zinc-900 pb-3">Convert a Checkpoint</h3>
        
        <PathField
          id="convert-input-path"
          name="input_path"
          label="Input Path"
          placeholder="/path/to/model.safetensors"
          helper="Path to the checkpoint file"
          pickerKind="existing_file"
          pickerPurpose="checkpoint_file"
          onresolve={async (candidate) => candidate}
        />

        <FormField label="Alias Name" for="convert-name" helper="Display name for this model">
          <Input
            id="convert-name"
            type="text"
            name="name"
            placeholder="my-model-name"
          />
        </FormField>

        <FormField label="Model Type" for="convert-model-type" required>
          <Select
            id="convert-model-type"
            name="model_type"
            options={[
              { value: '', label: '-- Select type --', disabled: true },
              { value: 'zimage', label: 'zimage' },
              { value: 'flux2-klein-4b', label: 'flux2-klein-4b' },
              { value: 'flux2-klein-9b', label: 'flux2-klein-9b' }
            ]}
          />
        </FormField>

        <FormField label="Base Model (optional)" for="convert-base-model" helper="Base model ID or path">
          <Input
            id="convert-base-model"
            type="text"
            name="base_model"
            placeholder="base model id or path"
          />
        </FormField>

        <div class="flex items-center gap-2">
          <input
            type="checkbox"
            name="copy"
            id="convert-copy"
            class="rounded border-zinc-700 bg-zinc-900 text-teal-500 focus:ring-teal-500 h-4 w-4"
          />
          <label class="text-xs text-zinc-400 cursor-pointer" for="convert-copy">Copy instead of moving</label>
        </div>

        <Button variant="primary" type="submit" disabled={formsBusy} loading={formsBusy} class="mt-auto w-full">
          Convert Checkpoint
        </Button>
      </form>

      <!-- Import Local LoRA -->
      <form class="admin-section flex flex-col gap-5" onsubmit={handleImportLoraLocal}>
        <h3 class="text-sm font-semibold text-zinc-100 border-b border-zinc-900 pb-3">Import Local LoRA</h3>
        
        <PathField
          id="import-local-source-path"
          name="source_path"
          label="Source Path"
          placeholder="/path/to/lora.safetensors"
          helper="Path to the LoRA file"
          pickerKind="existing_file"
          pickerPurpose="lora_file"
          onresolve={async (candidate) => candidate}
        />

        <FormField label="Alias Name" for="import-local-name" helper="Display name for this LoRA">
          <Input
            id="import-local-name"
            type="text"
            name="name"
            placeholder="my-lora"
          />
        </FormField>

        <Button variant="primary" type="submit" disabled={formsBusy} loading={formsBusy} class="mt-auto w-full">
          Import LoRA
        </Button>
      </form>

      <!-- Import HuggingFace LoRA -->
      <form class="admin-section flex flex-col gap-5" onsubmit={handleImportLoraHF}>
        <h3 class="text-sm font-semibold text-zinc-100 border-b border-zinc-900 pb-3">Import from HuggingFace</h3>
        
        <FormField label="Repository ID" for="import-hf-repo-id" required helper="Format: username/repository">
          <Input
            id="import-hf-repo-id"
            type="text"
            name="repo_id"
            placeholder="username/repository"
            required
          />
        </FormField>

        <FormField label="Filename" for="import-hf-filename" required helper="Name of the file in the repository">
          <Input
            id="import-hf-filename"
            type="text"
            name="filename"
            placeholder="model.safetensors"
            required
          />
        </FormField>

        <FormField label="Alias Name" for="import-hf-name" helper="Display name for this LoRA">
          <Input
            id="import-hf-name"
            type="text"
            name="name"
            placeholder="my-hf-lora"
          />
        </FormField>

        {#if !inventory.huggingface_configured}
          <div class="rounded-md border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
            Set <span class="font-mono">HF_TOKEN</span> for gated model downloads.
          </div>
        {/if}

        <Button variant="primary" type="submit" disabled={formsBusy} loading={formsBusy} class="mt-auto w-full">
          Download LoRA
        </Button>
      </form>
    </div>
  {/if}
</AdminPageShell>
