<script lang="ts">
  import { onMount } from 'svelte';
  import { getModelInventory, convertCheckpoint, importLoraLocal, importLoraHF, deleteModel, deleteLora } from '$lib/api/models';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { ModelDeleteInfo, ModelInventory, ModelStatusFields } from '$lib/types';
  import { Button, Icon, Input, Select, Tooltip } from '$lib/components/atoms';
  import { Alert, ConfirmDialog, FormField, KeyValueList, ModelStatusBadges, Panel, PathField } from '$lib/components/molecules';
  import { DOWNLOADED_TOOLTIP, NOT_DOWNLOADED_TOOLTIP } from '$lib/components/molecules/ModelStatusBadges.svelte';
  import { AdminPageShell } from '$lib/components/organisms';

  let inventory = $state<ModelInventory | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let notice = $state<{ tone: 'success' | 'error'; message: string } | null>(null);
  let formsBusy = $state(false);
  let checkpointPath = $state('');
  let localLoraPath = $state('');
  let checkpointPathReset = $state(0);
  let localLoraPathReset = $state(0);
  let pendingDelete = $state<{ type: 'model'; name: string; info: ModelDeleteInfo } | { type: 'lora'; name: string } | null>(null);
  let deleteOpen = $state(false);
  let deleting = $state(false);

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
      copy: fd.get('copy') === 'on',
      quantize: (fd.get('quantize') as string) ?? ''
    };
    formsBusy = true;
    notice = null;
    try {
      const result = await convertCheckpoint(data);
      if (result.tone === 'success') {
        notice = { tone: 'success', message: result.message || 'Checkpoint converted successfully.' };
        checkpointPath = '';
        checkpointPathReset += 1;
        form.reset();
        const pathControl = form.elements.namedItem('input_path');
        if (pathControl instanceof HTMLInputElement) pathControl.value = '';
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
        localLoraPath = '';
        localLoraPathReset += 1;
        form.reset();
        const pathControl = form.elements.namedItem('source_path');
        if (pathControl instanceof HTMLInputElement) pathControl.value = '';
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

  function deleteQuestion(target: NonNullable<typeof pendingDelete>): string {
    if (target.type === 'lora') return `Delete ${target.name}.safetensors from the LoRAs folder?`;
    if (target.info.kind === 'installed') return `Delete the model folder ${target.name} from the models folder?`;
    return `Delete the Hugging Face download of ${target.info.repo_id}?`;
  }

  function requestDelete(target: NonNullable<typeof pendingDelete>): void {
    pendingDelete = target;
    deleteOpen = true;
  }

  async function confirmDelete(): Promise<void> {
    if (!pendingDelete) return;
    const target = pendingDelete;
    deleting = true;
    notice = null;
    try {
      const result = target.type === 'model' ? await deleteModel(target.name) : await deleteLora(target.name);
      notice = { tone: 'success', message: result.message };
      addToast(result.message, 'success');
      await loadInventory();
    } catch (err) {
      notice = { tone: 'error', message: err instanceof Error ? err.message : 'Delete failed.' };
      addToast('Delete failed', 'error');
    } finally {
      deleting = false;
      deleteOpen = false;
      pendingDelete = null;
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

{#snippet modelName(m: ModelStatusFields & { name: string })}
  {#if m.downloaded === true || m.downloaded === false}
    <Tooltip text={`${m.name}\n${m.downloaded ? DOWNLOADED_TOOLTIP : NOT_DOWNLOADED_TOOLTIP}`} focusable={false} class="max-w-full items-center gap-1.5">
      {#if m.downloaded}
        <svg class="h-3.5 w-3.5 shrink-0 text-primary-main" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
        </svg>
      {:else}
        <svg class="h-3.5 w-3.5 shrink-0 text-text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10"></path>
        </svg>
      {/if}
      <span class="break-words {m.downloaded ? 'text-text-primary' : 'text-text-muted'}" data-testid="model-name" data-downloaded={String(m.downloaded)}>{m.name}</span>
    </Tooltip>
  {:else}
    <span class="block break-words text-text-primary" title={m.name} data-testid="model-name" data-downloaded="unknown">{m.name}</span>
  {/if}
{/snippet}

{#snippet deleteButton(label: string, onclick: () => void)}
  <button type="button" class="ui-btn ui-btn-row ui-btn-danger" aria-label={label} title={label} data-testid="delete-button" {onclick}>
    <Icon name="trash" size={14} />
  </button>
{/snippet}

{#snippet modelDeleteCell(m: { name: string; delete?: ModelDeleteInfo | null })}
  {#if m.delete}
    {@const info = m.delete}
    {@render deleteButton(info.kind === 'installed' ? `Delete ${m.name}` : `Delete the Hugging Face download of ${m.name}`, () => requestDelete({ type: 'model', name: m.name, info }))}
  {/if}
{/snippet}

{#snippet memoryFitCell(m: ModelStatusFields)}
  {#if m.memory_fit}
    <ModelStatusBadges memoryFit={m.memory_fit} tooltipAlign="end" />
  {:else}
    <span class="text-text-muted">—</span>
  {/if}
{/snippet}

{#snippet count(n: number)}
  <span class="font-mono text-meta text-text-muted">{n}</span>
{/snippet}

<AdminPageShell
  title="Models"
  description="Installed models and LoRAs. Convert checkpoints and import LoRAs."
  {loading}
  {error}
>
  {#if inventory}
    <div class="flex min-w-0 flex-col gap-4 pb-6">
      {#if notice}
        <Alert tone={notice.tone} live>{notice.message}</Alert>
      {/if}

      <Panel title="Folders and access" icon="folder">
        <KeyValueList items={[
          { label: 'Models folder', value: inventory.models_dir || '—', mono: true },
          { label: 'LoRAs folder', value: inventory.loras_dir || '—', mono: true },
          inventory.huggingface_configured
            ? { label: 'Hugging Face token', value: 'Available', tone: 'success', hint: `Read from ${inventory.huggingface_token_env_var ?? 'HF_TOKEN'}.` }
            : { label: 'Hugging Face token', value: 'Not set', tone: 'muted', hint: 'Set HF_TOKEN for gated model downloads.' },
        ]} />
      </Panel>

      <!-- Image models get the full width, so names are not cut short. -->
      <Panel title="Image models" icon="cube" flush>
        {#snippet actions()}{@render count(inventory!.image_models.length)}{/snippet}
        {#if inventory.image_models.length === 0}
          <p class="ui-help p-3 text-center">None found</p>
        {:else}
          <div class="overflow-x-auto px-1 pb-1">
            <table class="ui-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Family</th>
                  <th>Size</th>
                  <th>Memory</th>
                  <th class="w-8"><span class="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {#each inventory.image_models as m}
                  <tr>
                    <td>
                      {@render modelName(m)}
                      {#if m.stored_quant}
                        <span class="ui-help block font-normal" data-testid="stored-quant" title="Used automatically when {m.stored_quant.base_model} runs at q{m.stored_quant.bits}">q{m.stored_quant.bits} copy of {m.stored_quant.base_model}</span>
                      {/if}
                    </td>
                    <td class="font-mono">{m.family}</td>
                    <td class="font-mono">{m.size_label ?? '—'}</td>
                    <td>{@render memoryFitCell(m)}</td>
                    <td class="text-right">{@render modelDeleteCell(m)}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/if}
      </Panel>

      <div class="grid min-w-0 grid-cols-1 items-start gap-4 lg:grid-cols-2">
        <Panel title="Video models" icon="cube" flush>
          {#snippet actions()}{@render count(inventory!.video_models.length)}{/snippet}
          {#if inventory.video_models.length === 0}
            <p class="ui-help p-3 text-center">None found</p>
          {:else}
            <div class="overflow-x-auto px-1 pb-1">
              <table class="ui-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Family</th>
                    <th>Image to video</th>
                    <th>Memory</th>
                    <th class="w-8"><span class="sr-only">Actions</span></th>
                  </tr>
                </thead>
                <tbody>
                  {#each inventory.video_models as m}
                    <tr>
                      <td>{@render modelName(m)}</td>
                      <td class="font-mono">{m.family}</td>
                      <td>{m.supports_i2v ? 'Yes' : '—'}</td>
                      <td>{@render memoryFitCell(m)}</td>
                      <td class="text-right">{@render modelDeleteCell(m)}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}
        </Panel>

        <Panel title="LoRAs" icon="list" flush>
          {#snippet actions()}{@render count(inventory!.loras.length)}{/snippet}
          {#if inventory.loras.length === 0}
            <p class="ui-help p-3 text-center">None found</p>
          {:else}
            <div class="overflow-x-auto px-1 pb-1">
              <table class="ui-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Size</th>
                    <th class="w-8"><span class="sr-only">Actions</span></th>
                  </tr>
                </thead>
                <tbody>
                  {#each inventory.loras as l}
                    <tr>
                      <td class="break-words">{l.name}</td>
                      <td class="font-mono">{l.size_label ?? '—'}</td>
                      <td class="text-right">{@render deleteButton(`Delete ${l.name}`, () => requestDelete({ type: 'lora', name: l.name }))}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}
        </Panel>
      </div>

      <div class="grid min-w-0 grid-cols-1 items-start gap-4 md:grid-cols-2 xl:grid-cols-3">
        <Panel as="form" title="Convert a checkpoint" onsubmit={handleConvertCheckpoint}>
          <div class="flex flex-col gap-4">
            {#key checkpointPathReset}
              <PathField
                id="convert-input-path"
                name="input_path"
                label="Checkpoint file"
                value={checkpointPath}
                placeholder="/path/to/model.safetensors"
                pickerKind="existing_file"
                pickerPurpose="checkpoint_file"
                required
                onresolve={async (candidate) => candidate}
                onvaluechange={(value) => (checkpointPath = value)}
              />
            {/key}

            <FormField label="Name" for="convert-name" helper="How the model is listed in the app">
              <Input id="convert-name" type="text" name="name" placeholder="my-model-name" />
            </FormField>

            <FormField label="Model type" for="convert-model-type" required>
              <Select
                id="convert-model-type"
                name="model_type"
                required
                options={[
                  { value: '', label: 'Choose a type', disabled: true },
                  { value: 'zimage', label: 'zimage' },
                  { value: 'flux2-klein-4b', label: 'flux2-klein-4b' },
                  { value: 'flux2-klein-9b', label: 'flux2-klein-9b' },
                  { value: 'krea2-turbo', label: 'krea2-turbo' }
                ]}
              />
            </FormField>

            <FormField label="Base model (optional)" for="convert-base-model" helper="Base model ID or path">
              <Input id="convert-base-model" type="text" name="base_model" placeholder="base model id or path" />
            </FormField>

            {#if inventory?.stored_quants_supported}
              <FormField label="Quantized copy (optional)" for="convert-quantize" helper="Also save a q8 or q4 copy, used when that quantize level is selected">
                <Select
                  id="convert-quantize"
                  name="quantize"
                  options={[
                    { value: '', label: 'None' },
                    { value: '8', label: 'q8' },
                    { value: '4', label: 'q4' }
                  ]}
                />
              </FormField>
            {/if}

            <label class="flex cursor-pointer items-center gap-2 text-ui text-text-secondary" for="convert-copy">
              <input type="checkbox" name="copy" id="convert-copy" class="accent-primary-main h-3.5 w-3.5" />
              Copy instead of moving
            </label>

            <div class="flex justify-end">
              <Button variant="primary" type="submit" disabled={formsBusy} loading={formsBusy}>Convert checkpoint</Button>
            </div>
          </div>
        </Panel>

        <Panel as="form" title="Import a local LoRA" onsubmit={handleImportLoraLocal}>
          <div class="flex flex-col gap-4">
            {#key localLoraPathReset}
              <PathField
                id="import-local-source-path"
                name="source_path"
                label="LoRA file"
                value={localLoraPath}
                placeholder="/path/to/lora.safetensors"
                pickerKind="existing_file"
                pickerPurpose="lora_file"
                required
                onresolve={async (candidate) => candidate}
                onvaluechange={(value) => (localLoraPath = value)}
              />
            {/key}

            <FormField label="Name" for="import-local-name" helper="How the LoRA is listed in the app">
              <Input id="import-local-name" type="text" name="name" placeholder="my-lora" />
            </FormField>

            <div class="flex justify-end">
              <Button variant="primary" type="submit" disabled={formsBusy} loading={formsBusy}>Import LoRA</Button>
            </div>
          </div>
        </Panel>

        <Panel as="form" title="Download a LoRA from Hugging Face" onsubmit={handleImportLoraHF}>
          <div class="flex flex-col gap-4">
            <FormField label="Repository" for="import-hf-repo-id" required helper="For example owner/repository">
              <Input id="import-hf-repo-id" type="text" name="repo_id" placeholder="owner/repository" required />
            </FormField>

            <FormField label="File" for="import-hf-filename" required helper="Name of the file in the repository">
              <Input id="import-hf-filename" type="text" name="filename" placeholder="model.safetensors" required />
            </FormField>

            <FormField label="Name" for="import-hf-name" helper="How the LoRA is listed in the app">
              <Input id="import-hf-name" type="text" name="name" placeholder="my-hf-lora" />
            </FormField>

            {#if !inventory.huggingface_configured}
              <Alert tone="warning">Set <span class="font-mono">HF_TOKEN</span> for gated model downloads.</Alert>
            {/if}

            <div class="flex justify-end">
              <Button variant="primary" type="submit" disabled={formsBusy} loading={formsBusy}>Download LoRA</Button>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  {/if}
</AdminPageShell>

<ConfirmDialog
  bind:open={deleteOpen}
  question={pendingDelete ? deleteQuestion(pendingDelete) : ''}
  info="This cannot be undone."
  confirmLabel="Delete"
  pending={deleting}
  onconfirm={confirmDelete}
  oncancel={() => (pendingDelete = null)}
>
  {#if pendingDelete}
    <div class="space-y-2" data-testid="delete-dialog">
      {#if pendingDelete.type === 'model' && pendingDelete.info.kind === 'installed'}
        <p>Hugging Face files it links to are kept.</p>
      {:else if pendingDelete.type === 'model'}
        <p>The <span class="font-mono">{pendingDelete.name}</span> alias stays in your config and downloads the model again the next time you use it.</p>
        {#if pendingDelete.info.linked_by.length > 0}
          <Alert tone="warning" testId="delete-linked-warning">
            These converted models use files from this download and will stop working until it is downloaded again:
            <span class="font-mono">{pendingDelete.info.linked_by.join(', ')}</span>
          </Alert>
        {/if}
      {/if}
      {#if pendingDelete.type === 'model' && (pendingDelete.info.stored_quants?.length ?? 0) > 0}
        <p data-testid="delete-stored-quants">
          Its quantized copies are deleted too: <span class="font-mono">{pendingDelete.info.stored_quants?.join(', ')}</span>
        </p>
      {/if}
    </div>
  {/if}
</ConfirmDialog>
