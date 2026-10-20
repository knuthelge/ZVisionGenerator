<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { errorText } from '$lib/api/client';
  import { getModelInventory, convertCheckpoint, getConversionStatus, inspectLtxCheckpoint, importLoraLocal, importLoraHF, deleteModel, deleteEnhancerModel, deleteLora, deleteOtherDownload, freeStrandedCacheFiles } from '$lib/api/models';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { ConversionStatus, EnhancerModelEntry, LtxCheckpointInspection, ModelDeleteInfo, ModelInventory, ModelOperationResult, ModelStatusFields } from '$lib/types';
  import { Button, Icon, Input, Select, StepProgress, Tooltip } from '$lib/components/atoms';
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
  let pendingDelete = $state<{ type: 'model'; name: string; info: ModelDeleteInfo; downloadSizeLabel?: string | null } | { type: 'enhancer'; name: string; model: EnhancerModelEntry } | { type: 'download'; name: string; sizeLabel: string } | { type: 'stranded'; name: string; count: number; sizeLabel: string } | { type: 'lora'; name: string } | null>(null);
  let deleteOpen = $state(false);
  let deleting = $state(false);

  const LTX_TYPE = 'ltx-2.3';
  const CONVERSION_POLL_MS = 2000;
  const INSPECT_DELAY_MS = 400;
  let convertType = $state('');
  let ltxQuantize = $state('4');
  let ltxDistill = $state(false);
  let ltxInspection = $state<LtxCheckpointInspection | null>(null);
  let ltxInspectError = $state<string | null>(null);
  let conversion = $state<ConversionStatus | null>(null);
  let conversionTimer: ReturnType<typeof setTimeout> | null = null;
  let destroyed = false;
  let inspectTimer: ReturnType<typeof setTimeout> | null = null;
  let inspectRequest = 0;
  const isLtx = $derived(convertType === LTX_TYPE);
  const converting = $derived(conversion?.state === 'running');

  const ENHANCER_TIER_LABELS: Record<EnhancerModelEntry['tier'], string> = { fast: 'Fast', smart: 'Smart', custom: 'Custom' };


  onMount(async () => {
    await loadInventory();
    await pollConversion(false);
  });

  onDestroy(() => {
    destroyed = true;
    if (conversionTimer) clearTimeout(conversionTimer);
    if (inspectTimer) clearTimeout(inspectTimer);
  });

  /** Read the LTX conversion's status; while it runs (or the read failed), read it again shortly. A finished one is reported when *report* is set. */
  async function pollConversion(report: boolean): Promise<void> {
    try {
      conversion = await getConversionStatus();
    } catch {
      // A missed read (a dropped connection) says nothing about the conversion: try again while one may be running.
      if (!destroyed && conversion?.state === 'running') conversionTimer = setTimeout(() => void pollConversion(report), CONVERSION_POLL_MS);
      return;
    }
    if (destroyed) return;
    if (conversion.state === 'running') {
      conversionTimer = setTimeout(() => void pollConversion(true), CONVERSION_POLL_MS);
      return;
    }
    if (!report) return;
    if (conversion.state === 'done') {
      notice = { tone: 'success', message: conversion.message || 'Checkpoint converted.' };
      addToast('Conversion finished', 'success');
      await loadInventory();
    } else if (conversion.state === 'failed') {
      notice = { tone: 'error', message: conversion.message || 'Conversion failed.' };
      addToast('Conversion failed', 'error');
    }
  }

  /** Inspect the chosen LTX checkpoint shortly after the path or type last changed, preselecting the level and Distill. */
  function scheduleInspect(): void {
    if (inspectTimer) clearTimeout(inspectTimer);
    // A new path makes any answer still on its way for the old one stale.
    const request = ++inspectRequest;
    ltxInspection = null;
    ltxInspectError = null;
    const path = checkpointPath.trim();
    if (!isLtx || !path) return;
    inspectTimer = setTimeout(async () => {
      try {
        const found = await inspectLtxCheckpoint(path);
        if (request !== inspectRequest) return;
        ltxInspection = found;
        ltxQuantize = String(found.recommended_quantize);
        ltxDistill = found.recommend_distill;
      } catch (err) {
        if (request === inspectRequest) ltxInspectError = errorText(err, 'Could not read this checkpoint.');
      }
    }, INSPECT_DELAY_MS);
  }

  async function loadInventory(): Promise<void> {
    loading = true;
    error = null;
    try {
      inventory = await getModelInventory();
    } catch (e) {
      error = errorText(e, 'Failed to load model inventory');
    } finally {
      loading = false;
    }
  }

  /** The converted model's name, as the server picks it: the name given, else the checkpoint's file name without its extension. */
  function convertedName(name: string, inputPath: string): string {
    const file = inputPath.trim().split(/[\\/]/).pop() ?? '';
    return name.trim() || file.replace(/\.[^.]+$/, '');
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
      quantize: (fd.get('quantize') as string) ?? '',
      distill: fd.get('distill') === 'on'
    };
    formsBusy = true;
    notice = null;
    try {
      const result = await convertCheckpoint(data);
      if (result.tone === 'success' && result.background) {
        notice = { tone: 'success', message: result.message };
        addToast('Conversion started', 'success');
        // The server has started it, so it is running until a read says otherwise.
        conversion = { state: 'running', name: convertedName(data.name, data.input_path), stage: 'Starting', done: 0, total: 0 };
        await pollConversion(true);
      } else if (result.tone === 'success') {
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
      notice = { tone: 'error', message: errorText(err, 'Conversion failed.') };
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
      notice = { tone: 'error', message: errorText(err, 'Import failed.') };
      addToast('Operation failed', 'error');
    } finally {
      formsBusy = false;
    }
  }

  function notDownloadedTooltip(m: { source?: string; download_size_label?: string | null }): string {
    if (m.source === 'enhancer') return 'Not downloaded: it downloads the first time you enhance a prompt with it.';
    return m.source === 'seedvr2' && m.download_size_label
      ? `Not downloaded: the first plain upscale downloads it (${m.download_size_label}).`
      : NOT_DOWNLOADED_TOOLTIP;
  }

  function deleteQuestion(target: NonNullable<typeof pendingDelete>): string {
    if (target.type === 'lora') return `Delete ${target.name}.safetensors from the LoRAs folder?`;
    if (target.type === 'enhancer') return `Delete the ${target.name} enhancer model ${target.model.repo}?`;
    if (target.type === 'download') return `Delete the Hugging Face download of ${target.name}?`;
    if (target.type === 'stranded') return `Delete ${target.count} unused Hugging Face cache files?`;
    if (target.info.kind === 'installed') return `Delete the model folder ${target.name} from the models folder?`;
    if (target.info.kind === 'copy') return `Delete the prepared copy of ${target.name}?`;
    return `Delete the Hugging Face download of ${target.info.repo_id}?`;
  }

  function runDelete(target: NonNullable<typeof pendingDelete>): Promise<ModelOperationResult> {
    switch (target.type) {
      case 'model': return deleteModel(target.name);
      case 'enhancer': return deleteEnhancerModel(target.model.tier);
      case 'download': return deleteOtherDownload(target.name);
      case 'stranded': return freeStrandedCacheFiles();
      case 'lora': return deleteLora(target.name);
    }
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
      const result = await runDelete(target);
      notice = { tone: 'success', message: result.message };
      addToast(result.message, 'success');
      await loadInventory();
    } catch (err) {
      notice = { tone: 'error', message: errorText(err, 'Delete failed.') };
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
      notice = { tone: 'error', message: errorText(err, 'Download failed.') };
      addToast('Operation failed', 'error');
    } finally {
      formsBusy = false;
    }
  }
</script>

{#snippet modelName(m: ModelStatusFields & { name: string; source?: string; download_size_label?: string | null })}
  {#if m.downloaded === true || m.downloaded === false}
    <Tooltip text={`${m.name}\n${m.downloaded ? DOWNLOADED_TOOLTIP : notDownloadedTooltip(m)}`} focusable={false} class="max-w-full items-center gap-1.5">
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

{#snippet modelDeleteCell(m: { name: string; delete?: ModelDeleteInfo | null; source?: string; download_size_label?: string | null })}
  {#if m.delete}
    {@const info = m.delete}
    {@const downloadSizeLabel = m.source === 'seedvr2' ? m.download_size_label : null}
    {@render deleteButton(info.kind === 'huggingface' ? `Delete the Hugging Face download of ${m.name}` : info.kind === 'copy' ? `Delete the prepared copy of ${m.name}` : `Delete ${m.name}`, () => requestDelete({ type: 'model', name: m.name, info, downloadSizeLabel }))}
  {/if}
{/snippet}

{#snippet memoryFitCell(m: ModelStatusFields)}
  {#if m.memory_fit}
    <ModelStatusBadges memoryFit={m.memory_fit} tooltipAlign="end" />
  {:else}
    <span class="text-text-muted">—</span>
  {/if}
{/snippet}

{#snippet diskSizeCell(label: string | null | undefined)}
  {#if label}
    <span class="whitespace-nowrap font-mono" data-testid="disk-size" title="Disk space its files use">{label}</span>
  {:else}
    <span class="text-text-muted">—</span>
  {/if}
{/snippet}

{#snippet count(n: number)}
  <span class="font-mono text-meta text-text-muted">{n}</span>
{/snippet}

{#snippet submitRow(label: string, blocked: boolean)}
  <div class="flex justify-end">
    <Button variant="primary" type="submit" disabled={formsBusy || blocked} loading={formsBusy}>{label}</Button>
  </div>
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

      <!-- Models on the left, the longer LoRA list in its own narrower column. -->
      <div class="grid min-w-0 grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
        <div class="flex min-w-0 flex-col gap-4">
          <Panel title="Image models" icon="cube" flush>
            {#snippet actions()}{@render count(inventory!.image_models.length)}{/snippet}
            {#if inventory.image_models.length === 0}
              <p class="ui-help p-3 text-center">None found</p>
            {:else}
              <div class="overflow-x-auto px-1 pb-1">
                <table class="ui-table ui-table-comfortable">
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
                        <td>{@render diskSizeCell(m.disk_size_label)}</td>
                        <td>{@render memoryFitCell(m)}</td>
                        <td class="text-right">{@render modelDeleteCell(m)}</td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            {/if}
          </Panel>

          <Panel title="Video models" icon="cube" flush>
            {#snippet actions()}{@render count(inventory!.video_models.length)}{/snippet}
            {#if inventory.video_models.length === 0}
              <p class="ui-help p-3 text-center">None found</p>
            {:else}
              <div class="overflow-x-auto px-1 pb-1">
                <table class="ui-table ui-table-comfortable">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Family</th>
                      <th>Image to video</th>
                      <th>Size</th>
                      <th>Memory</th>
                      <th class="w-8"><span class="sr-only">Actions</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    {#each inventory.video_models as m}
                      <tr>
                        <td>
                          {@render modelName(m)}
                          {#if m.converted_bits}
                            <span class="ui-help block font-normal" data-testid="converted-level" title="Converted from a checkpoint; stored at this level only">Converted at q{m.converted_bits}</span>
                          {/if}
                        </td>
                        <td class="font-mono">{m.family}</td>
                        <td>{m.supports_i2v ? 'Yes' : '—'}</td>
                        <td>{@render diskSizeCell(m.disk_size_label)}</td>
                        <td>{@render memoryFitCell(m)}</td>
                        <td class="text-right">{@render modelDeleteCell(m)}</td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            {/if}
          </Panel>

          {#if inventory.enhancer_models?.length}
            <Panel title="Prompt enhancer" icon="cube" flush>
              {#snippet actions()}{@render count(inventory!.enhancer_models!.length)}{/snippet}
              <div class="overflow-x-auto px-1 pb-1">
                <table class="ui-table ui-table-comfortable">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Repository</th>
                      <th>Size</th>
                      <th class="w-8"><span class="sr-only">Actions</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    {#each inventory.enhancer_models as e (e.tier)}
                      {@const label = ENHANCER_TIER_LABELS[e.tier]}
                      <tr data-testid="enhancer-row" data-tier={e.tier}>
                        <td>
                          {@render modelName({ name: label, downloaded: e.downloaded, source: 'enhancer' })}
                          {#if e.unavailable}
                            <span class="ui-help block font-normal" data-testid="enhancer-unavailable">{e.unavailable}</span>
                          {/if}
                        </td>
                        <td class="break-all font-mono">{e.repo}</td>
                        <td class="whitespace-nowrap font-mono">
                          {#if e.disk_size_label}
                            {e.disk_size_label}
                          {:else if e.download_size_label}
                            <span class="text-text-muted" title="Size of the first-use download">≈{e.download_size_label}</span>
                          {:else}
                            <span class="text-text-muted">—</span>
                          {/if}
                        </td>
                        <td class="text-right">
                          {#if e.deletable}
                            {@render deleteButton(`Delete the ${label} enhancer model`, () => requestDelete({ type: 'enhancer', name: label, model: e }))}
                          {/if}
                        </td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            </Panel>
          {/if}

          {#if inventory.other_downloads && (inventory.other_downloads.repos.length || inventory.other_downloads.stranded)}
            {@const other = inventory.other_downloads}
            <Panel title="Other downloads" icon="download" flush>
              {#snippet actions()}{@render count(other.repos.length)}{/snippet}
              <p class="ui-help px-3 pt-2">
                In the Hugging Face cache (<span class="font-mono break-all">{other.cache_dir}</span>) but not used by any model here.
              </p>
              <div class="overflow-x-auto px-1 pb-1">
                <table class="ui-table ui-table-comfortable">
                  <thead>
                    <tr>
                      <th>Repository</th>
                      <th>Size</th>
                      <th class="w-8"><span class="sr-only">Actions</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    {#each other.repos as r (r.repo_id)}
                      <tr data-testid="other-download-row">
                        <td class="break-all font-mono">{r.repo_id}</td>
                        <td>{@render diskSizeCell(r.size_label)}</td>
                        <td class="text-right">{@render deleteButton(`Delete ${r.repo_id}`, () => requestDelete({ type: 'download', name: r.repo_id, sizeLabel: r.size_label }))}</td>
                      </tr>
                    {/each}
                    {#if other.stranded}
                      {@const stranded = other.stranded}
                      <tr data-testid="stranded-row">
                        <td>
                          Unused cache files
                          <span class="ui-help block font-normal">{stranded.count} files no download links to any more, left by earlier deletes</span>
                        </td>
                        <td>{@render diskSizeCell(stranded.size_label)}</td>
                        <td class="text-right">{@render deleteButton('Delete unused cache files', () => requestDelete({ type: 'stranded', name: 'Unused cache files', count: stranded.count, sizeLabel: stranded.size_label }))}</td>
                      </tr>
                    {/if}
                  </tbody>
                </table>
              </div>
            </Panel>
          {/if}
        </div>

        <Panel title="LoRAs" icon="list" flush>
          {#snippet actions()}{@render count(inventory!.loras.length)}{/snippet}
          {#if inventory.loras.length === 0}
            <p class="ui-help p-3 text-center">None found</p>
          {:else}
            <div class="overflow-x-auto px-1 pb-1">
              <table class="ui-table ui-table-comfortable">
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
                      <td class="whitespace-nowrap font-mono">{l.size_label ?? '—'}</td>
                      <td class="text-right">{@render deleteButton(`Delete ${l.name}`, () => requestDelete({ type: 'lora', name: l.name }))}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}
        </Panel>
      </div>


      <!-- The three ways to add something, side by side; fields stack inside each. -->
      <div class="grid min-w-0 grid-cols-1 items-start gap-4 lg:grid-cols-3">
      <Panel as="form" title="Convert a checkpoint" icon="cube" onsubmit={handleConvertCheckpoint}>
        <div class="flex flex-col gap-4">
          {#key checkpointPathReset}
            <PathField
              id="convert-input-path"
              name="input_path"
              label="Checkpoint file"
             
              value={checkpointPath}
              placeholder="/path/to/model.safetensors"
              purpose="checkpoint_file"
              required
              onresolve={async (candidate) => candidate}
              onvaluechange={(value) => {
                checkpointPath = value;
                scheduleInspect();
              }}
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
              bind:value={convertType}
              onchange={scheduleInspect}
              options={[
                { value: '', label: 'Choose a type', disabled: true },
                { value: 'zimage', label: 'zimage' },
                { value: 'flux2-klein-4b', label: 'flux2-klein-4b' },
                { value: 'flux2-klein-9b', label: 'flux2-klein-9b' },
                { value: 'krea2-turbo', label: 'krea2-turbo' },
                { value: LTX_TYPE, label: 'LTX-2.3 (video)' }
              ]}
            />
          </FormField>

          {#if isLtx}
            {#if ltxInspection}
              <Alert tone={ltxInspection.distillation === 'distilled' ? 'success' : 'warning'}>{ltxInspection.advice}</Alert>
            {:else if ltxInspectError}
              <Alert tone="error">{ltxInspectError}</Alert>
            {/if}

            <FormField label="Quantize level" for="convert-ltx-quantize" required helper="The only level stored. q8 needs about 10 GB more memory than q4.">
              <Select
                id="convert-ltx-quantize"
                name="quantize"
                required
                bind:value={ltxQuantize}
                options={[
                  { value: '4', label: 'q4' },
                  { value: '8', label: 'q8' }
                ]}
              />
            </FormField>

            <FormField label="Distillation" helper="Merges Lightricks' distilled LoRA (7.6 GB, downloaded once). Plain-dev checkpoints need it to run at 8 steps.">
              <label class="flex min-h-control cursor-pointer items-center gap-2 text-ui text-text-secondary" for="convert-distill">
                <input type="checkbox" name="distill" id="convert-distill" class="ui-checkbox" bind:checked={ltxDistill} />
                Distill
              </label>
            </FormField>
          {:else}
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

          <FormField label="Source file">
            <label class="flex min-h-control cursor-pointer items-center gap-2 text-ui text-text-secondary" for="convert-copy">
              <input type="checkbox" name="copy" id="convert-copy" class="ui-checkbox" />
              Copy instead of moving
            </label>
          </FormField>
          {/if}

          {#if converting && conversion}
            <StepProgress
              value={conversion.done ?? 0}
              max={conversion.total || 1}
              live
              busy={!conversion.total}
              label="Conversion progress"
              valuetext={conversion.stage}
            />
            <p class="text-ui text-text-secondary">
              Converting {conversion.name}: {conversion.stage}{conversion.total ? ` (${conversion.done}/${conversion.total})` : ''}
            </p>
          {/if}

          {@render submitRow('Convert checkpoint', converting)}
        </div>
      </Panel>

      <Panel as="form" title="Import a local LoRA" icon="folder" onsubmit={handleImportLoraLocal}>
        <div class="flex flex-col gap-4">
          {#key localLoraPathReset}
            <PathField
              id="import-local-source-path"
              name="source_path"
              label="LoRA file"
             
              value={localLoraPath}
              placeholder="/path/to/lora.safetensors"
              purpose="lora_file"
              required
              onresolve={async (candidate) => candidate}
              onvaluechange={(value) => (localLoraPath = value)}
            />
          {/key}

          <FormField label="Name" for="import-local-name" helper="How the LoRA is listed in the app">
            <Input id="import-local-name" type="text" name="name" placeholder="my-lora" />
          </FormField>

          {@render submitRow('Import LoRA', false)}
        </div>
      </Panel>

      <Panel as="form" title="Download a LoRA from Hugging Face" icon="download" onsubmit={handleImportLoraHF}>
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

          {@render submitRow('Download LoRA', false)}
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
      {#if pendingDelete.type === 'download'}
        <p data-testid="delete-download">Frees {pendingDelete.sizeLabel}. Another app on this computer that uses the Hugging Face cache may download it again.</p>
      {:else if pendingDelete.type === 'stranded'}
        <p data-testid="delete-stranded">Frees {pendingDelete.sizeLabel}. No downloaded model uses these files.</p>
      {/if}
      {#if pendingDelete.type === 'enhancer'}
        <p data-testid="delete-enhancer">
          Frees {pendingDelete.model.disk_size_label ?? 'its download'}. It downloads again the next time you use {pendingDelete.name} to enhance a prompt.
        </p>
      {/if}
      {#if pendingDelete.type === 'model' && pendingDelete.info.kind === 'installed' && pendingDelete.downloadSizeLabel}
        <p data-testid="delete-redownload">The next plain upscale downloads it again ({pendingDelete.downloadSizeLabel}).</p>
      {:else if pendingDelete.type === 'model' && pendingDelete.info.kind === 'installed'}
        <p>Hugging Face files it links to are kept.</p>
      {:else if pendingDelete.type === 'model' && pendingDelete.info.kind === 'copy'}
        <p data-testid="delete-copy">
          Deletes <span class="font-mono">{pendingDelete.info.stored_quants?.join(', ')}</span>. The download stays, since
          <span class="font-mono">{pendingDelete.info.shared_with?.join(', ')}</span> also uses it.
        </p>
      {:else if pendingDelete.type === 'model'}
        <p>The <span class="font-mono">{pendingDelete.name}</span> alias stays in your config and downloads the model again the next time you use it.</p>
        {#if pendingDelete.info.linked_by.length > 0}
          <Alert tone="warning" testId="delete-linked-warning">
            These converted models use files from this download and will stop working until it is downloaded again:
            <span class="font-mono">{pendingDelete.info.linked_by.join(', ')}</span>
          </Alert>
        {/if}
      {/if}
      {#if pendingDelete.type === 'model' && pendingDelete.info.kind !== 'copy' && (pendingDelete.info.stored_quants?.length ?? 0) > 0}
        <p data-testid="delete-stored-quants">
          Its quantized copies are deleted too: <span class="font-mono">{pendingDelete.info.stored_quants?.join(', ')}</span>
        </p>
      {/if}
    </div>
  {/if}
</ConfirmDialog>

