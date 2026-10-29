<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Icon } from '$lib/components/atoms';
  import { copyText } from '$lib/clipboard';
  import { addToast } from '$lib/state/toasts.svelte';
  import DetailGallery from './DetailGallery.svelte';
  import type { DetailView, MadeFor } from './discoverView';

  interface Props {
    view: DetailView;
    /** Which model the file was made for, when that is known. */
    madeFor?: MadeFor | null;
    /** A Hugging Face repository with several files: the choice, after Weight. */
    files?: Snippet;
    /** The file is on this computer: its name is "File name", and the trigger can be edited. */
    installed?: boolean;
    /** The trigger the user typed, to put back in the field; null when the shown one came from elsewhere. */
    typedTrigger?: string | null;
    onsavetrigger?: (trigger: string | null) => Promise<void> | void;
  }

  let { view, madeFor = null, files, installed = false, typedTrigger = null, onsavetrigger }: Props = $props();

  let editing = $state(false);
  let draftTrigger = $state('');
  let saving = $state(false);
  let inputEl = $state<HTMLInputElement | null>(null);

  async function startEdit(): Promise<void> {
    draftTrigger = typedTrigger ?? view.trigger ?? '';
    editing = true;
    await Promise.resolve();
    inputEl?.focus();
  }

  async function save(trigger: string | null): Promise<void> {
    saving = true;
    try {
      await onsavetrigger?.(trigger);
      editing = false;
    } finally {
      saving = false;
    }
  }

  async function copyCli(): Promise<void> {
    if (view.cli && (await copyText(view.cli))) addToast('Command copied', 'success');
    else addToast('Could not copy the command', 'error');
  }
</script>

<div class="disc-detail">
  <div><DetailGallery images={view.previews} title={view.title} emptyNote={view.emptyNote} /></div>
  <div class="disc-info">
    {#if view.lead}<p class="disc-lead" class:repo={view.leadIsRepo}>{view.lead}</p>{/if}
    <div>
      {#if madeFor}
        <div class="disc-kv">
          <span>Made for</span>
          <span class="v" role="status">
            {#if madeFor.kind === 'match'}<span class="ok"><Icon name="check" size={13} />{madeFor.text}</span>
            {:else}<span class="w"><Icon name="alert" size={13} />{madeFor.text}</span>{/if}
          </span>
        </div>
      {/if}
      {#if view.worksWith}<div class="disc-kv"><span>Works with</span><span class="v">{view.worksWith}</span></div>{/if}
      <div class="disc-kv">
        <span>Trigger</span>
        <span class="v">
          {#if editing}
            <span class="disc-edit">
              <input bind:this={inputEl} class="ui-field" type="text" aria-label="Trigger text" bind:value={draftTrigger} disabled={saving} onkeydown={(e) => { if (e.key === 'Enter') { e.preventDefault(); void save(draftTrigger.trim() || null); } else if (e.key === 'Escape') { e.stopPropagation(); editing = false; } }}>
              <button type="button" class="ui-btn ui-btn-sm" disabled={saving} onclick={() => void save(draftTrigger.trim() || null)}>Save</button>
              {#if typedTrigger}<button type="button" class="ui-btn ui-btn-sm" disabled={saving} onclick={() => void save(null)}>Clear</button>{/if}
              <button type="button" class="ui-btn ui-btn-quiet ui-btn-sm" disabled={saving} onclick={() => { editing = false; }}>Cancel</button>
            </span>
          {:else}
            <span class="disc-trig">
              {#if view.trigger}<span class="disc-code" title={view.trigger}>{view.trigger}</span>{:else}<span class="muted">None given</span>{/if}
              {#if installed}<button type="button" class="ui-btn ui-btn-quiet ui-btn-sm" onclick={() => void startEdit()}>Edit</button>{/if}
            </span>
          {/if}
        </span>
      </div>
      {#if view.weight !== null}<div class="disc-kv"><span>Weight</span><span class="v mono">{view.weight}</span></div>{/if}
      {#if files}{@render files()}{/if}
      <div class="disc-kv"><span>{installed ? 'File name' : 'Saves as'}</span><span class="v mono">{view.saveAs}</span></div>
      {#if view.size}<div class="disc-kv"><span>Size</span><span class="v mono">{view.size}</span></div>{/if}
      {#if view.downloads !== null}<div class="disc-kv"><span>Downloads</span><span class="v mono">{view.downloads.toLocaleString('en-US')}</span></div>{/if}
      {#if view.licence !== undefined}
        <div class="disc-kv">
          <span>Licence</span>
          <span class="v">{#if view.licence}{view.licence}{:else}<span class="muted">None given</span>{/if}{#if view.nonCommercial} <span class="ui-badge ui-badge-warning" style="margin-left: 4px">Non-commercial</span>{/if}</span>
        </div>
      {/if}
      {#if view.modelCardUrl}
        <div class="disc-kv"><span class="disc-wide"><a class="ui-link" href={view.modelCardUrl} target="_blank" rel="noopener noreferrer">Model card</a></span></div>
      {/if}
    </div>
    {#if view.cli}
      <div class="disc-cli">
        <code title={view.cli}>{view.cli}</code>
        <button type="button" class="ui-btn ui-btn-quiet ui-btn-icon ui-btn-sm" aria-label="Copy the command" title="Copy the command" onclick={() => void copyCli()}><Icon name="copy" size={13} /></button>
      </div>
    {/if}
  </div>
</div>
