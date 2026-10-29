<script lang="ts">
  import { tick, untrack } from 'svelte';
  import { Icon, Spinner } from '$lib/components/atoms';
  import { Alert, Modal, requestConfirm } from '$lib/components/molecules';
  import { pushEscapeLayer } from '$lib/actions/escapeStack';
  import { errorText } from '$lib/api/client';
  import { getCurated, getHfCheck, getHfRepo, setDiscoverSettings, setLoraTrigger, startDiscoverDownload, type DiscoverSource } from '$lib/api/discover';
  import { deleteLora } from '$lib/api/models';
  import { downloadStore } from '$lib/state/downloads.svelte';
  import { jobStore } from '$lib/state/job.svelte';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { loraItem } from '$lib/state/loraDownloads';
  import type { LoraChip } from '$lib/state/loras';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { HfCheck, HfItem, HfRepo, CuratedList, WorkspaceContext } from '$lib/types';
  import './discover.css';
  import CuratedListView from './CuratedList.svelte';
  import DetailFooter from './DetailFooter.svelte';
  import HfList from './HfList.svelte';
  import HfOptIn from './HfOptIn.svelte';
  import InstalledListView from './InstalledList.svelte';
  import LoraDetail from './LoraDetail.svelte';
  import {
    curatedDetail,
    curatedOf,
    footerView,
    hfDetail,
    installedDetail,
    installedForModel,
    installedMadeFor,
    loraStatus,
    madeFor as hfMadeFor,
    type DetailView,
    type DiscoverTarget,
    type MadeFor,
    type SectionFilter,
  } from './discoverView';

  /** What turning on "Use in LoRAs" hands back to the Workspace: the LoRA to put on a chip, at its weight, with the trigger to offer. */
  export interface LoraUse {
    name: string;
    weight: number;
    trigger: string | null;
    /** The file is still downloading, so its trigger is not on an installed row yet. */
    downloading: boolean;
  }

  interface Props {
    open?: boolean;
    context: WorkspaceContext;
    /** What the Workspace's model is; null when its LoRA family is not known. */
    target: DiscoverTarget | null;
    /** The LoRAs in use now. */
    chips: LoraChip[];
    onuse: (use: LoraUse) => void;
    onstop: (name: string) => void;
    /** A file was deleted: the Workspace takes its chips off. */
    ondeleted: (name: string) => void;
    /** The LoRA rows changed (a typed trigger): read them again. */
    onloraschange: () => void;
  }

  let { open = $bindable(false), context, target, chips, onuse, onstop, ondeleted, onloraschange }: Props = $props();

  type Tab = 'installed' | 'curated' | 'hf';
  type Detail = { kind: 'curated'; id: string; from: Tab } | { kind: 'hf'; repo: string; from: Tab } | { kind: 'installed'; name: string; from: Tab };
  const TAB_LABELS: Record<Tab, string> = { installed: 'Installed', curated: 'Curated', hf: 'Hugging Face' };

  let tab = $state<Tab>('curated');
  let detail = $state<Detail | null>(null);
  let curated = $state<CuratedList | null>(null);
  let curatedError = $state<string | null>(null);
  let hfEnabled = $state(false);
  let optInSaving = $state(false);
  let optInError = $state<string | null>(null);
  let turnOffSaving = $state(false);
  let turnOffError = $state<string | null>(null);
  let curatedQuery = $state('');
  let curatedFilter = $state<SectionFilter>('all');
  let installedQuery = $state('');
  let installedFilter = $state<SectionFilter>('all');
  let hfQuery = $state('');
  let busy = $state(false);
  let useError = $state<string | null>(null);
  // Changes when a switch request ended without the state the switch shows, so the footer draws it again.
  let switchEpoch = $state(0);
  let online = $state(typeof navigator === 'undefined' ? true : navigator.onLine);
  let opener: HTMLElement | null = null;

  // The Hugging Face detail: the repository, the file chosen, and what its header says about the model.
  let repo = $state<HfRepo | null>(null);
  let repoError = $state<string | null>(null);
  let chosenFile = $state<string | null>(null);
  let check = $state<HfCheck | null>(null);
  let repoRequest = 0;
  let checkRequest = 0;

  const entries = $derived(curated?.entries ?? []);
  const facts = $derived({ chips, installed: context.loras, items: downloadStore.items, unlisted: downloadStore.unlisted });
  const modelName = $derived(target?.name ?? 'this model');
  const layoutName = $derived(layout.mode);
  const installed = $derived(target ? installedForModel(context.loras, target) : { main: [], other: [] });

  async function loadCurated(): Promise<void> {
    if (!target) return;
    try {
      const result = await getCurated({ family: target.family, size: target.size, version: target.version });
      curated = result;
      hfEnabled = result.hf_enabled;
      curatedError = null;
    } catch (err) {
      curatedError = errorText(err, 'Could not read the curated list.');
    }
  }

  // Each opening starts on Curated at the top, locked to the Workspace's model; a model with nothing curated starts on Hugging Face.
  let wasOpen = false;
  $effect(() => {
    if (!open) {
      wasOpen = false;
      return;
    }
    if (wasOpen) return;
    wasOpen = true;
    untrack(() => {
      detail = null;
      curated = null;
      curatedError = null;
      curatedQuery = '';
      curatedFilter = 'all';
      installedQuery = '';
      installedFilter = 'all';
      hfQuery = '';
      useError = null;
      tab = 'curated';
      void loadCurated().then(() => {
        if (curated && curated.entries.length === 0) tab = 'hf';
      });
    });
  });

  // A download that finishes turns its tile into an installed one.
  $effect(() => {
    if (!open) return;
    return downloadStore.onDownloaded(() => void loadCurated());
  });

  // Each file of a repository saves under a name of its own, and which names are taken changes as downloads start, finish or fail: read the open repository again.
  let laneSignature: string | null = null;
  const loraLane = $derived(downloadStore.items.filter((item) => item.kind === 'lora').map((item) => `${item.id}:${item.state}`).join(','));
  $effect(() => {
    const signature = loraLane;
    const changed = laneSignature !== null && signature !== laneSignature;
    laneSignature = signature;
    if (changed) untrack(() => void refreshRepo());
  });

  $effect(() => {
    const on = (): void => { online = true; };
    const off = (): void => { online = false; };
    window.addEventListener('online', on);
    window.addEventListener('offline', off);
    return () => { window.removeEventListener('online', on); window.removeEventListener('offline', off); };
  });

  // Escape closes the detail first; the dialog's own layer, below this one, then closes the drawer.
  $effect(() => {
    if (!open || !detail) return;
    return pushEscapeLayer(back);
  });

  const curatedEntry = $derived.by(() => {
    const shown = detail;
    return shown?.kind === 'curated' ? (entries.find((entry) => entry.id === shown.id) ?? null) : null;
  });
  const hfFile = $derived(repo ? ([...repo.files, ...repo.other_files].find((file) => file.file === chosenFile) ?? null) : null);
  const ownLora = $derived.by(() => {
    const shown = detail;
    return shown?.kind === 'installed' ? (context.loras.find((lora) => lora.name === shown.name) ?? null) : null;
  });

  const view = $derived<DetailView | null>(curatedEntry ? curatedDetail(curatedEntry, modelName) : repo && hfFile ? hfDetail(repo, hfFile) : ownLora ? installedDetail(ownLora) : null);
  const savedAs = $derived(view?.saveAs ?? null);
  const served = $derived(curatedEntry ? curatedEntry.state : hfFile ? (hfFile.state === 'installed' ? 'installed' : hfFile.state === 'downloading' ? 'downloading' : 'available') : null);
  // A file that merely shares the entry's name is the user's own, not this entry's installed copy.
  const installedRow = $derived(savedAs && served !== 'name_taken' ? (context.loras.find((lora) => lora.name === savedAs) ?? null) : null);
  const shownView = $derived(view && installedRow?.trigger ? { ...view, trigger: installedRow.trigger } : view);
  const status = $derived(savedAs ? loraStatus(savedAs, served, facts) : null);
  const deleteBlocked = $derived(jobStore.jobsActive);
  const footer = $derived(view && status ? footerView({ status, name: savedAs ?? '', offline: !online, deleteBlocked, busy }) : null);
  const made = $derived<MadeFor | null>(
    !target ? null : curatedEntry ? { kind: 'match', text: target.name } : detail?.kind === 'hf' ? hfMadeFor(check, target) : ownLora ? installedMadeFor(ownLora, target) : null,
  );

  async function openDetail(next: Detail, trigger: HTMLElement): Promise<void> {
    opener = trigger;
    useError = null;
    detail = next;
    repo = null;
    repoError = null;
    chosenFile = null;
    check = null;
    if (next.kind === 'hf') await loadRepo(next.repo);
    await tick();
    document.querySelector<HTMLElement>('.disc-back')?.focus();
  }

  async function loadRepo(name: string): Promise<void> {
    const request = ++repoRequest;
    try {
      const found = await getHfRepo(name);
      if (request !== repoRequest) return;
      repo = found;
      chosenFile = found.preselected ?? found.files[0]?.file ?? null;
      void runCheck();
    } catch (err) {
      if (request !== repoRequest) return;
      repoError = errorText(err, 'Could not read this repository.');
    }
  }

  /** Read the open repository again for the name and state of each file, keeping the file chosen and its check. */
  async function refreshRepo(): Promise<void> {
    if (detail?.kind !== 'hf' || !repo) return;
    const name = detail.repo;
    const request = ++repoRequest;
    try {
      const found = await getHfRepo(name);
      if (request === repoRequest) repo = found;
    } catch {
      // The files keep what they showed; the next change of the lane reads them again.
    }
  }

  async function runCheck(): Promise<void> {
    if (!repo || !chosenFile || !target) return;
    const request = ++checkRequest;
    check = null;
    try {
      const result = await getHfCheck(repo.repo, chosenFile, { family: target.family, size: target.size });
      if (request === checkRequest) check = result;
    } catch {
      // Without an answer the row stays out; the file is checked again once it is here.
    }
  }

  function chooseFile(file: string): void {
    chosenFile = file;
    void runCheck();
  }

  function back(): void {
    detail = null;
    repoRequest++;
    checkRequest++;
    void tick().then(() => { if (opener?.isConnected) opener.focus(); });
  }

  function openCurated(id: string, trigger: HTMLElement): void {
    void openDetail({ kind: 'curated', id, from: tab }, trigger);
  }

  function openInstalled(name: string, trigger: HTMLElement): void {
    const lora = context.loras.find((one) => one.name === name);
    const entry = lora ? curatedOf(lora, entries) : null;
    void openDetail(entry ? { kind: 'curated', id: entry.id, from: 'installed' } : { kind: 'installed', name, from: 'installed' }, trigger);
  }

  function openHf(item: HfItem, trigger: HTMLElement): void {
    if (item.curated_id && entries.some((entry) => entry.id === item.curated_id)) void openDetail({ kind: 'curated', id: item.curated_id, from: 'hf' }, trigger);
    else void openDetail({ kind: 'hf', repo: item.repo, from: 'hf' }, trigger);
  }

  function searchHf(words: string): void {
    hfQuery = words;
    tab = 'hf';
  }

  async function showHf(): Promise<void> {
    optInSaving = true;
    optInError = null;
    try {
      hfEnabled = (await setDiscoverSettings(true)).hf_enabled;
    } catch (err) {
      optInError = errorText(err, 'Could not save your answer.');
    } finally {
      optInSaving = false;
    }
  }

  async function turnOffHf(): Promise<void> {
    turnOffSaving = true;
    turnOffError = null;
    try {
      hfEnabled = (await setDiscoverSettings(false)).hf_enabled;
    } catch (err) {
      turnOffError = errorText(err, 'Could not save your answer.');
    } finally {
      turnOffSaving = false;
    }
  }

  /** The chip's weight: the entry's recommended one, or 1. */
  const weight = $derived(view?.weight ?? 1);

  /** Start the download of the LoRA shown; returns the name it saves as, or null when it could not start. */
  async function startDownload(): Promise<{ name: string; trigger: string | null; downloading: boolean } | null> {
    if (!view) return null;
    const source: DiscoverSource = curatedEntry ? { curated_id: curatedEntry.id } : { repo: repo!.repo, file: chosenFile!, name: view.saveAs };
    try {
      const started = await startDiscoverDownload(source);
      if (started.item) downloadStore.adopt(started.item);
      return { name: started.name, trigger: started.trigger, downloading: started.status !== 'installed' };
    } catch (err) {
      useError = errorText(err, 'Could not start the download.');
      return null;
    }
  }

  /** Start a failed download again; true when it is on its way. */
  async function retryDownload(): Promise<boolean> {
    const item = view ? loraItem(view.saveAs, downloadStore.items) : null;
    return item ? await downloadStore.retryLora(item) : false;
  }

  /** The button on the left: Download (no chip), Retry download, or Delete file. */
  async function act(): Promise<void> {
    if (!footer || !view || busy) return;
    const kind = footer.action.kind;
    if (kind === 'delete') return void (await remove());
    busy = true;
    useError = null;
    try {
      if (kind === 'retry') await retryDownload();
      else if (kind === 'download') await startDownload();
    } finally {
      busy = false;
    }
  }

  /** The switch: on puts the LoRA on your chips (downloading the file first when it is not here), off takes the chip away and never stops a download. */
  async function toggleUse(on: boolean): Promise<void> {
    if (!view || !status || busy || status.file === 'taken') return;
    const name = view.saveAs;
    if (!on) {
      onstop(name);
      return;
    }
    busy = true;
    useError = null;
    try {
      if (status.file === 'installed') {
        onuse({ name, weight, trigger: shownView?.trigger ?? null, downloading: false });
      } else if (status.file === 'downloading') {
        onuse({ name, weight, trigger: shownView?.trigger ?? null, downloading: true });
      } else if (status.file === 'failed') {
        if (await retryDownload()) onuse({ name, weight, trigger: shownView?.trigger ?? null, downloading: true });
        else switchEpoch++;
      } else {
        const started = await startDownload();
        if (started) onuse({ name: started.name, weight, trigger: started.trigger, downloading: started.downloading });
        else switchEpoch++;
      }
    } finally {
      busy = false;
    }
  }

  async function remove(): Promise<void> {
    if (!view) return;
    const name = view.saveAs;
    const approved = await requestConfirm({
      question: `Delete ${name}?`,
      info: status?.inUse ? "It's also removed from your LoRAs." : undefined,
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!approved) return;
    try {
      await deleteLora(name);
    } catch (err) {
      useError = errorText(err, 'Could not delete the file.');
      return;
    }
    downloadStore.forgetLora(name);
    ondeleted(name);
    addToast(`${name} deleted`, 'success');
    back();
    void loadCurated();
  }

  async function saveTrigger(trigger: string | null): Promise<void> {
    if (!savedAs) return;
    try {
      await setLoraTrigger(savedAs, trigger);
      onloraschange();
    } catch (err) {
      addToast(errorText(err, 'Could not save the trigger.'), 'error');
      throw err;
    }
  }
</script>

{#snippet fileChoice()}
  {#if repo && (repo.files.length > 1 || repo.other_files.length > 0)}
    <div class="disc-kv">
      <span>File</span>
      <span class="v">
        <span class="disc-files" role="radiogroup" aria-label="File to download">
          {#each repo.files as file (file.file)}
            <label><input type="radio" name="disc-file" value={file.file} checked={chosenFile === file.file} onchange={() => chooseFile(file.file)}><span class="f">{file.file}</span><span class="s">{file.size_label}</span></label>
          {/each}
          {#if repo.other_files.length > 0}
            <details open={repo.other_files.some((file) => file.file === chosenFile)}>
              <summary>Other files ({repo.other_files.length})</summary>
              {#each repo.other_files as file (file.file)}
                <label><input type="radio" name="disc-file" value={file.file} checked={chosenFile === file.file} onchange={() => chooseFile(file.file)}><span class="f">{file.file}</span><span class="s">{file.size_label}</span></label>
              {/each}
            </details>
          {/if}
        </span>
      </span>
    </div>
  {/if}
{/snippet}

{#snippet detailFoot()}
  {#if footer}<DetailFooter view={footer} error={useError} epoch={switchEpoch} onaction={() => void act()} onuse={(on) => void toggleUse(on)} />{/if}
{/snippet}

<Modal bind:open title="Discover LoRAs" size="lg" placement="side" fill footer={detail && footer ? detailFoot : undefined}>
  <div class="disc" data-layout={layoutName}>
    {#if !target}
      <div class="disc-none"><p>Discover needs a model whose LoRA family is known. Choose another model.</p></div>
    {:else if detail}
      <div class="disc-bar">
        <button type="button" class="ui-btn ui-btn-quiet ui-btn-sm disc-back" onclick={back}><Icon name="chevleft" size={14} />{TAB_LABELS[detail.from]}</button>
        <h3>{view?.title ?? curatedEntry?.display_name ?? 'LoRA'}{#if view?.tool}<span class="disc-tag">Tool</span>{/if}</h3>
        {#if layoutName !== 'compact'}<span class="disc-for">for {modelName}</span>{/if}
      </div>
      <div class="disc-scroll">
        {#if view && shownView}
          <LoraDetail
            view={shownView}
            madeFor={made}
            files={detail.kind === 'hf' ? fileChoice : undefined}
            installed={installedRow !== null}
            typedTrigger={installedRow?.trigger_source === 'user' ? (installedRow.trigger ?? null) : null}
            onsavetrigger={saveTrigger}
          />
        {:else if repoError}
          <div style="padding-top: 14px"><Alert tone="error" live>{repoError}</Alert></div>
        {:else if detail.kind === 'hf'}
          <p class="disc-loading" role="status"><Spinner size="sm" />Reading the repository…</p>
        {:else}
          <div class="disc-none"><p>This LoRA is not here any more.</p></div>
        {/if}
      </div>
    {:else}
      <div class="disc-bar" role="presentation">
        <div class="ui-segmented ui-segmented-sm" role="tablist" aria-label="Where to look">
          <button type="button" role="tab" id="disc-tab-installed" aria-selected={tab === 'installed'} aria-controls="disc-panel" onclick={() => { tab = 'installed'; }}>Installed <span class="disc-n">{installed.main.length}</span></button>
          <button type="button" role="tab" id="disc-tab-curated" aria-selected={tab === 'curated'} aria-controls="disc-panel" onclick={() => { tab = 'curated'; }}>Curated <span class="disc-n">{entries.length}</span></button>
          <button type="button" role="tab" id="disc-tab-hf" aria-selected={tab === 'hf'} aria-controls="disc-panel" onclick={() => { tab = 'hf'; }}>Hugging Face</button>
        </div>
        {#if layoutName !== 'compact'}<span class="disc-for">for {modelName}</span>{/if}
      </div>
    {/if}
    <!-- The lists stay in the page while a detail is open, so Back returns to the same place. -->
    <div id="disc-panel" class="disc-scroll" role="tabpanel" aria-labelledby={`disc-tab-${tab}`} hidden={detail !== null || !target}>
      {#if tab === 'installed'}
        <InstalledListView
          main={installed.main}
          other={installed.other}
          {entries}
          {facts}
          {modelName}
          query={installedQuery}
          onquery={(value) => { installedQuery = value; }}
          filter={installedFilter}
          onfilter={(value) => { installedFilter = value; }}
          onopen={openInstalled}
          oncurated={() => { tab = 'curated'; }}
        />
      {:else if tab === 'curated'}
        {#if curatedError}
          <div style="padding-top: 14px"><Alert tone="error" live>{curatedError}</Alert></div>
        {:else if !curated}
          <p class="disc-loading" role="status"><Spinner size="sm" />Loading the curated list…</p>
        {:else if entries.length === 0}
          <div class="disc-none">
            <p>Nothing is curated for {target?.familyName} yet.</p>
            <button type="button" class="ui-btn" onclick={() => searchHf('')}><Icon name="search" size={14} />Search Hugging Face</button>
          </div>
        {:else}
          <CuratedListView
            {entries}
            {facts}
            query={curatedQuery}
            onquery={(value) => { curatedQuery = value; }}
            filter={curatedFilter}
            onfilter={(value) => { curatedFilter = value; }}
            onopen={openCurated}
            onsearchhf={searchHf}
          />
        {/if}
      {:else if target}
        {#if !hfEnabled}
          <HfOptIn saving={optInSaving} error={optInError} onshow={() => void showHf()} onnotnow={() => { tab = 'curated'; }} />
        {:else}
          <HfList model={{ family: target.family, size: target.size, version: target.version }} query={hfQuery} onquery={(value) => { hfQuery = value; }} onopen={openHf} saving={turnOffSaving} error={turnOffError} onturnoff={() => void turnOffHf()} />
        {/if}
      {/if}
    </div>
  </div>
</Modal>
