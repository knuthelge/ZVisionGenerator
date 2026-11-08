<script lang="ts">
  import { tick } from 'svelte';
  import { errorText } from '$lib/api/client';
  import { projectsStore } from '$lib/state/projects.svelte';
  import Modal from './Modal.svelte';

  interface Props {
    open?: boolean;
    /** The main button's label: `Create`, or `Create and move` from a Move list. */
    createLabel?: string;
    /** The project exists now (the server may have normalised its name). The dialog closes after this returns. */
    oncreated: (name: string) => void;
  }

  let { open = $bindable(false), createLabel = 'Create', oncreated }: Props = $props();

  let name = $state('');
  let busy = $state(false);
  let refusal = $state<string | null>(null);
  let field = $state<HTMLInputElement | null>(null);

  const trimmed = $derived(name.trim());
  const canCreate = $derived(trimmed !== '' && !busy);

  // Every opening starts with an empty field and no refusal.
  $effect(() => {
    if (!open) return;
    name = '';
    refusal = null;
    void tick().then(() => field?.focus());
  });

  async function create(): Promise<void> {
    if (!canCreate) return;
    busy = true;
    refusal = null;
    try {
      const project = await projectsStore.create(trimmed);
      oncreated(project.name);
      open = false;
    } catch (err) {
      refusal = errorText(err, 'Could not create the project.');
      void tick().then(() => field?.focus());
    } finally {
      busy = false;
    }
  }
</script>

<Modal bind:open title="New project" size="sm">
  {#snippet children()}
    <form class="project-name" onsubmit={(event) => { event.preventDefault(); void create(); }}>
      <label class="ui-label" for="project-name">Name</label>
      <input
        id="project-name"
        class="ui-field"
        bind:this={field}
        bind:value={name}
        oninput={() => { refusal = null; }}
        spellcheck="false"
        autocomplete="off"
        aria-invalid={refusal ? 'true' : undefined}
        aria-describedby={refusal ? 'project-name-error' : undefined}
      >
      {#if refusal}<p class="ui-help ui-help-error" id="project-name-error" role="alert">{refusal}</p>{/if}
    </form>
  {/snippet}
  {#snippet footer()}
    <button type="button" class="ui-btn" onclick={() => { open = false; }}>Cancel</button>
    <button type="button" class="ui-btn ui-btn-primary" data-action="create-project" disabled={!canCreate} onclick={() => void create()}>{createLabel}</button>
  {/snippet}
</Modal>

<style>
  .project-name { display: flex; flex-direction: column; gap: 8px; }
</style>
