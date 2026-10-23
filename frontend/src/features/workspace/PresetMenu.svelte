<script lang="ts">
  import { untrack } from 'svelte';
  import { errorText } from '$lib/api/client';
  import { Icon } from '$lib/components/atoms';
  import { ActionMenu, requestConfirm, type ActionMenuEntry } from '$lib/components/molecules';
  import { draft } from '$lib/state/draft.svelte';
  import { presetSettingsFromDraft } from '$lib/state/presets';
  import { presetsStore } from '$lib/state/presets.svelte';
  import { workflowMode } from '$lib/state/promptEnhance';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { SettingsPreset, WorkspaceContext } from '$lib/types';
  import PresetNameDialog from './PresetNameDialog.svelte';

  interface Props {
    context: WorkspaceContext | null;
    /** The preset the form started from, or null. */
    applied: SettingsPreset | null;
    /** Whether the form has moved off the applied preset. */
    changed: boolean;
    /** The phone's All sheet: the button fills the row. */
    stacked?: boolean;
    /** Render the toolbar button. Without it the menu is opened from outside through `open` (the phone's Preset chip). */
    button?: boolean;
    open?: boolean;
  }

  let { context, applied, changed, stacked = false, button = true, open = $bindable(false) }: Props = $props();

  let anchor = $state<HTMLButtonElement | null>(null);
  let hovered = $state(false);
  let nameDialogOpen = $state(false);
  let nameDialogKind = $state<'save' | 'rename'>('save');

  const mode = $derived(workflowMode(draft.state.workflow));
  const modePresets = $derived(presetsStore.list.filter((preset) => preset.mode === mode));
  const ready = $derived(context !== null && draft.authorityReady);

  // The list is read again each time the menu opens, so presets saved or removed elsewhere show up.
  $effect(() => {
    if (open) untrack(() => void presetsStore.load());
  });

  const items = $derived.by<ActionMenuEntry[]>(() => {
    const entries: ActionMenuEntry[] = [];
    if (modePresets.length > 0) {
      entries.push({ kind: 'heading', label: mode === 'image' ? 'Image presets' : 'Video presets' });
      for (const preset of modePresets) {
        entries.push({ kind: 'item', id: `preset:${preset.slug}`, label: preset.name, checked: preset.slug === applied?.slug, onselect: () => apply(preset) });
      }
      entries.push({ kind: 'separator' });
    }
    if (applied && changed) entries.push({ kind: 'item', id: 'preset-update', label: `Update "${applied.name}"`, onselect: () => void update(applied) });
    entries.push({ kind: 'item', id: 'preset-save-as', label: 'Save as…', onselect: () => openNameDialog('save') });
    if (applied) {
      entries.push({ kind: 'item', id: 'preset-rename', label: `Rename "${applied.name}"…`, onselect: () => openNameDialog('rename') });
      entries.push({ kind: 'item', id: 'preset-delete', label: `Delete "${applied.name}"`, danger: true, onselect: () => void remove(applied) });
    }
    return entries;
  });

  function toggle(): void {
    open = !open;
  }

  function apply(preset: SettingsPreset): void {
    if (!context) return;
    draft.applyPreset(context, preset);
  }

  async function update(preset: SettingsPreset): Promise<void> {
    if (!context) return;
    try {
      const { model, settings } = presetSettingsFromDraft(draft.state, context);
      const saved = await presetsStore.update(preset.mode, preset.slug, { name: preset.name, model, settings });
      addToast(`Updated "${saved.name}"`, 'success');
    } catch (err) {
      addToast(errorText(err, 'Could not update the preset.'), 'error');
    }
  }

  function openNameDialog(kind: 'save' | 'rename'): void {
    nameDialogKind = kind;
    nameDialogOpen = true;
  }

  async function submitName(name: string): Promise<void> {
    if (nameDialogKind === 'rename' && applied) {
      const renamed = await presetsStore.rename(applied.mode, applied.slug, name);
      draft.setAppliedPreset(renamed.slug);
      return;
    }
    if (!context) return;
    const { model, settings } = presetSettingsFromDraft(draft.state, context);
    const saved = await presetsStore.create({ name, mode, model, settings });
    draft.setAppliedPreset(saved.slug);
    addToast(`Saved "${saved.name}"`, 'success');
  }

  async function remove(preset: SettingsPreset): Promise<void> {
    const confirmed = await requestConfirm({ question: `Delete the preset "${preset.name}"?`, info: 'Your current settings stay as they are.', confirmLabel: 'Delete', danger: true });
    if (!confirmed) return;
    try {
      await presetsStore.remove(preset.mode, preset.slug);
      if (draft.state.appliedPreset === preset.slug) draft.setAppliedPreset(null);
    } catch (err) {
      addToast(errorText(err, 'Could not delete the preset.'), 'error');
    }
  }
</script>

{#if button}
  <!-- The look of the toolbar Select (shell variant); the name stands in for a label. -->
  <button
    type="button"
    bind:this={anchor}
    class="ui-select-shell relative flex h-control items-center justify-between gap-2 rounded-sm px-2 text-ui transition-colors text-text-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-main {stacked ? 'w-full' : 'w-40 shrink-0'}"
    data-testid="preset-button"
    data-hovered={hovered ? 'true' : 'false'}
    data-changed={changed}
    data-applied={applied?.slug ?? ''}
    aria-haspopup="menu"
    aria-expanded={open}
    aria-label="Preset: {applied?.name ?? 'none'}"
    disabled={!ready}
    onpointerenter={() => { hovered = true; }}
    onpointerleave={() => { hovered = false; }}
    onclick={toggle}
  >
    <span class="flex min-w-0 items-center gap-2">
      <Icon name="list" size={16} class="shrink-0 text-primary-main" />
      <span class="truncate" class:text-text-muted={!applied}>{applied?.name ?? 'Preset'}</span>
      {#if changed}<span class="preset-dot" aria-hidden="true"></span>{/if}
    </span>
    <Icon name="chevdown" size={14} class="shrink-0 text-text-muted" />
  </button>
{/if}

<ActionMenu {open} anchor={button ? anchor : null} align="start" label="Preset" {items} onclose={() => { open = false; }} />

<PresetNameDialog
  bind:open={nameDialogOpen}
  title={nameDialogKind === 'rename' ? 'Rename preset' : 'Save preset'}
  initialName={nameDialogKind === 'rename' ? applied?.name ?? '' : ''}
  confirmLabel={nameDialogKind === 'rename' ? 'Rename' : 'Save'}
  onsubmit={submitName}
/>

<style>
  .preset-dot { flex: none; width: 6px; height: 6px; border-radius: 50%; background: var(--color-primary-main); }
</style>
