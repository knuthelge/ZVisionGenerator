<script lang="ts">
  import { appendPromptEntry } from '$lib/api/promptFiles';
  import { errorText } from '$lib/api/client';
  import { Icon } from '$lib/components/atoms';
  import { ActionMenu, NewPromptFileDialog, PathBrowser, type ActionMenuEntry } from '$lib/components/molecules';
  import { fileName } from '$lib/components/molecules/assetDetails';
  import { lastSaveTarget, recentFiles, rememberRecent, rememberSaveTarget } from '$lib/state/promptFileStorage';
  import { router } from '$lib/state/router.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import { folderOf, saveTargets, type SaveSelection } from './savePrompt';

  interface Props {
    /** What to save, or why nothing can be saved. */
    selection: SaveSelection;
    /** The prompt file selected in the Workspace, offered as a target. */
    workspaceFile: string | null;
    /** The Enhanced tab is in view; only the wording of the disabled hint changes. */
    enhancedView?: boolean;
  }

  let { selection, workspaceFile, enhancedView = false }: Props = $props();

  let buttonEl = $state<HTMLButtonElement | null>(null);
  let menuOpen = $state(false);
  let browsing = $state(false);
  let newFileOpen = $state(false);
  let saving = $state(false);
  let targets = $state<string[]>([]);

  const blocked = $derived('block' in selection);
  const title = $derived.by(() => {
    if (saving) return 'Saving…';
    if ('block' in selection) {
      if (selection.block === 'enhancing') return 'Wait for the enhancement to finish';
      if (selection.block === 'empty') return enhancedView ? 'Nothing enhanced to save' : 'Write a prompt first';
    }
    return 'Save to prompt file…';
  });
  const items = $derived<ActionMenuEntry[]>([
    ...(targets.length
      ? [
          { kind: 'heading', label: 'Recent' } as const,
          ...targets.map((path): ActionMenuEntry => ({ kind: 'item', id: `target-${path}`, label: path, onselect: () => void saveTo(path) })),
          { kind: 'separator' } as const,
        ]
      : []),
    { kind: 'item', id: 'browse', label: 'Browse…', onselect: () => { browsing = true; } },
    { kind: 'item', id: 'new', label: 'New file…', onselect: () => { newFileOpen = true; } },
  ]);

  function open(): void {
    targets = saveTargets(lastSaveTarget(), workspaceFile, ...recentFiles());
    menuOpen = !menuOpen;
  }

  /** Append the selection to `path`, then remember the file and offer to open it. */
  async function saveTo(path: string): Promise<void> {
    if ('block' in selection || saving) return;
    saving = true;
    try {
      const res = await appendPromptEntry({ path, ...selection.entry });
      rememberRecent(res.path);
      rememberSaveTarget(res.path);
      addToast(`Saved to ${fileName(res.path)}.`, 'success', {
        timeout: 6000,
        action: { label: 'Open in Prompts', run: () => router.navigate('prompts', { path: res.path }) },
      });
    } catch (err) {
      addToast(errorText(err, 'Could not save the prompt.'), 'error');
    } finally {
      saving = false;
    }
  }
</script>

<button
  bind:this={buttonEl}
  type="button"
  id="ws-save-prompt"
  class="ui-btn ui-btn-sm"
  aria-label="Save to prompt file…"
  aria-haspopup="menu"
  aria-expanded={menuOpen}
  disabled={blocked || saving}
  {title}
  onclick={open}
><Icon name="file" size={14} />Save…</button>
<ActionMenu open={menuOpen} anchor={buttonEl} {items} label="Save to prompt file" onclose={() => { menuOpen = false; }} />
<PathBrowser bind:open={browsing} purpose="prompt_file" initialPath={targets[0] ?? null} onchoose={(path) => void saveTo(path)} />
<NewPromptFileDialog bind:open={newFileOpen} initialDirectory={folderOf(targets[0] ?? null)} oncreated={(path) => void saveTo(path)} />
