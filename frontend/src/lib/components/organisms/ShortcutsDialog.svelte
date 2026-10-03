<script lang="ts">
  import { ShortcutList } from '$lib/components/atoms';
  import Modal from '$lib/components/molecules/Modal.svelte';
  import { VIEWER_SHORTCUTS } from '$lib/components/molecules/viewerShortcuts';
  import { acceptsPageShortcut, APP_SHORTCUTS, stepGoChord, type ShortcutGroup } from '$lib/keyboard';
  import type { PageId } from '$lib/types';

  interface Props {
    currentPage: PageId;
    onnavigate: (page: PageId) => void;
  }

  let { currentPage, onnavigate }: Props = $props();

  const GROUPS: readonly ShortcutGroup[] = [...APP_SHORTCUTS, { title: 'Asset viewer', entries: VIEWER_SHORTCUTS }];

  let open = $state(false);

  // `?` opens this list; `G` then a page key navigates. Both stay out of fields and open dialogs.
  $effect(() => {
    let waitingSince: number | null = null;
    function handleKeydown(event: KeyboardEvent): void {
      if (!acceptsPageShortcut(event)) {
        waitingSince = null;
        return;
      }
      if (event.key === '?') {
        event.preventDefault();
        waitingSince = null;
        open = true;
        return;
      }
      const step = stepGoChord(waitingSince, event.key, Date.now());
      const started = step.waitingSince !== null && waitingSince === null;
      waitingSince = step.waitingSince;
      if (step.page) {
        event.preventDefault();
        if (step.page !== currentPage) onnavigate(step.page);
      } else if (started) {
        event.preventDefault();
      }
    }
    document.addEventListener('keydown', handleKeydown);
    return () => document.removeEventListener('keydown', handleKeydown);
  });
</script>

<Modal bind:open title="Keyboard shortcuts" size="lg">
  <div class="shortcut-groups" data-testid="shortcut-groups">
    {#each GROUPS as group (group.title)}
      <section>
        <h3 class="field-label mb-2">{group.title}</h3>
        <ShortcutList entries={group.entries} />
      </section>
    {/each}
  </div>
</Modal>

<style>
  .shortcut-groups { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px 28px; }
</style>
