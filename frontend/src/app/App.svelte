<script lang="ts">
  import { onMount } from 'svelte';
  import { router } from '$lib/state/router.svelte';
  import { getWorkspaceContext } from '$lib/api/workspace';
  import type { PageId } from '$lib/types';
  import TopNav from '$lib/components/organisms/TopNav.svelte';
  import ToastContainer from '$lib/components/molecules/ToastContainer.svelte';
  import WorkspacePage from '$features/workspace/WorkspacePage.svelte';
  import GalleryPage from '$features/gallery/GalleryPage.svelte';
  import ConfigPage from '$features/config/ConfigPage.svelte';
  import ModelsPage from '$features/models/ModelsPage.svelte';

  const _validPages: PageId[] = ['workspace', 'gallery', 'config', 'models'];

  let ready = $state(false);

  onMount(async () => {
    // Only apply startup_view when no explicit URL destination was specified
    if (!window.location.hash && !window.location.search.includes('page=')) {
      try {
        const ctx = await getWorkspaceContext();
        const startupView = ctx.config?.startup_view ?? 'workspace';
        if (_validPages.includes(startupView as PageId) && startupView !== 'workspace') {
          router.replace(startupView as PageId);
        }
      } catch {
        // ignore — stay on workspace
      }
    }
    ready = true;
  });
</script>

{#if ready}
  <div class="h-screen flex flex-col overflow-hidden bg-zinc-950 text-zinc-50 font-sans">
    <TopNav currentPage={router.page} />
    <div class="flex-1 flex flex-col min-h-0 overflow-hidden">
      {#if router.page === 'workspace'}
        <WorkspacePage />
      {:else if router.page === 'gallery'}
        <GalleryPage />
      {:else if router.page === 'config'}
        <ConfigPage />
      {:else if router.page === 'models'}
        <ModelsPage />
      {/if}
    </div>
    <ToastContainer />
  </div>
{:else}
  <div class="flex items-center justify-center min-h-screen bg-zinc-950">
    <div class="animate-spin rounded-full h-8 w-8 border-t-2 border-teal-500"></div>
  </div>
{/if}
