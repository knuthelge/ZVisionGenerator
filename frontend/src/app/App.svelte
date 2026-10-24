<script lang="ts">
  import { onMount } from 'svelte';
  import { router } from '$lib/state/router.svelte';
  import { getConfig } from '$lib/api/config';
  import { shouldApplyStartupView } from './startupRouting';
  import { jobStore } from '$lib/state/job.svelte';
  import { documentTitle, isReviewing } from '$lib/state/jobView';
  import { requestJobSettings } from '$lib/state/pendingJobSettings.svelte';
  import type { JobLogEntry, PageId, Workflow } from '$lib/types';
  import Spinner from '$lib/components/atoms/Spinner.svelte';
  import TopNav from '$lib/components/organisms/TopNav.svelte';
  import TopStrip from '$lib/components/organisms/TopStrip.svelte';
  import NavDrawer from '$lib/components/organisms/NavDrawer.svelte';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { compactUi } from '$lib/state/compactUi.svelte';
  import { scrollToTop } from '$lib/scrollToTop';
  import ToastContainer from '$lib/components/molecules/ToastContainer.svelte';
  import ShortcutsDialog from '$lib/components/organisms/ShortcutsDialog.svelte';
  import JobLogDrawer from '$features/jobLog/JobLogDrawer.svelte';
  import WorkspacePage from '$features/workspace/WorkspacePage.svelte';
  import GalleryPage from '$features/gallery/GalleryPage.svelte';
  import PromptsPage from '$features/prompts/PromptsPage.svelte';
  import ConfigPage from '$features/config/ConfigPage.svelte';
  import ModelsPage from '$features/models/ModelsPage.svelte';

  const _validPages: PageId[] = ['workspace', 'prompts', 'gallery', 'config', 'models'];

  let ready = $state(false);
  // The workflows the Workspace's contract lists; the Workspace fills it once its context has loaded.
  let workflows = $state<Workflow[] | null>(null);
  // Phones: the pages live in a drawer behind the strip's menu button.
  let drawerOpen = $state(false);
  let jobLogOpen = $state(false);
  // The page area: a tap on the phone's title scrolls whatever is scrolled in it back to the top.
  let pageEl = $state<HTMLDivElement | null>(null);

  /** Hand a logged job's settings to the Workspace, which applies them once it has loaded. */
  function loadJobLogSettings(entry: JobLogEntry): void {
    requestJobSettings(entry.settings);
    jobLogOpen = false;
    router.navigate('workspace');
  }

  const baseTitle = document.title;
  $effect(() => {
    document.title = documentTitle(baseTitle, isReviewing(jobStore.current));
  });

  onMount(async () => {
    // Only apply startup_view when no explicit URL destination was specified
    if (shouldApplyStartupView(window.location.hash, window.location.search)) {
      try {
        const appConfig = await getConfig();
        const startupView = appConfig.ui.startup_view ?? 'workspace';
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
  <div class="ui-viewport flex flex-col overflow-hidden bg-bg-base text-text-primary font-sans" data-layout={layout.mode}>
    {#if layout.compact}
      <TopStrip currentPage={router.page} {workflows} onmenu={() => { drawerOpen = true; }} onstatus={() => compactUi.requestJobSheet()} onjoblog={() => { jobLogOpen = true; }} ontitle={() => { if (pageEl) scrollToTop(pageEl); }} />
      <NavDrawer open={drawerOpen} currentPage={router.page} onnavigate={(page) => router.navigate(page)} onclose={() => { drawerOpen = false; }} />
    {:else}
      <TopNav currentPage={router.page} {workflows} onjoblog={() => { jobLogOpen = true; }} />
    {/if}
    <div bind:this={pageEl} class="flex-1 flex flex-col min-h-0 overflow-hidden">
      {#if router.page === 'workspace'}
        <WorkspacePage bind:workflows />
      {:else if router.page === 'prompts'}
        <PromptsPage />
      {:else if router.page === 'gallery'}
        <GalleryPage />
      {:else if router.page === 'config'}
        <ConfigPage />
      {:else if router.page === 'models'}
        <ModelsPage />
      {/if}
    </div>
    <ToastContainer />
    <ShortcutsDialog currentPage={router.page} onnavigate={(page) => router.navigate(page)} onopenjoblog={() => { jobLogOpen = true; }} />
    <JobLogDrawer bind:open={jobLogOpen} onloadsettings={loadJobLogSettings} />
  </div>
{:else}
  <div class="flex min-h-screen items-center justify-center bg-bg-base text-primary-main">
    <Spinner size="lg" label="Loading" />
  </div>
{/if}
