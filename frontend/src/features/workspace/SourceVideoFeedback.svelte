<script lang="ts">
  import { draft } from '$lib/state/draft.svelte';
  import { Alert } from '$lib/components/molecules';
  import { extendInspection } from '$lib/state/extendInspection.svelte';

  interface Props {
    /** Id for `aria-describedby` on the path field; leave it out where another copy already owns it. */
    id?: string;
  }

  let { id }: Props = $props();

  const sourcePath = $derived(draft.state.extendSourcePath);
  const sourceStatus = $derived(extendInspection.statusFor(sourcePath));
  const sourceFailure = $derived(extendInspection.failureFor(sourcePath));
  const problems = $derived(draft.extendSource?.problems ?? []);
</script>

<!-- Why the chosen source video cannot be extended, if it cannot. -->
<div {id} class="source-feedback" aria-live="polite">
  {#if draft.state.missingSourceVideo}
    <Alert tone="warning" testId="extend-source-moved">The source video has moved or been deleted; choose it again to extend.</Alert>
  {:else if sourceStatus === 'not_found'}
    <p class="ui-help ui-help-error">No video at this path.</p>
  {:else if sourceFailure}
    <p class="ui-help ui-help-error">{sourceFailure}</p>
  {:else if problems.length > 0}
    <Alert tone="error" testId="extend-source-problems">
      <p>This video can't be extended:</p>
      <ul class="problem-list">
        {#each problems as problem (problem.code)}<li>{problem.message}</li>{/each}
      </ul>
    </Alert>
  {/if}
</div>

<style>
  .problem-list { margin-top: 2px; padding-left: 16px; list-style: disc; }
</style>
