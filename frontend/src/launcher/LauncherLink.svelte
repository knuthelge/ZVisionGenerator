<script lang="ts">
  import type { LinkId } from './types';

  interface Props {
    /** The page it opens. */
    id: LinkId;
    label: string;
    onclick: () => void;
  }

  let { id, label, onclick }: Props = $props();
</script>

<!-- A link that opens the system browser: a button, because the window must never navigate itself. -->
<button type="button" class="launcher-link" data-link={id} {onclick}>{label}<span class="sr-only"> (opens in your browser)</span></button>

<style>
  .launcher-link {
    padding: 0;
    background: none;
    border: 0;
    font: inherit;
    color: var(--color-primary-main);
    text-decoration: underline;
    text-underline-offset: 2px;
    white-space: nowrap;
    cursor: pointer;
  }
</style>
