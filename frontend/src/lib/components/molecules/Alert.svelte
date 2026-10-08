<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Icon } from '$lib/components/atoms';

  interface Props {
    tone?: 'info' | 'success' | 'warning' | 'error';
    /** Errors are announced right away; other tones politely. */
    live?: boolean;
    testId?: string;
    class?: string;
    children?: Snippet;
  }

  let { tone = 'info', live = false, testId, class: extraClass = '', children }: Props = $props();

  const ICONS = { info: 'info', success: 'check', warning: 'alert', error: 'alert' } as const;
</script>

<div
  class="ui-alert ui-alert-{tone} {extraClass}"
  role={live ? (tone === 'error' ? 'alert' : 'status') : undefined}
  data-testid={testId}
>
  <Icon name={ICONS[tone]} size={14} />
  <div class="min-w-0 flex-1 break-words">{@render children?.()}</div>
</div>
