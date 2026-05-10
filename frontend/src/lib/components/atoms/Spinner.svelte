<script lang="ts">
  interface Props {
    size?: 'sm' | 'md' | 'lg';
    class?: string;
  }

  let {
    size = 'md',
    class: extraClass = ''
  }: Props = $props();

  const sizes: Record<string, string> = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8'
  };

  const cls = $derived(`animate-spin text-teal-400 ${sizes[size]} ${extraClass}`);
</script>

<svg
  class={cls}
  fill="none"
  viewBox="0 0 24 24"
  aria-label="Loading"
  role="status"
>
  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
  <path
    class="opacity-75"
    fill="currentColor"
    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
  ></path>
</svg>
