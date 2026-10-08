<script lang="ts">
  interface Props {
    /** 12px inside controls, 16px beside text, 24px for a page. */
    size?: 'sm' | 'md' | 'lg';
    /** Announce the spinner as a loading status; leave off when nearby text already says so. */
    label?: string;
    class?: string;
  }

  let { size = 'md', label, class: extraClass = '' }: Props = $props();

  const PX = { sm: 12, md: 16, lg: 24 } as const;
</script>

<span
  class="spinner {extraClass}"
  style:width="{PX[size]}px"
  style:height="{PX[size]}px"
  role={label ? 'status' : undefined}
  aria-label={label}
  aria-hidden={label ? undefined : 'true'}
></span>

<style>
  .spinner {
    display: inline-block;
    flex-shrink: 0;
    border: 2px solid currentColor;
    border-right-color: transparent;
    border-radius: 50%;
    animation: spinner-turn 0.7s linear infinite;
  }
  @keyframes spinner-turn {
    to { transform: rotate(360deg); }
  }
</style>
