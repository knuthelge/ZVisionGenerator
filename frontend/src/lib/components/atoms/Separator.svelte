<script lang="ts">
  interface Props {
    class?: string;
  }

  let { class: extraClass = '' }: Props = $props();
</script>

<hr class="border-0 border-b border-zinc-800 {extraClass}" />
