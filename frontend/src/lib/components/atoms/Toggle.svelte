<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    id?: string;
    name?: string;
    checked?: boolean;
    disabled?: boolean;
    label?: string;
    labelSnippet?: Snippet;
    class?: string;
    onchange?: (event: Event) => void;
  }

  let {
    id,
    name,
    checked = $bindable(false),
    disabled = false,
    label,
    labelSnippet,
    class: extraClass = '',
    onchange
  }: Props = $props();
</script>

<label class="inline-flex items-center gap-3 cursor-pointer {disabled ? 'opacity-50 cursor-not-allowed' : ''} {extraClass}">
  <div class="relative shrink-0">
    <input
      {id}
      {name}
      type="checkbox"
      bind:checked
      {disabled}
      class="sr-only peer"
      {onchange}
    />
    <!-- Track -->
    <div
      class="w-9 h-5 rounded-full bg-zinc-700 peer-checked:bg-teal-500 transition-colors
             peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-offset-2
             peer-focus-visible:ring-offset-zinc-950 peer-focus-visible:ring-teal-400"
    ></div>
    <!-- Thumb -->
    <div
      class="absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow-sm
             transition-transform peer-checked:translate-x-4"
    ></div>
  </div>
  {#if label}
    <span class="text-sm text-zinc-300 select-none">{label}</span>
  {:else if labelSnippet}
    {@render labelSnippet()}
  {/if}
</label>
