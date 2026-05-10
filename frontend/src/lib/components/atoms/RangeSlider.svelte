<script lang="ts">
  interface Props {
    id?: string;
    name?: string;
    value?: number;
    min?: number;
    max?: number;
    step?: number;
    disabled?: boolean;
    label?: string;
    showValue?: boolean;
    formatValue?: (v: number) => string;
    class?: string;
    oninput?: (event: Event) => void;
    onchange?: (event: Event) => void;
  }

  let {
    id,
    name,
    value = $bindable(0),
    min = 0,
    max = 100,
    step = 1,
    disabled = false,
    label,
    showValue = true,
    formatValue = (v) => String(v),
    class: extraClass = '',
    oninput,
    onchange
  }: Props = $props();

  const pct = $derived(((value - min) / (max - min)) * 100);
  const displayVal = $derived(formatValue(value));
</script>

<div class="flex flex-col gap-1 {extraClass}">
  {#if label || showValue}
    <div class="flex items-center justify-between">
      {#if label}
        <span class="text-xs text-zinc-400 font-medium uppercase tracking-wider">{label}</span>
      {/if}
      {#if showValue}
        <span class="text-xs font-mono text-zinc-300">{displayVal}</span>
      {/if}
    </div>
  {/if}
  <div class="relative h-5 flex items-center">
    <!-- Track background -->
    <div class="absolute inset-x-0 h-1.5 rounded-full bg-zinc-800 overflow-hidden">
      <!-- Filled portion -->
      <div
        class="h-full rounded-full bg-cyan-400 transition-all duration-75"
        style="width: {pct}%"
      ></div>
    </div>
    <!-- Native range input (transparent, on top) -->
    <input
      {id}
      {name}
      type="range"
      bind:value
      {min}
      {max}
      {step}
      {disabled}
      class="relative w-full h-1.5 appearance-none bg-transparent cursor-pointer disabled:cursor-not-allowed focus-visible:focus-ring rounded-full
             [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5
             [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-zinc-100
             [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-cyan-400
             [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:transition-transform
             [&::-webkit-slider-thumb]:hover:scale-110
             [&::-moz-range-thumb]:w-3.5 [&::-moz-range-thumb]:h-3.5
             [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-zinc-100
             [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-cyan-400
             [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:border-none"
      {oninput}
      {onchange}
    />
  </div>
</div>
