<script lang="ts">
  import { fitName, measurerFor, watchName } from './nameFit';

  interface Props {
    /** The file's full name. */
    name: string;
    class?: string;
  }

  let { name, class: extraClass = '' }: Props = $props();

  let element = $state<HTMLSpanElement | null>(null);
  // The text on screen: the name whole until the element has been measured.
  let shown = $state<string | null>(null);
  const text = $derived(shown ?? name);

  // Fit when the name or the element's width changes. The width is the layout's to give, never the text's, so the cut
  // text cannot change the width it was cut for.
  $effect(() => {
    const el = element;
    if (!el) return;
    const current = name;
    let fittedWidth = -1;
    return watchName(el, () => {
      const width = el.clientWidth;
      if (width === fittedWidth && shown !== null) return;
      const measure = width > 0 ? measurerFor(el) : null;
      fittedWidth = width;
      shown = measure ? fitName(current, width, measure) : current;
    });
  });
</script>

<!-- One text node, "start…tail", cut to the element's width; the full name is the title and the accessible name. -->
<span bind:this={element} class="file-name {extraClass}" title={name}><span class="sr-only">{name}</span><span aria-hidden="true">{text}</span></span>

<style>
  .file-name { display: block; min-width: 0; overflow: hidden; white-space: nowrap; }
</style>
