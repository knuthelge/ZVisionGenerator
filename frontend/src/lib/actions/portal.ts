/**
 * Move an overlay's root to `<body>` so no ancestor's stacking context, transform or overflow can trap it.
 *
 * Sheets, drawers, menus and dialogs are fixed to the viewport; rendered in place they inherit the
 * stacking context of whatever opened them (an isolated tile, a positioned pane) and paint under
 * their neighbours. Moving the node out keeps the z-index ladder in `global.css` the only order
 * that matters. The node must be the only root of its `{#if}` block, so Svelte removes it from its
 * new parent when the block tears down.
 */
export function portal(node: HTMLElement): { destroy: () => void } {
  const target = document.body;
  target.appendChild(node);
  return {
    destroy() {
      node.remove();
    },
  };
}
