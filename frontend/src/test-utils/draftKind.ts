import { draft } from '$lib/state/draft.svelte';
import type { ImageKind, VideoKind, WorkspaceContext } from '$lib/types';

/** Open a kind of the Image or Video tab the way the Workspace does: switch it, then settle the model and defaults for it, as the page's tab-change effect does. */
export function openKind(tab: 'image' | 'video', kind: ImageKind | VideoKind, ctx: WorkspaceContext | null): void {
  draft.setKind(tab, kind, ctx);
  if (ctx) draft.onTabChange(ctx);
}
