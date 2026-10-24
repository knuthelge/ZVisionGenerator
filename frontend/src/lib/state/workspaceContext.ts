import type { WorkspaceContext } from '$lib/types';

// The last context the server answered with. The Workspace draws from it the moment it mounts and refreshes it in the
// background, so a return to the page is complete from its first frame (pages remember).
let _remembered: WorkspaceContext | null = null;

/** The context a previous visit loaded, or null before the first answer. */
export function rememberedContext(): WorkspaceContext | null {
  return _remembered;
}

/** Keep a context for the next visit. */
export function rememberContext(ctx: WorkspaceContext): void {
  _remembered = ctx;
}

/** Drop the remembered context (tests). */
export function forgetContext(): void {
  _remembered = null;
}
