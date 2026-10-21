// Whether the Workspace offers the first-run dialog, and whether the user has put it away for this page load.

import type { WorkspaceContext } from '$lib/types';

export interface FirstRunFacts {
  /** The download lane has been read, so an active download is known to be absent rather than not yet seen. */
  downloadsSeeded: boolean;
  /** An image model is queued or downloading. */
  activeImageDownload: boolean;
  /** The user closed, skipped or answered the dialog since the page loaded. */
  dismissed: boolean;
}

/** Closed for this page load; module-level so a remount of the Workspace (SPA navigation) does not reopen it. */
let _dismissed = $state(false);

export const firstRun = {
  get dismissed(): boolean { return _dismissed; },
  /** Put the dialog away until the page reloads. */
  dismiss(): void { _dismissed = true; },
  /** Forget the dismissal, for tests. */
  reset(): void { _dismissed = false; },
};

/**
 * Whether to offer the first-run dialog: every image model is known to be missing and none is on its way, the user
 * has not skipped it, and the context and the download lane have both loaded.
 */
export function shouldOfferFirstRun(context: WorkspaceContext | null, facts: FirstRunFacts): boolean {
  if (!context?.first_run || context.first_run.skipped) return false;
  if (context.image_models.length === 0 || !context.image_models.every((model) => model.downloaded === false)) return false;
  return facts.downloadsSeeded && !facts.activeImageDownload && !facts.dismissed;
}
