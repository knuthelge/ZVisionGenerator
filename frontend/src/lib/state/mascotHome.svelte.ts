import { draft } from './draft.svelte';
import { jobStore } from './job.svelte';
import { homePlacement, REACTION_DURATION_MS, type MascotPlacement, type MascotReaction } from './mascot';

/** The Workspace's word on the mascot; null while the Workspace is closed. */
let _workspace = $state<MascotPlacement | null>(null);
/** A reaction to a job that ended while the Workspace was closed. */
let _reaction = $state<MascotReaction | null>(null);
let _reactionTimer: ReturnType<typeof setTimeout> | undefined;

/** The mascot's home in the logo: the Workspace sends him out to a spot and calls him back. */
export const mascotHome = {
  get placement(): MascotPlacement {
    return homePlacement(_workspace, jobStore.current, draft.state.workflow, _reaction);
  },
  publish(placement: MascotPlacement): void {
    _workspace = placement;
  },
  clear(): void {
    _workspace = null;
  },
  /** React at home for a moment, for a job that ended off the Workspace. */
  react(reaction: MascotReaction): void {
    clearTimeout(_reactionTimer);
    _reaction = reaction;
    _reactionTimer = setTimeout(() => { _reaction = null; }, REACTION_DURATION_MS[reaction]);
  },
};
