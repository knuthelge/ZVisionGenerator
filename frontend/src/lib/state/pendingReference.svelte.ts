// Reactive, so a Workspace that is already mounted notices a request made after it loaded.
let pending = $state<string | null>(null);

/** Ask the Workspace to add an image to its references once it has loaded (a route's params would replace the list). */
export function requestReference(path: string): void {
  pending = path;
}

/** Whether a request is waiting; reading it inside an effect re-runs the effect when one arrives. */
export function hasPendingReference(): boolean {
  return pending !== null;
}

/** Take the pending image path, if any; it is handed out once. */
export function takeReference(): string | null {
  const path = pending;
  pending = null;
  return path;
}
