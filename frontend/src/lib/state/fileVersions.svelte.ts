// How many times a file on the host was rewritten in place from this page (an Adjust that replaced it). A preview URL
// built for it carries the count, so a browser that cached the old picture asks again. The server ignores the value:
// its own cache already follows the file's modified time.
let _versions = $state<Record<string, number>>({});

export const fileVersions = {
  /** The count of rewrites of *path* seen so far; 0 until it has been rewritten. */
  of(path: string): number {
    return _versions[path] ?? 0;
  },

  /** Record that *path* was rewritten. */
  bump(path: string): void {
    _versions[path] = (_versions[path] ?? 0) + 1;
  },
};
