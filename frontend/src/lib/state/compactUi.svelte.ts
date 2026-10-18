/** Requests between the phone's shell and the Workspace that have no parent in common: the strip's job state opens the job sheet. */
let _jobSheetRequests = $state(0);

export const compactUi = {
  /** Bumps each time the job sheet (the running job's details and the queue) is asked for. */
  get jobSheetRequests(): number { return _jobSheetRequests; },
  requestJobSheet(): void {
    _jobSheetRequests += 1;
  },
};
