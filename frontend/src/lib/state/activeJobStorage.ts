const ACTIVE_JOB_STORAGE_KEY = 'ziv-active-job-id-v1';

function getStorage(): Storage | null {
  if (typeof sessionStorage === 'undefined') return null;
  return sessionStorage;
}

export function readActiveJobId(): string | null {
  try {
    const value = getStorage()?.getItem(ACTIVE_JOB_STORAGE_KEY)?.trim() ?? '';
    return value || null;
  } catch {
    return null;
  }
}

export function writeActiveJobId(jobId: string): void {
  try {
    getStorage()?.setItem(ACTIVE_JOB_STORAGE_KEY, jobId);
  } catch {
    // Storage can be unavailable in private or restricted browser contexts.
  }
}

export function clearActiveJobId(jobId?: string): void {
  try {
    const storage = getStorage();
    if (!storage) return;
    if (jobId && storage.getItem(ACTIVE_JOB_STORAGE_KEY) !== jobId) return;
    storage.removeItem(ACTIVE_JOB_STORAGE_KEY);
  } catch {
    // Storage can be unavailable in private or restricted browser contexts.
  }
}