import type { JobMemory, MemoryFitStatus, MemoryRow } from '$lib/types';

const WORDS: Record<MemoryFitStatus, string> = { fits: 'Fits', tight: 'Tight', too_large: 'Too large' };
const LABELS: Record<string, string> = { none: 'None', '8': 'q8', '4': 'q4', on: 'On', off: 'Off', only: '' };

/**
 * One row of a memory answer as the server sends it: its key, status and figures (GPU, then RAM; null for none), with the words the server
 * adds. The words are stand-ins for the server's, so a test checks that the client shows what it was sent, not what the server says.
 */
export function memoryRow(key: string, status: MemoryFitStatus, gpu_gb: number, ram_gb: number | null): MemoryRow {
  return { key, status, gpu_gb, ram_gb, word: WORDS[status], label: LABELS[key] ?? key, sentence: `The sentence of ${status}.`, details: `Details at ${key}` };
}

/** A discrete-GPU image answer with the None, q8 and q4 rows of an image whose current level is `current`. */
export function imageMemory(current = 'none', overrides: Partial<JobMemory> = {}): JobMemory {
  const rows = [memoryRow('none', 'tight', 8, 32.5), memoryRow('8', 'fits', 8, 17), memoryRow('4', 'too_large', 10.4, 5.9)];
  return {
    kind: 'discrete',
    budget: { gpu_gb: 10, ram_gb: 30.3 },
    control: 'quantize',
    current,
    rows,
    title: 'Memory title of the image',
    control_title: 'Quant',
    table_heading: 'Table heading of the quants',
    download_size_label: null,
    expected_seconds: null,
    ...overrides,
  };
}

/** A single-row answer for a model with no choice (MiniMax H3, audio): its status and figures. */
export function singleMemory(status: MemoryFitStatus, overrides: Partial<JobMemory> = {}): JobMemory {
  const row = memoryRow('only', status, status === 'fits' ? 6.7 : 12, null);
  return {
    kind: 'discrete',
    budget: { gpu_gb: 10, ram_gb: 30.3 },
    control: null,
    current: 'only',
    rows: [row],
    title: 'Memory title of the track',
    control_title: null,
    table_heading: null,
    download_size_label: null,
    expected_seconds: null,
    ...overrides,
  };
}
