import type { ActiveJobState, MemoryReading } from '$lib/types';
import { oneDecimal } from './memoryEstimate';

/** One figure of the readout, such as `GPU 7.1 / 10 GB`. */
export interface MemoryReadoutPart {
  /** `Memory`, `GPU` or `RAM`. */
  label: string;
  /** The used amount in GB, one decimal with a trailing `.0` dropped; `—` while it is not known. */
  figure: string;
  /** Set when the used amount is not known yet (the GPU before CUDA has started), so a screen reader is not read a dash. */
  unknown?: true;
  /** The total it is measured against in GB, formatted the same way. */
  budget: string;
}

export interface MemoryReadout {
  parts: MemoryReadoutPart[];
  /** The whole line, e.g. `GPU 7.1 / 10 GB · RAM 21.4 / 30.3 GB`. */
  text: string;
}

/** Shown for a used amount that is not known yet. */
const UNKNOWN_FIGURE = '—';

function isFiniteAll(...values: unknown[]): boolean {
  return values.every((value) => typeof value === 'number' && Number.isFinite(value));
}

function liveParts(memory: MemoryReading): MemoryReadoutPart[] | null {
  if (memory.kind === 'unified') {
    if (!isFiniteAll(memory.used_gb, memory.budget_gb)) return null;
    return [{ label: 'Memory', figure: oneDecimal(memory.used_gb), budget: oneDecimal(memory.budget_gb) }];
  }
  if (memory.kind !== 'discrete' || !isFiniteAll(memory.gpu_budget_gb, memory.system_gb, memory.system_budget_gb)) return null;
  if (memory.gpu_gb !== null && !isFiniteAll(memory.gpu_gb)) return null;
  const gpu = memory.gpu_gb === null ? { figure: UNKNOWN_FIGURE, unknown: true as const } : { figure: oneDecimal(memory.gpu_gb) };
  return [
    { label: 'GPU', ...gpu, budget: oneDecimal(memory.gpu_budget_gb) },
    { label: 'RAM', figure: oneDecimal(memory.system_gb), budget: oneDecimal(memory.system_budget_gb) },
  ];
}

/**
 * Build the memory line of a job card: plain figures for the machine's load (everything in use, not only this app's share), with no verdict.
 *
 * A job that is still active shows its newest reading: on unified memory, the memory in use over the machine's total memory (`Memory 18.2 / 32 GB`);
 * on a discrete GPU, `GPU 7.1 / 10 GB · RAM 21.4 / 30.3 GB`, with `GPU — / 10 GB` while the card's use is not known yet. A job that has ended
 * shows none. Returns null when there is nothing to show.
 */
export function memoryReadout(job: Pick<ActiveJobState, 'status' | 'memory'>): MemoryReadout | null {
  const live = job.status === 'queued' || job.status === 'running' || job.status === 'paused';
  const parts = live && job.memory ? liveParts(job.memory) : null;
  if (!parts) return null;
  return { parts, text: parts.map((part) => `${part.label} ${part.figure} / ${part.budget} GB`).join(' · ') };
}
