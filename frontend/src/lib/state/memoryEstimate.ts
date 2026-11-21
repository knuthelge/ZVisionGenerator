import type { JobMemory, MemoryFitStatus, MemoryRow } from '$lib/types';

/** The badge colour of a status. The words come from the server. */
export const MEMORY_VARIANTS: Record<MemoryFitStatus, 'success' | 'warning' | 'error'> = { fits: 'success', tight: 'warning', too_large: 'error' };

/** One figure column of the table: its resource and the budget it is judged against. */
export interface MemoryColumn {
  label: 'GPU' | 'RAM' | 'Memory';
  /** The budget in GB, e.g. `10` or `30.3`. */
  budget: string;
}

/** What a table cell shows for a figure that was not measured. */
const NO_FIGURE = '—';

/** One row of the table: a Quant level or a Low memory setting of this same job, or the job itself when it has one level. */
export interface MemoryTableRow {
  key: string;
  /** The level as its control names it (the server's word); empty when the job has a single level (audio, MiniMax H3). */
  label: string;
  /** GPU (Windows and Linux) or Memory (Mac), in GB. */
  gpu: string;
  /** RAM in GB; absent on a Mac and where no row has a RAM figure; a dash for a row without one. */
  ram?: string;
  status: MemoryFitStatus;
  /** The status as the server words it. */
  word: string;
  current: boolean;
}

/** Everything the memory estimate shows, worked out from the server's answer. */
export interface MemoryEstimateView {
  status: MemoryFitStatus;
  /** What the answer is about, as the server titles it. */
  title: string;
  /** What was judged at the selected level, small and muted; empty when there is nothing to name. */
  details: string;
  /** One calm sentence about the selected level, as the server words it. No advice: the table beneath shows the other levels. */
  sentence: string;
  /** The title of the level column (`Quant`, `Low memory`); null for a single level, which has no level column. */
  control: string | null;
  /** The small heading over the table; null for a single level. */
  heading: string | null;
  /** The figure columns, each with its budget: GPU and RAM on Windows and Linux (no RAM for audio), Memory on a Mac. */
  columns: MemoryColumn[];
  /** Every level of this job, in the order the server sends; one row when there is a single level. */
  rows: MemoryTableRow[];
  /** The key of the row the status is of: the level asked for when the answer has it, else the one the answer says the job falls on. */
  selected: string;
}

const gb = (value: number): string => value.toFixed(1);

/** A figure with one decimal and a trailing `.0` dropped (`10`, `30.3`): GB amounts and seconds. */
export const oneDecimal = (value: number): string => String(Number(value.toFixed(1)));

/** The row key of a Quant setting: `none` for unquantized, else the bits. */
export const quantLevelKey = (quantize: number | null): string => (quantize === null ? 'none' : String(quantize));

/** The row key of a Low memory setting. */
export const lowMemoryLevelKey = (on: boolean): string => (on ? 'on' : 'off');

/** The row the tag judges: the level asked for when the answer has it, else the one the answer says the job falls on. */
export function currentRow(memory: JobMemory, level: string | null | undefined): MemoryRow {
  return memory.rows.find((row) => row.key === level) ?? memory.rows.find((row) => row.key === memory.current) ?? memory.rows[0];
}

/** Lay out one answer: the status, the column budgets and the rows of the table, with the server's words. */
export function memoryEstimateView(memory: JobMemory, level?: string | null): MemoryEstimateView {
  const selected = currentRow(memory, level);
  const mac = memory.kind === 'unified';
  const hasRam = !mac && memory.budget.ram_gb !== null && memory.rows.some((row) => row.ram_gb !== null);
  const columns: MemoryColumn[] = mac
    ? [{ label: 'Memory', budget: oneDecimal(memory.budget.gpu_gb) }]
    : [{ label: 'GPU', budget: oneDecimal(memory.budget.gpu_gb) }, ...(hasRam ? [{ label: 'RAM' as const, budget: oneDecimal(memory.budget.ram_gb ?? 0) }] : [])];
  const many = memory.control !== null && memory.rows.length > 1;
  const rows = many ? memory.rows : [selected];
  return {
    selected: selected.key,
    status: selected.status,
    title: memory.title,
    details: selected.details,
    sentence: selected.sentence,
    control: many ? memory.control_title : null,
    heading: many ? memory.table_heading : null,
    columns,
    rows: rows.map((row) => ({
      key: row.key,
      label: many ? row.label : '',
      gpu: gb(row.gpu_gb),
      ...(hasRam ? { ram: row.ram_gb === null ? NO_FIGURE : gb(row.ram_gb) } : {}),
      status: row.status,
      word: row.word,
      current: row.key === selected.key,
    })),
  };
}

/** What a screen reader hears for one row: `Quant q8 (selected): Fits, GPU 8.0, RAM 17.0 GB.` A button row leaves `(selected)` to its pressed state (`marked` off). */
export function memoryRowSpeech(view: MemoryEstimateView, row: MemoryTableRow, marked = true): string {
  const figures = row.ram === undefined ? `${row.gpu} GB` : row.ram === NO_FIGURE ? `GPU ${row.gpu} GB` : `GPU ${row.gpu}, RAM ${row.ram} GB`;
  const name = row.label ? `${view.control} ${row.label}${marked && row.current ? ' (selected)' : ''}: ` : '';
  return `${name}${row.word}, ${figures}.`;
}

/**
 * The sentence a screen reader hears for the estimate, in place of the laid-out table. Rows the user can choose are left out when
 * `withRows` is off: they are buttons, each named by `memoryRowSpeech`.
 */
export function memoryEstimateSpeech(memory: JobMemory, view: MemoryEstimateView, withRows = true): string {
  const parts = [`${view.title}:`, view.details ? `${view.details}.` : '', memory.download_size_label !== null ? `Not downloaded, about ${memory.download_size_label}.` : '', view.sentence, ...(memory.notes ?? [])];
  if (view.heading) parts.push(`${view.heading}.`);
  parts.push(`Budget: ${view.columns.map((c) => `${c.label} ${c.budget} GB`).join(', ')}.`);
  if (withRows) for (const row of view.rows) parts.push(memoryRowSpeech(view, row));
  return parts.filter(Boolean).join(' ');
}
