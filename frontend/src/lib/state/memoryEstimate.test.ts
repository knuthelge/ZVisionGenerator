import { describe, expect, it } from 'vitest';

import type { JobMemory } from '$lib/types';
import { imageMemory, memoryRow, singleMemory } from '../../test-utils/memory';
import { currentRow, lowMemoryLevelKey, memoryEstimateSpeech, memoryEstimateView, quantLevelKey } from './memoryEstimate';

const MAC: JobMemory = {
  ...imageMemory('4'),
  kind: 'unified',
  budget: { gpu_gb: 21.3, ram_gb: null },
  rows: [memoryRow('none', 'too_large', 40.1, null), memoryRow('4', 'tight', 24.4, null)].map((row) => ({ ...row, resource: 'unified' as const })),
};

describe('the memory estimate view', () => {
  it('lays the rows out in the order the server sends them and marks the current one', () => {
    const view = memoryEstimateView(imageMemory('4'));

    expect(view.rows.map((row) => [row.key, row.current])).toEqual([['none', false], ['8', false], ['4', true]]);
    expect(view.rows.map((row) => [row.label, row.word])).toEqual(imageMemory('4').rows.map((row) => [row.label, row.word]));
    expect(view.control).toBe('Quant');
    expect(view.columns).toEqual([{ label: 'GPU', budget: '10' }, { label: 'RAM', budget: '30.3' }]);
    expect(view.rows[0]).toMatchObject({ gpu: '8.0', ram: '32.5', status: 'tight' });
  });

  it('takes the status from the level asked for, not the answer\'s', () => {
    const memory = imageMemory('none');

    expect(memoryEstimateView(memory, '8').status).toBe('fits');
    expect(memoryEstimateView(memory, '4').rows.find((row) => row.current)?.key).toBe('4');
    expect(currentRow(memory, 'nope').key).toBe('none');
  });

  it('has one memory column on a Mac and no RAM figures', () => {
    const view = memoryEstimateView(MAC);

    expect(view.columns).toEqual([{ label: 'Memory', budget: '21.3' }]);
    expect(view.rows.every((row) => row.ram === undefined)).toBe(true);
  });

  it('has no RAM column for audio on Windows and Linux, and one row with no level column', () => {
    const view = memoryEstimateView(singleMemory('fits'));

    expect(view.columns.map((column) => column.label)).toEqual(['GPU']);
    expect(view.control).toBeNull();
    expect(view.rows).toHaveLength(1);
    expect(view.rows[0].label).toBe('');
  });

  it('shows a dash for a level whose RAM was not measured', () => {
    const memory = imageMemory('none');
    memory.rows[1] = { ...memory.rows[1], ram_gb: null };

    expect(memoryEstimateView(memory).rows.map((row) => row.ram)).toEqual(['32.5', '—', '5.9']);
  });

  it('does not read a dash aloud for a level whose RAM was not measured', () => {
    const memory = imageMemory('none');
    memory.rows[1] = { ...memory.rows[1], ram_gb: null };

    const speech = memoryEstimateSpeech(memory, memoryEstimateView(memory));

    expect(speech).not.toContain('—');
    expect(speech).toContain('RAM 32.5 GB');
    expect(speech).toContain('GPU 10.4, RAM 5.9 GB');
    expect(speech).toContain('Quant q8: Fits, GPU 8.0 GB.');
  });

  it('titles the level column with the server\'s word for the control', () => {
    const memory: JobMemory = { ...singleMemory('fits'), control: 'low_memory', control_title: 'Low memory', current: 'on', rows: [memoryRow('on', 'fits', 6, 20), memoryRow('off', 'tight', 9, 20)] };

    const view = memoryEstimateView(memory);

    expect(view.control).toBe('Low memory');
    expect(view.rows.map((row) => row.label)).toEqual(['On', 'Off']);
  });
});

describe('the words come from the server', () => {
  it('shows the title, the heading and the selected row\'s sentence and details as sent', () => {
    const memory = imageMemory('8');
    const view = memoryEstimateView(memory);

    expect(view.title).toBe(memory.title);
    expect(view.heading).toBe(memory.table_heading);
    expect(view.sentence).toBe(memory.rows[1].sentence);
    expect(view.details).toBe(memory.rows[1].details);
  });

  it('takes the sentence and the details of the level the form holds, not the answer\'s', () => {
    const memory = imageMemory('8');
    const view = memoryEstimateView(memory, '4');

    expect(view.sentence).toBe(memory.rows[2].sentence);
    expect(view.details).toBe(memory.rows[2].details);
  });

  it('has no column title or heading for a single level, whatever the answer carries', () => {
    const view = memoryEstimateView(imageMemory('8', { rows: [memoryRow('8', 'fits', 8, 17)] }));

    expect(view.control).toBeNull();
    expect(view.heading).toBeNull();
  });

  it('speaks what the server sent, the level the form holds and the notes after the sentence', () => {
    const memory = imageMemory('4', { notes: ['A note.'] });
    const speech = memoryEstimateSpeech(memory, memoryEstimateView(memory, '8'));

    expect(speech).toContain(memory.title);
    expect(speech).toContain(memory.rows[1].sentence);
    expect(speech).not.toContain(memory.rows[2].sentence);
    expect(speech.indexOf('A note.')).toBeGreaterThan(speech.indexOf(memory.rows[1].sentence));
    expect(speech).toContain(memory.table_heading!);
  });

  it('names a quant level and a Low memory level by the key the server judges', () => {
    expect(quantLevelKey(null)).toBe('none');
    expect(quantLevelKey(4)).toBe('4');
    expect(lowMemoryLevelKey(true)).toBe('on');
    expect(lowMemoryLevelKey(false)).toBe('off');
  });
});
