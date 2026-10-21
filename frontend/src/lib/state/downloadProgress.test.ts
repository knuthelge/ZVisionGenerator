import { describe, it, expect } from 'vitest';
import type { DownloadItem } from '$lib/types';
import { downloadLine, downloadStatusText, partLabel, sizeLabel } from './downloadProgress';

const base = { bytes_done: 0, bytes_total: null, bytes_per_sec: null, download_eta_secs: null };

describe('downloadLine', () => {
  it('shows GB with one decimal, speed and ETA', () => {
    expect(downloadLine({ bytes_done: 12.4e9, bytes_total: 34.7e9, bytes_per_sec: 48e6, download_eta_secs: 470 }))
      .toBe('12.4 / 34.7 GB · 48 MB/s · about 8 min');
  });

  it('uses decimal units in the total\'s unit', () => {
    expect(downloadLine({ ...base, bytes_done: 500e6, bytes_total: 1.5e9 })).toBe('0.5 / 1.5 GB');
    expect(downloadLine({ ...base, bytes_done: 120e6, bytes_total: 800e6 })).toBe('120 / 800 MB');
    expect(downloadLine({ ...base, bytes_done: 12e3, bytes_total: 99e3 })).toBe('12 / 99 KB');
  });

  it('shows one decimal of speed below 10 MB/s', () => {
    expect(downloadLine({ ...base, bytes_done: 1e6, bytes_total: 2e6, bytes_per_sec: 4.25e6 })).toBe('1 / 2 MB · 4.3 MB/s');
    expect(downloadLine({ ...base, bytes_done: 1e6, bytes_total: 2e6, bytes_per_sec: 10.4e6 })).toBe('1 / 2 MB · 10 MB/s');
  });

  it('omits null speed and ETA segments', () => {
    expect(downloadLine({ ...base, bytes_done: 1e9, bytes_total: 2e9, download_eta_secs: 120 })).toBe('1.0 / 2.0 GB · about 2 min');
    expect(downloadLine({ ...base, bytes_done: 1e9, bytes_total: 2e9, bytes_per_sec: 50e6 })).toBe('1.0 / 2.0 GB · 50 MB/s');
  });

  it('shows done and speed only when the total is unknown', () => {
    expect(downloadLine({ ...base, bytes_done: 3.2e9, bytes_per_sec: 20e6, download_eta_secs: 300 })).toBe('3.2 GB · 20 MB/s');
    expect(downloadLine({ ...base, bytes_done: 40e6 })).toBe('40 MB');
  });

  it('bands the ETA', () => {
    const eta = (secs: number) => downloadLine({ ...base, bytes_done: 1e6, bytes_total: 2e6, download_eta_secs: secs });
    expect(eta(0)).toBe('1 / 2 MB · less than a minute');
    expect(eta(59)).toBe('1 / 2 MB · less than a minute');
    expect(eta(60)).toBe('1 / 2 MB · about 1 min');
    expect(eta(89 * 60)).toBe('1 / 2 MB · about 89 min');
    expect(eta(90 * 60)).toBe('1 / 2 MB · about 1 h 30 min');
    expect(eta(3 * 3600 + 5 * 60)).toBe('1 / 2 MB · about 3 h 5 min');
  });
});

describe('downloadLine unit and speed boundaries', () => {
  it('switches to GB at exactly 1e9 and to MB at exactly 1e6', () => {
    expect(downloadLine({ ...base, bytes_done: 5e8, bytes_total: 1e9 })).toBe('0.5 / 1.0 GB');
    expect(downloadLine({ ...base, bytes_done: 5e8, bytes_total: 999_999_999 })).toBe('500 / 1000 MB');
    expect(downloadLine({ ...base, bytes_done: 5e5, bytes_total: 1e6 })).toBe('1 / 1 MB');
    expect(downloadLine({ ...base, bytes_done: 5e5, bytes_total: 999_999 })).toBe('500 / 1000 KB');
  });

  it('shows a done value in the MB range in the total\'s GB unit', () => {
    expect(downloadLine({ ...base, bytes_done: 40e6, bytes_total: 2e9 })).toBe('0.0 / 2.0 GB');
    expect(downloadLine({ ...base, bytes_done: 250e6, bytes_total: 2e9 })).toBe('0.3 / 2.0 GB');
  });

  it('uses the done value\'s unit when the total is unknown', () => {
    expect(downloadLine({ ...base, bytes_done: 2e9, bytes_per_sec: 5e6 })).toBe('2.0 GB · 5.0 MB/s');
    expect(downloadLine({ ...base, bytes_done: 12e3 })).toBe('12 KB');
  });

  it('drops the decimal of speed at exactly 10 MB/s and keeps it just below', () => {
    const speed = (bps: number) => downloadLine({ ...base, bytes_done: 1e6, bytes_total: 2e6, bytes_per_sec: bps });
    expect(speed(10e6)).toBe('1 / 2 MB · 10 MB/s');
    expect(speed(9.9e6)).toBe('1 / 2 MB · 9.9 MB/s');
    expect(speed(0)).toBe('1 / 2 MB · 0.0 MB/s');
  });

  it('omits both speed and ETA when both are null', () => {
    expect(downloadLine({ ...base, bytes_done: 1e9, bytes_total: 2e9 })).toBe('1.0 / 2.0 GB');
  });
});

describe('sizeLabel', () => {
  it('uses decimal units like the server disk size labels', () => {
    expect(sizeLabel(34.7e9)).toBe('34.7 GB');
    expect(sizeLabel(92.4e6)).toBe('92 MB');
    expect(sizeLabel(4200)).toBe('4 KB');
    expect(sizeLabel(10)).toBe('1 KB');
  });
});

describe('partLabel', () => {
  it('labels only downloads with several parts', () => {
    expect(partLabel({ part: 1, parts: 2 })).toBe('part 1 of 2');
    expect(partLabel({ part: 2, parts: 3 })).toBe('part 2 of 3');
    expect(partLabel({ part: 1, parts: 1 })).toBe('');
  });
});

describe('downloadStatusText', () => {
  const item = (overrides: Partial<DownloadItem> = {}): DownloadItem => ({ id: 'd1', kind: 'model', name: 'klein9b', label: 'klein9b', state: 'running', message: null, ...overrides });

  it('has no text without a queued or running download', () => {
    expect(downloadStatusText(null)).toBeNull();
    expect(downloadStatusText(item({ state: 'done' }))).toBeNull();
    expect(downloadStatusText(item({ state: 'failed' }))).toBeNull();
    expect(downloadStatusText(item({ state: 'cancelled' }))).toBeNull();
  });

  it('says a queued download is queued', () => {
    expect(downloadStatusText(item({ state: 'queued' }))).toBe('Download queued');
  });

  it('shows the rounded percentage of a running download of known size', () => {
    expect(downloadStatusText(item({ bytes_done: 12.4e9, bytes_total: 29.5e9 }))).toBe('Downloading 42%');
    expect(downloadStatusText(item({ bytes_done: 0, bytes_total: 10 }))).toBe('Downloading 0%');
    expect(downloadStatusText(item({ bytes_done: 12, bytes_total: 10 }))).toBe('Downloading 100%');
  });

  it('shows no percentage before progress or a size is known', () => {
    expect(downloadStatusText(item())).toBe('Downloading');
    expect(downloadStatusText(item({ bytes_done: 5e6, bytes_total: null }))).toBe('Downloading');
    expect(downloadStatusText(item({ bytes_done: 5e6, bytes_total: 0 }))).toBe('Downloading');
  });
});
