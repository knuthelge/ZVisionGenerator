import { describe, expect, it } from 'vitest';
import type { DownloadItem, LoraInfo } from '$lib/types';
import { chipDownload, downloadPercent, failureText, loraBlock, loraItem, waitText } from './loraDownloads';

const item = (overrides: Partial<DownloadItem> = {}): DownloadItem => ({ id: 'd1', kind: 'lora', name: 'style', label: 'LoRA style', state: 'running', message: null, bytes_done: 40, bytes_total: 100, ...overrides });
const file = (name: string): LoraInfo => ({ name, path: `/loras/${name}.safetensors` });

describe('downloadPercent', () => {
  it('reads the bytes done against the total, rounded down and held to 0..100', () => {
    expect(downloadPercent(item())).toBe(40);
    expect(downloadPercent(item({ bytes_done: 999, bytes_total: 100 }))).toBe(100);
  });

  it('is 0 while queued, 100 when done, and unknown without a size', () => {
    expect(downloadPercent(item({ state: 'queued', bytes_done: undefined }))).toBe(0);
    expect(downloadPercent(item({ state: 'done' }))).toBe(100);
    expect(downloadPercent(item({ bytes_done: undefined, bytes_total: null }))).toBeNull();
  });

  it('falls back to the size the download expects', () => {
    expect(downloadPercent(item({ bytes_total: null, size_bytes: 200 }))).toBe(20);
  });
});

describe('chipDownload', () => {
  it('is ready for an installed file, whatever the lane says', () => {
    expect(chipDownload('style', [file('style')], [item()])).toEqual({ state: 'ready' });
  });

  it('is downloading while queued or running, with the percent', () => {
    expect(chipDownload('style', [], [item()])).toMatchObject({ state: 'downloading', percent: 40 });
    expect(chipDownload('style', [], [item({ state: 'queued' })])).toMatchObject({ state: 'downloading', percent: 0 });
  });

  it('keeps a finished download downloading until the file is listed, so Generate never races the refresh', () => {
    expect(chipDownload('style', [], [item({ state: 'done' })])).toMatchObject({ state: 'downloading', percent: 100 });
  });

  it('is failed with the server reason, or a cancelled notice', () => {
    expect(chipDownload('style', [], [item({ state: 'failed', message: 'The connection dropped.' })])).toMatchObject({ state: 'failed', reason: 'The connection dropped.' });
    expect(chipDownload('style', [], [item({ state: 'cancelled' })])).toMatchObject({ state: 'failed' });
  });

  it('ignores model downloads of the same name', () => {
    expect(chipDownload('style', [], [item({ kind: 'model' })])).toEqual({ state: 'ready' });
  });
});

describe('loraItem', () => {
  const failed = item({ id: 'dl-3', state: 'failed', message: 'The connection dropped.' });
  const done = item({ id: 'dl-5', state: 'done', message: null });

  it('is the newest attempt when the server lists finished items newest first', () => {
    expect(loraItem('style', [done, failed])?.id).toBe('dl-5');
    expect(chipDownload('style', [], [done, failed])).toMatchObject({ state: 'downloading', percent: 100 });
  });

  it('is the newest attempt wherever the page put it, so a retry just handed over does not lose to a failure listed after it', () => {
    expect(loraItem('style', [failed, done])?.id).toBe('dl-5');
    expect(loraItem('style', [done, failed, item({ id: 'dl-4', state: 'cancelled' })])?.id).toBe('dl-5');
  });

  it('is the newest of two active items, and an active item over a newer finished one', () => {
    expect(loraItem('style', [item({ id: 'dl-7' }), item({ id: 'dl-6' })])?.id).toBe('dl-7');
    expect(loraItem('style', [item({ id: 'dl-2' }), done])?.id).toBe('dl-2');
  });

  it('is null without an item of the name', () => {
    expect(loraItem('style', [item({ name: 'other' }), item({ kind: 'model' })])).toBeNull();
  });
});

describe('a finished download whose file is not listed', () => {
  const done = item({ id: 'dl-5', state: 'done' });

  it('blocks Generate until the page gives up waiting, then no longer', () => {
    const chips = [{ name: 'style', weight: 1 }];

    expect(loraBlock(chips, [], [done], new Set())).toMatchObject({ blocked: true, waiting: { name: 'style', percent: 100 } });
    expect(chipDownload('style', [], [done], new Set(['dl-5']))).toEqual({ state: 'ready' });
    expect(loraBlock(chips, [], [done], new Set(['dl-5']))).toMatchObject({ blocked: false, waiting: null, failures: [] });
  });

  it('is only about that attempt: a new download of the name blocks again', () => {
    const retry = item({ id: 'dl-6', state: 'running' });

    expect(chipDownload('style', [], [done, retry], new Set(['dl-5']))).toMatchObject({ state: 'downloading' });
    expect(chipDownload('style', [], [item({ id: 'dl-5', state: 'failed' })], new Set(['dl-5']))).toMatchObject({ state: 'failed' });
  });
});

describe('loraBlock', () => {
  const chips = [{ name: 'a', weight: 1 }, { name: 'b', weight: 1 }];

  it('is not blocked when every chip is ready', () => {
    expect(loraBlock(chips, [file('a'), file('b')], [])).toEqual({ waiting: null, failures: [], blocked: false });
  });

  it('waits for the first downloading chip', () => {
    const block = loraBlock(chips, [], [item({ name: 'b', bytes_done: 10 }), item({ id: 'd2', name: 'a', bytes_done: 70 })]);
    expect(block.blocked).toBe(true);
    expect(block.waiting).toEqual({ name: 'a', percent: 70 });
  });

  it('blocks on a failure and lists every failed name', () => {
    const block = loraBlock(chips, [], [item({ name: 'a', state: 'failed', message: 'no' }), item({ id: 'd2', name: 'b', state: 'failed', message: 'nope' })]);
    expect(block.waiting).toBeNull();
    expect(block.blocked).toBe(true);
    expect(block.failures.map((failure) => failure.name)).toEqual(['a', 'b']);
  });
});

describe('loraBlock: a failure wins over waiting', () => {
  it('does not wait for another chip while one has failed, whatever their order', () => {
    const failed = item({ id: 'd1', name: 'a', state: 'failed', message: 'no' });
    const running = item({ id: 'd2', name: 'b', bytes_done: 0 });

    for (const chips of [[{ name: 'a', weight: 1 }, { name: 'b', weight: 1 }], [{ name: 'b', weight: 1 }, { name: 'a', weight: 1 }]]) {
      const block = loraBlock(chips, [], [failed, running]);
      expect(block.waiting).toBeNull();
      expect(block.failures.map((failure) => failure.name)).toEqual(['a']);
      expect(block.blocked).toBe(true);
    }
  });
});

describe('waitText', () => {
  it('names the LoRA on desktop, with the percent always apart, and the full text for the tooltip', () => {
    expect(waitText({ name: 'technically_color', percent: 40 }, false)).toEqual({ lead: 'Waiting for technically_color', percent: '40 %', full: 'Waiting for technically_color · 40 %' });
  });

  it('drops the name on a phone but keeps the full text', () => {
    expect(waitText({ name: 'technically_color', percent: 40 }, true)).toEqual({ lead: 'Waiting', percent: '40 %', full: 'Waiting for technically_color · 40 %' });
  });

  it('has no percent while the size is unknown', () => {
    expect(waitText({ name: 'x', percent: null }, true).percent).toBe('');
  });
});

describe('failureText', () => {
  it('says what to do', () => {
    expect(failureText({ name: 'style' })).toBe("style didn't download. Retry it or remove it from LoRAs.");
  });
});
