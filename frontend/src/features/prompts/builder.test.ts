import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import type { PromptDocument, PromptDocumentLoad, PromptDocumentPreview } from '$lib/types';
import { PromptBuilder, type PreviewFn } from './builder.svelte';

function loadPayload(document: PromptDocument): PromptDocumentLoad & { document: PromptDocument } {
  return {
    path: '/p.yaml',
    revision: 'r1',
    raw_text: 'text',
    document,
    enhance_matrix: { axes: [], defaults: { style: 'keep', mood: 'keep', details: [], length: 'same', motion: [] } },
  };
}

function sample(): PromptDocument {
  return {
    snippets: [{ id: 'n0', name: 'light', value: { kind: 'text', text: 'soft light' } }],
    sets: [
      { id: 's0', name: 'portrait', entries: [
        { id: 's0.e0', prompt: { kind: 'text', text: 'a woman, $light' }, negative: null, active: true, enhance: null },
        { id: 's0.e1', prompt: { kind: 'text', text: 'a man' }, negative: null, active: false, enhance: null },
      ] },
      { id: 's1', name: 'scene', entries: [
        { id: 's1.e0', prompt: { kind: 'structured', data: { A: ['$light'] } }, negative: null, active: true, enhance: true },
      ] },
    ],
  };
}

const EMPTY_PREVIEW: PromptDocumentPreview = { entries: {}, problems: [], snippet_uses: {}, rolled: null };

describe('PromptBuilder', () => {
  let preview: Mock<PreviewFn>;
  let builder: PromptBuilder;

  beforeEach(() => {
    vi.useFakeTimers();
    preview = vi.fn<PreviewFn>().mockResolvedValue(EMPTY_PREVIEW);
    builder = new PromptBuilder(preview);
    builder.load(loadPayload(sample()));
  });

  afterEach(() => {
    builder.dispose();
    vi.useRealTimers();
  });

  it('loads a document clean, selects the first entry and previews it', async () => {
    expect(builder.dirty).toBe(false);
    expect(builder.selectedId).toBe('s0.e0');
    await vi.runAllTimersAsync();
    expect(preview).toHaveBeenCalledTimes(1);
  });

  it('records one undo step per edit and restores the document', () => {
    builder.toggleActive('s0.e1');
    builder.addSet();
    expect(builder.doc.sets).toHaveLength(3);
    expect(builder.dirty).toBe(true);

    builder.undo();
    expect(builder.doc.sets).toHaveLength(2);
    builder.undo();
    expect(builder.doc.sets[0].entries[1].active).toBe(false);
    expect(builder.undo()).toBe(false);
  });

  it('records no undo step for a typing session without typing', () => {
    builder.beginTyping('s0.e0:prompt');
    builder.endTyping();

    expect(builder.undo()).toBe(false);
    expect(builder.dirty).toBe(false);
  });

  it('keeps edits made while a save was running, under the saved ids', () => {
    builder.toggleActive('s0.e1');
    const sent = builder.snapshot();
    // Typed after the save started, before its answer arrived.
    builder.beginTyping('s0.e0:prompt');
    builder.type((document) => { const value = document.sets[0].entries[0].prompt; if (value.kind === 'text') value.text = 'typed during save'; });

    const savedDocument = JSON.parse(sent) as PromptDocument;
    savedDocument.sets.reverse();
    builder.applySaved({
      path: '/p.yaml', revision: 'r2', raw_text: 'saved', document: savedDocument,
      ids: { s0: 's1', 's0.e0': 's1.e0', 's0.e1': 's1.e1', s1: 's0', 's1.e0': 's0.e0', n0: 'n0' }, option_id_map: {}, warnings: [],
    }, sent);

    expect(builder.dirty).toBe(true);
    expect(builder.revision).toBe('r2');
    expect(builder.baseText).toBe('saved');
    expect(builder.doc.sets[0].id).toBe('s1');
    expect(builder.doc.sets[0].entries[0]).toMatchObject({ id: 's1.e0', prompt: { kind: 'text', text: 'typed during save' } });
    expect(builder.editingKey).toBe('s1.e0:prompt');
    expect(builder.undo()).toBe(false);
  });

  it('gives items brought back by undo during a save fresh ids, so none collide', () => {
    builder.deleteEntry('s0.e0');
    const sent = builder.snapshot();
    builder.undo(); // undone while the save is still running

    const saved = JSON.parse(sent) as PromptDocument;
    builder.applySaved({
      path: '/p.yaml', revision: 'r2', raw_text: 'saved', document: saved,
      ids: { n0: 'n0', s0: 's0', 's0.e1': 's0.e0', s1: 's1', 's1.e0': 's1.e0' }, option_id_map: {}, warnings: [],
    }, sent);

    const ids = builder.doc.sets.flatMap((set) => set.entries.map((entry) => entry.id));
    expect(new Set(ids).size).toBe(ids.length);
    expect(builder.doc.sets[0].entries.map((entry) => entry.id)).toEqual([expect.stringMatching(/^new-/), 's0.e0']);
    expect(builder.doc.sets[0].entries[0].prompt).toEqual({ kind: 'text', text: 'a woman, $light' });
  });

  it('records a typing session as one undo step', () => {
    builder.beginTyping('s0.e0:prompt');
    builder.type((document) => { const value = document.sets[0].entries[0].prompt; if (value.kind === 'text') value.text = 'a'; });
    builder.type((document) => { const value = document.sets[0].entries[0].prompt; if (value.kind === 'text') value.text = 'ab'; });
    builder.endTyping();

    builder.undo();
    expect(builder.doc.sets[0].entries[0].prompt).toEqual({ kind: 'text', text: 'a woman, $light' });
  });

  it('moves entries between sets and onto the end of a set', () => {
    builder.moveEntry('s0.e0', 's1', 's1.e0', 'before');
    expect(builder.doc.sets[1].entries.map((entry) => entry.id)).toEqual(['s0.e0', 's1.e0']);
    builder.moveEntry('s0.e1', 's1', null, 'after');
    expect(builder.doc.sets[1].entries.map((entry) => entry.id)).toEqual(['s0.e0', 's1.e0', 's0.e1']);
    expect(builder.doc.sets[0].entries).toEqual([]);
  });

  it('reorders sets and snippets', () => {
    builder.moveSet('s0', 's1', 'after');
    expect(builder.doc.sets.map((set) => set.id)).toEqual(['s1', 's0']);
    const id = builder.addSnippet();
    builder.moveSnippet(id, 'n0', 'before');
    expect(builder.doc.snippets.map((snippet) => snippet.id)).toEqual([id, 'n0']);
  });

  it('refuses bad set names and renames good ones', () => {
    expect(builder.renameSet('s0', 'scene')).not.toBeNull();
    expect(builder.renameSet('s0', 'snippets')).not.toBeNull();
    expect(builder.renameSet('s0', 'people')).toBeNull();
    expect(builder.doc.sets[0].name).toBe('people');
  });

  it('renames a snippet with its references, unless a structured value uses it', () => {
    builder.edit((document) => { document.sets[1].entries = []; });
    expect(builder.renameSnippet('n0', 'glow')).toBeNull();
    expect(builder.doc.sets[0].entries[0].prompt).toEqual({ kind: 'text', text: 'a woman, $glow' });

    builder.undo();
    builder.undo();
    expect(builder.renameSnippet('n0', 'glow')).not.toBeNull();
    expect(builder.renameSnippet('n0', '2x')).not.toBeNull();
  });

  it('duplicates sets with unique names and fresh ids', () => {
    builder.duplicateSet('s0');
    builder.duplicateSet('s0');
    const names = builder.doc.sets.map((set) => set.name);
    expect(names).toEqual(['portrait', 'portrait_copy_2', 'portrait_copy', 'scene']);
    const ids = builder.doc.sets.flatMap((set) => set.entries.map((entry) => entry.id));
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('switches text to one unnamed field without inventing a name', () => {
    builder.setValueMode('s0.e0', 'prompt', 'fields');
    expect(builder.doc.sets[0].entries[0].prompt).toEqual({ kind: 'fields', fields: [{ key: '', value: 'a woman, $light' }] });
    builder.addField('s0.e0', 'prompt');
    const prompt = builder.doc.sets[0].entries[0].prompt;
    expect(prompt.kind === 'fields' && prompt.fields.at(-1)).toEqual({ key: '', value: '' });
  });

  it('switches fields back to the text they flatten to', () => {
    builder.setValueMode('s0.e0', 'prompt', 'fields');
    builder.renameField('s0.e0', 'prompt', 0, 'Subject');
    builder.setValueMode('s0.e0', 'prompt', 'text');
    expect(builder.doc.sets[0].entries[0].prompt).toEqual({ kind: 'text', text: 'Subject: a woman, $light' });
  });

  it('inserts a snippet reference into text and fields prompts', () => {
    builder.insertSnippetRef('s0.e1', 'light');
    expect(builder.doc.sets[0].entries[1].prompt).toEqual({ kind: 'text', text: 'a man $light' });
    builder.setValueMode('s0.e1', 'prompt', 'fields');
    builder.insertSnippetRef('s0.e1', 'light');
    const prompt = builder.doc.sets[0].entries[1].prompt;
    expect(prompt.kind === 'fields' && prompt.fields.at(-1)).toEqual({ key: '', value: '$light' });
  });

  it('adds an entry and starts editing it', () => {
    const id = builder.addEntry('s1');
    expect(builder.selectedId).toBe(id);
    expect(builder.editingKey).toBe(`${id}:prompt`);
  });

  it('takes a saved document and moves the selection to its new id', () => {
    builder.toggleActive('s0.e1');
    builder.select('s0.e1');
    builder.applySaved({
      path: '/p.yaml', revision: 'r2', raw_text: 'new', document: sample(), ids: { 's0.e1': 's0.e0' }, option_id_map: {}, warnings: [],
    }, builder.snapshot());
    expect(builder.dirty).toBe(false);
    expect(builder.revision).toBe('r2');
    expect(builder.baseText).toBe('new');
    expect(builder.selectedId).toBe('s0.e0');
    expect(builder.undo()).toBe(false);
  });

  it('keeps a roll for its entry', async () => {
    preview.mockResolvedValueOnce({ ...EMPTY_PREVIEW, entries: { 's0.e0': { prompt: 'a {x|y}', negative: null } }, rolled: { entry_id: 's0.e0', prompt: 'a x' } });
    await builder.roll('s0.e0');
    expect(preview).toHaveBeenLastCalledWith(expect.anything(), 's0.e0');
    expect(builder.rolled).toEqual({ entryId: 's0.e0', base: 'a {x|y}', prompt: 'a x' });
  });

  it('ignores a preview that a newer request replaced', async () => {
    let resolveFirst: (value: PromptDocumentPreview) => void = () => undefined;
    preview.mockReturnValueOnce(new Promise((resolve) => { resolveFirst = resolve; }));
    preview.mockResolvedValueOnce({ ...EMPTY_PREVIEW, snippet_uses: { n0: 2 } });
    builder.refreshPreview(0);
    await vi.advanceTimersByTimeAsync(0);
    builder.refreshPreview(0);
    await vi.advanceTimersByTimeAsync(0);
    resolveFirst({ ...EMPTY_PREVIEW, snippet_uses: { n0: 1 } });
    await vi.runAllTimersAsync();
    expect(builder.preview?.snippet_uses).toEqual({ n0: 2 });
  });
});
