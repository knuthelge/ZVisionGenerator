import { describe, expect, it } from 'vitest';
import { applyPromptRun, requestPromptRun, takePromptRun } from './pendingPromptRun';

describe('pendingPromptRun', () => {
  it('hands a requested run out once', () => {
    requestPromptRun({ path: '/p.yaml', optionId: 'a:0' });
    expect(takePromptRun()).toEqual({ path: '/p.yaml', optionId: 'a:0' });
    expect(takePromptRun()).toBeNull();
  });

  it('points a form at one prompt-file prompt', () => {
    const form = new FormData();
    form.set('prompt_source', 'inline');
    form.set('prompt', 'typed prompt');
    form.set('negative_prompt', 'blur');
    form.set('json_prompt', '{}');
    form.append('prompt_option_id', 'b:0');
    form.append('prompt_option_id', 'b:1');
    form.set('steps', '8');

    applyPromptRun(form, { path: '/p.yaml', optionId: 'a:2' });

    expect(form.get('prompt_source')).toBe('file');
    expect(form.get('prompts_file')).toBe('/p.yaml');
    expect(form.getAll('prompt_option_id')).toEqual(['a:2']);
    expect(['prompt', 'negative_prompt', 'json_prompt'].some((key) => form.has(key))).toBe(false);
    expect(form.get('steps')).toBe('8');
  });
});
