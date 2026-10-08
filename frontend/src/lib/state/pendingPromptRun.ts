/** One prompt the Prompts page asked the Workspace to queue with its current settings. */
export interface PendingPromptRun {
  path: string;
  optionId: string;
}

let pending: PendingPromptRun | null = null;

/** Ask the Workspace to queue one prompt of a file the next time it is ready to submit. */
export function requestPromptRun(run: PendingPromptRun): void {
  pending = run;
}

/** Take the pending run, if any; it is handed out once. */
export function takePromptRun(): PendingPromptRun | null {
  const run = pending;
  pending = null;
  return run;
}

/**
 * Point a Workspace form submission at one prompt of a file, leaving the user's own prompt selection alone.
 * Inline-prompt and JSON-caption fields are dropped, since the server refuses them in prompt-file mode.
 */
export function applyPromptRun(form: FormData, run: PendingPromptRun): void {
  form.set('prompt_source', 'file');
  form.set('prompts_file', run.path);
  form.delete('prompt_option_id');
  form.append('prompt_option_id', run.optionId);
  for (const key of ['prompt', 'negative_prompt', 'json_prompt']) form.delete(key);
}
