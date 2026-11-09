// The New voice tab's form, held in memory: it survives closing the Voices window and is lost with a page reload.
// The description and the preview line stay after Make 3, so Make 3 more asks for the same voice again.

class NewVoiceForm {
  description = $state('');
  /** The line the user wrote, or null while they have not: the tab then shows the default line for the take's text. */
  #line = $state<string | null>(null);

  /** The preview line the user wrote, or *fallback* while they have not written one. */
  lineOr(fallback: string): string {
    return this.#line ?? fallback;
  }

  /** The user wrote the preview line. */
  setLine(line: string): void {
    this.#line = line;
  }

  /** Forget both fields. */
  clear(): void {
    this.description = '';
    this.#line = null;
  }
}

export const newVoiceForm = new NewVoiceForm();
