import { describe, expect, it } from 'vitest';

import { textWriting } from './textWriting.svelte';

describe('textWriting', () => {
  it('has no reason while nothing is being written', () => {
    expect(textWriting.reason).toBeNull();
  });

  it('says what is being written and that Generate waits for it', () => {
    const end = textWriting.begin('Enhancing');

    expect(textWriting.reason).toBe("Enhancing… Generate when it's done.");

    end();
    expect(textWriting.reason).toBeNull();
  });

  it('holds Generate until the last of several runs has ended, and ending twice changes nothing', () => {
    const enhance = textWriting.begin('Enhancing');
    const helper = textWriting.begin('Writing');
    expect(textWriting.reason).toBe("Enhancing… Generate when it's done.");

    enhance();
    enhance();
    expect(textWriting.reason).toBe("Writing… Generate when it's done.");

    helper();
    expect(textWriting.reason).toBeNull();
  });
});
