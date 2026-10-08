/** A bounded stack of undo snapshots (oldest dropped first). */
export class UndoHistory<T> {
  #stack: T[] = [];
  readonly #limit: number;

  constructor(limit = 100) {
    this.#limit = limit;
  }

  push(snapshot: T): void {
    this.#stack.push(snapshot);
    if (this.#stack.length > this.#limit) this.#stack.shift();
  }

  pop(): T | undefined {
    return this.#stack.pop();
  }

  clear(): void {
    this.#stack = [];
  }
}
