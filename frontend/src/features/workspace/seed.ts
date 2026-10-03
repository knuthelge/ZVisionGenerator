/** Return a random seed in the full unsigned 32-bit range. */
export function randomSeed(): number {
  return Math.floor(Math.random() * 2 ** 32);
}
