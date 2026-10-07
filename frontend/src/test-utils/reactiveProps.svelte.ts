/** Wrap component props in a rune proxy so tests can change them after mount. */
export function reactiveProps<T extends object>(initial: T): T {
  const props = $state(initial);
  return props;
}
