import type { PrefillParams } from '$lib/types';

/** Parse a query string; a name that repeats (`reference_path`) becomes a list in order, a single one stays a string. */
export function parseQuery(query: string): PrefillParams {
  const params: PrefillParams = {};
  new URLSearchParams(query).forEach((value, key) => {
    const seen = params[key];
    params[key] = seen === undefined ? value : [...(Array.isArray(seen) ? seen : [seen]), value];
  });
  return params;
}

/** Build a query string from *params*; a list repeats its name once per value. */
export function buildQuery(params: PrefillParams): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    for (const item of listOf(value)) query.append(key, item);
  }
  return query.toString();
}

/** Return the values of one parameter as a list: itself for a list, a one-item list for a string. */
export function listOf(value: string | string[]): string[] {
  return Array.isArray(value) ? value : [value];
}

/** Return *params* with each list reduced to its first value (an empty list to an empty string). */
export function firstValues(params: PrefillParams): Record<string, string> {
  return Object.fromEntries(Object.entries(params).map(([key, value]) => [key, listOf(value)[0] ?? '']));
}
