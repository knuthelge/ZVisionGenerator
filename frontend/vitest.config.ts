import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { resolve } from 'path';

export default defineConfig({
  plugins: [svelte()],
  resolve: {
    // Use the browser export condition so Svelte resolves to its client runtime.
    // Without this, vitest resolves 'svelte' to the server runtime where
    // lifecycle hooks like onMount are no-ops, breaking component tests.
    conditions: ['browser'],
    alias: {
      '$lib': resolve(import.meta.dirname, 'src/lib'),
      '$features': resolve(import.meta.dirname, 'src/features'),
      '$app': resolve(import.meta.dirname, 'src/app')
    }
  },
  test: {
    environment: 'jsdom',
    // VM threads build jsdom once per worker instead of once per file (about 60% less CPU); a worker that grows past
    // the limit is replaced, so memory stays bounded.
    pool: 'vmThreads',
    vmMemoryLimit: '512MB',
    setupFiles: ['./src/test-setup.ts'],
    globals: true
  }
});
