import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { resolve } from 'path';

export default defineConfig({
  plugins: [svelte({ hot: false })],
  resolve: {
    // Use the browser export condition so Svelte resolves to its client runtime.
    // Without this, vitest resolves 'svelte' to the server runtime where
    // lifecycle hooks like onMount are no-ops, breaking component tests.
    conditions: ['browser'],
    alias: {
      '$lib': resolve(__dirname, 'src/lib'),
      '$features': resolve(__dirname, 'src/features'),
      '$app': resolve(__dirname, 'src/app')
    }
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
    globals: true
  }
});
