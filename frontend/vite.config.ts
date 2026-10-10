import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import tailwindcss from '@tailwindcss/vite';
import { resolve } from 'path';

export default defineConfig({
  plugins: [
    tailwindcss(),
    svelte()
  ],
  base: '/app-static/',
  build: {
    outDir: '../zvisiongenerator/web/static/app',
    emptyOutDir: true,
    rolldownOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html')
      }
    }
  },
  resolve: {
    alias: {
      '$lib': resolve(import.meta.dirname, 'src/lib'),
      '$features': resolve(import.meta.dirname, 'src/features'),
      '$app': resolve(import.meta.dirname, 'src/app')
    }
  },
  server: {
    // Object form with changeOrigin: false keeps the browser's Host header, so it matches the Origin header
    // the backend's request guard checks on writes (the string shorthand sets changeOrigin: true).
    proxy: {
      '/api': { target: 'http://localhost:8765', changeOrigin: false },
      '/jobs': { target: 'http://localhost:8765', changeOrigin: false }
    }
  }
});
