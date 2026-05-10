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
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html')
      }
    }
  },
  resolve: {
    alias: {
      '$lib': resolve(__dirname, 'src/lib'),
      '$features': resolve(__dirname, 'src/features'),
      '$app': resolve(__dirname, 'src/app')
    }
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8765',
      '/jobs': 'http://localhost:8765'
    }
  }
});
