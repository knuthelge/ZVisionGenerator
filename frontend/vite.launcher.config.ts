import { mergeConfig } from 'vite';
import { resolve } from 'path';
import main from './vite.config.ts';

// The launcher window is its own build: shares the main config's plugins and aliases, but not its `build`
// (mergeConfig would merge `rolldownOptions.input` and also build index.html) or its dev-server proxy.
const { build: _build, server: _server, ...shared } = main;

export default mergeConfig(shared, {
  base: './',
  publicDir: false,
  build: {
    outDir: '../launcher/dist',
    emptyOutDir: true,
    rolldownOptions: {
      input: {
        launcher: resolve(import.meta.dirname, 'launcher.html')
      }
    }
  }
});
