import './app/global.css';
import App from './app/App.svelte';
import { mount } from 'svelte';
import { watchViewport } from './lib/state/viewport';

// iOS Safari only applies :active (the pressed state a finger gets in place of hover) once the page listens for touches.
document.addEventListener('touchstart', () => {}, { passive: true });

// iOS can still zoom the page with a pinch despite `touch-action`; its own gesture events are the way to stop it.
document.addEventListener('gesturestart', (event) => event.preventDefault());

// The app is always the size of what is visible: above the keyboard, and in step with the browser's toolbar.
watchViewport(window);

const app = mount(App, {
  target: document.getElementById('app')!
});

export default app;
