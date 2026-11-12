// Entry of the launcher window: the shared tokens, the bundled font, then the app.
import '../app/global.css';
import './launcher.css';
import { mount } from 'svelte';
import LauncherApp from './LauncherApp.svelte';

const app = mount(LauncherApp, { target: document.getElementById('app')! });

export default app;
