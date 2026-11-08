import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const layoutMock = vi.hoisted(() => ({ compact: false, mode: 'wide', coarse: false }));
const listProjects = vi.hoisted(() => vi.fn());
const createProject = vi.hoisted(() => vi.fn());

vi.mock('$lib/state/layoutMode.svelte', () => ({ layout: layoutMock }));
vi.mock('$lib/api/projects', () => ({ listProjects, createProject }));
vi.mock('$lib/api/gallery', () => ({ getGallery: vi.fn().mockResolvedValue({ assets: [], page: 1, total_pages: 1, total_count: 0 }) }));

import { ApiError } from '$lib/api/client';
import { projectPick } from '$lib/state/projectPick.svelte';
import { projectsStore } from '$lib/state/projects.svelte';
import { buttonByText, mountComponent, settle } from '../../../test-utils/mount';
import ProjectPicker from './ProjectPicker.svelte';

const NIGHT_BUS = { name: 'Night Bus', path: '/out/Night Bus' };
const ZOO = { name: 'Zoo', path: '/out/Zoo' };

const trigger = () => document.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]')!;
const menu = () => document.querySelector<HTMLElement>('[role="menu"]');
const row = (id: string) => document.querySelector<HTMLElement>(`[data-action="${id}"]`);
const labels = () => Array.from(document.querySelectorAll('[role="menu"] [role="menuitemradio"], [role="menu"] [role="menuitem"]')).map((item) => item.textContent?.trim());
const field = () => document.querySelector<HTMLInputElement>('#project-name')!;
const createButton = () => document.querySelector<HTMLButtonElement>('[data-action="create-project"]')!;

function type(value: string): void {
  field().value = value;
  field().dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
}

async function openMenu(): Promise<void> {
  trigger().click();
  await settle();
}

beforeEach(() => {
  Object.assign(layoutMock, { compact: false, mode: 'wide', coarse: false });
  localStorage.clear();
  projectPick.set(null);
  listProjects.mockReset().mockResolvedValue([NIGHT_BUS, ZOO]);
  createProject.mockReset();
});

describe('ProjectPicker nav variant', () => {
  it('shows All files as a plain button, without the picked look', () => {
    mountComponent(ProjectPicker, { variant: 'nav' });

    expect(trigger().textContent?.trim()).toBe('All files');
    expect(trigger().getAttribute('aria-label')).toBe('Project: All files');
    expect(trigger().classList.contains('ui-btn')).toBe(true);
    expect(trigger().classList.contains('ui-btn-on')).toBe(false);
    expect(trigger().getAttribute('aria-expanded')).toBe('false');
  });

  it('shows the picked project in the teal state with its full name as the tooltip', () => {
    projectPick.set('Night Bus');
    mountComponent(ProjectPicker, { variant: 'nav', tight: true });

    expect(trigger().textContent?.trim()).toBe('Night Bus');
    expect(trigger().getAttribute('aria-label')).toBe('Project: Night Bus');
    expect(trigger().title).toBe('Night Bus');
    expect(trigger().classList.contains('ui-btn-on')).toBe(true);
    expect(trigger().classList.contains('tight')).toBe(true);
  });

  it('opens a list of All files, the projects, and New project with the pick checked', async () => {
    projectPick.set('Zoo');
    mountComponent(ProjectPicker, { variant: 'nav' });

    await openMenu();

    expect(menu()?.getAttribute('aria-label')).toBe('Project');
    expect(trigger().getAttribute('aria-expanded')).toBe('true');
    expect(labels()).toEqual(['All files', 'Night Bus', 'Zoo', 'New project…']);
    expect(row('project:Zoo')?.getAttribute('aria-checked')).toBe('true');
    expect(row('project-none')?.getAttribute('aria-checked')).toBe('false');
    expect(listProjects).toHaveBeenCalledOnce();
  });

  it('picks a project, closes the list and reports the pick', async () => {
    const onpicked = vi.fn();
    mountComponent(ProjectPicker, { variant: 'nav', onpicked });
    await openMenu();

    row('project:Night Bus')!.click();
    await settle();

    expect(projectPick.current).toBe('Night Bus');
    expect(onpicked).toHaveBeenCalledOnce();
    expect(menu()).toBeNull();
    expect(trigger().classList.contains('ui-btn-on')).toBe(true);
  });

  it('returns to All files from the none row', async () => {
    projectPick.set('Zoo');
    mountComponent(ProjectPicker, { variant: 'nav' });
    await openMenu();

    row('project-none')!.click();
    await settle();

    expect(projectPick.current).toBeNull();
    expect(trigger().textContent?.trim()).toBe('All files');
  });

  it('leaves the pick alone when the list is closed without choosing', async () => {
    const onpicked = vi.fn();
    mountComponent(ProjectPicker, { variant: 'nav', onpicked });
    await openMenu();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await settle();

    expect(menu()).toBeNull();
    expect(projectPick.current).toBeNull();
    expect(onpicked).not.toHaveBeenCalled();
  });

  it('creates a project from the dialog and picks it', async () => {
    createProject.mockResolvedValue({ name: 'Bus Stop', path: '/out/Bus Stop' });
    const onpicked = vi.fn();
    mountComponent(ProjectPicker, { variant: 'nav', onpicked });
    await openMenu();

    row('project-new')!.click();
    await settle();

    expect(menu()).toBeNull();
    expect(document.querySelector('[role="dialog"]')?.textContent).toContain('New project');
    expect(buttonByText('Create').textContent?.trim()).toBe('Create');
    expect(createButton().disabled).toBe(true);

    type('  Bus Stop ');
    expect(createButton().disabled).toBe(false);
    createButton().click();
    await settle();

    expect(createProject).toHaveBeenCalledWith('Bus Stop');
    expect(projectPick.current).toBe('Bus Stop');
    expect(onpicked).toHaveBeenCalledOnce();
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(menu()).toBeNull();
    expect(projectsStore.list.map((project) => project.name)).toContain('Bus Stop');
  });

  it('keeps the dialog and the pick on a refusal, shows the server text, and clears it on the next edit', async () => {
    createProject.mockRejectedValue(new ApiError('conflict', 409, 'A project named Zoo already exists.'));
    projectPick.set('Night Bus');
    const onpicked = vi.fn();
    mountComponent(ProjectPicker, { variant: 'nav', onpicked });
    await openMenu();
    row('project-new')!.click();
    await settle();

    type('Zoo');
    createButton().click();
    await settle();

    const error = document.querySelector<HTMLElement>('[role="alert"]');
    expect(error?.textContent).toBe('A project named Zoo already exists.');
    expect(field().value).toBe('Zoo');
    expect(field().getAttribute('aria-invalid')).toBe('true');
    expect(field().getAttribute('aria-describedby')).toBe(error?.id);
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();
    expect(projectPick.current).toBe('Night Bus');
    expect(onpicked).not.toHaveBeenCalled();

    type('Zoo 2');

    expect(document.querySelector('[role="alert"]')).toBeNull();
    expect(field().hasAttribute('aria-invalid')).toBe(false);
  });

  it('submits on Enter and closes on Cancel', async () => {
    createProject.mockResolvedValue({ name: 'Quiet', path: '/out/Quiet' });
    listProjects.mockResolvedValue([NIGHT_BUS, { name: 'Quiet', path: '/out/Quiet' }, ZOO]);
    mountComponent(ProjectPicker, { variant: 'nav' });
    await openMenu();
    row('project-new')!.click();
    await settle();

    type('Quiet');
    document.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();
    expect(projectPick.current).toBe('Quiet');

    await openMenu();
    row('project-new')!.click();
    await settle();
    buttonByText('Cancel').click();
    await settle();

    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(menu()).toBeNull();
    expect(projectPick.current).toBe('Quiet');
  });
});

describe('ProjectPicker drawer variant', () => {
  beforeEach(() => {
    layoutMock.compact = true;
    layoutMock.mode = 'compact';
  });

  it('is a row with the Project label and the name, named like the nav button', () => {
    projectPick.set('Night Bus');
    mountComponent(ProjectPicker, { variant: 'drawer' });

    const rowEl = trigger();
    expect(rowEl.classList.contains('drawer-project')).toBe(true);
    expect(rowEl.getAttribute('aria-label')).toBe('Project: Night Bus');
    expect(rowEl.textContent).toContain('Project');
    expect(rowEl.querySelector('.drawer-project-name')?.textContent).toBe('Night Bus');
    expect(rowEl.hasAttribute('data-picked')).toBe(true);
  });

  it('shows All files unpicked', () => {
    mountComponent(ProjectPicker, { variant: 'drawer' });

    expect(trigger().querySelector('.drawer-project-name')?.textContent).toBe('All files');
    expect(trigger().hasAttribute('data-picked')).toBe(false);
  });

  it('opens the same list as a sheet and reports a pick', async () => {
    const onpicked = vi.fn();
    mountComponent(ProjectPicker, { variant: 'drawer', onpicked });

    await openMenu();

    expect(document.querySelector('[data-testid="action-sheet"]')).not.toBeNull();
    expect(labels()).toEqual(['All files', 'Night Bus', 'Zoo', 'New project…']);

    row('project:Zoo')!.click();
    await settle();

    expect(projectPick.current).toBe('Zoo');
    expect(onpicked).toHaveBeenCalledOnce();
  });
});
