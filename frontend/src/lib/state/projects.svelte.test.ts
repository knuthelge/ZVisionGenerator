import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { GalleryPage, Project } from '$lib/types';

const listProjects = vi.fn<() => Promise<Project[]>>();
const createProject = vi.fn<(name: string) => Promise<Project>>();
const getGallery = vi.fn<(page?: number, filter?: string, sort?: string, project?: string | null) => Promise<GalleryPage>>();
const addToast = vi.fn();
vi.mock('$lib/api/projects', () => ({ listProjects: () => listProjects(), createProject: (name: string) => createProject(name) }));
vi.mock('$lib/api/gallery', () => ({
  getGallery: (page?: number, filter?: string, sort?: string, project?: string | null) => getGallery(page, filter, sort, project),
}));
vi.mock('$lib/state/toasts.svelte', () => ({ addToast: (...args: unknown[]) => addToast(...args) }));

const NIGHT_BUS: Project = { name: 'Night Bus', path: '/out/Night Bus' };
const ZOO: Project = { name: 'Zoo', path: '/out/Zoo' };
const emptyPage: GalleryPage = { assets: [], page: 1, total_pages: 1, total_count: 0 };

type Modules = {
  projects: typeof import('./projects.svelte').projectsStore;
  pick: typeof import('./projectPick.svelte').projectPick;
  gallery: typeof import('./gallery.svelte').galleryStore;
  memory: typeof import('./pageMemory').galleryMemory;
};

async function load(): Promise<Modules> {
  vi.resetModules();
  return {
    projects: (await import('./projects.svelte')).projectsStore,
    pick: (await import('./projectPick.svelte')).projectPick,
    gallery: (await import('./gallery.svelte')).galleryStore,
    memory: (await import('./pageMemory')).galleryMemory,
  };
}

beforeEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
  listProjects.mockReset();
  createProject.mockReset();
  getGallery.mockReset().mockResolvedValue(emptyPage);
  addToast.mockReset();
});

describe('the pick in storage', () => {
  it('starts as All files (null) when nothing or an empty value is stored', async () => {
    expect((await load()).pick.current).toBeNull();
    localStorage.setItem('ziv-project-v1', '');
    expect((await load()).pick.current).toBeNull();
  });

  it('survives a reload', async () => {
    const first = await load();
    first.pick.set('Night Bus');
    expect(localStorage.getItem('ziv-project-v1')).toBe('Night Bus');
    expect((await load()).pick.current).toBe('Night Bus');
  });

  it('removes the stored value for All files', async () => {
    const { pick } = await load();
    pick.set('Night Bus');
    pick.set(null);
    expect(localStorage.getItem('ziv-project-v1')).toBeNull();
    expect(pick.current).toBeNull();
  });

  it('works in memory when storage throws', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('denied'); });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('denied'); });
    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => { throw new Error('denied'); });
    const { pick } = await load();
    expect(pick.current).toBeNull();
    pick.set('Night Bus');
    expect(pick.current).toBe('Night Bus');
    pick.set(null);
    expect(pick.current).toBeNull();
  });
});

describe('inProject', () => {
  it('admits everything with All files and only that project otherwise', async () => {
    vi.resetModules();
    const { inProject } = await import('./projectPick.svelte');
    expect(inProject({ project: 'A' }, null)).toBe(true);
    expect(inProject({ project: null }, null)).toBe(true);
    expect(inProject({ project: 'A' }, 'A')).toBe(true);
    expect(inProject({ project: 'B' }, 'A')).toBe(false);
    expect(inProject({ project: null }, 'A')).toBe(false);
  });
});

describe('projectsStore.load', () => {
  it('keeps the list and exposes the picked folder', async () => {
    const { projects, pick } = await load();
    listProjects.mockResolvedValue([NIGHT_BUS, ZOO]);
    pick.set('Zoo');
    await projects.load();
    expect(projects.list).toEqual([NIGHT_BUS, ZOO]);
    expect(projects.currentPath).toBe('/out/Zoo');
    pick.set(null);
    expect(projects.currentPath).toBeNull();
    expect(addToast).not.toHaveBeenCalled();
  });

  it('returns a gone pick to All files and says so', async () => {
    const { projects, pick } = await load();
    pick.set('Night Bus');
    listProjects.mockResolvedValue([ZOO]);
    await projects.load();
    expect(pick.current).toBeNull();
    expect(localStorage.getItem('ziv-project-v1')).toBeNull();
    expect(addToast).toHaveBeenCalledWith(expect.stringContaining('Night Bus'), 'warning');
  });

  it('changes nothing when the read fails', async () => {
    const { projects, pick } = await load();
    pick.set('Night Bus');
    listProjects.mockResolvedValue([NIGHT_BUS]);
    await projects.load();
    listProjects.mockRejectedValue(new Error('offline'));
    await projects.load();
    expect(pick.current).toBe('Night Bus');
    expect(projects.list).toEqual([NIGHT_BUS]);
    expect(addToast).not.toHaveBeenCalled();
    expect(getGallery).not.toHaveBeenCalled();
  });
});

describe('projectsStore.choose', () => {
  it('does nothing when the pick is unchanged', async () => {
    const { projects, pick } = await load();
    projects.choose(null);
    pick.set('Zoo');
    projects.choose('Zoo');
    expect(getGallery).not.toHaveBeenCalled();
  });

  it('sets the pick and scrolls the remembered Gallery to the top keeping filter and sort, without asking the server for a list', async () => {
    const { projects, pick, memory } = await load();
    memory.set({ filter: 'video', sort: 'oldest', starred: false, scrollTop: 480 });
    projects.choose('Zoo');
    expect(pick.current).toBe('Zoo');
    expect(memory.get()).toEqual({ filter: 'video', sort: 'oldest', starred: false, scrollTop: 0 });
    expect(getGallery).not.toHaveBeenCalled();
  });

  it('drops the lists and the responses from before it', async () => {
    const { projects, gallery } = await load();
    let release: (page: GalleryPage) => void = () => {};
    getGallery.mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }));
    const before = gallery.refresh();
    gallery.setView({ filter: 'image', sort: 'newest', starred: false }, { assets: [], page: 2, totalPages: 3, totalCount: 9 });

    getGallery.mockResolvedValue({ assets: [{ id: 'z1', project: 'Zoo', media_type: 'image' } as never], page: 1, total_pages: 1, total_count: 1 });
    projects.choose('Zoo');
    const after = gallery.refresh();
    expect(gallery.view({ filter: 'image', sort: 'newest', starred: false }).totalCount).toBeNull();
    release({ assets: [{ id: 'stale', project: null, media_type: 'image' } as never], page: 1, total_pages: 1, total_count: 1 });
    await before;
    await after;
    expect(gallery.assets.map((asset) => asset.id)).toEqual(['z1']);
  });
});

describe('projectsStore.create', () => {
  it('adds the project to the sorted list without picking it', async () => {
    const { projects, pick } = await load();
    listProjects.mockResolvedValue([NIGHT_BUS, ZOO]);
    await projects.load();
    createProject.mockResolvedValue({ name: 'archive', path: '/out/archive' });
    await expect(projects.create('archive')).resolves.toEqual({ name: 'archive', path: '/out/archive' });
    expect(projects.list.map((project) => project.name)).toEqual(['archive', 'Night Bus', 'Zoo']);
    expect(pick.current).toBeNull();
  });

  it('rejects with the server error and leaves the list alone', async () => {
    const { projects } = await load();
    createProject.mockRejectedValue(new Error('exists'));
    await expect(projects.create('Zoo')).rejects.toThrow('exists');
    expect(projects.list).toEqual([]);
  });
});
