import { createProject, listProjects } from '$lib/api/projects';
import { galleryStore } from '$lib/state/gallery.svelte';
import { galleryMemory } from '$lib/state/pageMemory';
import { projectPick } from '$lib/state/projectPick.svelte';
import { addToast } from '$lib/state/toasts.svelte';
import type { Project } from '$lib/types';

let _list = $state<Project[]>([]);

/**
 * The projects, and the one place the pick changes. The pick is a name, or null for All files (the picker's label for
 * it); `projectPick.current` holds it.
 */
export const projectsStore = {
  /** The projects as last read, sorted by name without regard to case. */
  get list(): Project[] { return _list; },

  /** The folder of the picked project, or null with All files or while the list does not have it. */
  get currentPath(): string | null {
    const pick = projectPick.current;
    return pick === null ? null : (_list.find((project) => project.name === pick)?.path ?? null);
  },

  /**
   * Read the project list. A failed read changes nothing. A picked project that is not in the list any more
   * (its folder was removed or renamed) returns the pick to All files, with a warning.
   */
  async load(): Promise<void> {
    let projects: Project[];
    try {
      projects = await listProjects();
    } catch {
      return;
    }
    _list = projects;
    const pick = projectPick.current;
    if (pick !== null && !projects.some((project) => project.name === pick)) {
      projectsStore.choose(null);
      addToast(`${pick} is no longer there. Showing All files.`, 'warning');
    }
  },

  /**
   * Pick a project, or All files with null. Nothing happens when the pick is unchanged. The Gallery page starts at the
   * top again but keeps its filter and sort; lists from before the change are dropped, and so are their responses still in
   * flight. The pages that show a list load it themselves: the Gallery page when the pick changes under it, the Workspace
   * when it is open or opens next. Refreshing here too would ask the server for the same page twice.
   */
  choose(name: string | null): void {
    if (name === projectPick.current) return;
    projectPick.set(name);
    const visit = galleryMemory.get();
    if (visit) galleryMemory.set({ ...visit, scrollTop: 0 });
    galleryStore.reset();
  },

  /** Create a project and add it to the list; it does not pick it. Rejects with the server's reason. */
  async create(name: string): Promise<Project> {
    const project = await createProject(name);
    if (!_list.some((item) => item.name === project.name)) {
      _list = [..._list, project].sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
    }
    return project;
  },
};
