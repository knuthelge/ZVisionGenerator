import { api } from './client';
import type { Project } from '$lib/types';

/** The projects on the server: the folders directly under the output folder, sorted by name. */
export function listProjects(): Promise<Project[]> {
  return api.get<{ projects: Project[] }>('/api/projects').then((reply) => reply.projects);
}

/** Create a project folder; rejects with the server's reason when the name is taken (409) or not allowed (422). */
export function createProject(name: string): Promise<Project> {
  return api.post<Project>('/api/projects', { name });
}
