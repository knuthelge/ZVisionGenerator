import { api } from './client';
import type { AppConfig } from '$lib/types';

export function getConfig(): Promise<AppConfig> {
  return api.get<AppConfig>('/api/config');
}

export function updateConfig(config: Partial<AppConfig>): Promise<AppConfig> {
  return api.post<AppConfig>('/api/config', config);
}
