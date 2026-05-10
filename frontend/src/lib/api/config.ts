import { api } from './client';
import type { AppConfig, WritableConfigValue } from '$lib/types';

export type WritableConfigPatch = Record<string, WritableConfigValue>;

export function getConfig(): Promise<AppConfig> {
  return api.get<AppConfig>('/api/config');
}

export function updateConfig(config: WritableConfigPatch): Promise<AppConfig> {
  return api.post<AppConfig>('/api/config', config);
}
