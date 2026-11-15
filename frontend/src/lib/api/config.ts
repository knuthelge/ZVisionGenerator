import { api } from './client';
import type { AppConfig, WritableConfigValue } from '$lib/types';

export type WritableConfigPatch = Record<string, WritableConfigValue>;

export function getConfig(): Promise<AppConfig> {
  return api.get<AppConfig>('/api/config');
}

export function updateConfig(config: WritableConfigPatch): Promise<AppConfig> {
  return api.post<AppConfig>('/api/config', config);
}

/** Check a Hugging Face token with Hugging Face and save it on the server's computer; answers where the token now comes from. */
export function saveHfToken(token: string): Promise<{ huggingface_token_source: string | null }> {
  return api.post<{ huggingface_token_source: string | null }>('/api/huggingface/token', { token });
}
