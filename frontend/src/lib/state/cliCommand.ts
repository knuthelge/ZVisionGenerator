import type { ExtendSourceInfo, JobSettings } from '$lib/types';
import { checked, first, numberOrNull, values } from './jobSettings';

/** A submitted setting the CLI cannot express exactly; the copied command differs from the Web job there. */
export type CliCaveat =
  /** The CLI takes negative prompts only from a prompts file. */
  | 'negative_prompt'
  /** The CLI runs every prompt in the file, and only some of them were chosen. */
  | 'prompt_options'
  /** The reference image was uploaded from the browser; the command names the uploaded file, not its path. */
  | 'uploaded_image';

export interface CliCommand {
  /** One shell line, with every argument quoted for a POSIX shell. */
  command: string;
  caveats: CliCaveat[];
}

export interface CliCommandOptions {
  /** Directory the Web UI writes to; passed as `--output` so CLI results land in the same gallery. */
  outputDir?: string | null;
  /** How many prompts the chosen prompts file holds; unknown counts as more than were chosen. */
  promptFileOptionCount?: number | null;
  /**
   * Extend only: what the source video recorded. `ziv video --extend` fills a missing `--scene` from the source's scene
   * and a missing `--prompt` from a legacy source's prompt, so a blank Scene or Prompt is then sent as an empty flag.
   */
  extendSource?: Pick<ExtendSourceInfo, 'scene' | 'legacy'> | null;
}

const SAFE_ARG = /^[A-Za-z0-9_\-+=.,:/@%]+$/;

/** Quote an argument for a POSIX shell; plain words stay bare. */
export function shellQuote(arg: string): string {
  if (SAFE_ARG.test(arg)) return arg;
  return `'${arg.replaceAll("'", `'\\''`)}'`;
}

/** Collect a form's text fields as job settings; an uploaded file is recorded by its file name. */
export function formSettings(form: FormData): JobSettings {
  const settings: Record<string, string[]> = {};
  for (const [key, value] of form.entries()) {
    (settings[key] ??= []).push(typeof value === 'string' ? value : value.name);
  }
  return Object.fromEntries(Object.entries(settings).map(([key, list]) => [key, list.length === 1 ? list[0] : list]));
}

/** Return a field's first value, or undefined when it is missing or blank. */
function text(settings: JobSettings, key: string): string | undefined {
  const value = first(settings, key)?.trim();
  return value ? value : undefined;
}

/** Serialize submitted enhancement options to the `--enhance` SPEC grammar (`parts` and `motion` only for video). */
function enhanceSpec(raw: string | undefined, isVideo: boolean, isExtend: boolean): string {
  if (!raw) return '';
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return '';
  }
  const parts: string[] = [];
  // The CLI rewrites an extension's action only and any other video's scene and action unless told otherwise, so only the other choice is spelled out.
  if (isVideo && parsed.parts !== undefined && parsed.parts !== (isExtend ? 'action' : 'both')) parts.push(`parts=${String(parsed.parts)}`);
  for (const key of ['style', 'mood', 'details', 'length', ...(isVideo ? ['motion'] : [])]) {
    const value = parsed[key];
    if (value === undefined || value === null) continue;
    parts.push(`${key}=${Array.isArray(value) ? value.join('+') : String(value)}`);
  }
  return parts.join(',');
}

/** Append the flags for an optional-amount toggle (`--sharpen [X]` / `--no-sharpen`) relative to its CLI default. */
function amountToggle(args: string[], settings: JobSettings, name: string, defaultOn: boolean): void {
  if (settings[`${name}_enabled`] === undefined) return;
  const enabled = first(settings, `${name}_enabled`) === 'true';
  const amount = numberOrNull(first(settings, `${name}_amount`));
  if (!enabled) {
    if (defaultOn) args.push(`--no-${name}`);
  } else if (amount !== null) {
    args.push(`--${name}`, String(amount));
  } else if (!defaultOn) {
    args.push(`--${name}`);
  }
}

/**
 * Build the `ziv image` / `ziv video` command that matches the settings a Workspace form would submit.
 *
 * Settings that equal the CLI's own defaults are left out; settings the CLI cannot express are listed as caveats.
 *
 * @throws {Error} When the settings carry no prompt, prompt caption, or prompts file, since the CLI would then
 *   fall back to its default `prompts.yaml`.
 */
export function cliCommand(settings: JobSettings, options: CliCommandOptions = {}): CliCommand {
  const isVideo = first(settings, 'mode') === 'video';
  // An extension continues a chosen video and takes its size, frame rate and length from it.
  const isExtend = isVideo && first(settings, 'workflow') === 'extend';
  const args: string[] = ['ziv', isVideo ? 'video' : 'image'];
  const caveats: CliCaveat[] = [];
  const flag = (name: string, value: string | undefined): void => {
    if (value !== undefined) args.push(name, value);
  };

  if (isExtend) {
    const source = text(settings, 'source_video');
    if (source === undefined) throw new Error('Choose a video to extend first.');
    args.push('--extend', source);
    flag('--extend-frames', text(settings, 'extend_frames'));
  }

  flag('--model', text(settings, 'model'));
  if (!isVideo) flag('--quantize', text(settings, 'quantize'));

  // Prompt: inline text, a structured JSON caption (sent only in caption mode, even when blank), or a prompts file.
  // A video's scene may stand in for a blank inline prompt; in prompt-file mode it is the default scene for entries without one.
  const isJson = settings.json_prompt !== undefined;
  const isFile = !isJson && first(settings, 'prompt_source') === 'file';
  const scene = isVideo && !isJson ? text(settings, 'scene') : undefined;
  const [promptFlag, promptValue] = isJson ? ['--json-prompt', text(settings, 'json_prompt')]
    : isFile ? ['--prompts-file', text(settings, 'prompts_file')]
    : ['--prompt', text(settings, 'prompt')];
  // An extension may continue with neither: the source's scene carries it.
  if (promptValue === undefined && (isFile || (scene === undefined && !isExtend))) {
    throw new Error(isJson ? 'Write a JSON caption first.' : isFile ? 'Choose a prompts file first.' : isVideo ? 'Write a scene or an action first.' : 'Write a prompt first.');
  }
  if (promptValue !== undefined) args.push(promptFlag, promptValue);
  else if (isExtend && !isFile && options.extendSource?.legacy) args.push('--prompt', '');
  flag('--scene', scene);
  if (isExtend && !isJson && scene === undefined && options.extendSource?.scene) args.push('--scene', '');
  if (isFile && values(settings, 'prompt_option_id').length < (options.promptFileOptionCount ?? Infinity)) caveats.push('prompt_options');
  if (!isJson && !isFile && text(settings, 'negative_prompt') !== undefined) caveats.push('negative_prompt');

  // Size: a custom width and height stand on their own; otherwise the ratio and preset. An extension has neither: it takes both from its source.
  const width = text(settings, 'width');
  const height = text(settings, 'height');
  if (!isExtend && (width !== undefined || height !== undefined)) {
    flag('--width', width);
    flag('--height', height);
  } else if (!isExtend) {
    flag('--ratio', text(settings, 'ratio'));
    flag('--size', text(settings, 'size'));
  }
  if (isVideo && !isExtend) flag('--frames', text(settings, 'frames'));

  const runs = numberOrNull(first(settings, 'runs'));
  if (runs !== null && runs !== 1) args.push('--runs', String(runs));
  flag('--steps', text(settings, 'steps'));
  if (!isVideo) {
    flag('--guidance', text(settings, 'guidance'));
    flag('--scheduler', text(settings, 'scheduler'));
    const sigma = numberOrNull(first(settings, 'first_sigma'));
    if (sigma !== null) args.push('--first-sigma', String(sigma));
  }
  flag('--seed', text(settings, 'seed'));
  flag('--lora', text(settings, 'lora'));

  // Reference image: an upload wins over a typed path, as on the server. It has no host path, so the command names it.
  const upload = isExtend ? undefined : text(settings, 'image_file');
  const imagePath = isExtend ? undefined : upload ?? text(settings, 'image_path');
  if (upload !== undefined) caveats.push('uploaded_image');
  flag('--image', imagePath);
  if (!isVideo && imagePath !== undefined) flag('--image-strength', text(settings, 'image_strength'));

  if (isVideo) {
    if (settings.audio !== undefined && !checked(settings, 'audio')) args.push('--no-audio');
    if (settings.low_memory !== undefined && !checked(settings, 'low_memory')) args.push('--no-low-memory');
    if (!isExtend) {
      const upscale = text(settings, 'video_upscale_factor') ?? text(settings, 'upscale');
      flag('--upscale', upscale);
      if (upscale !== undefined && checked(settings, 'review')) args.push('--review');
    }
  } else {
    amountToggle(args, settings, 'sharpen', true);
    amountToggle(args, settings, 'contrast', false);
    amountToggle(args, settings, 'saturation', false);
    const upscale = text(settings, 'upscale');
    if (upscale !== undefined) {
      args.push('--upscale', upscale);
      flag('--upscale-denoise', text(settings, 'upscale_denoise'));
      flag('--upscale-steps', text(settings, 'upscale_steps'));
      flag('--upscale-guidance', text(settings, 'upscale_guidance'));
      if (first(settings, 'upscale_sharpen') === 'false') args.push('--no-upscale-sharpen');
      if (checked(settings, 'review')) args.push('--review');
    }
  }

  // Auto-enhance rewrites every prompt before the model loads; JSON captions are never enhanced.
  if (!isJson && checked(settings, 'enhance_auto')) {
    const spec = enhanceSpec(first(settings, 'enhance_settings'), isVideo, isExtend);
    args.push('--enhance');
    if (spec) args.push(spec);
  }

  if (options.outputDir) args.push('--output', options.outputDir);
  return { command: args.map(shellQuote).join(' '), caveats };
}
