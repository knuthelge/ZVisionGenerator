import { describe, expect, it } from 'vitest';

import { cliCommand, formSettings, shellQuote } from './cliCommand';

describe('shellQuote', () => {
  it('leaves plain words bare and single-quotes everything else', () => {
    expect(shellQuote('2:3')).toBe('2:3');
    expect(shellQuote('/models/z-image@main')).toBe('/models/z-image@main');
    expect(shellQuote('a red fox')).toBe("'a red fox'");
    expect(shellQuote("the fox's den")).toBe("'the fox'\\''s den'");
    expect(shellQuote('$HOME')).toBe("'$HOME'");
    expect(shellQuote('')).toBe("''");
  });
});

describe('formSettings', () => {
  it('keeps repeated fields as lists and records uploads by file name', () => {
    const form = new FormData();
    form.append('mode', 'image');
    form.append('audio', 'on');
    form.append('audio', 'false');
    form.append('image_file', new File(['x'], 'ref.png'));

    expect(formSettings(form)).toEqual({ mode: 'image', audio: ['on', 'false'], image_file: 'ref.png' });
  });
});

describe('cliCommand', () => {
  it('builds an image command with every submitted setting', () => {
    const { command, caveats } = cliCommand({
      mode: 'image', workflow: 'img2img', model: 'zit', quantize: '8', prompt_source: 'inline', prompt: 'a lake at dawn',
      ratio: '2:3', size: 'm', runs: '3', steps: '9', guidance: '3.5', seed: '42', scheduler: 'beta', lora: 'style:0.7',
      image_path: '/refs/lake.png', image_strength: '0.6',
      sharpen_enabled: 'true', sharpen_amount: '0.8', contrast_enabled: 'true', contrast_amount: '1.1', saturation_enabled: 'false',
      upscale: '2', upscale_denoise: '0.3', upscale_steps: '', upscale_guidance: '', upscale_sharpen: 'false',
    }, { outputDir: '/home/me/ziv out' });

    expect(command).toBe(
      "ziv image --model zit --quantize 8 --prompt 'a lake at dawn' --ratio 2:3 --size m --runs 3 --steps 9 --guidance 3.5"
      + ' --scheduler beta --seed 42 --lora style:0.7 --image /refs/lake.png --image-strength 0.6'
      + ' --sharpen 0.8 --contrast 1.1 --upscale 2 --upscale-denoise 0.3 --no-upscale-sharpen'
      + " --output '/home/me/ziv out'",
    );
    expect(caveats).toEqual([]);
  });

  it('leaves out settings that match the CLI defaults', () => {
    const { command } = cliCommand({
      mode: 'image', model: 'zit', prompt: 'a lake', runs: '1', seed: '', scheduler: '', first_sigma: '',
      sharpen_enabled: 'true', contrast_enabled: 'false', saturation_enabled: 'false', upscale_sharpen: 'true',
    });

    expect(command).toBe('ziv image --model zit --prompt \'a lake\'');
  });

  it('turns default-on post-processing off and default-off post-processing on', () => {
    const { command } = cliCommand({ mode: 'image', prompt: 'a', sharpen_enabled: 'false', saturation_enabled: 'true' });

    expect(command).toBe('ziv image --prompt a --no-sharpen --saturation');
  });

  it('uses a custom width and height instead of the ratio preset', () => {
    const { command } = cliCommand({ mode: 'image', prompt: 'a', ratio: '2:3', size: 'm', width: '1024', height: '768' });

    expect(command).toBe('ziv image --prompt a --width 1024 --height 768');
  });

  it('sends a structured JSON caption without prompt enhancement', () => {
    const { command } = cliCommand({ mode: 'image', json_prompt: '{"high_level_description": "a lake"}', enhance_auto: 'true' });

    expect(command).toBe('ziv image --json-prompt \'{"high_level_description": "a lake"}\'');
  });

  it('reports a negative prompt the CLI cannot take inline', () => {
    const { command, caveats } = cliCommand({ mode: 'image', prompt: 'a lake', negative_prompt: 'blur' });

    expect(command).toBe("ziv image --prompt 'a lake'");
    expect(caveats).toEqual(['negative_prompt']);
  });

  it('runs a prompts file and reports it only when some of its prompts were chosen', () => {
    const settings = { mode: 'image', prompt_source: 'file', prompts_file: '/p/prompts.yaml', prompt_option_id: ['a:0', 'b:1'], prompt: 'ignored' };

    expect(cliCommand(settings, { promptFileOptionCount: 2 })).toEqual({ command: 'ziv image --prompts-file /p/prompts.yaml', caveats: [] });
    expect(cliCommand(settings, { promptFileOptionCount: 3 }).caveats).toEqual(['prompt_options']);
    expect(cliCommand(settings).caveats).toEqual(['prompt_options']);
  });

  it('refuses settings without a prompt instead of falling back to the default prompts file', () => {
    expect(() => cliCommand({ mode: 'image', prompt_source: 'inline', prompt: '  ' })).toThrow(Error);
    expect(() => cliCommand({ mode: 'image', json_prompt: '', prompt: 'hidden' })).toThrow('JSON caption');
    expect(() => cliCommand({ mode: 'image', prompt_source: 'file', prompts_file: '' })).toThrow(Error);
  });

  it('names an uploaded reference image over a typed path, as the server does, and reports it', () => {
    const { command, caveats } = cliCommand({ mode: 'image', prompt: 'a', image_path: '/a/old.png', image_file: 'my ref.png', image_strength: '0.5' });

    expect(command).toBe("ziv image --prompt a --image 'my ref.png' --image-strength 0.5");
    expect(caveats).toEqual(['uploaded_image']);
  });

  it('serializes auto-enhance options to the SPEC grammar, with motion only for video', () => {
    const settings = { style: 'photo', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: ['lighting', 'camera'], length: 'longer', motion: ['action'] };

    expect(cliCommand({ mode: 'image', prompt: 'a', enhance_auto: 'true', enhance_settings: JSON.stringify(settings) }).command)
      .toBe('ziv image --prompt a --enhance style=photo,mood=keep,theme=keep,details=lighting+camera,length=longer');
    expect(cliCommand({ mode: 'video', prompt: 'a', enhance_auto: 'true', enhance_settings: JSON.stringify(settings) }).command)
      .toBe('ziv video --prompt a --enhance style=photo,mood=keep,theme=keep,details=lighting+camera,length=longer,camera=keep,pace=keep,motion=action');
    expect(cliCommand({ mode: 'image', prompt: 'a', enhance_auto: 'true' }).command).toBe('ziv image --prompt a --enhance');
  });

  it('writes the theme, camera, pace and sound, and the direction as --enhance-note', () => {
    const settings = { style: 'cinematic', theme: 'noir', details: [], length: 'same', camera: 'orbit', pace: 'slow', motion: ['action'], sound: ['ambient'], note: 'in the rain' };
    const { command } = cliCommand({ mode: 'video', prompt: 'a', enhance_auto: 'true', enhance_settings: JSON.stringify(settings) });
    expect(command).toContain('--enhance style=cinematic,theme=noir,details=,length=same,camera=orbit,pace=slow,motion=action,sound=ambient');
    expect(command).toContain('--enhance-note');
    expect(command).toContain('in the rain');
    const image = cliCommand({ mode: 'image', prompt: 'a', enhance_auto: 'true', enhance_settings: JSON.stringify(settings) }).command;
    expect(image).not.toMatch(/camera=|pace=|motion=|sound=/);
  });

  it('names the Smart enhancer and JSON captions', () => {
    expect(cliCommand({ mode: 'image', prompt: 'a', enhance_auto: 'true', enhance_model: 'smart', enhance_format: 'json' }).command)
      .toBe('ziv image --prompt a --enhance --enhance-model smart --enhance-format json');
    expect(cliCommand({ mode: 'video', prompt: 'a', enhance_auto: 'true', enhance_model: 'smart', enhance_format: 'json' }).command)
      .toBe('ziv video --prompt a --enhance --enhance-model smart');
  });

  it('writes JSON captions for a prompt file without auto-enhance with the format alone', () => {
    expect(cliCommand({ mode: 'image', prompt_source: 'file', prompts_file: '/p/prompts.yaml', enhance_format: 'json' }).command)
      .toBe('ziv image --prompts-file /p/prompts.yaml --enhance-format json');
  });

  it('spells out the Rewrite choice only where it is not the default of the workflow', () => {
    const settings = (parts: string) => JSON.stringify({ style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: ['lighting'], length: 'same', motion: ['action'], parts });
    const video = { mode: 'video', prompt: 'a', enhance_auto: 'true' };
    expect(cliCommand({ ...video, enhance_settings: settings('both') }).command).toBe('ziv video --prompt a --enhance style=keep,mood=keep,theme=keep,details=lighting,length=same,camera=keep,pace=keep,motion=action');
    expect(cliCommand({ ...video, enhance_settings: settings('action') }).command).toContain('--enhance parts=action,style=keep');
    const extend = { ...video, workflow: 'extend', source_video: '/clips/walk.mp4' };
    expect(cliCommand({ ...extend, enhance_settings: settings('action') }).command).not.toContain('parts=');
    expect(cliCommand({ ...extend, enhance_settings: settings('both') }).command).toContain('parts=both');
    // An image has no scene to keep.
    expect(cliCommand({ mode: 'image', prompt: 'a', enhance_auto: 'true', enhance_settings: settings('action') }).command).not.toContain('parts=');
  });

  describe('scene', () => {
    it('adds --scene for video beside the prompt', () => {
      expect(cliCommand({ mode: 'video', prompt: 'a knight rides in', scene: 'A castle at dusk.' }).command)
        .toBe("ziv video --prompt 'a knight rides in' --scene 'A castle at dusk.'");
    });

    it('runs a scene alone without a --prompt', () => {
      expect(cliCommand({ mode: 'video', prompt: '', scene: 'A castle at dusk.' }).command).toBe("ziv video --scene 'A castle at dusk.'");
    });

    it('needs a scene or a prompt for video', () => {
      expect(() => cliCommand({ mode: 'video', prompt: ' ', scene: ' ' })).toThrow(Error);
    });

    it('passes the default scene beside a prompts file, but never without a file', () => {
      expect(cliCommand({ mode: 'video', prompt_source: 'file', prompts_file: '/p/city.yaml', scene: 'A rainy city.' }).command)
        .toBe("ziv video --prompts-file /p/city.yaml --scene 'A rainy city.'");
      expect(() => cliCommand({ mode: 'video', prompt_source: 'file', prompts_file: '', scene: 'A rainy city.' })).toThrow(Error);
    });

    it('leaves a blank scene out, and ignores a scene for images', () => {
      expect(cliCommand({ mode: 'video', prompt: 'a', scene: '  ' }).command).toBe('ziv video --prompt a');
      expect(cliCommand({ mode: 'image', prompt: 'a', scene: 'A castle.' }).command).toBe('ziv image --prompt a');
    });
  });

  it('builds a video command with frames, toggles, and upscale', () => {
    const { command } = cliCommand({
      mode: 'video', workflow: 'img2vid', model: 'ltx', prompt: 'waves', ratio: '16:9', size: 'm', frames: '97', steps: '8',
      image_path: '/refs/sea.png', image_strength: '0.5', guidance: '3', quantize: '8',
      audio: 'false', low_memory: ['on', 'false'], upscale: '2', video_upscale_factor: '2',
    });

    expect(command).toBe('ziv video --model ltx --prompt waves --ratio 16:9 --size m --frames 97 --steps 8 --image /refs/sea.png --no-audio --upscale 2');
  });

  it('adds --review for an upscale that pauses for review', () => {
    const base = { mode: 'image', model: 'zit', prompt: 'a lake', upscale: '2' };
    expect(cliCommand({ ...base, review: 'true' }).command).toContain('--upscale 2 --review');
    expect(cliCommand(base).command).not.toContain('--review');
    // The Workspace never posts review without upscale, but a stray field must not make the CLI refuse the command.
    expect(cliCommand({ mode: 'image', model: 'zit', prompt: 'a lake', review: 'true' }).command).not.toContain('--review');
  });

  it('adds --review to a video command with an upscale that pauses for review', () => {
    const base = { mode: 'video', model: 'ltx', prompt: 'waves', video_upscale_factor: '2', upscale: '2' };
    expect(cliCommand({ ...base, review: 'true' }).command).toBe('ziv video --model ltx --prompt waves --upscale 2 --review');
    expect(cliCommand(base).command).not.toContain('--review');
    // A stray review field without an upscale must not make the CLI refuse the command.
    expect(cliCommand({ mode: 'video', model: 'ltx', prompt: 'waves', review: 'true' }).command).not.toContain('--review');
  });

  describe('extend', () => {
    const extend = { mode: 'video', workflow: 'extend', source_video: '/clips/walk one.mp4', extend_frames: '48', model: 'ltx', steps: '8' };

    it('continues the source with the new frames and leaves out the size and length flags', () => {
      const { command } = cliCommand({
        ...extend, prompt: 'She turns and waves.', scene: 'A rainy street.',
        ratio: '16:9', size: 'm', width: '848', height: '480', frames: '97', image_path: '/refs/a.png', video_upscale_factor: '2', upscale: '2',
        audio: 'false',
      });

      expect(command).toBe("ziv video --extend '/clips/walk one.mp4' --extend-frames 48 --model ltx --prompt 'She turns and waves.' --scene 'A rainy street.' --steps 8 --no-audio");
    });

    it('leaves --prompt out when the prompt is empty and still runs without any text', () => {
      expect(cliCommand({ ...extend, prompt: '', scene: '' }).command).toBe("ziv video --extend '/clips/walk one.mp4' --extend-frames 48 --model ltx --steps 8");
      expect(cliCommand({ ...extend, prompt: '', scene: 'A rainy street.' }).command).toContain("--scene 'A rainy street.'");
    });

    it('sends an empty --scene or --prompt where the CLI would fill them from the source', () => {
      const blank = { ...extend, prompt: '', scene: '' };
      expect(cliCommand(blank, { extendSource: { scene: 'A rainy street.', legacy: false } }).command).toContain(" --scene ''");
      expect(cliCommand(blank, { extendSource: { scene: 'A rainy street.', legacy: false } }).command).not.toContain('--prompt');
      expect(cliCommand(blank, { extendSource: { scene: null, legacy: true } }).command).toContain(" --prompt ''");
      expect(cliCommand(blank, { extendSource: { scene: null, legacy: true } }).command).not.toContain('--scene');
      expect(cliCommand(blank, { extendSource: { scene: null, legacy: false } }).command).not.toMatch(/--(prompt|scene)/);
    });

    it('refuses an extension without a source', () => {
      expect(() => cliCommand({ mode: 'video', workflow: 'extend', prompt: 'a', source_video: ' ' })).toThrow('video to extend');
    });
  });
});
