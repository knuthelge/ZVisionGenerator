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
    const settings = { style: 'photo', mood: 'keep', details: ['lighting', 'camera'], length: 'longer', motion: ['action'] };

    expect(cliCommand({ mode: 'image', prompt: 'a', enhance_auto: 'true', enhance_settings: JSON.stringify(settings) }).command)
      .toBe('ziv image --prompt a --enhance style=photo,mood=keep,details=lighting+camera,length=longer');
    expect(cliCommand({ mode: 'video', prompt: 'a', enhance_auto: 'true', enhance_settings: JSON.stringify(settings) }).command)
      .toBe('ziv video --prompt a --enhance style=photo,mood=keep,details=lighting+camera,length=longer,motion=action');
    expect(cliCommand({ mode: 'image', prompt: 'a', enhance_auto: 'true' }).command).toBe('ziv image --prompt a --enhance');
  });

  it('builds a video command with frames, toggles, and upscale', () => {
    const { command } = cliCommand({
      mode: 'video', workflow: 'img2vid', model: 'ltx', prompt: 'waves', ratio: '16:9', size: 'm', frames: '97', steps: '8',
      image_path: '/refs/sea.png', image_strength: '0.5', guidance: '3', quantize: '8',
      audio: 'false', low_memory: ['on', 'false'], upscale: '2', video_upscale_factor: '2',
    });

    expect(command).toBe('ziv video --model ltx --prompt waves --ratio 16:9 --size m --frames 97 --steps 8 --image /refs/sea.png --no-audio --upscale 2');
  });
});
