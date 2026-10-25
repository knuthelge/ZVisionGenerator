# Batches and prompt files

A prompt file holds many prompts, grouped in named sets. Run a file and every active prompt is generated; add `-r` to run the whole file several times.

## Run a prompt file

```yaml title="prompts.yaml"
fox:
  - prompt: A red fox sitting in fresh snow, early morning light
  - prompt: A red fox {leaping|sleeping} in a {birch|pine} forest

lighthouse:
  - prompt: A lighthouse in a storm, waves crashing
```

```bash
ziv-image -m klein9b -p prompts.yaml        # every prompt once
ziv-image -m klein9b -p prompts.yaml -r 3   # the whole file three times
ziv-video -m ltx-4 -p clips.yaml            # video takes the same files
```

- `-p` defaults to `prompts.yaml` in the current folder.
- `--prompt` overrides `-p`.
- Each run picks the `{a|b}` choices anew and uses a new seed, so `-r 3` gives three different images per prompt.
- The set name starts the file name: `fox_2026-10-04_14-03-22.png`.

The full format, with snippets, structured prompts, negative prompts and video scenes, is in [Prompts and prompt files](../concepts/prompts.md). The Web UI's [Prompts page](../web-ui/prompts.md) edits the same files without YAML.

## Turn prompts on and off

```yaml
fox:
  - prompt: A red fox sitting in fresh snow
  - active: false
    prompt: A red fox at the beach   # skipped
```

## Controls during a run

While `ziv-image` runs a batch, single keys control it:

| Key | Action | What happens |
|---|---|---|
| ++n++ | Skip | Stops the current image and moves to the next prompt |
| ++q++ | Quit | Stops the current image and ends the batch |
| ++p++ | Pause | Finishes the current image, then waits for a key |
| ++r++ | Repeat | Finishes the current image, then makes it again with a new seed (same random choices and enhanced prompt) |

The keys also work while prompts are enhanced and while the model loads. During enhancement, ++n++ skips the current rewrite (that image uses your original prompt), ++q++ stops before the model loads, ++p++ pauses, and ++r++ does nothing.

`ziv-video` stops with ++ctrl+c++.

## Enhance a batch

```bash
ziv-image -m klein9b -p prompts.yaml --enhance style=photo,length=longer
```

Every prompt is rewritten before the model loads. Prompt files can also set enhancement per entry with `enhance:`. See [Enhance prompts](enhance.md).
