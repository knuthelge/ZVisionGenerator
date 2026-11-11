---
search:
  exclude: true
---

# Prompt-enhancer bench

The unit tests mock the enhancer, so they cover how a rewrite is asked for and parsed, not what a real model writes. The bench in `bench/enhance/` runs the real enhancer over fixed cases and counts how well the rewrites follow the request. Run it before and after changing the enhancer's wording (rules, option instructions, length plans) or a pinned enhancer model, and put the compare summary in the commit body or PR.

```bash
make enhance-bench ARGS="run --label main-fast"                  # all cases, Fast, 3 seeds each
make enhance-bench ARGS="run --label main-smart --model smart"
make enhance-bench ARGS="run --label try --tags quotes,short --seeds 1"   # tags include audio, lyrics, script and fill
make enhance-bench ARGS="compare main-fast try"                  # writes bench/enhance/runs/compare-main-fast-try.md
make enhance-bench ARGS="rescore main-fast try"                  # recompute checks from stored outputs, no model
make enhance-bench ARGS="sheet main-fast try --label look --tags quotes"   # render the rewrites of image cases as a picture sheet
```

- **Cases** live in `bench/enhance/cases.yaml`: a prompt (and a scene for video), the `enhance:` settings, the job context (ratio, frames, fps, audio, reference image, the extend source's prompt) and optional expectations (`keep`, `count`, `forbid`). Seeds come from the case id, so a run repeats with the same model, platform and settings.
- **A run** loads the enhancer once and calls the same functions preflight uses (`enhance_prompt`, `enhance_video_prompt`, `enhance_caption`, `enhance_audio_caption`, `improve_lyrics`, `edit_script`, `fill_in`). It writes `results.jsonl` (one line per case and seed, with the system prompt, the output, every model call and the checks) and `meta.json` (commit, model, revision, temperature) to `bench/enhance/runs/<label>/`, which is ignored by version control.
- **Checks:** usable, first try (no retry), format (scene and action pair, caption shape, audio shape), length within the planned range, quoted text kept exactly, `keep`, `count`, `forbid`, and for extensions no repeat of the earlier prompt. A silent video checks only quoted lettering, since it drops quoted speech by design. The audio modes add their own, below.
- **Audio cases** name a `family`, the model whose rules the rewrite follows (`yue2`, `stable_audio_3` or `qwen3_tts`), and run through the function of their task. Each mode takes a `settings` shape and a `context`:

| Mode | Runs | `settings` | `context` | Checks |
|---|---|---|---|---|
| `audio-sound` | `enhance_audio_caption` | Sound axes | `seconds`, `sound_type` | `format`, `length_in_range`, `banned` |
| `audio-style` | `enhance_audio_caption` | Music axes | `lyrics`, `instrumental`, `tempo`, `key` | `format`, `length_in_range`, `banned`, `marks_kept` |
| `audio-lyrics` | `improve_lyrics` | `improve`, `note`, `section`, `action` | `style` | `format`, `tags_kept`, `scope_kept`, `quotes_kept` |
| `audio-script` | `edit_script` | `actions`, `seconds`, `pace` | none | `format`, `words_kept`, `marks_known`, `fit_in_range`, `marks_kept`, `quotes_kept` |
| `audio-fill` | `fill_in` | `fields` | what the user already set | `fields_filled`, `only_empty`, `values_valid`, `fill_tags`, `banned`, `format` |

  A case the task would refuse fails to load. `keep`, `count`, `forbid` and quoted text work as for images.
- **Audio checks.** `format` is the expected shape (plain text, no markdown). `length_in_range` is the word range of the family's shape. `banned` is true when the text avoids the words the family's rules rule out. `marks_kept` is true when every `[mark]` or section tag of the input survives. `tags_kept` is true when the output has the same section tags in the same order. `scope_kept` is true when the task stayed in its scope. `words_kept` is true when Add delivery changed no word. `marks_known` is true when no bracketed word is one the server would refuse. `fit_in_range` is true when a Fit reads within 15% of the target time, and `length_ratio` shows whether a model runs long or short. `fields_filled` is the share of the asked fields the answer holds; `only_empty`, `values_valid` and `fill_tags` read the first answer.
- **`keep`** items match case-insensitively, each word as a stem ("river" finds "riverbank"), with up to two words in between ("green lamp" finds "green desk lamp"); `|` lists alternatives ("old man|elderly man"). **`count`** accepts a number word or digit with up to two words before the noun ("three fluffy cats").
- **Rescore** recomputes the checks of stored runs from their outputs, after a check or a case's expectations change. A video's format result is kept, since the stored text can't show whether the pair needed the fallback.
- **Compare** reports each check overall and per tag, speed, and the ten cases that changed most, side by side with a diff of their system prompts.
- **Sheet** answers what the checks cannot: whether a different rewrite makes a different picture. Given the labels of one or more runs, it renders every image case's rewrites through the image CLI (`--model`, default `zit`; this loads that model, so it needs the hardware) and lays the pictures out in `index.html` in `bench/enhance/runs/<label>/`, one row per case and one column per run after the user's own prompt. One seed, ratio and size apply to every image (`--seed`, `--ratio`, `--size`), so only the prompt differs.

The bench loads one enhancer model (and, for `sheet`, one image model). `tests/prompts/test_enhance_bench.py` covers its checks and plumbing with a fake model.

## Other benches

Two more benches in `bench/` are spikes for one decision each, run by hand and not part of any check:

- `bench/transparent/` measures background-removal methods on rendered cases. Its `render` steps drive the image CLI in the app's environment; its `matte` and `sheet` steps need matting libraries and run in a separate environment, so the app's dependencies stay as they are. Its module docstring lists the commands.
- `bench/video_upscale/` (macOS) renders source clips, upscales them with LTX's stage 2 against a bicubic baseline and records time and peak memory per phase; `sheet` and `review` write a blind comparison page. Its module docstring lists the commands.
