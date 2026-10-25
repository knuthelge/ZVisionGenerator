# Prompt-enhancer bench

The unit tests mock the enhancer, so they cover how a rewrite is asked for and parsed, not what a real model writes. The bench in `bench/enhance/` runs the real enhancer over fixed cases and counts how well the rewrites follow the request. Run it before and after changing the enhancer's wording (rules, option instructions, length plans) or a pinned enhancer model, and put the compare summary in the commit body or PR.

```bash
make enhance-bench ARGS="run --label main-fast"                  # all cases, Fast, 3 seeds each
make enhance-bench ARGS="run --label main-smart --model smart"
make enhance-bench ARGS="run --label try --tags quotes,short --seeds 1"
make enhance-bench ARGS="compare main-fast try"                  # writes bench/enhance/runs/compare-main-fast-try.md
make enhance-bench ARGS="rescore main-fast try"                  # recompute checks from stored outputs, no model
```

- **Cases** live in `bench/enhance/cases.yaml`: a prompt (and a scene for video), the `enhance:` settings, the job context (ratio, frames, fps, audio, reference image, the extend source's prompt) and optional expectations (`keep`, `count`, `forbid`). Seeds come from the case id, so a run repeats with the same model, platform and settings.
- **A run** loads the enhancer once and calls the same functions preflight uses (`enhance_prompt`, `enhance_video_prompt`, `enhance_caption`). It writes `results.jsonl` (one line per case and seed, with the system prompt, the output, every model call and the checks) and `meta.json` (commit, model, revision, temperature) to `bench/enhance/runs/<label>/`, which is ignored by version control.
- **Checks:** usable, first try (no retry), format (scene and action pair, caption shape), length within the planned range, quoted text kept exactly, `keep`, `count`, `forbid`, and for extensions no repeat of the earlier prompt. A silent video checks only quoted lettering, since it drops quoted speech by design.
- **`keep`** items match case-insensitively, each word as a stem ("river" finds "riverbank"), with up to two words in between ("green lamp" finds "green desk lamp"); `|` lists alternatives ("old man|elderly man"). **`count`** accepts a number word or digit with up to two words before the noun ("three fluffy cats").
- **Rescore** recomputes the checks of stored runs from their outputs, after a check or a case's expectations change. A video's format result is kept, since the stored text can't show whether the pair needed the fallback.
- **Compare** reports each check overall and per tag, speed, and the ten cases that changed most, side by side with a diff of their system prompts.

The bench loads one enhancer model and nothing else. `tests/test_enhance_bench.py` covers its checks and plumbing with a fake model.
