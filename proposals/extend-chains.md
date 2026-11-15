# Extend chains: make a long video in parts

**Status:** Proposed (2026-11-15)

## Problem

A MiniMax H3 clip that is too long for the machine gets a **Too large** note under **Duration (s)**, and nothing else. The note only warns and H3 has no length cap, so the job runs out of memory or crawls. The way round it is manual: make a 5 s clip, wait, Extend it, wait, Extend that, and so on. Each Extend is set up again by hand, and each leaves another file in the Gallery that holds the whole video so far.

The app already has every piece of the loop:

- Extend continues from a clip's last frames and carries the scene and, on H3, the references. Its output is the whole video so far (`join_clips_stage` splices the source and the new part).
- An Extend can follow an audio track from any point in it (`AudioTrack.start`).
- On H3 each Extend's window needs about the same memory as a new clip of the same length. The memory and time estimates already exist (`clip_memory_fit` in `utils/model_memory.py`, `utils/h3_memory.py`).

What's missing is one job that does the loop.

## What the user sees

### Workspace

On the **Video** tab, in the **New** kind, with **MiniMax H3**: when the Duration note says **Tight** or **Too large**, it offers the fix.

> At this size and length: needs ~13 GB of 10 GB of GPU memory. Expect an out-of-memory error. Use fewer frames or a smaller size. **Make it in 5 parts**

- **Make it in 5 parts** turns the job into a chain: a first clip, then Extends of it until the video is as long as asked. The count is always the fewest parts whose window fits, so the user never sets a number.
- **The note then reads:** "In 5 parts of 5.2 s, each continuing the one before. About 34 min (estimated)." It ends with **Make it in one**, which turns the chain off. When the length changes, the count follows. When it fits in one part again, the line goes away and the job is a plain clip.
- **No offer appears** when the job has a last frame, when **Upscale** is on, or when no part size fits (the note then says what it says today). An Extend can't end on a last frame or be upscaled, and an upscaled clip can't be extended.
- **Everything else is set as for one clip:** start image, references, audio track, scene and action. Every part gets the scene and action exactly as typed, quotes included, so a quoted line is spoken in each part. With Enhance on, the enhancer rewrites them once before the job, and every part uses that text.

The job card shows **Part 2 of 5** beside the step progress. The job has one card and one **Cancel**.

**Each part shows as it finishes.** When part 2 is saved (a video of about 9 s that holds parts 1 and 2), it replaces part 1 as the job's output: on the card, in the result strip and for Reuse. The file of part 1 is removed. The user can watch each seam as it arrives and cancel early. The Gallery always holds one file for the job, and it grows.

**Controls, failure and Cancel:**

- **Skip** and **Cancel** stop the running part at its next step, as they stop a video today. The saved file stays as the job's output, and the card says "Stopped in part 3 of 5". The finished parts can be half an hour of GPU time, so the job keeps them.
- **A failure in part 3** leaves the file of parts 1–2 as the job's output. The card and the Job log say "Part 3 of 5 failed: *reason*". To go on, the user Extends that file by hand, as today.
- **Pause** waits after the current part, not the whole video: a chain can take an hour.
- **Repeat** makes the whole video again with a new seed, after the last part.

**Reuse** of the result loads the New form as it was submitted: start image, references, audio track and its start, scene, action, seed, the whole length and **In parts**. **Extend** of the result works as for any clip.

**Layouts:**

- **Desktop (1024 px and wider):** the offer and the parts line sit in the Duration note in the settings sidebar's Sampling section.
- **Tablet (640–1023 px):** the same note in the settings half under the stage, with the buttons at touch size.
- **Phone:** the **Length** chip reads "20 s · 5 parts". It keeps its warning mark while the clip is in one part and doesn't fit. Its sheet holds the Duration row and the same note and buttons. The **All** sheet holds the whole section. The compact job card shows **Part 2 of 5** as the full card does.

### CLI

```
ziv-video -m h3 --image start.png --reference portrait.png --audio-file song.flac --frames 481 --parts 5 --scene "…" --prompt "…"
```

- `--parts N` makes the clip in N parts: the first clip, then N−1 Extends.
- When the clip is too large, the existing warning names the count: "… Use fewer --frames or a smaller --size, or --parts 5."
- Each part prints a line: `Part 2 of 5: +102 frames, seed=43`.
- `--parts` is refused with `--end-image`, `--upscale` or `--extend`, and on a model that can't make parts.

## How the parts are cut

Every part is generated in a window of the same size, so every part needs the same memory:

- Each Extend adds *e* new frames, a multiple of the model's Extend step. Its window is those frames plus the frames it carries from the part before.
- The first clip is as long as one Extend's window (`window_frames(e, family)`).
- *e* is the smallest step multiple for which the first clip plus N−1 Extends reach the asked length. The last Extend adds only what is left, so the video is as long as asked.

On H3 at 24 fps, a 20 s clip is 481 frames, and a 5 s window is 124 frames:

| Platform | Frames carried | Parts | Each part's window | Each Extend adds |
|---|---|---|---|---|
| Windows/Linux | 22 | 5 (a clip, then 4 Extends) | 124 frames, 5.2 s | 102 frames, 4.25 s (the last 51) |
| macOS | 1 | 4 (a clip, then 3 Extends) | 124 frames, 5.2 s | 119 frames, 5 s |

On Windows/Linux an Extend repeats the last 22 frames of the part before inside its window, so a 5 s window adds 4.25 s. "20 s on a card that fits 5 s" is a clip and 4 Extends.

## How it works

### One job, looping in the runner

The parts loop runs inside `video_runner.run_video_batch`, once for each iteration of the batch. A batch of 3 runs makes three videos in parts. It is not a set of queued jobs, one per part:

| | One job, loop in the runner | One queued job per part |
|---|---|---|
| CLI | The same code; `ziv-video` has no queue | Needs its own loop anyway |
| A part's source | The file the last part just wrote | A file that doesn't exist when the job is queued; the queue would need jobs that wait for another job's output |
| Cancel and failure | One Cancel; the loop stops | Cancel or failure must drop the later jobs too |
| Other jobs | Can't run between parts | A job queued meanwhile runs between parts and loads its own model |
| Job card and Job log | One | N |

Jobs that wait for another job's output belong to [Studio](studio.md#the-queue), where each shot has its own prompt.

### Model handles

The H3 backends resolve files in `load_model` and load the models for each clip, but the handle is made for one mode:

- **Windows/Linux:** `_require_mode` refuses an Extend on a `t2v` or `i2v` handle.
- **macOS:** `extend_video` refuses only a `t2v` handle, and the task tag doesn't read the mode. A job that starts from an image could Extend on its first handle.

The callers already load the model (`video_cli.main`, `WebRunner._run_video_request`). For a job in parts they also load it with `mode="extend"` on both platforms, so the caller code is the same everywhere, and pass that handle to `run_video_batch` as `extend_model`. The second load is light: on Windows/Linux it resolves the files and checks the stored copy; on macOS it also writes the combined user-LoRA file again, which takes seconds.

### Each part

- **Part 1** is today's request, made with `num_frames` = the part window.
- **Part k > 1** is an Extend request:
  - its source is `load_extend_source(file of part k−1)`, with the recorded prompt cleared. A file made without a scene would otherwise count as `legacy`, and `_extend_action` would strip the quotes from the action. The action is sent as typed.
  - its frames are the part's *e*;
  - its audio track is the job's track from its start plus the frames so far over the frame rate. The runner works this out itself rather than reading back the metadata it rewrote.
  - it has no start image;
  - its references and text are the same as part 1's.

  It runs through the Extend workflow (`build_video_workflow` gains a way to build it for the same job) on the Extend handle. No stage changes.
- **Seed:** part k uses the job's seed + k − 1, wrapped into the seed range. With one seed for every part, each Extend would start from the same noise in a window of the same shape. Only the job's seed is recorded; the parts' seeds follow from it, so Reuse makes the same video.
- **Files:** after part k is saved, the runner removes the file of part k−1. A missing file is ignored. The newest part lives in the output folder, not in a temporary one, because a hard kill (out of memory) runs no `finally` that could move it into place.
- **Events:** `generation_started` and the step events carry `part` and `n_parts`, the names audio already uses. A new `part_finished` event carries each saved part's `output_path` and, from part 2 on, `replaces`: the previous part's path. `generation_finished` fires once per video, as today.
- **Controls:** `SkipSignal` holds one action. Quit and Next stop the running part at its next step, as today. Between parts the runner consumes Pause, Quit and Next and acts on them at once:
  - Pause waits (`pause_until_resumed`);
  - Quit and Next stop the chain, and the saved file stays the output.

  Without this, a Quit or Next pressed as a part ends would carry into the next part and be seen only at its first step, after the encodes. On a Mac text-to-video part, it would not be seen at all. Only Repeat stays queued, so it runs after the last part.

### The job's output in the Web UI

Today `WebRunner._publish_event` registers an output only on a successful `generation_finished`, and Quit emits only `batch_cancelled`. `part_finished` adds the new part to the job's outputs and makes it the result path. When the event carries `replaces`, the entry with that path is removed. The frontend does the same in the job's outputs and the result strip. Because the entry is matched by path, the finished video of run 1 in a batch of 2 is never touched. A later stop or failure leaves the newest part in place, and the card shows "Stopped in part 3 of 5" or "Part 3 of 5 failed".

### Recorded settings

A file made in parts records the job, not its last part. After each part is saved, the runner embeds part 1's payload again with the existing `build_video_config_payload` and `embed_mp4_config`, with:

- the file's real frame count and the matching `audio_track_end`;
- the job's total generation time so far, not part 1's;
- `parts`, the number of parts the file holds.

So the record holds part 1's workflow (`txt2vid` or `img2vid`), start image, references, audio track and its start, scene, action and seed. It records no `source_video`, so Reuse never points at a removed part, and Extend of the result continues the track. `RecordedSettings` gains `parts`. The request type, the stages and the Extend branch of `utils/provenance.py` don't change.

### Capability and server

- `parts: true` goes in the `h3` family's capabilities in `utils/model_metadata.yaml`. The web layer reads it, as the architecture invariants ask, and the CLI refuses `--parts` without it.
- A new pure module, `utils/video_parts.py`, holds two functions, shared by the CLI, the server and the runner:
  - `part_lengths(total, parts, family)` returns the first clip's frames and each Extend's frames.
  - `fewest_parts(total, family, fits)` returns the fewest parts whose window fits, using a fit check passed in. The check takes the larger of two estimates at the window, and requires `fits`, not `tight`:
    - part 1's mode (`i2v` counts the keyframe rows);
    - `extend` (with references, also the tail's vision block).
- `POST /api/video/memory-fit` takes `in_parts` and returns `parts`: the fewest parts that fit (1 when the whole clip does), or `null` when the job can't be made in parts or no part fits. A system-memory fit has no count, because a shorter window doesn't change it. With `in_parts` and more than one part, `memory_fit` describes one part's window, and `expected_seconds` is the first clip plus the Extends.
- The video form takes `parts`, and `_parse_video_job` applies the same refusals as the CLI.

### Frontend

- The draft gains `inParts`. It travels with Duration in Reuse, **Load settings** and presets.
- The clip memory request sends it, and the form submits the count the note shows.
- The Duration note gets the two buttons, and the Length chip gets the count.
- `job.svelte.ts` reads `part` and `n_parts` for the card, and `part_finished` for the output.
- `cliCommand.ts` writes `--parts`.

### macOS

The loop, the files, the events and the recorded settings are shared, so macOS ships with Windows/Linux. On macOS each H3 Extend starts from the last frame only (`platform_carry_frames: { darwin: 1 }`), so motion can change more at a seam. The guide says so. Carrying more frames on macOS is its own [backlog](backlog.md) item.

## Decisions

- **One action for all parts.**
  - Per-part actions need a list of text boxes, which is too much for the phone sheet, and they are a story editor, which is what Studio is for.
  - The guide says to write an action that can go on ("he walks along the beach"), not one that ends ("he sits down"). An action that ends is done again in every part.
  - The enhancer's rewrite is planned for part 1's mode, so if it describes the opening picture, every part gets that description. The guide says so in one line.
  - The text is never changed per part.
- **The split is an offer in the Duration note**, with no parts control. The count is always the fewest parts that fit. The CLI takes an explicit `--parts N`, and its warning names the count.
- **Intermediate files:** only the newest file is kept. It always holds every finished part and is the job's output.
- **Failure or Cancel in part k:** the parts finished so far stay as one file, recorded as a video of k−1 parts.
- **Seed:** the job's seed + k − 1 for part k.
- **Time and memory:** the memory note describes one part's window, and the time note covers all parts.
- **LTX:** not in this change, on either platform. The need comes from H3, whose Extend also carries references. On Windows/Linux, LTX loads one pipeline per mode (`_PIPELINES_BY_MODE` in `backends/cuda/video/backend.py`), so a second handle is a second pipeline in system memory.

## Not in this change

- **The Extend kind in parts** (extending a clip by 20 s). Extend already sets its own length, and adding parts there doubles the Web UI and server work (the extend inspection route). The loop itself would take it unchanged.
- **Parts with Upscale.** An Extend is made at the source's size. Upscale the finished video from the Gallery instead.
- **A different action per part.** See Studio.

## Alternatives considered

- **One queued job per part:** see the table above.
- **A Parts control (a number) next to Duration.** It means one more setting on every layout, and the right number depends on the card. The offer gives the number when it's needed.
- **Keep every part's file.** The Gallery would fill with copies of the same video at growing lengths, and Reuse of the last one would only make its last Extend again.
- **Keep the parts in a temporary folder and save only the final file.** An out-of-memory kill in part 5 would lose four parts, because no `finally` runs to move part 4 into place.
- **Name the flag `--chain`.** "Parts" is the user's word. `--enhance parts=…` is a key inside the enhance spec, not a flag, so the two don't clash on the command line. The docs keep them apart.

## Steps

1. **Core:**
   - `utils/video_parts.py`;
   - the `parts` capability;
   - `parts` in `RecordedSettings`;
   - the loop in `run_video_batch`, with the second handle, the payload written again after each part, and `part_finished`.
2. **CLI:** `--parts`, the refusals and the count in the warning.
3. **Web server:**
   - the form field and the memory-fit fields;
   - the second load in `WebRunner._run_video_request`;
   - `part_finished` in `_publish_event`, which replaces the previous part's output by path.
4. **Web UI:**
   - the Duration note and the Length chip;
   - the job card's part label and stop and failure lines;
   - Reuse and **Load settings**;
   - the CLI command;
   - `make frontend-build`.
5. **Docs and changelog.**
6. **Hand check on each platform:**
   - Windows/Linux and a Mac each make a 20 s H3 video in parts, from a start image, with one reference and an audio track;
   - check the seams by eye and the sound by ear;
   - confirm the joins stream-copied the video;
   - Cancel in part 3 and check the file that stays.

## Tests

- `tests/video/test_video_parts.py`:
  - `part_lengths` for H3 on both platforms' carry and for the 20 s example;
  - a total that is a whole number of steps, and one that isn't;
  - `fewest_parts` with a stub fit: the larger of the two estimates decides, and `tight` doesn't count as fitting.
- `tests/video/test_video_runner.py`, with a mocked backend and workflow:
  - the parts run in order, on the right handles;
  - each Extend's source is the file before it, with no recorded prompt;
  - each Extend's track starts where the frames so far end;
  - the seeds are seed + k − 1;
  - the earlier file is removed;
  - after each part, part 1's payload is embedded again with the file's frames, `parts` and the total time;
  - `part`, `n_parts` and `part_finished` are on the events;
  - Pause between parts;
  - a Quit queued at the end of part 2 makes no part 3, and the file of parts 1–2 stays;
  - a Repeat queued during part 2 runs after part N;
  - Quit in part 3 keeps the file of parts 1–2;
  - a failure in part 3 reports it;
  - a batch of 2 runs makes two videos in parts.
- `tests/common/test_provenance.py`: `parts` reads back into `RecordedSettings`.
- `tests/video/test_video_cli.py`:
  - `--parts` with `--end-image`, `--upscale` or `--extend` is refused;
  - `--parts` on LTX is refused;
  - the too-large warning names the count.
- `tests/web/`:
  - `part_finished` replaces the previous part's output by path;
  - a batch of 2 in parts ends with two outputs;
  - a later `batch_cancelled` or failure leaves it in place;
  - the memory-fit `parts` field: a count, 1, and `null` for a last frame, Upscale and system memory;
  - the form's `parts` and its refusals;
  - the workspace contract's capability.
- Frontend:
  - the offer appears for a too-large H3 clip and not with a last frame or Upscale;
  - **Make it in one** removes it;
  - the chip's value;
  - the card's part label, and its outputs and result strip after `part_finished` (the replaced file is gone);
  - Reuse restores **In parts**;
  - the CLI command has `--parts`.

## Docs

- `docs/guides/video.md`: a short "Make a long video in parts" section after "Fit a clip in memory". It covers:
  - when the offer appears;
  - one action for all parts: write one that can go on, a quoted line is spoken in each part, and an enhancer rewrite of the opening picture repeats in each part;
  - what happens on a failure or Cancel;
  - that a Mac's seams are weaker.
- `docs/concepts/memory.md#video`: one sentence that the note offers parts on MiniMax H3.
- `docs/web-ui/workspace.md`: the job card's part label, each part replacing the output, and Pause between parts.
- `docs/cli/video.md`: a recipe. `docs/cli/reference.md`: `--parts`.
- `docs/reference/troubleshooting.md`: the `--parts` refusals and the "Part k of N failed" message.
- The CHANGELOG.

## Risks

- **A seam can cut to another shot**, depending on the action and the seed (seen while building references). A chain has N−1 seams, so the chance adds up. Each part shows as it finishes, so the user sees a cut early and can cancel. The guide says to write an action that can go on.
- **Each join can encode the video again.** `splice_clips` stream-copies the video only when `can_stream_copy` passes. Otherwise it encodes the whole growing video again at crf 12 for every join, so part 1 goes through it N−1 times. The hand check confirms the stream copy.
- **Each join encodes the sound again.** The joined sound is encoded once more as AAC at every Extend, so part 1's sound goes through it N−1 times. The hand check listens for it before anything is changed.
