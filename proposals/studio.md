# Studio: generate on a timeline

**Status:** Proposed (2026-11-05)

## Idea

A **Studio** page with a viewport over a timeline of video lanes and audio lanes, where you generate *in place*:

- **Click a point** on a video lane to make an image there: a keyframe.
- **Drag a range** to make a clip that fills it. The frames visible at the start and end of the range become the clip's first and last frame. The audio under the range (music, voice, sound effects, or any mix of them) becomes the soundtrack the clip follows.
- **Drag a range on an audio lane** to make music, speech or a sound effect there.

Everything goes through the existing queue and lands where you asked for it. You can queue a chain of shots, leave the machine running, and come back to a timeline that has filled in.

## Why

In one test session, a person and an agent made five short films and music videos with the CLI. The generators did the work; what was missing was a place where time is visible.

- **Every clip's place in the film was arithmetic.** A lip-synced clip only matches the song at its own song time, so each clip's audio start was computed by hand, and its first frame was picked by hand from earlier stills.
- **Films made clip by clip came out choppy.** They worked once the whole piece was laid out on one time axis first: the song or narration, then where each shot sits in it.
- **Continuity came from frames.** Each shot's first frame came from a still made for it, often composed from earlier stills used as references. A clip ending where the next one starts needs the same frame on both sides. First and last frame (`--end-image`) and following an audio track (`--audio-file`, `--audio-start`) already exist; they just have no shared timeline to read from.
- **Several angles of the same moment are lanes.** The music video that worked had four long takes of the same stretch of song from different angles, cut together on the beat. On a timeline that is four video lanes over one song lane, and the cut is which lane shows where.
- **Retakes were the most common action.** "Same frames, same stretch of song, try again" should be one click.

## What the user sees and does

### The screen

```
┌────────────────────────────── Studio · "Night Bus" ─────────────────────────┐
│                         Viewport                        │ Generate          │
│     ┌────────────────────────────────────────────┐      │ Range 0:33.0–38.1  │
│     │                                            │      │ First frame  [▣] ✕ │
│     │   (shows what's at the playhead: the top   │      │  Video 2 · 0:33.0  │
│     │    video lane with something on it)        │      │ Last frame   [▣] ✕ │
│     └────────────────────────────────────────────┘      │  Video 1 · 0:38.1  │
│        ◀◀   ▶   ▶▶    00:35.2 / 01:04.0                  │ Sound ☑ Music      │
│                                                          │       ☑ Voice      │
│                                                          │       ☐ Rain       │
│                                                          │ References [+]     │
│                                                          │ Model  H3 ▾  ×2    │
│                                                          │ Prompt …           │
│                                                          │ [Generate]         │
├──────────────────────────────────────────────────────────┴──────────────────┤
│ Video 2 │              [▣]────────────────[▣]                               │
│ Video 1 │[▣]──clip──[▣][▣]──clip──────────[▣]  [▣]   ░queued░  [▣]          │
│ Music   │▁▂▃▅▆▇▆▅▆▇▇▆▅▆▇▇▇▆▅▆▇▇▇▆▇▇▇▆▇▇▇▆▇▇▇▆▇▇▇▆▇▇▇▆▇▇▇▆▇▇▇▆▇▇▇▆▇▇▇▆▇▇  │
│ Voice   │        ═══ line 1 ═══           ═══ line 2 ═══                    │
│ Rain    │   ~~rain~~~~~~~~~~~~~~~~~~                                        │
└──────────────────────────────────────────────────────────────────────────────┘
   [▣] an image (keyframe)     ──clip── a video clip     ░queued░ a slot waiting for its job
```

- **Viewport** plays the timeline. At any moment it shows the topmost video lane that has something there, with all unmuted audio lanes playing under it. Where no lane has anything, the viewport is black.
- **Video lanes** hold images and clips. **Audio lanes** hold music, voice and sound, and each lane has a name, a volume and a mute.
- The **Generate panel** fills in from what you selected.

**What shows when:**

- **Clips** cover their own span.
- **Images** have a hold length: 2 s when placed, and you can drag it longer or shorter.
- **Gaps** on a lane are empty, so the lane below shows through.
- **An image used as a clip's first or last frame** becomes part of that clip and no longer shows on its own. It stays reachable as a small thumbnail at that end of the clip, where you can open it, retake it or pick another take of it. If the clip is deleted, the image shows on its own again.
- **At a boundary**, where one clip ends exactly as the next begins, the frame there is the outgoing clip's last frame. When the next clip starts from it, that shared frame plays once.

**Making a timeline:**

- Studio is a page in the top navigation, with a list of the project's timelines to open.
- **New timeline** asks for a name, a size (16:9, 9:16 or 1:1) and a frame rate (24 fps), and saves it in the current [project](projects.md) folder. A project can hold several timelines. With no project picked, New timeline asks for one first.
- Everything generated from a timeline is saved in the timeline's project, whatever project is picked in the top bar.

**The frame:**

- Each size maps to one pixel size per model: the video model's size for that ratio.
- Keyframes are generated at the video size, not the image size, so a keyframe is exactly the frame its clip starts from. The video models crop a first frame to fill when its shape differs, so a keyframe made at the image size would lose its edges.
- Imported images and clips of another shape fill the frame the same way, cropped to the centre.
- Render writes at the timeline's size, scaling each item to fill it.

### Put things on the timeline

**Generate on it:**

- **Click a point** on a video lane. The Generate panel opens for an image, with the usual image settings.
- Add **references** if the model takes them (Klein: up to four), dragged from the timeline, the gallery or your files. A person, place or prop made earlier is reused this way.
- **Generate.** The image goes on the lane at that point as a keyframe.

**Bring things in:** drag an image, a clip or an audio file from the gallery or from your files onto a lane, for example a song you already have.

### Drag a range: make a clip

Drag across a video lane from one point to another. The Generate panel fills in from the timeline:

| On the timeline | Becomes |
|---|---|
| The frame visible at the start of the range | **First frame**, or Text to video if nothing is visible there |
| The frame visible at the end of the range | **Last frame** (LTX and H3 take one). A last frame needs a first frame. |
| The audio lanes under the range | **The soundtrack the clip follows**, mixed from the lanes you tick. A clear voice gives the best lip-sync; music alone gives movement in time with it. |
| The range's length | **Clip length** |

**Rules for the frames:**

- **The visible frame wins:** what the viewport shows at that moment, from the topmost lane with something there.
- **Once a clip is made, its sources are fixed.** It remembers which item, and which take of it, gave its first frame, last frame and sound. Later changes to the lanes, such as a new lane added on top, don't change what it came from.
- **The two frames are independent.** The first and last frame can come from different lanes, from an image and a clip, or from two different clips.
- **Images at the ends of the range** become the clip's frames. They don't count as something already in the range.
- **Images inside the range** are not part of the clip.

You write the action and pick the model. Each part is a switch you can turn off: start from scratch, from the first frame only, or from both frames, with or without the sound.

**Picking the frames without hunting:**

- **Snapping.** The ends of a range snap to images and to the start and end of clips, so dragging close to an image is enough to start on it.
- **The panel shows both frames** as thumbnails, with the lane and time each comes from. For either one you can step it a frame earlier or later, or clear it.
- Frames are taken by the server at an exact frame, so what the thumbnail shows is what the model gets.

**Length:**

- The models make clips in fixed steps: 8k+1 frames for LTX (9, 17, 25, …), 17k+5 for H3 (22, 39, 56, …). How the range fits depends on whether it has a last frame:
  - **With a last frame,** the clip must end exactly on it. The range snaps to the nearest length the model can make, and the panel shows by how much the end moved.
  - **Without a last frame,** the clip is made at the next length up and trimmed to the range. The extra frames stay in the clip as a handle, so the end can be dragged out later.
- Changing the model re-fits the range the same way.
- How long a clip can be depends on memory, not on a fixed cap. The panel warns when a range is longer than the model is likely to manage on this machine. A longer stretch is made as several clips, or by continuing one.

**Where the clip goes:**

- If the lane is free across the range, the clip goes there.
- If a clip, or an image that isn't one of the range's end frames, overlaps the range on that lane, the new clip goes on a new video lane at the top of the stack. Nothing is replaced or lost, and since the new lane is on top, the new clip is what you see. A short note says "Added Video 3".
- Lanes left empty can be removed with **Remove empty lanes**.

**Continue a clip.** Continuing is a different thing from making a clip between two frames. It is the existing Extend: the new part follows on from the end of the clip, its motion and sound, rather than starting from one still frame. On macOS, H3 continues from the clip's last frame only.

- If the range starts exactly at the end of an untrimmed clip, the panel offers a choice: **Continue that clip**, or **a new clip from frames**, which uses the first and last frame as above.
- Continue isn't offered for an upscaled clip, or one shorter than about a second, which Extend can't take.
- The continued part follows the audio under the range from the range's start.
- On the timeline, the continued clip replaces the original, and the original stays in the gallery.

### Several angles of the same moment

A clip made from a range follows the audio at exactly that range's time. Two clips made for the same range on two lanes follow the same stretch of sound, so they stay in step with each other and with the audio, lip-sync included.

Cutting between angles is then a matter of lanes:

1. Make a range on **Video 1**, for example the singer seen wide, following the song from 0:33 to 0:38.
2. For a second angle, click 0:33 on **Video 2** and make a close-up keyframe there. For its hold length it is the visible frame.
3. Drag 0:33 to 0:38 on Video 2. The first frame is the close-up, since it is visible. At 0:38 the visible frame is the wide shot on Video 1, so clear the last frame or put a close-up keyframe at 0:38 too. The panel shows where each frame comes from. The keyframes become part of the new clip, so it stays on Video 2.
4. Video 2 is on top, so the viewport shows the close-up. To show the wide shot for a moment, cut the Video 2 clip twice with the cut tool and **hide** the piece between. The wide shot on Video 1 shows through, at the same moment of the song. Show the piece again and the close-up is back, so trying a different cut costs nothing.

It works for a song, a conversation, a chase or a dance.

### Drag a range on an audio lane

Drag across an audio lane and choose what to make:

| Audio | How the range is used |
|---|---|
| **Sound** (Stable Audio) | Takes the range's length directly. |
| **Speech** (Higgs) | Starts at the range's start and runs as long as the reading takes. |
| **Music** (YuE2) | Starts at the range's start and runs as long as the song is. Its length follows the lyrics and section tags, not seconds. |

Speech and music that run past the range can be trimmed where you want them to end. Audio lanes follow the same rule as video lanes: new audio that would overlap something on its lane goes on a new audio lane, and nothing is replaced. Since speech and music only know their length once they are made, that check happens when they land.

### Markers from the audio

Audio on the timeline shows markers that ranges snap to, with nothing to place by hand:

- **Songs made in the app** carry YuE2's score. YuE2 writes a score before it makes the audio: tempo, bars, chords, the vocal melody, and the sections of the lyrics (intro, verse, chorus). Stretched to the take's real length, it gives:
  - section starts;
  - bars and beats;
  - lyric lines.

  In a test with two takes, the section starts from the score landed within half a second of where they were heard, most within a tenth. The audio runs 2–3% faster than the score's written tempo, which the stretch corrects.
- **Speech, and songs brought in from outside,** get word and line markers from [speech recognition](speech-recognition.md). It is reliable on speech, and rougher on singing over music.

The same markers show under the waveform when the take is opened in the gallery's viewer.

**Lay the song down first.** Clips made from a song follow that take of it. Retaking the song later marks every clip made from it as changed, so the song is best settled before shots are built on it.

### Takes and retakes

- **Selecting an item** (a clip, an image or audio) shows its settings in the Generate panel: prompt, model, frames and sound sources. **Retake** is there.
- **Retake** makes the item again over the same range with the same settings and a new seed, as a **take** of that item. The item shows a small count. Cycling through the takes in place, playing in the viewport with the sound, picks one.
- **Which take shows:** the first take to finish is picked. Later takes are added without changing the pick.
- **Retake uses the same sources,** with whichever take of them is picked now. If you picked a better keyframe take or retook the song since, the retake uses that.
- **Changed sources are marked.** A clip whose source has a different take picked since it was made shows that its sources changed.
- **Retake ×N** (the ×2 beside the model) queues several at once.
- **Editing the prompt** in the panel first retakes with the change.
- **Cut clips:** cutting a clip cuts its takes the same way. Retaking one piece of a cut clip retakes only that piece's range, with its frames taken from the picked take at the cut points.

### The queue

Generating from the timeline adds ordinary jobs to the existing queue, which runs one after another for as long as the app is running.

- **Waiting for what it starts from.** If a range's first frame, last frame or sound comes from a slot that is still queued (an earlier clip, a keyframe, a song), the range is tied to that slot. The server holds it back until that slot has finished, then adds it to the end of the queue, where it reads its frames and sound from that slot's picked take. The panel shows "from Video 1 · 0:28–0:33 (queued)" instead of a thumbnail. You can queue a whole chain (keyframes, then the clips between them) and leave.
- **If what it waits for fails, is cancelled or is deleted,** it doesn't run, and it shows why.
- **Retry** on a failed slot queues it again, followed by everything that was waiting on it, in order.
- **Slots show their state on the timeline:** queued, waiting for another slot, running, failed.
- **The server records results.** When a job finishes, the server writes the take into the timeline file, so the browser doesn't need to be open. The server owns the file: the browser sends its edits to the server instead of saving the file itself, so the two never overwrite each other.
- **Missing files.** If a take the timeline uses is deleted from the gallery, its slot shows as missing, with Retake.
- **Restarts:** the timeline file records which job fills which slot. If the app restarts, unfinished slots show as failed and can be queued again with one click.
- **Undo:** undoing the creation of a slot that is still queued cancels its job. Deleting a finished clip can be undone; its file stays in the gallery either way.

### Sound of a clip

- **A clip made from the audio lanes plays silent.** Its own sound is that same audio, so playing it would double it.
- **A clip made without sound keeps its generated sound** (rain, footsteps, a door) and plays it.
- Either way, a switch on the clip turns its own sound on or off.

### Editing

Only what generating on a timeline needs:

- **Clips:** move, trim and delete.
- **Cut tool:** click a clip anywhere to cut it in two there.
- **Hide:** a hidden clip or piece stays on its lane, greyed out, but doesn't show or play, so the lane below shows through. It is left out of Render. Show it again with one click.
- **Lanes:** add, remove, mute, and set a volume per audio lane.
- **Snapping:** to other items and to the playhead.
- **Undo and redo.**

Moving a clip that follows the audio takes it out of sync with that audio. The clip shows how far it has moved, and a click puts it back.

### Render

**Render** is a job in the queue. It writes the timeline to one MP4 in the project folder, at the timeline's size and frame rate and a high bitrate, as the viewport plays it:

- the top visible video lane at each moment, with black where there is nothing;
- all unmuted audio lanes, plus the sound of clips that have it on, mixed;
- slots that are queued or failed render as their keyframe or as black.

### Layouts

- **Desktop:** as above. The Generate panel sits on the right, and the timeline resizes vertically.
- **Tablet:** the Generate panel becomes a sheet that opens when you select something. A long press starts a range selection; handles at both ends adjust it.
- **Phone:** the viewport over a simplified timeline that shows lanes as rows. On a phone you play the timeline, cycle through takes and pick one, retake, and retry failed slots.

## How it ties into what exists

| The Studio uses | What exists today |
|---|---|
| Image at a point | Image jobs, with up to four references on Klein |
| Clip in a range | Video jobs with a first frame, a last frame, an audio track with a start in seconds, and a length (LTX and H3) |
| Continue | Extend, including continuing with an audio track. It writes the whole clip plus the new part as one file, and needs 24 fps on H3. |
| Sound, speech, music on a lane | Audio jobs: Stable Audio with a length, Higgs, YuE2 |
| Takes | Gallery assets. Every output carries its settings, including its first and last frame, references and audio track with start and end, so a retake can be built from it. |
| Queue | The existing job queue. It is held in memory, and finished jobs are dropped from its list after a while, which is why the timeline file keeps track of its own slots. |

The Studio needs two proposals that aren't built yet:

- **[Projects](projects.md):** a timeline lives in a project folder. A project can hold several timelines.
- **[Grabbing a frame](video-frame-grab.md):** taking an exact frame out of a clip, here for use as a first or last frame as well.

**Genuinely new:**

- **The timeline file.** What is on which lane and when, which take is picked, and which job fills which slot.
- **The sound under a range.** The ticked lanes for the range are mixed into one file that becomes the clip's audio track. When the range sits inside a single item on a single ticked lane, that item's file and start are used directly.
  - The mixed files and grabbed frames are saved in a hidden working folder next to the timeline, so a retake can use them again.
  - The gallery skips that folder, so they don't show up as outputs. This differs from a frame grabbed by hand in the viewer, which is an output in its own right.
- **Ranges held back until the slots they depend on finish.** The queue runs jobs in order and can't reorder or defer them, so the server listens for jobs finishing and queues a waiting range once its source is done, even with no browser open. The grabbed frames and the audio mix are then made as part of its job, and its audio start is passed explicitly.
- **Playback.** Only the visible video lane is decoded at a time, and the audio lanes are mixed in the browser.
- **Rendering the timeline.** One MP4 from the lanes.

## Not in this proposal

- Transitions, grades, titles, effects and mixing beyond a volume per lane.
- Automatic cutting to the beat.
- Export to other editors.
- Generating a script or shot list from an idea.
