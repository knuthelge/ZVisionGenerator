# Projects

A project groups a chunk of work. Pick one and the Web UI shows and saves that project's files, so new work doesn't mix with everything before it.

## What a project is

A project is a folder directly under the [output folder](../reference/data-folder.md#output-files), `~/.ziv/outputs` by default. The list of projects is the list of those folders, and the files in one are the project.

```text
outputs/
├── Night Bus/                      the project "Night Bus"
│   ├── portrait_2026-11-08_10-15-00.png
│   └── portrait_2026-11-08_10-15-00_2x.png
├── Harbour Songs/                  the project "Harbour Songs"
│   └── music_2026-11-08_11-00-02.flac
└── fox_2026-11-04_09-30-00.png     in no project
```

- A folder already in the output folder is a project straight away, for example one made with `ziv-image -o ~/.ziv/outputs/tests`.
- A file in a folder inside a project belongs to that project.
- Folders whose names start with a dot are never projects.
- A file at the top of the output folder is in **No project**. That is a place, not a choice: see [All files](#all-files).

## Pick a project

Every page has a project picker, a button after the logo that shows the project's name or **All files**. It lists **All files**, then each project from A to Z, then **New project…**.

**On a phone:** the picker is in the menu.

The pick is kept in the browser, so a phone and a computer can work in different projects. Changing it never changes a file. With a project picked:

- The Gallery and the Workspace strip, preview and latest-file choices show only its files. An empty project says *Nothing in Night Bus yet*.
- New work is saved into its folder.
- A file browser for a reference image, an audio track or a source video opens in its folder when the field is empty.
- **Copy as a command** writes the project's folder as `-o`.

A job keeps the project it was submitted with, so changing the pick while jobs wait doesn't move them. **Reuse settings** and **Load settings** never change the pick. The job card and each queued job say which project they save to.

The queue, the job log, recent prompts, uploaded references, recently used LoRAs and the last seed are the same in every project. A project groups files, not settings.

### All files { #all-files }

**All files** is the view with no project picked. It shows everything, saves new work at the top of the output folder, and each Gallery tile says which project it is in. To see only the files in no project, choose **Not in a project** in the Gallery's media filter; it is offered with **All files**.

## Make a new project

Choose **New project…** in the picker. A dialog asks for a name; **Create** makes the folder and switches to the project. The same item in a [Move](#move-a-file-to-another-project) list makes the project and moves the files into it (**Create and move**) without changing the pick.

## Where results land { #made-from-outputs }

Work made from a file stays beside it, whichever project is picked:

- An [upscale](upscaling.md#upscaling-an-existing-image) is saved next to the file it came from.
- An [Extend](../guides/video.md#extend-a-clip) is saved next to its source clip when the clip is in the output folder. A source from anywhere else is extended into the picked project.

A result that belongs to a different project than the pick, because you changed the pick while the job ran or because it was saved beside its source, doesn't appear in the Gallery or the strip. A notice says where it went: *Saved to Night Bus*.

## Move a file to another project { #move-a-file-to-another-project }

Move sorts existing files into projects, or takes them out of one. The steps are in [Move to project](../web-ui/gallery.md#move-to-project) on the Gallery page: **Move to project…** in a file's **⋯** menu, or **Move selected** for several. The target list holds **No project**, each project and **New project…**.

A move is a rename inside the output folder:

- The file keeps its modified time, so the order in the Gallery doesn't change.
- Nothing is overwritten. A file whose name is taken in the target gets the next free one, such as `portrait_2026-11-08_10-15-00_2.png`.
- A file from a folder inside a project goes to the top of the target project.
- If one of several files can't be moved, the others still are; the ones that couldn't stay selected.

Three things to know:

- Don't move a file a queued job uses, just as you wouldn't delete it: the job keeps the path it was given.
- Upscales, edits and extensions record where their source was. After you move the source, **Reuse settings** on them reports it missing, as it does when the source is deleted.
- A Workspace slot that held a moved file, such as a start image, a reference or an extend source, follows it to its new place. A deleted file empties the slot.

## Projects on the command line

There is no `ziv-image` option for projects. To put command-line work in a project, name its folder with `-o`:

```bash
ziv-image --prompt "A night bus in the rain" -o "$HOME/.ziv/outputs/Night Bus"
```

The folder is made if it doesn't exist, and the Web UI shows it as a project at the next load.

You can rename, remove or reorganise project folders in your file manager; the Web UI cannot. A project whose folder is gone returns the picker to **All files**, with a notice.
