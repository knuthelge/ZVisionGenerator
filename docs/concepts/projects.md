# Projects

A project groups a chunk of work. Pick one and the Web UI shows and saves that project's files, so new work doesn't mix with everything before it.

## What a project is

A project is a folder directly under the [output folder](data-folder.md#output-files). The list of projects is the list of those folders, and the files in one are the project.

```text
outputs/
├── Night Bus/                      the project "Night Bus"
│   ├── portrait_2026-11-08_10-15-00.png
│   └── portrait_2026-11-08_10-15-00_2x.png
├── Harbour Songs/                  the project "Harbour Songs"
│   └── music_2026-11-08_11-00-02.flac
└── fox_2026-11-04_09-30-00.png     in no project
```

- Folders already in the output folder are projects straight away, for example one made with `ziv image -o outputs/tests`.
- A file in a folder inside a project belongs to that project.
- Folders whose names start with a dot are never projects.
- A file at the top of the output folder is in **No project**, which is a place rather than a choice: see [All files](#all-files).

## Pick a project

Every page has a project picker. It lists **All files**, then each project from A to Z, then **New project…**, which asks for a name, makes the project and switches to it. The pick is kept in the browser, so a phone and a computer can work in different projects. Changing it never changes a file.

With a project picked:

- The [Gallery](../web-ui/gallery.md) and the [Workspace](../web-ui/workspace.md) strip, preview and latest-file choices show only its files.
- New work is saved into its folder.
- A file browser for a reference image, an audio track or a source video opens in its folder when the field is empty.
- **Copy as a command** writes the project's folder as `-o`.

A job keeps the project it was submitted with, so changing the pick while jobs wait doesn't move them. **Reuse settings** and **Load settings** never change the pick.

The queue, the job log, recent prompts, uploaded references, recently used LoRAs and the last seed are the same in every project. A project groups files, not settings.

### All files

**All files** is the view with no project picked. It shows everything, new work is saved at the top of the output folder, and each Gallery tile says which project it is in.

To see only the files in no project, choose **Not in a project** in the Gallery's media filter. It is offered with **All files**.

### When a result lands somewhere else

A result that belongs to a different project than the pick, because you changed the pick while the job ran or because [Extend](#made-from-outputs) keeps it beside its source, doesn't appear in the Gallery or the strip. A notice says where it went: *Saved to Night Bus*.

## Made-from outputs

Work made from a file stays beside it:

- An [upscale](upscaling.md#upscaling-an-existing-image) is saved next to the file it came from, whichever project is picked.
- An [Extend](video.md#extending-a-video) is saved next to its source clip when the clip is in the output folder. A source from anywhere else is extended into the picked project.

## Move a file to another project

Move sorts files into projects, or takes them out of one; [how to do it](../web-ui/gallery.md#move-to-project) is on the Gallery page. The target list holds **No project**, each project and **New project…**, which makes the project without changing the pick.

A move is a rename inside the output folder:

- The file keeps its modified time, so the order in the Gallery doesn't change.
- Nothing is overwritten. A file whose name is taken in the target gets the next free one, such as `portrait_2026-11-08_10-15-00_2.png`.
- A file from a folder inside a project goes to the top of the target project.
- If one of several files can't be moved, the others still are.

A queued job keeps the path it was given, so don't move a file a waiting job uses, just as you wouldn't delete it.

Upscales and extensions record where their source was. After a move that link shows as a missing source, as it does when the source is deleted.

A Workspace slot that held a moved file, such as a start image, a reference or an extend source, follows it to its new place. A deleted file empties the slot.

## Names

A project name is the folder name, without spaces at either end. The New project dialog explains a name it refuses. The rules are the same on every system. Names that differ only in capitals count as the same, so `Night Bus` and `night bus` can't both exist.

## On disk and in the CLI

There is no `ziv` option for projects. To put CLI work in a project, name its folder with `-o`:

```bash
ziv image --prompt "A night bus in the rain" -o "$HOME/.ziv/outputs/Night Bus"
```

The Web UI shows it in the project at the next load. The folder is made if it doesn't exist.

You can rename, remove or reorganise project folders in your file manager; the Web UI cannot. A project whose folder is gone returns the picker to **All files**, with a notice.
