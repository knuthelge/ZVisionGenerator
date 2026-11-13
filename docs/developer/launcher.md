# Desktop launcher

The desktop launcher is a small Tauri 2 app, in `launcher/`, that installs Z-Vision Generator for someone who never opens a terminal, then starts, stops, updates, repairs and uninstalls it. It is a separate program from the Python package and has its own version and its own releases: it cannot be Python, because Python is not on the machine until the launcher installs it. This page holds the rules the launcher follows and how to build, test and release it. What a module does is in its `//!` comment at the top of the file.

## How it fits together

The launcher downloads a pinned standalone uv, and uv installs its own Python and `z-vision-generator` as a tool, the way the [install guide](../get-started/install.md) does. Everything lives in one folder of its own, on a short path so that torch's long file names stay under Windows' path limit. The launcher then runs `ziv ui --no-browser --port <port>` as a child process and owns that server: it knows its port, waits until it answers, and stops its whole process tree. The app itself is unchanged; the launcher uses only the public flags of `ziv ui`. The data folder and the Hugging Face cache stay where the app puts them, so the launcher and a terminal install share them.

The window is a Svelte page, built from `frontend/launcher.html` and `frontend/src/launcher/` with its own Vite config, `frontend/vite.launcher.config.ts`. It reuses the Web UI's tokens and components. The build goes into the launcher's dist folder, which is not committed, and nothing from it reaches `zvisiongenerator/web/static/app/` or the wheel. The sdist excludes `launcher/` (`pyproject.toml`).

## Build and test

You need Rust (stable, edition 2024) beside the Node and pnpm of the [frontend](frontend.md). On Linux, install `libwebkit2gtk-4.1-dev`, `libayatana-appindicator3-dev`, `librsvg2-dev` and `patchelf` first. The Tauri CLI comes from the frontend's packages, so `make frontend-install` is the only setup.

```bash
make launcher-dev        # build the window UI, then run the launcher with hot reload
make launcher-test       # cargo fmt --check, clippy -D warnings and cargo test
make launcher-build      # build the installers for this platform
make launcher-build LAUNCHER_BUNDLES=appimage,deb   # or only some of them
make launcher-clean      # remove the Rust build output and launcher/dist
```

`make launcher-test` builds the window UI first, because the Tauri build embeds it. The launcher is not part of `make check`: it needs Rust and, on Linux, WebKitGTK, and a Tauri build is slow on the macOS runner. The window's Vitest tests do run in `make frontend-test`. The Rust checks run in the launcher release workflow, on its Linux leg, before the installers are built, and nowhere else. Run `make launcher-test` before you push a change in `launcher/`.

A test never downloads from the network, runs uv or starts a server. The one exception is a test marked ignored that downloads each Linux pin and checks it against the manifest; run it with `cargo test -- --ignored` when you change a pin.

## Rules

- **One concept per module, in `launcher/src-tauri/src/`.** Pure logic (parsing, choosing, classifying, building arguments) takes plain inputs and is unit-tested in the same file. Anything that touches the machine (files, processes, the network, Tauri) is a thin edge function around it, so a test needs no mock of a system. The controller is the only module that holds working state, and it runs one action at a time; the command adapters hold none.
- **Rust sends codes and raw values, the window owns every word.** A snapshot carries phases, error codes and numbers, never sentences, so wording changes touch only the frontend.
- **Every child process is built by `spawn.rs`.** No shell, arguments and paths stay as OS strings, and on Windows no console window opens. Run from an AppImage, a child gets the launcher's own environment back: `spawn.rs` removes the variables `AppRun` and the GTK hook export (`PYTHONHOME`, `GTK_*` and the like) and drops the `$APPDIR` entries from `PATH`, `LD_LIBRARY_PATH` and the other folder lists, because a Python that inherits `PYTHONHOME` cannot start.
- **Every web page opens through `opener.rs`.** The window's links and the Web UI after a start use its one function. On Linux it runs `xdg-open` as a child built by `spawn.rs`, so under an AppImage `xdg-open` and the helpers it starts do not inherit the `$APPDIR` library paths and GIO variables; macOS and Windows use the system's own opener.
- **Start at login follows the launcher.** At launch, an enabled start-at-login entry is written again, so it names the launcher's current path: an AppImage that was moved or replaced would otherwise leave an entry that starts nothing. A launcher on a macOS disk image leaves it alone.

## Invariants

The Rust modules named here and below are in `launcher/src-tauri/src/`. Where a test enforces an invariant it is named; where none does, review is the check.

- **Every download is pinned, and checked before it is used.** `manifest.rs` holds each tool's URL and our own SHA-256. A download streams to a temporary file while it is hashed and is extracted only on a match; a mismatch deletes it. Shape tests in the manifest and the verify tests in `download.rs` cover it, and the ignored live test checks the real files.
- **The launcher stops only the server it started.** It records the child's process ID and start time, and stops that tree only when the process is alive and its start time still matches (`process_tree::same_process` is the one place that compares them, and `stop_tree` takes the expected start time of the root); otherwise it deletes the record. A reused process ID is therefore never killed. Unit tests in `process_tree.rs` and `server.rs`.
- **It never edits `PATH`, and writes nothing outside its folder except the start-at-login entry and Tauri's own webview data.** The tools are found by full path, and ffmpeg reaches only the app it starts. The layout tests in `paths.rs` pin the folder's contents; nothing tests the absence of a write elsewhere, so review for it.
- **Every uv call carries the same environment.** It points uv's tools, Python and cache into the launcher's folder, forbids a system Python, ignores the user's own uv settings and sets a long network timeout. `uv.rs` builds it in one place, and its tests assert the exact variables and arguments of each command on each platform, including `--torch-backend cu132` on Windows, which takes only PyTorch from the CUDA index and everything else from PyPI. Update is `uv tool install --upgrade z-vision-generator` on every platform, not `uv tool upgrade`: `uv tool upgrade` cannot take `--torch-backend` and would bring back a PyTorch without CUDA on Windows. The package requirement on the command line replaces the one the tool's receipt stored, and `--upgrade` implies `--refresh`, so uv looks past its cached PyPI index. With the beta preference on, install and update also pass `--prerelease allow`. Repair always reinstalls exactly the installed version (`z-vision-generator==<version>`, which uv accepts without `--prerelease` even for a pre-release), and only an unknown installed version falls back to the plain package, so neither Repair nor Update ever downgrades.
- **The server keeps a stable port.** The launcher reuses its last port while it is free, else scans from the app's default, so a bookmark keeps working. Tests in `server.rs`.
- **The window can show only what the snapshot allows.** The controls of each state are derived from the snapshot once, in `snapshot.rs`, and travel inside it; the window only reads them (`frontend/src/launcher/types.ts` holds the contract's types). `launcher/contract/snapshot_cases.json` holds one snapshot per state, controls included: Rust tests that `controls()` reproduces each one, and the window's tests render them. Change the contract on both sides and in that file together; a command whose control is not offered returns an error and does nothing.
- **Every way out stops what the launcher started, once, before the app exits.** The tray's and the window's Quit, Cmd+Q, the Dock's Quit, logout, SIGTERM, SIGINT, SIGHUP and the exit after a Windows uninstall all reach `begin_quit` in `app.rs` (the run event handler holds back an exit that did not come from it). It hides the window, calls the controller's `shutdown`, which does its work once however often it is called, and then exits the app. Unit tests in `app.rs` cover the one-time gate and `controller.rs` covers the shutdown; the signal and logout paths are checked by hand.
- **The Windows uninstaller deletes the data only on a real uninstall.** The installer hook `hooks.nsh` in `launcher/src-tauri/windows` runs the launcher with `--uninstall-cleanup` before any file goes, which stops the server and the other launcher processes. A real uninstall (Settings, Apps, or Uninstall in the app) runs `uninstall.exe` from a copy in `%TEMP%`, and only then does the hook also delete the data folder and the start-at-login registry values. The installer's "Uninstall before installing" choice runs the old uninstaller in place; there the hook adds `--keep-data`, so the install that follows finds its data. Tauri's template deletes the start-at-login value on that path, so the hook saves it and writes it back after the template's delete. The check is `$EXEDIR != $INSTDIR`, which can only be tried on Windows.
- **Every docs page the window opens exists.** `links.rs` maps each link to a docs anchor, and its test reads `docs/` for it. Renaming a heading the launcher links to fails the test, so update both.

## Pins

The pins are in `manifest.rs`: uv and the ffmpeg builds, each by URL and SHA-256, plus the package name, the Python version and the size estimates the progress ring uses. To move a pin, download the new file into a scratch folder, compare its SHA-256 with the one upstream publishes, put both into the manifest and run `make launcher-test` and the ignored live test. ffmpeg must be a GPL build with libx264, because the app encodes H.264 and AAC. Windows and Linux use a dated release of [BtbN's FFmpeg-Builds](https://github.com/FFmpeg-Builds/FFmpeg-Builds); macOS uses Martin Riedl's versioned folder, whose `.sha256` files sit beside each zip.

Downloads are made over https only, redirects included. A download step records the tools' SHA-256 pins as `toolPins` in the launcher's preferences (`Prefs` in `records.rs`) as soon as it succeeds, so a Retry after a later step failed does not download them again. Install and update download the tools only when that record differs from the current manifest or is missing, or when a tool file is gone; Repair always downloads them. Moving a pin therefore reaches existing installs with the launcher's next update; `plan.rs` decides it and the controller tests cover it.

An upstream release can disappear. The launcher then reports that the download is no longer available and points to a newer launcher, so a launcher release re-pins. Pin ffmpeg to a dated release tag, never to `latest`.

## Releasing

The launcher has its own version, in `launcher/src-tauri/Cargo.toml`, and its own workflow, `.github/workflows/launcher-release.yml`, started by `launcher-v*` tags. See [Releasing](releasing.md#launcher-releases) for the checklist.

A manual run of the workflow (Actions, **Launcher release**, **Run workflow**) is the test build: it builds the installers for all three platforms and uploads them as workflow artifacts, and publishes nothing, because the release job runs only on a tag. Use it to try a change on macOS and Windows before a tag.
