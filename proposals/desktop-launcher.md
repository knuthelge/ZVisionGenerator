# A desktop launcher

**Status:** In progress (2026-11-13). Round 1 (unsigned, for testing) is built and verified end to end on Linux. It awaits the macOS and Windows checks below and has no `launcher-v*` release yet. The launcher needs no change to the app's code: it uses only `ziv ui`'s public flags (the app's version now reads only `v*` tags).

## Why

Today the app can only be installed and started from a terminal: install uv, run `uv tool install` (on Windows with an extra `--torch-backend cu132` flag), install ffmpeg with Homebrew, winget or apt, then run `ziv ui` and keep the terminal open. A launcher lets someone who never opens a terminal install the app, start it, stop it and update it.

## What the user sees and does

- **Install:** downloads the launcher for their platform from the link in the install guide (a `.dmg` on macOS, an installer `.exe` on Windows, an AppImage, `.deb` or `.rpm` on Linux) and opens it. A small window checks the machine, then sets up the app with a progress bar. On macOS, if the launcher is running from the mounted `.dmg` or a quarantined download location, it first asks the user to move it to Applications, because start at login and later starts need a fixed location. It says up front that the first install downloads several GB. When it's done, the Web UI opens in the browser, and the existing first-run dialog offers an image model as today.
- **Start:** opens the launcher. The app starts and the Web UI opens in the browser. Opening the launcher again while the app is running brings its window forward.
- **Use:** works in the Web UI as today.
- **Stop:** chooses Quit in the launcher window or the tray menu. Every other way the launcher exits (Cmd+Q, Quit from the Dock, logging out or shutting down) stops the app the same way.
- **Update:** chooses Update in the launcher window when it's offered. The app stops, updates and starts again.
- **Start with the machine:** ticks Start at login in the launcher window. The app then starts in the background at login, with the tray icon showing and no browser window (on a desktop with no tray icon nothing shows; opening the launcher brings up its window). Unticking it stops that.
- **Something broke:** Repair in the launcher window stops the app, downloads uv and ffmpeg again (antivirus sometimes quarantines them) and reinstalls the app.
- **Uninstall:** one path per platform. It stops the app and removes the app, uv, Python, ffmpeg and the start-at-login entry, and says that the user's outputs, settings and models stay, naming the default places (`~/.ziv` and the Hugging Face cache) and linking to the data-folder page in the docs. On Windows, Uninstall in the launcher window starts the normal Windows uninstaller (also reachable from Settings → Apps). Its pre-uninstall step runs the launcher in a headless stop-and-clean mode, so the server is stopped and its files unlocked before the folder is deleted and the launcher removed. On macOS and Linux, Uninstall in the window does it, then quits with "Move Z-Vision Generator to the Trash to finish" (on Linux: delete the AppImage or remove the package).

An unsupported machine gets a clear message before anything is downloaded. Windows and Linux need an NVIDIA RTX 20-series or newer with driver 580 or newer: the launcher reads both the driver version and the compute capability (7.5 or higher) from `nvidia-smi`, and no `nvidia-smi` means no usable NVIDIA GPU. macOS needs Apple silicon with macOS 14 or newer. These are the app's requirements today; there is no CPU fallback. The launcher also checks for enough free disk space (about 10 GB) on the drive it installs to.

### The launcher window

One small fixed-size window holds everything. It shows the app's state (checking, installing, starting, running, stopped, update failed) with progress while installing, and the installed version. Its controls:

- **Open Z-Vision Generator**: opens the Web UI in the browser, starting the app first if it has stopped, for example after a crash.
- **Update**: only when a new version is out.
- **Start at login**: a checkbox, off by default.
- **Repair** and **Uninstall**, in the ⋯ menu with **Get beta versions**. The menu is locked while the app is starting.
- **Quit**: stops the app and closes the launcher.

Closing the window only hides it; the app keeps running. It opens on first install and when the launcher is started by hand (on macOS also from the Dock icon), not when it's started at login.

The launcher watches the server it started. If the server stops on its own, for example after running out of memory, the window shows "stopped" with the end of the server's log, and Open starts it again.

### The tray icon

A shortcut while the app is running (the menu bar on macOS), with three items: **Open Z-Vision Generator**, **Show window** (which reads "Update available" when there is one) and **Quit**. Open behaves as in the window. Linux desktops that show no tray icon (stock GNOME, for example on Fedora, without the AppIndicator extension) just have the window, reached from the dock or app grid like any other app.

## How it works

### Built with Tauri 2

The launcher is a small Tauri 2 app in a new `launcher/` folder. It can't be a Python app, because Python isn't on the machine until the launcher installs it. Tauri builds are about 3–10 MB (an AppImage, which carries its own WebKitGTK, is about 70 MB), against 100 MB or more for Electron. The tray is built into Tauri, and its official plugins cover start at login and a single running copy (opening it again brings that copy forward). On Linux the tray uses AppIndicator, so Ubuntu and KDE show it.

### Installing: the launcher installs uv, and uv installs the rest

This is the pattern ComfyUI Desktop, the InvokeAI launcher and Stability Matrix have all settled on. None of them freeze torch into one bundle: that would make every release a multi-GB download per platform, and PyInstaller builds are often flagged by Windows Defender.

1. The launcher downloads a pinned standalone uv.
2. uv installs its own Python and installs `z-vision-generator` as a tool, the same way the install guide does (plain PyPI on macOS and Linux, PyTorch from the cu132 index on Windows), plus `--python 3.14` on install, Repair and Update. Without the pin, uv would pick the newest Python it knows, and once that is 3.15 the install fails wherever torch or mlx have no 3.15 wheels yet. Moving to a newer Python is a launcher release.
3. Everything lives in a launcher-owned folder on a short path (`%LOCALAPPDATA%\ZVision` on Windows, `~/Library/Application Support/ZVision` on macOS, `~/.local/share/zvision` on Linux): uv, its Python, its cache, the app and its `ziv` program. A short path keeps the torch files under Windows' 260-character path limit. The launcher runs uv with `UV_TOOL_DIR`, `UV_TOOL_BIN_DIR`, `UV_PYTHON_INSTALL_DIR` and `UV_CACHE_DIR` pointing there, `UV_MANAGED_PYTHON=1`, `UV_PYTHON_INSTALL_BIN=0` and `UV_PYTHON_INSTALL_REGISTRY=0` so it never uses a system Python (Homebrew's, for example) or exposes its own one (on Windows, to the `py` launcher), and `UV_NO_CONFIG=1` so the user's own uv settings don't apply. Nothing is put on the user's `PATH`, and a user's own `uv tool install` of `ziv` is left alone. The launcher runs `ziv` by its full path.
4. On Windows, install, Repair and Update all use the same argument as the install guide, `--torch-backend cu132`, which takes only PyTorch from the CUDA index; without it PyTorch comes without CUDA. `uv tool upgrade` cannot take it, so Update is `uv tool install --upgrade` on every platform.
5. The launcher downloads ffmpeg and ffprobe into the same folder and puts them on the app's `PATH`, which is where the app looks for them. It has to be a GPL build with libx264, because the app encodes H.264 and AAC; LGPL builds lack libx264, and some macOS builds ship ffprobe separately. The source and SHA-256 of each download (uv and ffmpeg, per platform) are pinned in the launcher. The Web UI needs ffmpeg for video and audio jobs and doesn't install it.

The installed app is about 6.5 GB on Linux (measured, not counting uv's cache), so the launcher shows progress and always runs uv with a generous `UV_HTTP_TIMEOUT`. A failed install shows the error with a Retry button, which runs uv again. Repair is one button: it downloads uv and ffmpeg again, runs `uv cache clean`, then `uv tool install --reinstall`. After each install or update it runs `uv cache prune --ci`, which drops downloaded wheels, so old torch downloads don't pile up in the folder; the installed app keeps its own copies of the files. The user's data stays where it is today (`~/.ziv` and the Hugging Face cache), so a user who later switches between the launcher and the CLI keeps their models, outputs and settings.

### Starting and stopping: the launcher owns its server

The launcher needs to know three things about the server it started: that it is ready, which port it is on, and which processes to stop. It handles all three itself, using only public flags, so the app doesn't change:

- **Port:** the launcher picks a free port on `127.0.0.1` and runs `ziv ui --no-browser --port <port>`. The server keeps its default host, so it isn't reachable from the network.
- **Ready:** it waits until `http://127.0.0.1:<port>/` answers, or the process exits, with a long timeout (a first start can take minutes). If another program takes the port in the moment between the launcher's pick and the server's start, the app moves to the next port and the launcher times out; the window shows the error and Open tries again.
- **Which processes:** it records its child's process ID and start time in a small file in its own folder. On Windows the child is the `ziv.exe` wrapper, which starts `python.exe` as its own child, so stopping always covers the whole process tree.

The launcher runs the server with `PYTHONIOENCODING=utf-8` and sends its output to a log file in its folder, cleared at each start (a server started at login can run for days). On Windows the server gets a hidden console of its own (`CREATE_NO_WINDOW`, never `DETACHED_PROCESS`), so the ffmpeg and converter processes it starts don't pop up console windows. A failed start shows the end of the log. Once the server answers, the launcher opens the browser, except when it was started at login. It reads the installed version with `ziv --version`.

Quit stops the whole process tree, collecting the list of child processes before stopping anything: SIGTERM on macOS and Linux, with a hard stop only after about 15 seconds (the server's own shutdown takes up to 13), and a hard stop on Windows. The launcher then deletes its file. Because a stopped server exits with a signal status, the launcher tells "stopped on its own" from "stopped by me" by whether it asked, not by the exit code. The file can still be left behind, after a Windows shutdown or a launcher crash, so at start the launcher stops the recorded process only if it is alive and its start time matches; otherwise it just deletes the file. This is the same guard against reused process IDs that the app's own instance records use.

Start at login uses the Tauri autostart plugin: a LaunchAgent on macOS, the `Run` registry key on Windows and an XDG autostart entry on Linux (pointing at the AppImage itself, not its temporary mount). It starts the launcher with a flag that skips opening the browser.

### Updates

**The app.** The launcher checks the PyPI JSON API for `z-vision-generator` at start and once a day, and offers Update when a newer stable release is out. A "Get beta versions" item in the ⋯ menu, off by default, also offers pre-releases; it never downgrades. Update stops the server, runs `uv tool install --upgrade z-vision-generator --python 3.14` (on Windows with `--torch-backend cu132`) and starts the server again. If the upgrade or the restart fails, the window says so and offers Open, which starts the app again; the update stays on offer, and Repair is in the ⋯ menu.

A future app release may need something this launcher doesn't provide, such as a newer Python. Then the update succeeds without installing it. So after upgrading, the launcher checks that the installed version actually changed; if it didn't although a newer one was offered, the update counts as failed, like any other failed update: the window shows the end of uv's output, Open starts the app again and the update stays on offer.

**The launcher.** It doesn't check for newer launchers in the first round: it should rarely change, and testers are told when it does. The Tauri updater can come with signing.

### Models

The launcher downloads no models. The first-run dialog already picks an image model that fits the machine and downloads it in the browser. The built-in models need no Hugging Face account.

### Releases

The launcher has its own release workflow, started by `launcher-v*` tags, separate from the app's `v*` release. It builds the launcher for macOS (arm64), Windows and Linux (an AppImage, a `.deb` and an `.rpm`) and publishes them as a GitHub release that isn't marked as the latest, so app releases stay on top. A separate workflow means an app release doesn't rebuild the launcher, and a failed launcher build can't leave a half-finished app release behind after the PyPI publish. The existing workflow's prerelease rule (any tag containing `a`, `b` or `rc`) must not be copied: it matches every `launcher-` tag.

Because a launcher release is never "latest", the install guide links to the exact `launcher-v…` release, and that link is updated as part of each launcher release. Asset file names stay the same from release to release.

The launcher pins the uv and ffmpeg versions it downloads. Which app version it installs comes from PyPI, so a new app release doesn't need a new launcher.

The install guide gets a launcher section next to the terminal install. The troubleshooting page gets entries for the launcher's own messages (unsupported machine, failed install) and says to use Repair where the app's messages suggest `uv tool install --reinstall`.

## Unsigned first round

- **macOS:** the `.dmg` is ad-hoc signed. In Tauri this is opt-in (`bundle.macOS.signingIdentity: "-"`), and it matters: unsigned arm64 apps are reported as "damaged" with no way past except a terminal command. From macOS 15, right-click → Open no longer gets past Gatekeeper: users open the app once, then choose Open Anyway in System Settings → Privacy & Security.
- **Windows:** SmartScreen shows "Windows protected your PC"; users choose More info → Run anyway.
- **Linux:** an AppImage needs to be marked executable; a `.deb` or `.rpm` installs with a double-click.

The install page in the docs shows these steps with screenshots. Signing later costs $99 a year for Apple (with notarization) and about $10 a month for Azure Artifact Signing on Windows; whether an individual in the EU qualifies for the Azure service is unconfirmed.

## Checks

Linux was verified end to end. What remains is the macOS and Windows pass, plus a few Linux distros:

- **macOS 15 and 26 (Apple silicon):** first open from the `.dmg` shows the move-to-Applications message; Gatekeeper blocks the first open and Open Anyway works (`codesign -dv` reports an ad-hoc signature); the uv, Python and ffmpeg the launcher downloads aren't quarantined; install, start, Quit, start at login (LaunchAgent), the menu-bar icon, Update, Repair and Uninstall work.
- **Windows 11 (RTX 20-series or newer):** SmartScreen allows the setup `.exe` (per-user, no admin); no console window appears; CUDA torch survives an Update; a username with spaces or non-ASCII characters works; start at login (Run value) and the tray icon work; Quit, a shutdown with the app running and Uninstall (from the window and from Settings, Apps) leave nothing running or behind.
- **Measured size:** the install size and first-install time on both, to replace the Windows and macOS `EXPECTED_CACHE_BYTES` placeholders in `launcher/src-tauri/src/manifest.rs` (Linux is measured), and the folder size staying flat after an Update that changes torch.
- **Third-party notices:** before the first `launcher-v*` tag, a generated notice file for the Rust crates compiled into the launcher and the npm packages in the bundled window (for example with cargo-about), shipped in the bundle or the release.
- **Linux distros:** the AppImage on a fresh Ubuntu 24.04 (no `libfuse2`) and on Fedora, the tray on KDE and on Fedora GNOME without AppIndicator, and the `.deb` from Ubuntu's App Center.

## Alternatives

- **A frozen bundle (PyInstaller, Nuitka, Briefcase).** One self-contained app, but several GB per platform for every release, and frequent antivirus false positives.
- **Electron.** What ComfyUI Desktop and the InvokeAI launcher use, with a familiar web stack, but 100 MB or more per download.
- **Go with a tray library.** About as small as Tauri, but start at login, the window and updates would be written by hand.
- **Install into a fresh folder per version and switch only after it starts.** A failed update would leave the working version running, at the cost of more launcher code. Rejected in favour of upgrading in place plus Repair.
- **A `--status-file` flag in the app.** The server would write its real port and process once it's listening, which removes the port race, at the cost of an app change, a contract and a test. Not needed: the race is rare and Open retries.
- **Read the app's existing instance records.** Makes a private record format and the private `ZIV_UI_LAUNCH_ID` variable a contract, and the records are written before the server answers.
- **Leave ffmpeg to the user.** Less to maintain, but it breaks the promise that the user never opens a terminal.

## Out of scope

- Warning before Update, Repair or Quit stops a running job or model download, adopting a server left behind by a crashed launcher instead of stopping it, and noticing a `ziv ui` already running from a terminal. Left out of the test round.
- Data and model folders moved with `ZIV_DATA_DIR` or `HF_HOME` in a shell profile: a launcher started from the Finder or at login doesn't see them.
- Rolling back to the previous version after a bad update. Repair reinstalls the installed version, or the update on offer when there is one, never an older one; a rollback would re-download several GB, because the cache is pruned after each update.
- Gated models that need `hf auth login`; none of the built-in models do.
- Reaching the app from other devices (`--host local`).
- A `ziv` command on the user's `PATH`; CLI users keep installing with uv.
- Signing, notarization and the launcher updating itself.
