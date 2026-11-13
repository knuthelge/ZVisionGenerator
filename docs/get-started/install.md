# Install

Install the app and open it in your browser. No account or payment is needed.

## 1. Check your machine { #check-your-machine }

- **Mac:** Apple silicon (M1 or newer) and macOS 14 Sonoma or newer.
- **Windows and Linux:** an NVIDIA RTX 20-series or newer, with driver 580 or newer. There is no CPU fallback.

=== "macOS"

    Open the Apple menu and choose **About This Mac**. It names the chip and the macOS version. An Intel Mac isn't supported.

=== "Windows"

    In PowerShell, run `nvidia-smi`. The table names your card and shows the driver version at the top. To update an old driver, install the newest one from the [NVIDIA driver page](https://www.nvidia.com/Download/index.aspx) or with the NVIDIA App.

=== "Linux"

    In a terminal, run `nvidia-smi`. The table names your card and shows the driver version at the top. To update an old driver, use your distribution's NVIDIA driver package or the [NVIDIA driver page](https://www.nvidia.com/Download/index.aspx).

Models download into the Hugging Face cache; to move it to another drive, see [Data folder](../reference/data-folder.md#models-and-the-hugging-face-cache).

## Install with the launcher { #install-with-the-launcher }

The launcher is a small app that installs Z-Vision Generator and ffmpeg for you, starts and stops the app, and opens it in your browser. You never open a terminal. It needs about 10 GB of free disk space. It checks your machine first, as in step 1, and tells you what is missing before it downloads anything.

The launcher installs its own copy of the app, even if `ziv` is already installed with uv. The two copies share the same [data folder](../reference/data-folder.md) and downloaded models.

### Get the launcher { #get-the-launcher }

The launcher is not signed yet, so your system warns you the first time you open it. The warning is expected; the steps below get past it.

=== "macOS"

    1. Download [the launcher for macOS](https://github.com/knuthelge/ZVisionGenerator/releases/download/launcher-v0.1.0/Z-Vision-Generator-Launcher-macos-arm64.dmg) (`.dmg`).
    2. Open the `.dmg` and drag **Z-Vision Generator** into **Applications**. Do this before you open it: the launcher has to live in Applications to start at login and to update.
    3. Open **Z-Vision Generator** from **Applications**. macOS says it can't verify the app and doesn't open it.
    4. Open **System Settings → Privacy & Security**, scroll to the message about Z-Vision Generator and choose **Open Anyway**. Confirm, and the launcher opens. You do this only once.

    ![The Privacy and Security settings with Open Anyway beside the Z-Vision Generator message](../assets/screenshots/placeholder.svg "launcher-gatekeeper")

=== "Windows"

    1. Download [the launcher for Windows](https://github.com/knuthelge/ZVisionGenerator/releases/download/launcher-v0.1.0/Z-Vision-Generator-Launcher-windows-x64-setup.exe) (`.exe`) and run it.
    2. Windows shows **Windows protected your PC**. Choose **More info**, then **Run anyway**.
    3. Follow the installer, then open **Z-Vision Generator** from the Start menu.

    ![The Windows protected your PC dialog with More info opened and the Run anyway button](../assets/screenshots/placeholder.svg "launcher-smartscreen")

=== "Linux"

    Pick the format that suits your distribution:

    - [AppImage](https://github.com/knuthelge/ZVisionGenerator/releases/download/launcher-v0.1.0/Z-Vision-Generator-Launcher-linux-x86_64.AppImage): works on most distributions. Allow it to run as a program (in your file manager: **Properties**, then **Allow executing file as program**; or `chmod +x` on the file), then open it.
    - [`.deb`](https://github.com/knuthelge/ZVisionGenerator/releases/download/launcher-v0.1.0/Z-Vision-Generator-Launcher-linux-amd64.deb): for Ubuntu, Debian and their relatives. Open it with your software installer.
    - [`.rpm`](https://github.com/knuthelge/ZVisionGenerator/releases/download/launcher-v0.1.0/Z-Vision-Generator-Launcher-linux-x86_64.rpm): for Fedora and its relatives. Open it with your software installer.

    After a `.deb` or `.rpm`, open **Z-Vision Generator** from your applications menu.

The launcher's window checks your machine. If it is ready, the window says so and shows an **Install** button. The first install downloads several gigabytes, which can take a while on a slow connection. You can close the window: the install carries on, and your browser opens when it is ready. If the window says it can't install, see [The launcher says this computer isn't supported](../reference/troubleshooting.md#launcher-unsupported); if the install stops, see [The launcher's install failed](../reference/troubleshooting.md#launcher-install-failed).

The first time, the app offers to download an image model picked for your machine. [Your first image](first-image.md) takes it from there.

### Start and stop { #launcher-start-and-stop }

Open the launcher to start the app. Your browser opens at the app when it is ready. While the window says **Running**, **Open Z-Vision Generator** opens it again; if the app has stopped, it starts it first.

Closing the window only hides it; the app keeps running. Find the launcher again in the menu bar (macOS), the notification area (Windows) or the tray (Linux), or open it like any other app to bring its window forward. Some Linux desktops, such as stock GNOME, show no tray icon; open the launcher from the applications menu instead.

**Quit**, in the window or in the tray menu, stops the app and closes the launcher. If the app stops on its own, for example when it runs out of memory, the window says **Stopped** and shows the end of its log; **Open Z-Vision Generator** starts it again.

### Start at login { #launcher-start-at-login }

Tick **Start at login** in the launcher window. The app then starts in the background when you log in, with the tray icon showing and no browser window; open it from the launcher when you want it. On a desktop with no tray icon nothing shows, and opening the launcher brings up its window. Untick it to stop.

### Update the app { #launcher-update }

When a new version is out, the window offers **Update**. The app stops, updates and starts again. Your data and models stay. If the update doesn't finish, **Open** starts the app again and the update stays on offer. To reinstall, choose **Repair** in **More**.

To try pre-release versions, open **More** and tick **Get beta versions**; it is there before you install too. The window then offers betas as updates (**Beta 0.15.0b1 is out.**) and marks an installed beta next to its version. Unticking it downloads nothing and never goes back to an older version: a beta stays installed, **Repair** reinstalls that same beta, and **Update** offers the next stable version once one newer than your beta is out.

The launcher doesn't update itself. If a new version of the app needs a newer launcher, the window says so; see [The launcher says an update needs a newer launcher](../reference/troubleshooting.md#launcher-needs-newer).

### Repair the launcher's install { #launcher-repair }

If the app doesn't start, or reports a missing or outdated package or a missing ffmpeg, open **More** in the launcher window and choose **Repair**. The app stops, the launcher downloads its tools again and reinstalls the app. Your data folder and downloaded models stay as they are.

### Uninstall the launcher's install { #launcher-uninstall }

Open **More** and choose **Uninstall…**. It stops the app and removes the app, its tools and the start-at-login entry. Your data folder, with your LoRAs, outputs and settings, and the downloaded models stay; both locations are in [Data folder](../reference/data-folder.md).

=== "macOS"

    The window then asks you to move Z-Vision Generator to the Trash to finish.

=== "Windows"

    The launcher starts the Windows uninstaller, which asks before it removes anything. **Settings → Apps** reaches the same uninstaller.

=== "Linux"

    The window then asks you to delete the AppImage, or to remove the package with your software installer.

### Prefer a terminal?

Steps 2 to 6 install the same app from a terminal instead of the launcher. Use them if you want the `ziv` commands on your path, or if you prefer commands. Skip them if you used the launcher.

## 2. Open a terminal { #open-a-terminal }

The rest of the steps are commands.

=== "macOS"

    Open **Terminal**: press ++cmd+space++, type `Terminal` and press ++enter++.

=== "Windows"

    Open **PowerShell**: press the Start key, type `PowerShell` and press ++enter++.

=== "Linux"

    Open your terminal application.

## 3. Get uv { #get-uv }

[uv](https://docs.astral.sh/uv/) installs the app and brings its own Python, so you need nothing else first.

=== "macOS"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

Close the terminal and open a new one, so that it finds `uv`.

## 4. Install { #install }

=== "macOS"

    ```bash
    uv tool install z-vision-generator
    ```

=== "Windows"

    ```powershell
    uv tool install z-vision-generator --torch-backend cu132
    ```

    PyPI's PyTorch for Windows has no CUDA, so an install from PyPI alone ends with ["… cannot run: the installed PyTorch has no CUDA"](../reference/troubleshooting.md#pytorch-has-no-cuda) even on an NVIDIA card. `--torch-backend cu132` takes PyTorch, and only PyTorch, from PyTorch's CUDA index. It needs uv 0.11.29 or newer (`uv self update` updates an older uv), and uv prints a warning that the option is experimental, which is expected.

=== "Linux"

    ```bash
    uv tool install z-vision-generator
    ```

## 5. Get ffmpeg { #ffmpeg }

[ffmpeg](https://ffmpeg.org/) is a free tool that video and audio need to write their files.

=== "macOS"

    ```bash
    brew install ffmpeg
    ```

=== "Windows"

    ```powershell
    winget install --id Gyan.FFmpeg --exact
    ```

    Open a new PowerShell window afterwards.

=== "Linux"

    ```bash
    sudo apt install ffmpeg
    ```

## 6. Start it { #start-it }

```bash
ziv ui
```

Your browser opens at the address the terminal prints, usually `http://127.0.0.1:8080/`. Keep the terminal open while you use the app; ++ctrl+c++ in it stops the app. To use it again tomorrow, run `ziv ui` again.

The first time, the app offers to download an image model picked for your machine. [Your first image](first-image.md) takes it from there.

To use it from a phone or tablet, see [Phones, tablets and other computers](../web-ui/devices.md).

## If it doesn't start { #if-it-doesnt-start }

See [Troubleshooting](../reference/troubleshooting.md).

## Update { #update }

=== "macOS"

    ```bash
    uv tool upgrade z-vision-generator
    ```

=== "Windows"

    ```powershell
    uv tool install --upgrade z-vision-generator --torch-backend cu132
    ```

    Use this command, not `uv tool upgrade`. `uv tool upgrade` can't take `--torch-backend`, so on Windows it would bring back a PyTorch without CUDA.

=== "Linux"

    ```bash
    uv tool upgrade z-vision-generator
    ```

If you installed with the launcher, use **Update** in its window instead; see [Update the app](#launcher-update).

Your LoRAs, settings and outputs live in the [data folder](../reference/data-folder.md), and downloaded models in the Hugging Face cache, not in the install, so they survive an update.

## Repair the install { #repair-the-install }

If the app reports a missing or outdated package, reinstall it. Your data folder and downloaded models stay as they are.

=== "macOS"

    ```bash
    uv tool install --reinstall z-vision-generator
    ```

=== "Windows"

    ```powershell
    uv tool install --reinstall z-vision-generator --torch-backend cu132
    ```

    Keep `--torch-backend cu132`. Without it, uv takes PyTorch from PyPI, which has no CUDA on Windows.

=== "Linux"

    ```bash
    uv tool install --reinstall z-vision-generator
    ```

In a checkout, run `uv sync` instead. If you installed with the launcher, use **Repair** in its window instead; see [Repair the launcher's install](#launcher-repair).

## Uninstall { #uninstall }

```bash
uv tool uninstall z-vision-generator
```

If you installed with the launcher, use **Uninstall…** in its window instead; see [Uninstall the launcher's install](#launcher-uninstall).

The data folder stays, with your LoRAs, outputs and converted models, and so do the downloaded models in the Hugging Face cache. Delete them to free the space; both locations are in [Data folder](../reference/data-folder.md).

## Other ways to install { #other-ways-to-install }

??? note "The latest code, or a checkout"

    The latest code from the repository:

    ```bash
    uv tool install git+https://github.com/knuthelge/ZVisionGenerator.git
    ```

    On Windows, add `--torch-backend cu132`, as in step 4.

    A checkout for working on the code: see [Developer](../developer/index.md).

    ```bash
    git clone https://github.com/knuthelge/ZVisionGenerator && cd ZVisionGenerator
    uv sync
    uv run ziv ui
    ```
