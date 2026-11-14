# Install

Install the app and open it in your browser. No account or payment is needed.

## 1. Check your machine { #check-your-machine }

- **Mac:** Apple silicon (M1 or newer) and macOS 14 Sonoma or newer.
- **Windows and Linux:** an NVIDIA RTX 20-series or newer, with driver 580 or newer. There is no CPU fallback.
- **Disk:** about 10 GB free for the app, plus room for the models you download.

=== "macOS"

    Open the Apple menu and choose **About This Mac**. It names the chip and the macOS version. An Intel Mac isn't supported.

=== "Windows"

    In PowerShell, run `nvidia-smi`. The table names your card and shows the driver version at the top. To update an old driver, install the newest one from the [NVIDIA driver page](https://www.nvidia.com/Download/index.aspx) or with the NVIDIA App.

=== "Linux"

    In a terminal, run `nvidia-smi`. The table names your card and shows the driver version at the top. To update an old driver, use your distribution's NVIDIA driver package or the [NVIDIA driver page](https://www.nvidia.com/Download/index.aspx).

Models download into the Hugging Face cache; to move it to another drive, see [Data folder](../reference/data-folder.md#models-and-the-hugging-face-cache).

## Install with the launcher { #install-with-the-launcher }

The launcher installs the app and ffmpeg, starts the app and opens it in your browser, so you never open a terminal. It isn't signed yet, so your system warns you the first time you open it; the steps below get past it.

### Get the launcher { #get-the-launcher }

=== "macOS"

    1. Download [the launcher for macOS](https://github.com/knuthelge/ZVisionGenerator/releases/download/launcher-v0.1.0/Z-Vision-Generator-Launcher-macos-arm64.dmg) (`.dmg`).
    2. Open it, drag **Z-Vision Generator** into **Applications**, then open it from **Applications**. macOS says it can't verify the app.
    3. Open **System Settings → Privacy & Security**, scroll to the message about Z-Vision Generator and choose **Open Anyway**.

    ![The Privacy and Security settings with Open Anyway beside the Z-Vision Generator message](../assets/screenshots/placeholder.svg "launcher-gatekeeper")

=== "Windows"

    1. Download [the launcher for Windows](https://github.com/knuthelge/ZVisionGenerator/releases/download/launcher-v0.1.0/Z-Vision-Generator-Launcher-windows-x64-setup.exe) (`.exe`) and run it.
    2. At the **Windows protected your PC** warning, choose **More info**, then **Run anyway**.
    3. Follow the installer, then open **Z-Vision Generator** from the Start menu.

=== "Linux"

    - [AppImage](https://github.com/knuthelge/ZVisionGenerator/releases/download/launcher-v0.1.0/Z-Vision-Generator-Launcher-linux-x86_64.AppImage): in its **Properties**, tick **Allow executing file as program**, then open it.
    - [`.deb`](https://github.com/knuthelge/ZVisionGenerator/releases/download/launcher-v0.1.0/Z-Vision-Generator-Launcher-linux-amd64.deb) (Ubuntu, Debian) or [`.rpm`](https://github.com/knuthelge/ZVisionGenerator/releases/download/launcher-v0.1.0/Z-Vision-Generator-Launcher-linux-x86_64.rpm) (Fedora): open it with your software installer, then open **Z-Vision Generator** from your applications menu.

Choose **Install** in the launcher window. When it is done, your browser opens at the app, which offers an image model picked for your machine; [Your first image](first-image.md) takes it from there. Next time, open the launcher to start the app; **Quit** stops it. If it can't install, see [Troubleshooting](../reference/troubleshooting.md#launcher-install-failed).

**Update**, **Repair**, **Uninstall…** and **Get beta versions** are in the launcher window (the last three in the **⋯** menu) and keep your data and models. Use **Uninstall…** before you delete the launcher; if you already did, see [The launcher's folder](../reference/data-folder.md#the-launchers-folder). The launcher doesn't update itself: quit it and download it again from [Get the launcher](#get-the-launcher); on macOS, choose **Open Anyway** again.

### Prefer a terminal?

Steps 2 to 6 install the same app from a terminal; use them for the `ziv` commands, for [phones, tablets and other computers](../web-ui/devices.md), or to keep the [data folder](../reference/data-folder.md#use-another-folder) or models on another drive on macOS or Linux, and skip them if you used the launcher.

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

If you installed with the launcher, use **Update** in its window instead.

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

In a checkout, run `uv sync` instead. If you installed with the launcher, use **Repair** in its window instead.

## Uninstall { #uninstall }

```bash
uv tool uninstall z-vision-generator
```

If you installed with the launcher, use **Uninstall…** in its window instead; if you already deleted the launcher, see [The launcher's folder](../reference/data-folder.md#the-launchers-folder).

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
