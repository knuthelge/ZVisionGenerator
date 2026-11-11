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
    uv tool install z-vision-generator --index https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match
    ```

    PyPI's PyTorch for Windows has no CUDA, so an install from PyPI alone ends with ["… cannot run: the installed PyTorch has no CUDA"](../reference/troubleshooting.md#pytorch-has-no-cuda) even on an NVIDIA card. The extra index fixes that; the word `unsafe` in the command is expected.

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

```bash
uv tool upgrade z-vision-generator
```

Your LoRAs, settings and outputs live in the [data folder](../reference/data-folder.md), and downloaded models in the Hugging Face cache, not in the install, so they survive an update.

## Repair the install { #repair-the-install }

If the app reports a missing or outdated package, reinstall it. Your data folder and downloaded models stay as they are.

=== "macOS"

    ```bash
    uv tool install --reinstall z-vision-generator
    ```

=== "Windows"

    ```powershell
    uv tool install --reinstall z-vision-generator --index https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match
    ```

    Keep the `--index` options. Without them, uv takes PyTorch from PyPI, which has no CUDA on Windows.

=== "Linux"

    ```bash
    uv tool install --reinstall z-vision-generator
    ```

In a checkout, run `uv sync` instead.

## Uninstall { #uninstall }

```bash
uv tool uninstall z-vision-generator
```

The data folder stays, with your LoRAs, outputs and converted models, and so do the downloaded models in the Hugging Face cache. Delete them to free the space; both locations are in [Data folder](../reference/data-folder.md).

## Other ways to install { #other-ways-to-install }

??? note "The latest code, or a checkout"

    The latest code from the repository:

    ```bash
    uv tool install git+https://github.com/knuthelge/ZVisionGenerator.git
    ```

    On Windows, add the same `--index` and `--index-strategy` options as in step 4.

    A checkout for working on the code: see [Developer](../developer/index.md).

    ```bash
    git clone https://github.com/knuthelge/ZVisionGenerator && cd ZVisionGenerator
    uv sync
    uv run ziv ui
    ```
