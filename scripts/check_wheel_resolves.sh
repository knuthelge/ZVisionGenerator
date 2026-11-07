#!/usr/bin/env bash
# Build the wheel and resolve it from an empty directory, as `uv tool install z-vision-generator` does on a user's machine.
#
# Nothing of this repository's uv configuration applies there (no lock file, no [tool.uv] overrides), so a dependency that only
# resolves inside the repository fails here, before a release is published.
#
# Usage: scripts/check_wheel_resolves.sh
#   Resolves the same three targets from any host, as docs/install.md tells users to install:
#     linux    x86_64-unknown-linux-gnu, PyPI only (Linux gets its CUDA PyTorch build from PyPI)
#     windows  x86_64-pc-windows-msvc, with PyTorch's CUDA index and --index-strategy unsafe-best-match
#     macos    aarch64-apple-darwin at macOS 14 (uv assumes macOS 13 for that target, while mlx only ships macOS 14+ wheels)
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

uv build --wheel --out-dir "$work/dist" "$root" >/dev/null
wheel=$(ls "$work"/dist/*.whl)
mkdir "$work/empty"
printf '%s\n' "$wheel" >"$work/empty/requirements.in"
cd "$work/empty"

# Resolve for one platform; fail when it does not resolve or a package the wheel must not need comes back.
resolve() {
    local name=$1 platform=$2
    shift 2
    uv pip compile requirements.in --no-config --python-version 3.14 --no-header --python-platform "$platform" "$@" >"$name.txt"
    if grep -Eiq '^(gradio|qwen-tts-hf|faster-qwen3-tts)==' "$name.txt"; then
        echo "FAIL $name: the resolution contains a package that is vendored or not needed" >&2
        exit 1
    fi
    echo "ok   $name ($(grep -Ec '^[A-Za-z0-9_.-]+==' "$name.txt") packages)"
}

resolve linux x86_64-unknown-linux-gnu
resolve windows x86_64-pc-windows-msvc --index https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match
MACOSX_DEPLOYMENT_TARGET=14.0 resolve macos aarch64-apple-darwin
