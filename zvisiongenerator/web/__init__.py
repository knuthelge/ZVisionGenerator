"""Run the local Web UI server for Z-Vision Generator."""

from __future__ import annotations

import argparse
import importlib.util
import socket
import threading
import time
import webbrowser


_WEB_RUNTIME_MODULES = {
    "fastapi": "fastapi",
    "python-multipart": "multipart",
    "uvicorn": "uvicorn",
}


def _missing_web_runtime_dependencies() -> list[str]:
    """Return required Web UI runtime dependencies that are not installed."""
    return [package for package, module in _WEB_RUNTIME_MODULES.items() if importlib.util.find_spec(module) is None]


def _missing_web_runtime_message(*, prog: str, missing: list[str]) -> str:
    """Build a user-facing recovery hint for incomplete Web UI installs."""
    missing_list = ", ".join(missing)
    return (
        f"{prog} could not start because this installation is missing required Web UI dependencies: {missing_list}.\n"
        "Reinstall or upgrade the base package:\n"
        "  uv tool install --reinstall z-vision-generator\n"
        "or, from a repository checkout:\n"
        "  uv sync"
    )


def _ensure_web_runtime_dependencies(*, prog: str) -> None:
    """Raise when required Web UI runtime dependencies are not installed."""
    missing = _missing_web_runtime_dependencies()
    if missing:
        raise RuntimeError(_missing_web_runtime_message(prog=prog, missing=missing))


def _build_parser(*, prog: str) -> argparse.ArgumentParser:
    """Build the Web UI launcher parser."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Launch the Z-Vision Generator Web UI.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind the Web UI server to.")
    parser.add_argument("--port", type=int, default=8080, help="Preferred local port for the Web UI server.")
    parser.add_argument("--no-browser", action="store_true", help="Start the server without opening a browser tab.")
    return parser


def _find_available_port(host: str, preferred_port: int) -> int:
    """Return the preferred port or the next available local port."""
    for port in range(preferred_port, preferred_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            try:
                candidate.bind((host, port))
            except OSError:
                continue
        return port
    raise RuntimeError(f"No available port found starting at {preferred_port}")


def _wait_for_server(host: str, port: int, *, attempts: int = 100, delay: float = 0.1) -> bool:
    """Wait for the HTTP server port to accept connections."""
    for _ in range(attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(delay)
            if probe.connect_ex((host, port)) == 0:
                return True
        time.sleep(delay)
    return False


def _open_browser(url: str, host: str, port: int) -> None:
    """Open the default browser once the server is accepting connections."""
    if not _wait_for_server(host, port):
        print(f"Web UI available at {url}")
        return

    try:
        opened = webbrowser.open_new_tab(url)
    except webbrowser.Error:
        opened = False

    if not opened:
        print(f"Web UI available at {url}")


def run_server(*, host: str = "127.0.0.1", port: int = 8080, open_browser: bool = True) -> None:
    """Launch the FastAPI Web UI server."""
    _ensure_web_runtime_dependencies(prog="The Web UI launcher")
    import uvicorn

    selected_port = _find_available_port(host, port)
    url = f"http://{host}:{selected_port}"

    print(f"Starting Z-Vision Generator Web UI at {url}")

    if open_browser:
        opener = threading.Thread(target=_open_browser, args=(url, host, selected_port), daemon=True)
        opener.start()

    uvicorn.run("zvisiongenerator.web.server:app", host=host, port=selected_port, log_level="info")


def main(argv: list[str] | None = None, *, prog: str = "ziv-ui") -> None:
    """Parse launcher options and start the Web UI server."""
    parser = _build_parser(prog=prog)
    args = parser.parse_args(argv)

    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")

    try:
        _ensure_web_runtime_dependencies(prog=prog)
    except RuntimeError as exc:
        parser.exit(status=1, message=f"{exc}\n")

    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)


__all__ = ["main", "run_server"]
