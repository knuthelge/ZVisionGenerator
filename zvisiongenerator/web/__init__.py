"""Run the local Web UI server for Z-Vision Generator."""

from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import errno
import socket
import sys
import threading
import time
import webbrowser

from zvisiongenerator.utils.app_log import log_file_path, setup_logging, uvicorn_log_config


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


def _socket_family(host: str) -> socket.AddressFamily:
    return socket.AF_INET6 if ":" in host else socket.AF_INET


def _connect_host(host: str) -> str:
    """Return an address that reaches a server bound to *host* (loopback for wildcard binds)."""
    return {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "::1"}.get(host, host)


def _port_is_available(host: str, port: int) -> bool:
    """Return whether the Web UI can take *port* without colliding with another live server.

    A plain bind fails both for a live listener and for TIME_WAIT leftovers of a just-stopped server (a quick
    restart), which must not move the UI to another port and strand open pages. So an "address in use" bind is
    retried the way uvicorn binds (SO_REUSEADDR): if even that fails the port is busy. On macOS SO_REUSEADDR also
    lets 127.0.0.1:P bind while another process listens on 0.0.0.0:P (e.g. Docker), so finally check nothing answers.
    """
    family = _socket_family(host)

    def _bind(*, reuse: bool) -> None:
        with socket.socket(family, socket.SOCK_STREAM) as candidate:
            if reuse:
                candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            candidate.bind((host, port))

    try:
        _bind(reuse=False)
        return True
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE or sys.platform == "win32":
            # On Windows SO_REUSEADDR would allow binding over an active listener, and TIME_WAIT does not block binds.
            return False
    try:
        _bind(reuse=True)
    except OSError:
        return False
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((_connect_host(host), port)) != 0


def _find_available_port(host: str, preferred_port: int) -> int:
    """Return the preferred port or the next available local port."""
    for port in range(preferred_port, preferred_port + 100):
        if _port_is_available(host, port):
            return port
    raise RuntimeError(f"No available port found starting at {preferred_port}")


def _wait_for_server(host: str, port: int, *, attempts: int = 100, delay: float = 0.1) -> bool:
    """Wait for the HTTP server port to accept connections."""
    for _ in range(attempts):
        with socket.socket(_socket_family(host), socket.SOCK_STREAM) as probe:
            probe.settimeout(delay)
            if probe.connect_ex((_connect_host(host), port)) == 0:
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


_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", ""})


def _machine_hostnames() -> list[str]:
    """Return this machine's name as reported plus bare and mDNS forms, e.g. ``mymac.lan``, ``mymac``, ``mymac.local``."""
    machine = socket.gethostname().strip().lower()
    if not machine:
        return []
    # gethostname() may carry a DHCP/search domain (mymac.lan) or .local; other devices typically use the bare name.
    short = machine.split(".")[0]
    return list(dict.fromkeys([machine, short, f"{short}.local"]))


def _bound_hostnames(host: str) -> list[str]:
    """Return the hostnames the Web UI should answer to for a given ``--host``.

    A non-IP ``--host`` is allowed as-is. A wildcard or non-loopback IP bind (LAN
    access) also allows this machine's hostname, so opening the UI by name works.
    """
    if host in _WILDCARD_HOSTS:
        return _machine_hostnames()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return [host]
    return [] if address.is_loopback else _machine_hostnames()


def run_server(*, host: str = "127.0.0.1", port: int = 8080, open_browser: bool = True) -> None:
    """Launch the FastAPI Web UI server."""
    _ensure_web_runtime_dependencies(prog="The Web UI launcher")
    import uvicorn

    from zvisiongenerator.web.request_guard import configure_allowed_hostnames

    configure_allowed_hostnames(_bound_hostnames(host))
    selected_port = _find_available_port(host, port)
    url = f"http://[{host}]:{selected_port}" if ":" in host else f"http://{host}:{selected_port}"

    print(f"Starting Z-Vision Generator Web UI at {url}")
    log_path = log_file_path()
    if log_path is not None:
        print(f"Error log: {log_path}")

    if open_browser:
        opener = threading.Thread(target=_open_browser, args=(url, host, selected_port), daemon=True)
        opener.start()

    uvicorn.run("zvisiongenerator.web.server:app", host=host, port=selected_port, log_level="info", log_config=uvicorn_log_config())


def main(argv: list[str] | None = None, *, prog: str = "ziv-ui") -> None:
    """Parse launcher options and start the Web UI server."""
    parser = _build_parser(prog=prog)
    args = parser.parse_args(argv)

    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")

    setup_logging("ui")
    try:
        _ensure_web_runtime_dependencies(prog=prog)
    except RuntimeError as exc:
        parser.exit(status=1, message=f"{exc}\n")

    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)


__all__ = ["main", "run_server"]
