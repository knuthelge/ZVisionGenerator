"""Run the local Web UI server for Z-Vision Generator."""

from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import errno
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser

from zvisiongenerator.utils.app_log import log_file_path, setup_logging, uvicorn_log_config
from zvisiongenerator.utils.paths import get_ziv_data_dir
from zvisiongenerator.utils.subprocess_env import utf8_child_env
from zvisiongenerator.web.instances import UiInstance, instances_dir, list_instances, register_instance, stop_instances, unregister_instance


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


_HOST_ALIASES = {"local": "0.0.0.0"}


def _resolve_host(value: str) -> str:
    """Return the bind address for a ``--host`` value, expanding aliases such as ``local``."""
    return _HOST_ALIASES.get(value.strip().lower(), value)


def _build_parser(*, prog: str) -> argparse.ArgumentParser:
    """Build the Web UI launcher parser."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Launch the Z-Vision Generator Web UI.",
    )
    parser.add_argument(
        "--host",
        type=_resolve_host,
        default="127.0.0.1",
        help="Host interface to bind the Web UI server to ('local' is short for 0.0.0.0, reachable from your network).",
    )
    parser.add_argument("--port", type=int, default=8080, help="Preferred local port for the Web UI server.")
    parser.add_argument("--no-browser", action="store_true", help="Start the server without opening a browser tab.")
    background = parser.add_mutually_exclusive_group()
    background.add_argument(
        "--background",
        action="store_true",
        help="Run the server in the background and return to the prompt; it keeps running after the terminal closes.",
    )
    background.add_argument(
        "--background-stop",
        action="store_true",
        help="Stop every running Web UI server, in the background or in another terminal, including any job it is running.",
    )
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


def _lan_address(host: str) -> str | None:
    """Return this machine's network IP when *host* is a wildcard bind, or ``None`` when there is none to show."""
    if host not in _WILDCARD_HOSTS:
        return None
    family, target = (socket.AF_INET6, "2001:4860:4860::8888") if host == "::" else (socket.AF_INET, "8.8.8.8")
    try:
        # Connecting a UDP socket sends nothing; it only picks the interface that routes outward.
        with socket.socket(family, socket.SOCK_DGRAM) as probe:
            probe.connect((target, 80))
            address = probe.getsockname()[0]
    except OSError:
        return None
    return None if ipaddress.ip_address(address).is_loopback else address


def _server_url(host: str, port: int) -> str:
    """Return the Web UI address for a server bound to *host*:*port*."""
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def _print_addresses(host: str, port: int, log_path: Path | str | None, *, heading: str) -> None:
    """Print where the Web UI can be reached and where its error log is."""
    print(f"{heading} {_server_url(host, port)}")
    lan_address = _lan_address(host)
    if lan_address is not None:
        lan_host = f"[{lan_address}]" if ":" in lan_address else lan_address
        print(f"On your network: http://{lan_host}:{port}")
    if log_path is not None:
        print(f"Error log: {log_path}")


_LAUNCH_ID_ENV = "ZIV_UI_LAUNCH_ID"
# Open job event streams never finish on their own; past this many seconds a stopping server drops them and shuts down.
_GRACEFUL_SHUTDOWN_SECONDS = 3


def run_server(*, host: str = "127.0.0.1", port: int = 8080, open_browser: bool = True) -> None:
    """Launch the FastAPI Web UI server."""
    _ensure_web_runtime_dependencies(prog="The Web UI launcher")
    import uvicorn

    from zvisiongenerator.web.request_guard import configure_allowed_hostnames

    configure_allowed_hostnames(_bound_hostnames(host))
    selected_port = _find_available_port(host, port)
    log_path = log_file_path()
    _print_addresses(host, selected_port, log_path, heading="Starting Z-Vision Generator Web UI at")

    if open_browser:
        opener = threading.Thread(target=_open_browser, args=(_server_url(host, selected_port), host, selected_port), daemon=True)
        opener.start()

    record = register_instance(
        instances_dir(get_ziv_data_dir()),
        host=host,
        port=selected_port,
        log=log_path,
        launch_id=os.environ.get(_LAUNCH_ID_ENV),
    )
    try:
        uvicorn.run(
            "zvisiongenerator.web.server:app",
            host=host,
            port=selected_port,
            log_level="info",
            log_config=uvicorn_log_config(),
            timeout_graceful_shutdown=_GRACEFUL_SHUTDOWN_SECONDS,
        )
    finally:
        unregister_instance(record)


_BACKGROUND_SERVER_CODE = "import sys; from zvisiongenerator.web import main; main(sys.argv[1:], prog='ziv ui')"
_BACKGROUND_START_TIMEOUT = 180.0


def _detach_options() -> dict[str, object]:
    """Return ``subprocess.Popen`` options that keep a child running after the terminal that started it closes."""
    if sys.platform == "win32":
        # A hidden console of its own, not DETACHED_PROCESS: without a console, every ffmpeg or converter the server
        # starts would open a visible console window.
        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _spawn_background_server(host: str, port: int, launch_id: str) -> subprocess.Popen[bytes]:
    """Start a detached Web UI server process tagged with *launch_id*.

    Its output goes nowhere, and Python would encode it with the locale's code page (cp1252 on Windows), where a
    library's non-ASCII print, such as the SeedVR2 core's status line, raises. UTF-8 output can encode anything.
    """
    return subprocess.Popen(
        [sys.executable, "-c", _BACKGROUND_SERVER_CODE, "--host", host, "--port", str(port), "--no-browser"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=utf8_child_env(**{_LAUNCH_ID_ENV: launch_id}),
        **_detach_options(),
    )


def _wait_for_background_server(process: subprocess.Popen[bytes], directory: Path, launch_id: str, *, timeout: float = _BACKGROUND_START_TIMEOUT, delay: float = 0.2) -> UiInstance | None:
    """Wait until the server tagged *launch_id* accepts connections; ``None`` when it exits or times out first."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return None
        instance = next((item for item in list_instances(directory) if item.launch_id == launch_id), None)
        if instance is not None and _wait_for_server(instance.host, instance.port, attempts=1, delay=delay):
            return instance
        time.sleep(delay)
    return None


def run_background(*, host: str = "127.0.0.1", port: int = 8080, open_browser: bool = True) -> int:
    """Start the Web UI server in the background and return once it is reachable.

    Returns:
        The process exit status: ``0`` when the server is up, ``1`` when it failed to start.
    """
    directory = instances_dir(get_ziv_data_dir())
    launch_id = uuid.uuid4().hex
    print("Starting Z-Vision Generator Web UI in the background...")
    process = _spawn_background_server(host, port, launch_id)
    instance = _wait_for_background_server(process, directory, launch_id)
    if instance is None:
        if process.poll() is None:
            process.kill()
        print("The Web UI did not start in the background. Run ziv ui without --background to see why.", file=sys.stderr)
        return 1

    _print_addresses(instance.host, instance.port, instance.log, heading="Web UI running in the background at")
    print("Stop it with: ziv ui --background-stop")
    if open_browser:
        _open_browser(_server_url(instance.host, instance.port), instance.host, instance.port)
    return 0


def stop_all_servers() -> int:
    """Stop every running Web UI server and report each one.

    Returns:
        The process exit status: ``0`` when every server stopped (or none was running), ``1`` otherwise.
    """
    running = list_instances(instances_dir(get_ziv_data_dir()))
    if not running:
        print("No Web UI is running.")
        return 0
    stopped = stop_instances(running)
    for instance in stopped:
        print(f"Stopped Web UI at {_server_url(instance.host, instance.port)} (process {instance.pid})")
    return 0 if len(stopped) == len(running) else 1


def main(argv: list[str] | None = None, *, prog: str = "ziv-ui") -> None:
    """Parse launcher options and start the Web UI server."""
    parser = _build_parser(prog=prog)
    args = parser.parse_args(argv)

    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")

    if args.background_stop:
        # No setup_logging here: it would clear the error log of the servers being stopped.
        parser.exit(status=stop_all_servers())

    try:
        _ensure_web_runtime_dependencies(prog=prog)
    except RuntimeError as exc:
        parser.exit(status=1, message=f"{exc}\n")

    if args.background:
        # The background server opens the error log itself.
        parser.exit(status=run_background(host=args.host, port=args.port, open_browser=not args.no_browser))

    setup_logging("ui")
    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)


__all__ = ["main", "run_background", "run_server", "stop_all_servers"]
