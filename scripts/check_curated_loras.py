"""Check the curated list against the Hugging Face Hub and refresh its download counts.

Run it before each release (``make check-curated``). It asks the Hub for each entry's repository at the pinned commit, with no
token, and compares the file's size and ``lfs.sha256`` with the list. It exits 1 and names every mismatch, missing file or
unreachable repository. Every entry whose repository answered also gets the repository's current download count written
into ``curated_loras.json``, which is what Discover shows on the cards.

    uv run --frozen python scripts/check_curated_loras.py [--only NAME,...] [--delay SECONDS] [--no-write]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from zvisiongenerator.utils.curated_loras import DATA_FILE, CuratedLora, load_curated_loras

_API = "https://huggingface.co/api/models"
_RETRY_STATUS = {429, 500, 502, 503, 504}
_ATTEMPTS = 5


def compare_entry(entry: CuratedLora, siblings: Sequence[Mapping[str, object]]) -> list[str]:
    """Return what differs between *entry* and the *siblings* the Hub lists for its pinned commit; empty when it matches."""
    listed = next((item for item in siblings if item.get("rfilename") == entry.file), None)
    if listed is None:
        return [f"{entry.file} is not in {entry.repo} at {entry.revision}"]
    problems = []
    if listed.get("size") != entry.size_bytes:
        problems.append(f"size is {listed.get('size')}, the list has {entry.size_bytes}")
    lfs = listed.get("lfs")
    hub_sha = lfs.get("sha256") if isinstance(lfs, Mapping) else None
    if hub_sha != entry.sha256:
        problems.append(f"sha256 is {hub_sha}, the list has {entry.sha256}")
    return problems


def fetch_model(entry: CuratedLora) -> dict:
    """Return the Hub's description of *entry*'s repository at its pinned commit: the files with sizes and LFS hashes (``siblings``) and the ``downloads`` count.

    No token is sent. A rate limit or a server error is retried with a growing wait.

    Raises:
        RuntimeError: When the Hub does not answer with the file list.
    """
    url = f"{_API}/{entry.repo}/revision/{entry.revision}?blobs=true"
    request = urllib.request.Request(url, headers={"User-Agent": "ziv-check-curated"})
    for attempt in range(_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in _RETRY_STATUS or attempt == _ATTEMPTS - 1:
                raise RuntimeError(f"the Hub answered {error.code} for {entry.repo}@{entry.revision}") from error
        except OSError as error:
            if attempt == _ATTEMPTS - 1:
                raise RuntimeError(f"the Hub is unreachable for {entry.repo}: {error}") from error
        time.sleep(2 + attempt * 3)
    raise RuntimeError(f"no answer for {entry.repo}")  # pragma: no cover


def check_entries(
    entries: Sequence[CuratedLora],
    fetch: Callable[[CuratedLora], dict] = fetch_model,
    delay: float = 0.3,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Return the problems of each entry that has any, by entry name, and the download count of each entry whose repository answered, by entry id."""
    failures: dict[str, list[str]] = {}
    downloads: dict[str, int] = {}
    for position, entry in enumerate(entries):
        if position:
            sleep(delay)
        try:
            model = fetch(entry)
            problems = compare_entry(entry, model.get("siblings", []))
        except RuntimeError as error:
            problems = [str(error)]
        else:
            count = model.get("downloads")
            if isinstance(count, int) and count >= 0:
                downloads[entry.id] = count
        if problems:
            failures[entry.name] = problems
    return failures, downloads


def refresh_downloads(path: Path, downloads: Mapping[str, int]) -> int:
    """Write *downloads* (entry id to count) into the curated list at *path*, keeping its layout; return how many entries changed."""
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for raw in data["entries"]:
        count = downloads.get(raw["id"])
        if count is not None and raw["downloads"] != count:
            raw["downloads"] = count
            changed += 1
    if changed:
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    """Check the packaged curated list, refresh its download counts and print the result; return the exit code."""
    parser = argparse.ArgumentParser(description="Compare the curated list's pinned files with the Hugging Face Hub.")
    parser.add_argument("--only", help="Comma-separated local names to check; default: every entry.")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds to wait between requests (default: 0.3).")
    parser.add_argument("--no-write", action="store_true", help="Check only: leave the download counts in the list as they are.")
    args = parser.parse_args(argv)
    entries = load_curated_loras()
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        entries = tuple(entry for entry in entries if entry.name in wanted)
        if len(entries) != len(wanted):
            print(f"Unknown names: {sorted(wanted - {entry.name for entry in entries})}", file=sys.stderr)
            return 2
    failures, downloads = check_entries(entries, delay=args.delay)
    for name, problems in failures.items():
        for problem in problems:
            print(f"{name}: {problem}")
    print(f"Checked {len(entries)} curated LoRAs: {len(failures)} with problems.")
    if not args.no_write:
        print(f"Refreshed the download counts of {refresh_downloads(DATA_FILE, downloads)} entries.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
