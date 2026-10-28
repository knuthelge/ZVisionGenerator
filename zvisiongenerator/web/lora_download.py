"""Install LoRAs through the download lane: choose the name, stage the file, verify it and rename it into place.

:class:`LoraInstaller` decides the name of a LoRA before any bytes move, so a clash is reported when the user presses
Use. A new download is one lane item of kind ``lora``. It stages its file in a folder of its own beside the LoRA
folder's files (``.download-<name>``, outside the Hugging Face cache), checks the size and SHA-256, renames the file to
``<name>.safetensors`` exactly once and records where it came from. A failed or stopped item leaves no file behind, and a staging folder a killed server left behind is swept by
:meth:`LoraInstaller.sweep_staging`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import glob
from pathlib import Path
import logging
import threading
from typing import Any, Literal
import warnings

from zvisiongenerator.converters.lora_import import validate_name
from zvisiongenerator.downloads.hub_access import describe_refusal
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.lora_install import install_staged
from zvisiongenerator.utils.lora_names import repo_slug, unique_name
from zvisiongenerator.utils.lora_trigger import card_triggers
from zvisiongenerator.web.discover_hub import RepoFile
from zvisiongenerator.web.download_lane import DownloadLane
from zvisiongenerator.web.hub_client import validated_file, validated_repo, validated_revision
from zvisiongenerator.web.lora_sources import LoraSourceRow, LoraSources, LoraSourcesError
from zvisiongenerator.web.lora_staging import STAGING_PREFIX, StagingClaim, sweep_abandoned

__all__ = ["KIND", "HubFacts", "LoraInstaller", "LoraSource", "NameDecision", "StartResult", "fetch_preview", "import_source", "staging_dir"]

logger = logging.getLogger(__name__)

KIND = "lora"
_SUFFIX = ".safetensors"
# The most of a preview image the download keeps; a larger one is skipped.
_PREVIEW_READ_BYTES = 8 * 1024 * 1024

type Status = Literal["free", "running", "installed"]


@dataclass(frozen=True)
class HubFacts:
    """What a Hugging Face repository showed when its file was chosen, kept with the downloaded file.

    Attributes:
        author: The user or organisation that published the repository.
        downloads: The repository's download count.
        licence: The repository's licence id, or None when it names none.
        image_url: The Hub URL of its first preview image, or None when it has none.
    """

    author: str
    downloads: int
    licence: str | None
    image_url: str | None


@dataclass(frozen=True)
class LoraSource:
    """One file to install and what is known about it.

    Attributes:
        repo: Hugging Face repository.
        revision: The 40-character commit to download.
        file: Path of the ``.safetensors`` file in the repository.
        size_bytes: Its size, or None when the Hub did not say.
        sha256: The SHA-256 it must have, or None when the source states none.
        curated_id: Id of the curated entry, or None for an open Hugging Face file.
        trigger: The trigger the source names, recorded with the file.
        trigger_source: ``"curated"`` or ``"card"`` for *trigger*.
        anonymous: Download without the user's Hugging Face token.
        facts: What the repository showed, to keep with the file (an open Hugging Face file); None for a curated entry.
    """

    repo: str
    revision: str
    file: str
    size_bytes: int | None = None
    sha256: str | None = None
    curated_id: str | None = None
    trigger: str | None = None
    trigger_source: str | None = None
    anonymous: bool = True
    facts: HubFacts | None = None


@dataclass(frozen=True)
class NameDecision:
    """The name a source would install under, and whether that needs a download.

    Attributes:
        name: The final local name.
        status: ``"free"`` (a download would start), ``"running"`` (the same source is already downloading under the
            name; *item* is its lane item) or ``"installed"`` (the same file is already installed under the name).
        renamed: Whether the name differs from the one asked for, because that one was taken.
        item: The running lane item, or None.
    """

    name: str
    status: Status
    renamed: bool = False
    item: dict[str, Any] | None = None


@dataclass(frozen=True)
class StartResult:
    """The outcome of pressing Use: the final name, and the lane item when a download is queued or running."""

    name: str
    status: Literal["queued", "running", "installed"]
    renamed: bool
    item: dict[str, Any] | None


def staging_dir(loras_dir: Path, name: str) -> Path:
    """Return the folder the download of LoRA *name* is staged in: hidden, beside the LoRA files, so the final rename stays on one drive."""
    return loras_dir / f"{STAGING_PREFIX}{name}"


def _same_source(item: dict[str, Any], source: LoraSource) -> bool:
    """Return whether the lane *item* downloads *source*."""
    if source.curated_id is not None:
        return item.get("curated_id") == source.curated_id
    return (item.get("source_repo"), item.get("source_file"), item.get("source_revision")) == (source.repo, source.file, source.revision)


class LoraInstaller:
    """Start LoRA downloads on a lane, one decision at a time.

    Args:
        lane: The download lane that runs the items.
        loras_dir: Returns the LoRA folder (resolved at each call, as the data folder can move).
        sources: Where the source of each installed file is recorded.
        preview: Returns ``(content type, bytes)`` of the small preview picture at a Hub image URL, or None; defaults to :func:`fetch_preview`.
        background: Runs a function off the lane thread, for the preview fetch that follows a finished download; defaults to a daemon thread.
    """

    def __init__(
        self,
        lane: DownloadLane,
        *,
        loras_dir: Callable[[], Path],
        sources: LoraSources,
        preview: Callable[[str], tuple[str, bytes] | None] | None = None,
        background: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self._lane = lane
        self._loras_dir = loras_dir
        self._sources = sources
        self._preview = fetch_preview if preview is None else preview
        self._background = _in_thread if background is None else background
        self._lock = threading.Lock()

    def repo_rows(self, repo: str) -> list[LoraSourceRow]:
        """Return the recorded sources of the LoRAs installed from *repo* (none when the store cannot be read), to pass to :meth:`decide` for each of its files."""
        try:
            return self._sources.for_repo(repo)
        except LoraSourcesError:
            return []

    def decide(self, source: LoraSource, *, name: str | None = None, suffix_on_clash: bool = True, repo_rows: Sequence[LoraSourceRow] | None = None) -> NameDecision:
        """Return the name *source* would install under, or raise when it cannot.

        A curated entry always uses its curated *name*. An open file with no *name* is installed already when a recorded
        row has the same repository, file and SHA-256 and its file is on disk at the recorded size, whatever the file
        is called now and whichever commit the repository is at; a replaced file (same repository and file, other
        SHA-256) is a new version and gets a free name. Otherwise an open file uses *name*, else a slug of its repository name.

        Args:
            source: The file.
            name: The name asked for: required for a curated entry, optional otherwise.
            suffix_on_clash: Give an open file ``-2``, ``-3`` and so on when its name is taken; when False a taken name is refused.
            repo_rows: The result of :meth:`repo_rows` for the source's repository, to read the store once for several files; read here when None.

        Raises:
            ValueError: If the name is not a valid file name.
            FileExistsError: If the name is taken by something else and may not be changed, or a curated entry's name is held by another file.
        """
        with self._lock:
            return self._decide(source, name, suffix_on_clash, repo_rows)

    def start(self, source: LoraSource, *, name: str | None = None, suffix_on_clash: bool = True, default_name: str | None = None) -> StartResult:
        """Queue the download of *source* unless the same file is already installed or downloading, and return the outcome.

        The name is decided and the item queued under one lock, so two presses of Use make one download, and a
        press of Use and an import of one name make one file or a refusal. Nothing is fetched when this raises.

        Args:
            source: The file.
            name: The name asked for, as in :meth:`decide`.
            suffix_on_clash: As in :meth:`decide`.
            default_name: The name an open file takes when *name* is None, in place of a slug of its repository name; it
                is validated like a typed name but may be numbered and does not stop an installed copy being found under another name.

        Raises:
            ValueError: As :meth:`decide`.
            FileExistsError: As :meth:`decide`.
            RuntimeError: If the download lane has shut down.
        """
        with self._lock:
            decision = self._decide(source, name, suffix_on_clash, None, default_name)
            if decision.status == "installed":
                return StartResult(decision.name, "installed", decision.renamed, None)
            if decision.status == "running":
                return StartResult(decision.name, "running", decision.renamed, decision.item)
            loras_dir = self._loras_dir()
            staging = staging_dir(loras_dir, decision.name)
            claim = StagingClaim(staging)
            item = self._lane.submit(
                kind=KIND,
                name=decision.name,
                label=f"LoRA {decision.name}",
                plan=lambda: (self._target(source, claim.acquire()),),
                finalize=lambda: self._finalize(source, decision.name, claim.folder, loras_dir),
                cleanup=claim.release,
                extra={
                    "curated_id": source.curated_id,
                    "source_repo": source.repo,
                    "source_file": source.file,
                    "source_revision": source.revision,
                    "size_bytes": source.size_bytes,
                    "anonymous": source.anonymous,
                },
            )
            return StartResult(decision.name, "queued", decision.renamed, item)

    def sweep_staging(self) -> None:
        """Remove the staging folders no download holds: what a server that was killed mid-download left behind (see :func:`~zvisiongenerator.web.lora_staging.sweep_abandoned`)."""
        sweep_abandoned(self._loras_dir())

    def pending_sizes(self) -> dict[str, int]:
        """Return ``{name: size in bytes}`` of the LoRAs still downloading whose size is known."""
        return {item["name"]: item["size_bytes"] for item in self._lane.active_items(KIND) if isinstance(item.get("size_bytes"), int)}

    def _decide(self, source: LoraSource, name: str | None, suffix_on_clash: bool, repo_rows: Sequence[LoraSourceRow] | None, default_name: str | None = None) -> NameDecision:
        """Return the decision for *source*; the caller holds the lock."""
        loras_dir = self._loras_dir()
        on_disk = {path.stem for path in _lora_files(loras_dir)}
        held = {item["name"]: item for item in self._lane.active_items(KIND)}
        if source.curated_id is not None:
            if name is None:
                raise ValueError("A curated LoRA is installed under its own name.")
            return self._decide_fixed(source, validate_name(name), loras_dir, on_disk, held)
        base = validate_name(name) if name else validate_name(default_name) if default_name else repo_slug(source.repo)
        running_name = _running_name(held, source, base, any_name=not name)
        if running_name is not None:
            return NameDecision(running_name, "running", item=held[running_name])
        rows = self.repo_rows(source.repo) if repo_rows is None else repo_rows
        present = _installed_name(rows, source, on_disk, loras_dir, only=base if name else None)
        if present is not None:
            return NameDecision(present, "installed")
        taken = {*on_disk, *held}
        if _match(taken, base) is None:
            return NameDecision(base, "free")
        if not suffix_on_clash:
            raise FileExistsError(f"A LoRA named '{base}' already exists or is downloading. Pick another name.")
        return NameDecision(unique_name(base, taken), "free", renamed=True)

    def _decide_fixed(self, source: LoraSource, name: str, loras_dir: Path, on_disk: set[str], held: dict[str, dict[str, Any]]) -> NameDecision:
        """Decide for a curated entry, whose name is fixed."""
        running_name = _match(held, name)
        if running_name is not None:
            if _same_source(held[running_name], source):
                return NameDecision(running_name, "running", item=held[running_name])
            raise FileExistsError(f"Another download is already saving a LoRA as '{name}'.")
        present = _match(on_disk, name)
        if present is None:
            return NameDecision(name, "free")
        if source.size_bytes is not None and (loras_dir / f"{present}{_SUFFIX}").stat().st_size == source.size_bytes:
            return NameDecision(present, "installed")
        raise FileExistsError(f"A different LoRA called '{present}' is already installed. Rename or delete it first.")

    @staticmethod
    def _target(source: LoraSource, staging: Path) -> DownloadTarget:
        """Return the Hub download of *source* into its staging folder.

        The Hub matches allow patterns as globs, so the file's path is escaped: a name such as
        ``[FLUX.2.Klein]Style.safetensors`` must match itself, not a character class.
        """
        return DownloadTarget(source.repo, revision=source.revision, allow_patterns=(glob.escape(source.file),), local_dir=str(staging), anonymous=source.anonymous)

    def _finalize(self, source: LoraSource, name: str, staging: Path, loras_dir: Path) -> None:
        """Verify the staged file, rename it into the LoRA folder and record its source (runs on the lane thread).

        The preview picture is fetched after the item is done, off the lane thread: it never holds the download back or fails it.
        """
        destination = loras_dir / f"{name}{_SUFFIX}"
        digest = install_staged(staging / source.file, destination, expected_sha256=source.sha256, expected_size=source.size_bytes)
        trigger = source.trigger
        facts = source.facts
        size = destination.stat().st_size
        row = LoraSourceRow(
            name=name,
            size_bytes=size,
            curated_id=source.curated_id,
            repo=source.repo,
            revision=source.revision,
            file=source.file,
            sha256=digest,
            auto_trigger=trigger,
            trigger_source=source.trigger_source if trigger is not None else None,
            author=facts.author if facts else None,
            downloads=facts.downloads if facts else None,
            licence=facts.licence if facts else None,
        )
        try:
            self._sources.record(row)
        except LoraSourcesError as exc:
            warnings.warn(f"Could not record where LoRA '{name}' came from ({exc}).", stacklevel=2)
            return
        if facts is not None and facts.image_url:
            image_url = facts.image_url
            self._background(lambda: self._keep_preview(name, size, image_url))

    def _keep_preview(self, name: str, size: int, image_url: str) -> None:
        """Fetch the preview picture at *image_url* and keep it with the row of *name*; any failure is logged and ignored."""
        try:
            image = self._preview(image_url)
            if image is not None:
                self._sources.set_image(name, size, image)
        except Exception:  # noqa: BLE001 - a preview is optional and must never disturb an installed download
            logger.warning("Could not keep the preview picture of LoRA '%s'.", name, exc_info=True)


def _in_thread(function: Callable[[], None]) -> None:
    """Run *function* on a daemon thread of its own."""
    threading.Thread(target=function, name="lora-preview", daemon=True).start()


def fetch_preview(url: str) -> tuple[str, bytes] | None:
    """Return the small preview picture of the Hub image at *url*, or None when it cannot be read (the download is not held back by it).

    The image is read without any credential, at most ``_PREVIEW_READ_BYTES`` of it, and shrunk by ``make_preview``.
    """
    from zvisiongenerator.utils.lora_preview import make_preview
    from zvisiongenerator.web.hub_client import HubError, http_fetch

    try:
        response = http_fetch(url, max_bytes=_PREVIEW_READ_BYTES)
    except HubError, ValueError:
        return None
    if response.status != 200 or response.truncated:
        return None
    return make_preview(response.body)


def _lora_files(loras_dir: Path) -> list[Path]:
    """Return the ``.safetensors`` files directly in *loras_dir* (none when it is missing)."""
    try:
        return [path for path in loras_dir.iterdir() if path.suffix == _SUFFIX and (path.is_file() or path.is_symlink())]
    except OSError:
        return []


def _running_name(held: dict[str, dict[str, Any]], source: LoraSource, base: str, *, any_name: bool) -> str | None:
    """Return the name under which *source* is already downloading: *base* when a name was asked for, else any name it holds (a clash may have numbered it)."""
    if not any_name:
        found = _match(held, base)
        return found if found is not None and _same_source(held[found], source) else None
    return next((name for name, item in held.items() if _same_source(item, source)), None)


def _same_file(row: LoraSourceRow, source: LoraSource) -> bool:
    """Return whether *row* recorded the file *source* offers: same repository and file, and the same SHA-256 (the same commit when the Hub gave no SHA-256)."""
    if (row.repo, row.file) != (source.repo, source.file):
        return False
    return row.sha256 == source.sha256 if source.sha256 is not None else row.revision == source.revision


def _installed_name(rows: Iterable[LoraSourceRow], source: LoraSource, on_disk: set[str], loras_dir: Path, *, only: str | None) -> str | None:
    """Return the local name of the installed copy of *source*, or None.

    A copy counts when a row records the same file, the file is on disk and its size is the recorded one. *only*
    limits the search to the row of one name (a name the user asked for).
    """
    for row in rows:
        if not _same_file(row, source) or (only is not None and row.name.casefold() != only.casefold()):
            continue
        present = _match(on_disk, row.name)
        if present is None:
            continue
        try:
            if (loras_dir / f"{present}{_SUFFIX}").stat().st_size == row.size_bytes:
                return present
        except OSError:
            continue
    return None


def _match(names: Iterable[str], wanted: str) -> str | None:
    """Return the entry of *names* that equals *wanted* ignoring case, else None."""
    folded = wanted.casefold()
    return next((name for name in names if name.casefold() == folded), None)


def import_source(repo: str, filename: str | None, *, read_repo: Callable[[str], tuple[str, tuple[RepoFile, ...], object]] | None = None) -> LoraSource:
    """Return the :class:`LoraSource` of a file the user asks to import from a Hugging Face repository.

    The Hub is read with the user's own Hugging Face token, so a private repository they can open can be imported. When
    *filename* is None the repository must hold exactly one ``.safetensors`` file; the card's ``instance_prompt`` is
    kept as the trigger.

    Args:
        repo: Repository ID.
        filename: The ``.safetensors`` file in it, or None to take the only one.
        read_repo: Returns ``(commit, files, instance_prompt)`` for a repository; defaults to the Hub library.

    Raises:
        ValueError: If a name is not valid, the repository has no such file, or no unambiguous file.
        RuntimeError: If the Hub refuses or cannot be reached; a gated repository gets one line saying what to do.
    """
    validated_repo(repo)
    if filename is not None:
        validated_file(filename)
    commit, files, prompt = (read_repo or _read_repo_with_token)(repo)
    validated_revision(commit)
    if filename is None:
        if not files:
            raise ValueError(f"No .safetensors files found in repo '{repo}'.")
        if len(files) > 1:
            raise ValueError(f"Multiple .safetensors files in repo '{repo}': {', '.join(entry.file for entry in files)}. Specify one with the filename argument.")
        chosen = files[0]
    else:
        chosen = next((entry for entry in files if entry.file == filename), None)
        if chosen is None:
            raise ValueError(f"'{filename}' is not in repo '{repo}'.")
    trigger = card_triggers(prompt)
    return LoraSource(repo, commit, chosen.file, chosen.size_bytes, chosen.sha256, trigger=trigger, trigger_source="card" if trigger else None, anonymous=False)


def _read_repo_with_token(repo: str) -> tuple[str, tuple[RepoFile, ...], object]:
    """Read a repository with ``huggingface_hub`` and the user's token: its commit, ``.safetensors`` files and ``instance_prompt``."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo, files_metadata=True)
    except Exception as exc:
        raise RuntimeError(describe_refusal(exc, repo) or f"Could not read '{repo}' on Hugging Face: {exc}") from exc
    files = tuple(
        RepoFile(sibling.rfilename, sibling.size, sibling.lfs.sha256 if sibling.lfs is not None else None, None, False)
        for sibling in info.siblings or ()
        if sibling.rfilename.endswith(_SUFFIX) and sibling.size is not None
    )
    card = info.card_data.to_dict() if info.card_data is not None else {}
    return info.sha or "", files, card.get("instance_prompt")
