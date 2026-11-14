"""Stream files as one zip, without a temporary file or a size limit."""

from __future__ import annotations

import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

__all__ = ["zip_stream"]


class _Sink:
    """Collect what a ``ZipFile`` writes so the caller can take it chunk by chunk; it cannot seek, so the zip is streamed."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []

    def write(self, data: bytes) -> int:
        """Keep *data* until the next :meth:`drain` and return its length."""
        self._chunks.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        """Do nothing; the bytes wait for :meth:`drain`."""

    def drain(self) -> bytes:
        """Return everything written since the last drain and forget it."""
        data = b"".join(self._chunks)
        self._chunks.clear()
        return data


def zip_stream(entries: Iterable[tuple[Path, str]], *, chunk_size: int = 1 << 20) -> Iterator[bytes]:
    """Yield a zip of *entries* as it is built, holding about one chunk in memory.

    Entries are deflate streams at level 0 (no compression), not stored: a deflate stream ends with its own marker, so an
    unzipper finds where an entry ends although the sink cannot seek and the local header holds a data descriptor and
    zero sizes. Generated media is already compressed, so nothing is gained from a higher level.

    Each entry is read in chunks of *chunk_size* bytes and named by its archive name. A source that cannot be
    opened is skipped, so the zip holds only the files that were readable. Large archives use ZIP64 records.

    Args:
        entries: ``(path, arcname)`` pairs; *arcname* is the entry's name inside the zip.
        chunk_size: Bytes read from a source at a time.

    Yields:
        The zip's bytes, in order. No entries give a valid empty zip.
    """
    sink = _Sink()
    with zipfile.ZipFile(sink, "w", allowZip64=True) as archive:
        for path, arcname in entries:
            try:
                info = zipfile.ZipInfo.from_file(path, arcname, strict_timestamps=False)
                source = path.open("rb")
            except OSError:
                continue
            info.compress_type = zipfile.ZIP_DEFLATED
            info.compress_level = 0
            with source, archive.open(info, "w") as target:
                while chunk := source.read(chunk_size):
                    target.write(chunk)
                    if data := sink.drain():
                        yield data
    if data := sink.drain():
        yield data
