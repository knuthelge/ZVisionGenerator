"""Tests for the streamed zip."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from zvisiongenerator.web.gallery_zip import zip_stream


def _write(root: Path, name: str, data: bytes) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _open(chunks: list[bytes]) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(b"".join(chunks)))


class TestZipStream:
    def test_the_stream_is_a_valid_zip_of_level_0_deflate_entries_named_by_their_arcnames(self, tmp_path):
        first = _write(tmp_path, "a.png", b"\x89PNG-first" * 50)
        second = _write(tmp_path, "proj/b.mp4", bytes(range(256)) * 40)

        archive = _open(list(zip_stream([(first, "a.png"), (second, "proj/b.mp4")])))

        assert archive.testzip() is None
        assert archive.namelist() == ["a.png", "proj/b.mp4"]
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_DEFLATED}
        assert all(info.compress_size >= info.file_size for info in archive.infolist())
        assert archive.read("a.png") == first.read_bytes()
        assert archive.read("proj/b.mp4") == second.read_bytes()

    def test_a_file_larger_than_a_chunk_is_streamed_in_several_chunks_with_equal_bytes(self, tmp_path):
        data = bytes(range(251)) * 800
        source = _write(tmp_path, "big.png", data)

        chunks = list(zip_stream([(source, "big.png")], chunk_size=1000))

        assert len(chunks) > 3
        assert _open(chunks).read("big.png") == data

    def test_an_empty_file_is_an_entry_with_no_bytes(self, tmp_path):
        source = _write(tmp_path, "empty.png", b"")

        assert _open(list(zip_stream([(source, "empty.png")]))).read("empty.png") == b""

    def test_no_entries_give_a_valid_empty_zip(self):
        archive = _open(list(zip_stream([])))

        assert archive.namelist() == []

    def test_the_zip_is_built_while_it_is_consumed_not_up_front(self, tmp_path):
        source = _write(tmp_path, "a.png", b"x" * 5000)
        stream = zip_stream([(source, "a.png")], chunk_size=100)

        first = next(stream)

        assert len(first) < 5000
        stream.close()
