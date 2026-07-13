"""Security tests for archive extraction (L32).

extract_archive must reject path-traversal members (``../evil``) in both tar
and zip archives, while extracting well-formed archives faithfully.
"""

import tarfile
import zipfile

import pytest

from tinyfold.data.sources.dips_plus import extract_archive


def _make_tar_gz(path, members):
    """members: list of (arcname, data-bytes)."""
    import io
    with tarfile.open(path, "w:gz") as tar:
        for arcname, data in members:
            info = tarfile.TarInfo(name=arcname)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _make_zip(path, members):
    with zipfile.ZipFile(path, "w") as z:
        for arcname, data in members:
            z.writestr(arcname, data)


def test_tar_traversal_is_rejected(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    _make_tar_gz(archive, [("../escape.txt", b"pwned")])
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ValueError, match="path traversal"):
        extract_archive(archive, dest)
    # Nothing escaped the destination.
    assert not (tmp_path / "escape.txt").exists()


def test_zip_traversal_is_rejected(tmp_path):
    archive = tmp_path / "evil.zip"
    _make_zip(archive, [("../escape.txt", b"pwned")])
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ValueError, match="path traversal"):
        extract_archive(archive, dest)
    assert not (tmp_path / "escape.txt").exists()


def test_tar_benign_extraction_roundtrip(tmp_path):
    archive = tmp_path / "good.tar.gz"
    _make_tar_gz(archive, [("a.txt", b"hello"), ("sub/b.txt", b"world")])
    dest = tmp_path / "out"
    dest.mkdir()
    extract_archive(archive, dest)
    assert (dest / "a.txt").read_bytes() == b"hello"
    assert (dest / "sub" / "b.txt").read_bytes() == b"world"


def test_zip_benign_extraction_roundtrip(tmp_path):
    archive = tmp_path / "good.zip"
    _make_zip(archive, [("a.txt", b"hello"), ("sub/b.txt", b"world")])
    dest = tmp_path / "out"
    dest.mkdir()
    extract_archive(archive, dest)
    assert (dest / "a.txt").read_bytes() == b"hello"
    assert (dest / "sub" / "b.txt").read_bytes() == b"world"
