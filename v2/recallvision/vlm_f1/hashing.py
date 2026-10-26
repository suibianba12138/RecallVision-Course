"""Stable SHA256 helpers for F1 provenance and package repeats."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Hash one file without loading the whole file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_sha256(root: str | Path) -> str:
    """Hash package-relative paths and bytes in deterministic order."""

    package_root = Path(root)
    if not package_root.is_dir():
        raise ValueError(f"package root does not exist: {package_root}")
    digest = hashlib.sha256()
    files = sorted(path for path in package_root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
