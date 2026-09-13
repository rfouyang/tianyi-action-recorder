"""Atomic NumPy archive operations for compiled simulation trajectories."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class NumpyArchiveHelper:
    """Read and atomically write NPZ files without action business rules."""

    def write_npz(
        self,
        *,
        path: Path,
        arrays: Mapping[str, object],
        overwrite: bool = False,
    ) -> Path:
        if path.suffix != ".npz":
            raise ValueError(f"Trajectory archive path must end in .npz: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.stem}.",
                suffix=".npz",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            np.savez_compressed(temporary_path, **arrays)
            with temporary_path.open("rb") as archive_file:
                os.fsync(archive_file.fileno())
            if overwrite:
                os.replace(temporary_path, path)
            else:
                os.link(temporary_path, path)
                temporary_path.unlink()
            return path
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def read_npz(self, *, path: Path) -> dict[str, NDArray[np.generic]]:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}

    def list_npz_files(self, *, directory: Path) -> tuple[Path, ...]:
        if not directory.exists():
            return ()
        return tuple(sorted(directory.glob("*.npz"), key=lambda path: path.name))
