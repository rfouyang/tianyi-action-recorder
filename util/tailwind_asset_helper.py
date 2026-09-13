"""Build the local Tailwind and daisyUI bundle when sources change."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TailwindAssetHelper:
    """Keep generated CSS current without a separate launch command."""

    project_root: Path
    input_path: Path
    output_path: Path
    source_directories: tuple[Path, ...]

    @classmethod
    def for_tianyi_3d(cls, *, project_root: Path) -> TailwindAssetHelper:
        ui_root = project_root / "app" / "ui_tianyi_3d"
        return cls(
            project_root=project_root,
            input_path=ui_root / "static" / "css" / "input.css",
            output_path=ui_root / "static" / "css" / "app.css",
            source_directories=(ui_root / "templates", ui_root / "static" / "js"),
        )

    def ensure_built(self) -> bool:
        if not self.needs_build():
            return False
        if shutil.which("npm") is None:
            raise RuntimeError("The Tianyi UI CSS needs rebuilding, but npm is not installed.")

        tailwind_cli = self.project_root / "node_modules" / ".bin" / "tailwindcss"
        if not tailwind_cli.is_file():
            LOGGER.info("Installing local frontend build dependencies")
            subprocess.run(["npm", "install"], cwd=self.project_root, check=True)

        LOGGER.info("Building updated Tailwind and daisyUI styles")
        subprocess.run(["npm", "run", "build:css"], cwd=self.project_root, check=True)
        if not self.output_path.is_file():
            raise RuntimeError(f"CSS build did not create {self.output_path}")
        return True

    def needs_build(self) -> bool:
        if not self.output_path.is_file():
            return True
        output_modified_at = self.output_path.stat().st_mtime_ns
        return any(
            path.stat().st_mtime_ns > output_modified_at
            for path in self._source_paths()
            if path.is_file()
        )

    def _source_paths(self) -> tuple[Path, ...]:
        package_sources = (
            self.project_root / "package.json",
            self.project_root / "package-lock.json",
        )
        directory_sources = tuple(
            path for directory in self.source_directories for path in directory.rglob("*")
        )
        return (self.input_path, *package_sources, *directory_sources)
