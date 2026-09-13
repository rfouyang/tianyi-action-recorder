"""Validate the pinned Unitree G1 model used for retargeting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class G1AssetHelper:
    """Resolve and verify the locally vendored G1 kinematic assets."""

    MODEL_ID = "unitree_g1_29dof_rev_1_0_fake_hand"
    MJCF_SHA256 = "165fa7a5275745e45fa25e1bc78010c1055d81bb81c013b3c7bc8b9347a3b9c3"
    URDF_SHA256 = "4aa267edc4eb2458f43c15efb4a29590006112fa188005e8c5a79867f46eb629"
    RETARGET_BASELINE_SHA256 = (
        "9f2041cc861d7db611e6dbcf23408b5b4302cfe793c7484b24b3830c2d2c1a69"
    )

    def __init__(self, *, asset_dir: Path) -> None:
        self.asset_dir = asset_dir.resolve()
        self.metadata_path = self.asset_dir / "model_metadata.json"
        self.mjcf_path = self.asset_dir / "g1_29dof_fake_hand.xml"
        self.urdf_path = self.asset_dir / "g1_29dof_fake_hand.urdf"
        self.retarget_baseline_path = self.asset_dir / "retarget_baseline.json"

    def validate(self) -> None:
        required = (
            self.metadata_path,
            self.mjcf_path,
            self.urdf_path,
            self.retarget_baseline_path,
            self.asset_dir / "LICENSE",
            self.asset_dir / "UPSTREAM.md",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Pinned G1 retargeting assets are missing: {missing}")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_id") != self.MODEL_ID:
            raise ValueError("Pinned G1 metadata has an unexpected model_id")
        if metadata.get("mjcf") != self.mjcf_path.name:
            raise ValueError("Pinned G1 metadata does not identify the expected MJCF")
        if metadata.get("urdf") != self.urdf_path.name:
            raise ValueError("Pinned G1 metadata does not identify the expected URDF")
        if self._sha256(self.mjcf_path) != self.MJCF_SHA256:
            raise ValueError("Pinned G1 MJCF hash does not match the reviewed asset")
        if self._sha256(self.urdf_path) != self.URDF_SHA256:
            raise ValueError("Pinned G1 URDF hash does not match the reviewed asset")
        if self._sha256(self.retarget_baseline_path) != self.RETARGET_BASELINE_SHA256:
            raise ValueError("Pinned G1 retargeting baseline hash does not match")
        expected_hashes = {
            self.mjcf_path.name: self.MJCF_SHA256,
            self.urdf_path.name: self.URDF_SHA256,
            self.retarget_baseline_path.name: self.RETARGET_BASELINE_SHA256,
        }
        if metadata.get("reviewed_sha256") != expected_hashes:
            raise ValueError("Pinned G1 metadata does not record the reviewed hashes")
        mesh_count = len(tuple((self.asset_dir / "meshes").glob("*.STL")))
        if mesh_count != int(metadata.get("mesh_count", -1)):
            raise ValueError("Pinned G1 mesh count does not match model metadata")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
