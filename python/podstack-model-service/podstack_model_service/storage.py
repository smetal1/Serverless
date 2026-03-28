"""NFS storage management for downloaded models.

Path convention matches go/pkg/nfs/model_cache.go:
  {nfs_base}/base/{org}--{model}/
"""

import os
import shutil
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ModelInfo:
    name: str  # Original HuggingFace ID (e.g. "Qwen/Qwen2.5-8B-Instruct")
    path: str  # Absolute path on disk
    size_bytes: int
    file_count: int


class Storage:
    def __init__(self, nfs_base: str = "/models"):
        self.nfs_base = Path(nfs_base)
        self.base_dir = self.nfs_base / "base"

    def ensure_directories(self):
        """Create base/, lora/, snapshots/ subdirectories (matches Go EnsureDirectories)."""
        for sub in ("base", "lora", "snapshots"):
            (self.nfs_base / sub).mkdir(parents=True, exist_ok=True)

    def model_dir_name(self, model_id: str) -> str:
        """Convert HuggingFace model ID to directory name.

        'Qwen/Qwen2.5-8B-Instruct' -> 'Qwen--Qwen2.5-8B-Instruct'
        """
        return model_id.replace("/", "--")

    def model_id_from_dir(self, dir_name: str) -> str:
        """Convert directory name back to HuggingFace model ID.

        'Qwen--Qwen2.5-8B-Instruct' -> 'Qwen/Qwen2.5-8B-Instruct'
        """
        return dir_name.replace("--", "/", 1)

    def model_path(self, model_id: str) -> Path:
        """Get the full path for a model on NFS."""
        return self.base_dir / self.model_dir_name(model_id)

    def list_models(self) -> list[ModelInfo]:
        """List all downloaded models on NFS."""
        models = []
        if not self.base_dir.exists():
            return models

        for entry in sorted(self.base_dir.iterdir()):
            if not entry.is_dir():
                continue
            if "--" not in entry.name:
                continue

            size = 0
            count = 0
            for f in entry.rglob("*"):
                if f.is_file():
                    size += f.stat().st_size
                    count += 1

            models.append(ModelInfo(
                name=self.model_id_from_dir(entry.name),
                path=str(entry),
                size_bytes=size,
                file_count=count,
            ))

        return models

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Get info for a specific model."""
        path = self.model_path(model_id)
        if not path.exists():
            return None

        size = 0
        count = 0
        for f in path.rglob("*"):
            if f.is_file():
                size += f.stat().st_size
                count += 1

        return ModelInfo(
            name=model_id,
            path=str(path),
            size_bytes=size,
            file_count=count,
        )

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from NFS. Returns True if deleted."""
        path = self.model_path(model_id)
        if not path.exists():
            return False
        shutil.rmtree(path)
        return True
