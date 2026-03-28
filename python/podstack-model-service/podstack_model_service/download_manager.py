"""Background download manager with disk-based progress tracking."""

import logging
import os
import time
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Disable hf_xet transfer - it stalls writing large files to NFS mounts.
# Force classic HTTP downloads instead.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import snapshot_download, HfApi

from .storage import Storage

logger = logging.getLogger(__name__)


class DownloadStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadTask:
    id: str
    model_id: str
    local_dir: str = ""
    status: DownloadStatus = DownloadStatus.PENDING
    started_at: float = 0.0
    completed_at: float = 0.0
    total_bytes: int = 0
    downloaded_bytes: int = 0
    total_files: int = 0
    completed_files: int = 0
    error: str = ""
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict[str, Any]:
        elapsed = 0.0
        if self.started_at:
            end = self.completed_at or time.time()
            elapsed = end - self.started_at

        return {
            "id": self.id,
            "model_id": self.model_id,
            "status": self.status.value,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "total_files": self.total_files,
            "completed_files": self.completed_files,
            "elapsed_seconds": round(elapsed, 1),
            "error": self.error,
        }


def _scan_dir_size(path: str) -> tuple[int, int]:
    """Return (total_bytes, file_count) for all files under path."""
    total = 0
    count = 0
    try:
        for f in Path(path).rglob("*"):
            if f.is_file():
                total += f.stat().st_size
                count += 1
    except OSError:
        pass
    return total, count


class DownloadManager:
    def __init__(self, storage: Storage):
        self.storage = storage
        self._tasks: dict[str, DownloadTask] = {}
        self._hf_api = HfApi()

    def start_download(self, model_id: str) -> DownloadTask:
        """Start a background download for a HuggingFace model."""
        for task in self._tasks.values():
            if task.model_id == model_id and task.status in (
                DownloadStatus.PENDING, DownloadStatus.DOWNLOADING
            ):
                return task

        task = DownloadTask(
            id=str(uuid.uuid4())[:8],
            model_id=model_id,
            local_dir=str(self.storage.model_path(model_id)),
        )
        self._tasks[task.id] = task

        thread = threading.Thread(
            target=self._run_download,
            args=(task,),
            daemon=True,
        )
        thread.start()
        return task

    def get_task(self, task_id: str) -> DownloadTask | None:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[DownloadTask]:
        return list(self._tasks.values())

    def cancel_download(self, task_id: str) -> bool:
        """Cancel an active download."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        if task.status not in (DownloadStatus.PENDING, DownloadStatus.DOWNLOADING):
            return False
        task._cancel_event.set()
        task.status = DownloadStatus.CANCELLED
        task.completed_at = time.time()
        return True

    def _monitor_progress(self, task: DownloadTask):
        """Poll disk usage every second to update progress."""
        while task.status == DownloadStatus.DOWNLOADING:
            size, count = _scan_dir_size(task.local_dir)
            task.downloaded_bytes = size
            task.completed_files = count
            time.sleep(1)

    def _run_download(self, task: DownloadTask):
        """Execute the download in a background thread."""
        try:
            task.status = DownloadStatus.DOWNLOADING
            task.started_at = time.time()

            # Probe the repo to get file list and sizes
            try:
                repo_info = self._hf_api.repo_info(
                    task.model_id, repo_type="model", files_metadata=True
                )
                if repo_info.siblings:
                    task.total_files = len(repo_info.siblings)
                    task.total_bytes = sum(
                        s.size for s in repo_info.siblings if s.size
                    )
            except Exception as e:
                logger.warning(f"Could not fetch repo info for {task.model_id}: {e}")

            if task._cancel_event.is_set():
                return

            # Start disk monitor thread
            monitor = threading.Thread(
                target=self._monitor_progress,
                args=(task,),
                daemon=True,
            )
            monitor.start()

            snapshot_download(
                task.model_id,
                local_dir=task.local_dir,
                repo_type="model",
            )

            if task._cancel_event.is_set():
                return

            task.status = DownloadStatus.COMPLETED
            task.completed_at = time.time()

            # Final size from disk
            size, count = _scan_dir_size(task.local_dir)
            task.downloaded_bytes = size
            task.completed_files = count
            task.total_bytes = size
            task.total_files = count

            logger.info(f"Download completed: {task.model_id}")

        except Exception as e:
            if task._cancel_event.is_set():
                task.status = DownloadStatus.CANCELLED
            else:
                task.status = DownloadStatus.FAILED
                task.error = str(e)
                logger.error(f"Download failed for {task.model_id}: {e}")
            task.completed_at = time.time()
