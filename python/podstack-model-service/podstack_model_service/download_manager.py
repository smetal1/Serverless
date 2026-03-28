"""Background download manager with real-time progress tracking."""

import asyncio
import logging
import time
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import os

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
class FileProgress:
    filename: str
    downloaded_bytes: int = 0
    total_bytes: int = 0


@dataclass
class DownloadTask:
    id: str
    model_id: str
    status: DownloadStatus = DownloadStatus.PENDING
    started_at: float = 0.0
    completed_at: float = 0.0
    total_bytes: int = 0
    downloaded_bytes: int = 0
    total_files: int = 0
    completed_files: int = 0
    error: str = ""
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _file_progress: dict[str, FileProgress] = field(default_factory=dict, repr=False)

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


class DownloadManager:
    def __init__(self, storage: Storage):
        self.storage = storage
        self._tasks: dict[str, DownloadTask] = {}
        self._hf_api = HfApi()

    def start_download(self, model_id: str) -> DownloadTask:
        """Start a background download for a HuggingFace model."""
        # Check if already downloading this model
        for task in self._tasks.values():
            if task.model_id == model_id and task.status in (
                DownloadStatus.PENDING, DownloadStatus.DOWNLOADING
            ):
                return task

        task = DownloadTask(
            id=str(uuid.uuid4())[:8],
            model_id=model_id,
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

            local_dir = str(self.storage.model_path(task.model_id))

            # Use huggingface_hub's snapshot_download with a progress callback
            # The tqdm_class parameter lets us intercept progress
            snapshot_download(
                task.model_id,
                local_dir=local_dir,
                repo_type="model",
                tqdm_class=_make_progress_class(task),
            )

            if task._cancel_event.is_set():
                return

            task.status = DownloadStatus.COMPLETED
            task.completed_at = time.time()

            # Final size from disk
            model_info = self.storage.get_model(task.model_id)
            if model_info:
                task.downloaded_bytes = model_info.size_bytes
                task.total_bytes = model_info.size_bytes
                task.completed_files = model_info.file_count
                task.total_files = model_info.file_count

            logger.info(f"Download completed: {task.model_id}")

        except Exception as e:
            if task._cancel_event.is_set():
                task.status = DownloadStatus.CANCELLED
            else:
                task.status = DownloadStatus.FAILED
                task.error = str(e)
                logger.error(f"Download failed for {task.model_id}: {e}")
            task.completed_at = time.time()


def _make_progress_class(task: DownloadTask):
    """Create a tqdm subclass that intercepts progress updates into our DownloadTask."""

    from tqdm import tqdm as _tqdm_base

    class _ProgressTracker(_tqdm_base):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True  # suppress all terminal output
            super().__init__(*args, **kwargs)
            self._task = task

            if self.total and self.desc:
                fp = FileProgress(
                    filename=self.desc,
                    total_bytes=self.total,
                )
                self._task._file_progress[self.desc] = fp

        def update(self, n=1):
            super().update(n)
            if self.desc and self.desc in self._task._file_progress:
                fp = self._task._file_progress[self.desc]
                fp.downloaded_bytes += n

            total_downloaded = sum(
                fp.downloaded_bytes for fp in self._task._file_progress.values()
            )
            self._task.downloaded_bytes = total_downloaded

        def close(self):
            super().close()
            if self.desc and self.desc in self._task._file_progress:
                self._task.completed_files = sum(
                    1 for fp in self._task._file_progress.values()
                    if fp.downloaded_bytes >= fp.total_bytes
                )

    return _ProgressTracker
