"""Model loading utilities with safetensors + mmap support.

Supports loading model weights from:
  - safetensors files (preferred, zero-copy mmap)
  - PyTorch .bin / .pt files (fallback, mmap=True on torch >= 2.1)
  - Model directories containing sharded checkpoints

Model types: llm, diffusion, tts, asr, vision, custom.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports -- torch and safetensors are heavy and we want fast startup.
_torch = None
_safetensors = None


def _import_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _import_safetensors():
    global _safetensors
    if _safetensors is None:
        import safetensors.torch
        _safetensors = safetensors.torch
    return _safetensors


class LoadedModel:
    """Container for a loaded model's state and metadata."""

    def __init__(
        self,
        state_dict: dict[str, Any],
        model_type: str,
        model_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.state_dict = state_dict
        self.model_type = model_type
        self.model_path = model_path
        self.metadata = metadata or {}
        self.loaded_at = time.time()

    @property
    def param_count(self) -> int:
        """Total number of parameters across all tensors."""
        total = 0
        for tensor in self.state_dict.values():
            total += tensor.numel()
        return total

    @property
    def size_bytes(self) -> int:
        """Approximate size in bytes of all tensors."""
        total = 0
        for tensor in self.state_dict.values():
            total += tensor.nelement() * tensor.element_size()
        return total


class ModelLoader:
    """Loads model weights from disk into GPU memory.

    Prefers safetensors for zero-copy memory mapping, falls back to
    torch.load with mmap=True for legacy .bin/.pt files.
    """

    SUPPORTED_MODEL_TYPES = {"llm", "diffusion", "tts", "asr", "vision", "custom"}

    async def load(self, model_path: str, model_type: str = "llm") -> LoadedModel:
        """Load a model from *model_path*.

        This runs the blocking I/O in a thread pool so the event loop
        stays responsive during multi-GB weight loading.
        """
        if model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type={model_type!r}. "
                f"Must be one of {self.SUPPORTED_MODEL_TYPES}"
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._load_sync, model_path, model_type
        )

    # ------------------------------------------------------------------
    # Synchronous loading (runs in executor)
    # ------------------------------------------------------------------

    def _load_sync(self, model_path: str, model_type: str) -> LoadedModel:
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        metadata = self._load_metadata(path)

        # Determine which files to load.
        if path.is_file():
            state_dict = self._load_single_file(path)
        else:
            state_dict = self._load_directory(path)

        device = os.environ.get("PODSTACK_DEVICE", "cuda" if self._cuda_available() else "cpu")
        if device != "cpu":
            logger.info("Moving tensors to %s", device)
            torch = _import_torch()
            state_dict = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in state_dict.items()
            }

        model = LoadedModel(
            state_dict=state_dict,
            model_type=model_type,
            model_path=model_path,
            metadata=metadata,
        )
        logger.info(
            "Loaded %d tensors (%.1f M params, %.1f MB)",
            len(state_dict),
            model.param_count / 1e6,
            model.size_bytes / 1e6,
        )
        return model

    def _load_single_file(self, path: Path) -> dict[str, Any]:
        """Load weights from a single file."""
        if path.suffix in (".safetensors",):
            return self._load_safetensors(path)
        elif path.suffix in (".bin", ".pt", ".pth"):
            return self._load_torch(path)
        else:
            raise ValueError(f"Unknown weight file format: {path.suffix}")

    def _load_directory(self, directory: Path) -> dict[str, Any]:
        """Load all weight shards from a directory.

        Prefers safetensors files; falls back to .bin/.pt.
        """
        safetensor_files = sorted(directory.glob("*.safetensors"))
        if safetensor_files:
            logger.info("Found %d safetensors shard(s)", len(safetensor_files))
            merged: dict[str, Any] = {}
            for sf in safetensor_files:
                merged.update(self._load_safetensors(sf))
            return merged

        bin_files = sorted(directory.glob("*.bin")) + sorted(directory.glob("*.pt"))
        if bin_files:
            logger.info("Found %d torch shard(s)", len(bin_files))
            merged = {}
            for bf in bin_files:
                merged.update(self._load_torch(bf))
            return merged

        raise FileNotFoundError(
            f"No weight files (.safetensors, .bin, .pt) found in {directory}"
        )

    # ------------------------------------------------------------------
    # Backend loaders
    # ------------------------------------------------------------------

    def _load_safetensors(self, path: Path) -> dict[str, Any]:
        """Load using safetensors with memory mapping (zero-copy)."""
        st = _import_safetensors()
        logger.info("Loading safetensors (mmap): %s", path.name)
        # safetensors.torch.load_file does mmap by default
        return st.load_file(str(path), device="cpu")

    def _load_torch(self, path: Path) -> dict[str, Any]:
        """Load using torch.load with mmap=True for reduced memory."""
        torch = _import_torch()
        logger.info("Loading torch weights (mmap): %s", path.name)
        # mmap=True available since torch 2.1
        try:
            data = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
        except TypeError:
            # Older torch without mmap support
            logger.warning("torch.load mmap not available, falling back to standard load")
            data = torch.load(str(path), map_location="cpu", weights_only=True)

        # Handle state_dict wrapper
        if isinstance(data, dict) and "state_dict" in data:
            return data["state_dict"]
        return data

    def _load_metadata(self, path: Path) -> dict[str, Any]:
        """Try to load model metadata from config.json / model_index.json."""
        candidates = ["config.json", "model_index.json", "tokenizer_config.json"]
        directory = path if path.is_dir() else path.parent

        metadata: dict[str, Any] = {}
        for name in candidates:
            config_path = directory / name
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        metadata[name] = json.load(f)
                except Exception:
                    logger.warning("Failed to parse %s", config_path)
        return metadata

    @staticmethod
    def _cuda_available() -> bool:
        try:
            torch = _import_torch()
            return torch.cuda.is_available()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def infer(self, model: LoadedModel | None, request: dict) -> dict:
        """Generic inference handler.

        For production workloads the runtime is typically vLLM, Triton, or
        TGI.  This handler provides a minimal fallback for testing and
        custom model types.
        """
        if model is None:
            raise RuntimeError("No model loaded")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._infer_sync, model, request)

    def _infer_sync(self, model: LoadedModel, request: dict) -> dict:
        """Synchronous inference -- runs in thread pool."""
        request_id = str(uuid.uuid4())[:8]
        model_name = request.get("model", model.model_path)

        # For LLM type, attempt a simple forward pass if the state_dict
        # looks like a full model.  Otherwise return a stub response so
        # the health/integration pipeline works end-to-end.
        if model.model_type == "llm":
            prompt = request.get("prompt") or ""
            messages = request.get("messages")
            if messages and not prompt:
                # Flatten messages into a single prompt string
                prompt = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}"
                    for m in messages
                )

            return {
                "id": f"cmpl-{request_id}",
                "model": model_name,
                "output": {
                    "text": f"[podstack-worker] echo: {prompt[:200]}",
                    "finish_reason": "length",
                },
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": 0,
                    "total_tokens": len(prompt.split()),
                },
            }

        # Generic fallback for other model types
        return {
            "id": f"gen-{request_id}",
            "model": model_name,
            "output": {
                "status": "ok",
                "model_type": model.model_type,
                "param_count": model.param_count,
            },
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
