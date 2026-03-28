"""Decorator-based model deployment DSL.

Provides a Python-native way to define GPU model deployments that compile
down to Kubernetes ``ModelDeployment`` custom resources::

    from podstack import App, GPU, Scaling

    app = App("my-stack", namespace="production")

    @app.model(
        name="llama3-70b",
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        image="ghcr.io/podstack/vllm:latest",
        runtime="vllm",
        gpu=GPU(count=4, type="a100", memory_mb=81920),
        scaling=Scaling(min_replicas=0, max_replicas=10, idle_timeout=300),
        snapshot=True,
    )
    class Llama3:
        pass
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GPU:
    """GPU resource requirements for a model deployment."""

    count: int = 1
    memory_mb: int = 24000
    type: str = "l40s"
    cores_percent: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "memoryMB": self.memory_mb,
            "type": self.type,
            "coresPercent": self.cores_percent,
        }


@dataclasses.dataclass
class Scaling:
    """Autoscaling parameters for a model deployment."""

    min_replicas: int = 0
    max_replicas: int = 5
    idle_timeout: int = 300
    standby_pool: int = 1
    target_concurrency: int = 1
    scale_up_delay: int = 0
    scale_down_delay: int = 30

    def to_dict(self) -> dict[str, Any]:
        return {
            "minReplicas": self.min_replicas,
            "maxReplicas": self.max_replicas,
            "idleTimeout": self.idle_timeout,
            "standbyPool": self.standby_pool,
            "targetConcurrency": self.target_concurrency,
            "scaleUpDelay": self.scale_up_delay,
            "scaleDownDelay": self.scale_down_delay,
        }


# Default container images per runtime
_DEFAULT_IMAGES: dict[str, str] = {
    "vllm": "ghcr.io/podstack/vllm:latest",
    "tgi": "ghcr.io/podstack/tgi:latest",
    "triton": "ghcr.io/podstack/triton:latest",
    "diffusion": "ghcr.io/podstack/diffusion:latest",
    "whisper": "ghcr.io/podstack/whisper:latest",
    "custom": "ghcr.io/podstack/base:latest",
}


class App:
    """Container for a set of model deployments.

    Acts as a registry: the ``@app.model()`` decorator registers model
    definitions that can later be compiled to Kubernetes manifests and
    applied to a cluster.
    """

    def __init__(self, name: str, namespace: str = "default") -> None:
        self.name = name
        self.namespace = namespace
        self.models: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def model(
        self,
        name: str,
        model: str = "",
        image: str = "",
        runtime: str = "vllm",
        gpu: GPU | None = None,
        scaling: Scaling | None = None,
        snapshot: bool = True,
        model_type: str = "llm",
        model_source: str = "huggingface",
        env: dict[str, str] | None = None,
        args: list[str] | None = None,
    ) -> Callable:
        """Decorator to register a model deployment.

        Parameters
        ----------
        name : str
            Unique name for this model deployment.
        model : str
            Model identifier (e.g. HuggingFace repo ID).
        image : str
            Container image.  Defaults based on *runtime*.
        runtime : str
            Inference runtime: vllm, tgi, triton, diffusion, whisper, custom.
        gpu : GPU
            GPU resource requirements.
        scaling : Scaling
            Autoscaling parameters.
        snapshot : bool
            Whether to enable CUDA/CRIU snapshotting.
        model_type : str
            Type of model: llm, diffusion, tts, asr, vision, custom.
        model_source : str
            Where the model comes from: huggingface, s3, nfs, local.
        env : dict
            Extra environment variables for the container.
        args : list
            Extra command-line arguments for the runtime.
        """
        resolved_image = image or _DEFAULT_IMAGES.get(runtime, _DEFAULT_IMAGES["custom"])
        resolved_gpu = gpu or GPU()
        resolved_scaling = scaling or Scaling()

        def decorator(cls: type) -> type:
            self.models[name] = {
                "name": name,
                "model": model,
                "image": resolved_image,
                "runtime": runtime,
                "gpu": resolved_gpu,
                "scaling": resolved_scaling,
                "snapshot": snapshot,
                "model_type": model_type,
                "model_source": model_source,
                "env": env or {},
                "args": args or [],
                "cls": cls,
            }
            return cls

        return decorator

    # ------------------------------------------------------------------
    # Manifest generation
    # ------------------------------------------------------------------

    def to_k8s_manifests(self) -> list[dict[str, Any]]:
        """Convert all registered models to Kubernetes ModelDeployment manifests."""
        manifests: list[dict[str, Any]] = []

        for model_name, spec in self.models.items():
            manifest = {
                "apiVersion": "podstack.io/v1alpha1",
                "kind": "ModelDeployment",
                "metadata": {
                    "name": model_name,
                    "namespace": self.namespace,
                    "labels": {
                        "app.kubernetes.io/managed-by": "podstack-sdk",
                        "podstack.io/app": self.name,
                        "podstack.io/runtime": spec["runtime"],
                        "podstack.io/model-type": spec["model_type"],
                    },
                },
                "spec": {
                    "model": spec["model"],
                    "modelType": spec["model_type"],
                    "modelSource": spec["model_source"],
                    "runtime": spec["runtime"],
                    "image": spec["image"],
                    "gpu": spec["gpu"].to_dict(),
                    "scaling": spec["scaling"].to_dict(),
                    "snapshot": {
                        "enabled": spec["snapshot"],
                    },
                },
            }

            # Add optional env vars
            if spec["env"]:
                manifest["spec"]["env"] = [
                    {"name": k, "value": v} for k, v in spec["env"].items()
                ]

            # Add optional args
            if spec["args"]:
                manifest["spec"]["args"] = spec["args"]

            manifests.append(manifest)

        return manifests

    def to_yaml(self) -> str:
        """Render all manifests as a multi-document YAML string."""
        manifests = self.to_k8s_manifests()
        documents = []
        for m in manifests:
            documents.append(yaml.dump(m, default_flow_style=False, sort_keys=False))
        return "---\n".join(documents)

    # ------------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------------

    def deploy(self, kubeconfig: str | None = None, dry_run: bool = False) -> None:
        """Apply ModelDeployment CRDs to the Kubernetes cluster.

        Parameters
        ----------
        kubeconfig : str, optional
            Path to kubeconfig file.  Uses default if not specified.
        dry_run : bool
            If True, print manifests instead of applying.
        """
        yaml_content = self.to_yaml()

        if dry_run:
            print(yaml_content)
            return

        logger.info("Deploying %d model(s) to namespace %s", len(self.models), self.namespace)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            tmp_path = f.name

        try:
            cmd = ["kubectl", "apply", "-f", tmp_path]
            if kubeconfig:
                cmd.extend(["--kubeconfig", kubeconfig])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error("kubectl apply failed: %s", result.stderr.strip())
                raise RuntimeError(f"Deployment failed: {result.stderr.strip()}")

            logger.info("Deployment successful:\n%s", result.stdout.strip())
            print(result.stdout.strip())

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict[str, Any]]:
        """Return a summary of all registered models."""
        return [
            {
                "name": name,
                "model": spec["model"],
                "runtime": spec["runtime"],
                "model_type": spec["model_type"],
                "gpu": spec["gpu"].to_dict(),
                "scaling": spec["scaling"].to_dict(),
                "snapshot": spec["snapshot"],
            }
            for name, spec in self.models.items()
        ]

    def __repr__(self) -> str:
        model_names = list(self.models.keys())
        return f"<App name={self.name!r} namespace={self.namespace!r} models={model_names}>"


def load_app_from_file(file_path: str) -> App:
    """Import a Python file and find the ``App`` instance in it.

    Used by the CLI to load app definitions from user scripts.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"App file not found: {file_path}")

    spec = importlib.util.spec_from_file_location("_podstack_app", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_podstack_app"] = module
    spec.loader.exec_module(module)

    # Find the App instance
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, App):
            return attr

    raise ValueError(f"No podstack.App instance found in {file_path}")
