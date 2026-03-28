"""CLI for Podstack: deploy, status, logs, snapshot.

Usage::

    podstack deploy app.py
    podstack status
    podstack logs llama3-70b
    podstack snapshot create llama3-70b
    podstack snapshot list
    podstack models
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import click
import yaml

logger = logging.getLogger(__name__)


def _get_namespace() -> str:
    """Return the active Kubernetes namespace."""
    return os.environ.get("PODSTACK_NAMESPACE", "default")


def _kubectl(*args: str, namespace: str | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a kubectl command."""
    cmd = ["kubectl"]
    ns = namespace or _get_namespace()
    cmd.extend(["-n", ns])
    cmd.extend(args)

    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        timeout=30,
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def cli(verbose: bool) -> None:
    """Podstack Inference OS CLI.

    Deploy, manage, and monitor GPU inference models on Kubernetes.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
    )


# ======================================================================
# deploy
# ======================================================================


@cli.command()
@click.argument("app_file", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Print manifests without applying.")
@click.option("--kubeconfig", type=click.Path(), default=None, help="Path to kubeconfig.")
def deploy(app_file: str, dry_run: bool, kubeconfig: str | None) -> None:
    """Deploy models from a Python app file.

    The app file should contain a ``podstack.App`` instance with model
    registrations via the ``@app.model()`` decorator.

    Example::

        podstack deploy my_models.py
        podstack deploy my_models.py --dry-run
    """
    from .app import load_app_from_file

    try:
        app = load_app_from_file(app_file)
    except Exception as exc:
        click.echo(f"Error loading app file: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Found app '{app.name}' with {len(app.models)} model(s)")

    if dry_run:
        click.echo("\n--- Generated Manifests ---\n")
        click.echo(app.to_yaml())
        return

    try:
        app.deploy(kubeconfig=kubeconfig)
        click.echo("Deployment successful!")
    except Exception as exc:
        click.echo(f"Deployment failed: {exc}", err=True)
        sys.exit(1)


# ======================================================================
# status
# ======================================================================


@cli.command()
@click.option("--namespace", "-n", default=None, help="Kubernetes namespace.")
@click.option("--output", "-o", type=click.Choice(["table", "json", "yaml"]), default="table")
def status(namespace: str | None, output: str) -> None:
    """Show all model deployments and their current status.

    Queries the Kubernetes API for ModelDeployment custom resources and
    displays their phase, replica counts, and GPU usage.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        _rich_available = True
    except ImportError:
        _rich_available = False

    ns = namespace or _get_namespace()

    result = _kubectl(
        "get", "modeldeployments.podstack.io",
        "-o", "json",
        namespace=ns,
    )

    if result.returncode != 0:
        click.echo(f"Failed to query ModelDeployments: {result.stderr.strip()}", err=True)
        # Try to show pods as fallback
        click.echo("\nFalling back to pod listing:")
        pod_result = _kubectl("get", "pods", "-l", "podstack.io/managed=true", namespace=ns, capture=False)
        sys.exit(1 if pod_result.returncode != 0 else 0)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        click.echo("Failed to parse API response", err=True)
        sys.exit(1)

    items = data.get("items", [])

    if output == "json":
        click.echo(json.dumps(items, indent=2))
        return
    elif output == "yaml":
        click.echo(yaml.dump(items, default_flow_style=False))
        return

    # Table output
    if not items:
        click.echo("No model deployments found.")
        return

    if _rich_available:
        console = Console()
        table = Table(title=f"Model Deployments ({ns})")
        table.add_column("Name", style="cyan")
        table.add_column("Model", style="white")
        table.add_column("Runtime", style="blue")
        table.add_column("Phase", style="green")
        table.add_column("Replicas", style="yellow")
        table.add_column("GPU", style="magenta")
        table.add_column("Snapshot", style="white")

        for item in items:
            meta = item.get("metadata", {})
            spec = item.get("spec", {})
            st = item.get("status", {})
            gpu = spec.get("gpu", {})
            snapshot = spec.get("snapshot", {})

            phase = st.get("phase", "Unknown")
            ready = st.get("readyReplicas", 0)
            desired = st.get("replicas", 0)

            table.add_row(
                meta.get("name", ""),
                spec.get("model", ""),
                spec.get("runtime", ""),
                phase,
                f"{ready}/{desired}",
                f"{gpu.get('count', 0)}x {gpu.get('type', 'unknown')}",
                "enabled" if snapshot.get("enabled") else "disabled",
            )

        console.print(table)
    else:
        # Plain text fallback
        header = f"{'NAME':<25} {'MODEL':<40} {'RUNTIME':<10} {'PHASE':<12} {'REPLICAS':<10}"
        click.echo(header)
        click.echo("-" * len(header))
        for item in items:
            meta = item.get("metadata", {})
            spec = item.get("spec", {})
            st = item.get("status", {})
            ready = st.get("readyReplicas", 0)
            desired = st.get("replicas", 0)
            click.echo(
                f"{meta.get('name', ''):<25} "
                f"{spec.get('model', ''):<40} "
                f"{spec.get('runtime', ''):<10} "
                f"{st.get('phase', 'Unknown'):<12} "
                f"{ready}/{desired}"
            )


# ======================================================================
# logs
# ======================================================================


@cli.command()
@click.argument("model_name")
@click.option("--namespace", "-n", default=None, help="Kubernetes namespace.")
@click.option("--follow", "-f", is_flag=True, help="Follow log output.")
@click.option("--tail", "-t", default=100, help="Number of lines to show.")
@click.option("--container", "-c", default="worker", help="Container name.")
def logs(model_name: str, namespace: str | None, follow: bool, tail: int, container: str) -> None:
    """Stream container logs for a model deployment.

    Finds pods belonging to the given model deployment and streams their
    logs to stdout.
    """
    ns = namespace or _get_namespace()

    # Find pods for this model
    selector = f"podstack.io/model={model_name}"
    cmd = ["logs", f"-l{selector}", f"--tail={tail}", f"-c{container}"]
    if follow:
        cmd.append("-f")

    result = _kubectl(*cmd, namespace=ns, capture=False)
    if result.returncode != 0:
        click.echo(f"Failed to get logs for {model_name}", err=True)
        sys.exit(1)


# ======================================================================
# snapshot
# ======================================================================


@cli.group()
def snapshot() -> None:
    """Snapshot management commands.

    Create, list, and restore CUDA/CRIU snapshots for model deployments.
    """
    pass


@snapshot.command("create")
@click.argument("model_name")
@click.option("--namespace", "-n", default=None, help="Kubernetes namespace.")
def snapshot_create(model_name: str, namespace: str | None) -> None:
    """Manually trigger snapshot creation for a model deployment.

    Annotates the model's pod to signal the operator to create a
    CUDA/CRIU snapshot.
    """
    ns = namespace or _get_namespace()

    # Find the pod
    selector = f"podstack.io/model={model_name}"
    result = _kubectl(
        "get", "pods", f"-l{selector}", "-o", "jsonpath={.items[0].metadata.name}",
        namespace=ns,
    )

    if result.returncode != 0 or not result.stdout.strip():
        click.echo(f"No running pod found for model {model_name}", err=True)
        sys.exit(1)

    pod_name = result.stdout.strip()
    click.echo(f"Triggering snapshot for pod {pod_name}...")

    # Annotate the pod to trigger snapshot
    result = _kubectl(
        "annotate", "pod", pod_name,
        "podstack.io/snapshot-request=create",
        "--overwrite",
        namespace=ns,
    )

    if result.returncode != 0:
        click.echo(f"Failed to trigger snapshot: {result.stderr.strip()}", err=True)
        sys.exit(1)

    click.echo(f"Snapshot requested for {model_name} (pod: {pod_name})")


@snapshot.command("list")
@click.argument("model_name", required=False)
@click.option("--namespace", "-n", default=None, help="Kubernetes namespace.")
def snapshot_list(model_name: str | None, namespace: str | None) -> None:
    """List available snapshots for a model deployment."""
    ns = namespace or _get_namespace()

    # Query for snapshot PVCs or ConfigMaps
    selector = "podstack.io/snapshot=true"
    if model_name:
        selector += f",podstack.io/model={model_name}"

    result = _kubectl(
        "get", "pvc", f"-l{selector}", "-o", "json",
        namespace=ns,
    )

    if result.returncode != 0:
        click.echo(f"Failed to list snapshots: {result.stderr.strip()}", err=True)
        sys.exit(1)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        click.echo("No snapshots found.")
        return

    items = data.get("items", [])
    if not items:
        click.echo("No snapshots found.")
        return

    click.echo(f"{'SNAPSHOT':<30} {'MODEL':<25} {'SIZE':<10} {'CREATED':<25}")
    click.echo("-" * 90)
    for item in items:
        meta = item.get("metadata", {})
        labels = meta.get("labels", {})
        spec = item.get("spec", {})
        status = item.get("status", {})

        click.echo(
            f"{meta.get('name', ''):<30} "
            f"{labels.get('podstack.io/model', ''):<25} "
            f"{status.get('capacity', {}).get('storage', 'N/A'):<10} "
            f"{meta.get('creationTimestamp', ''):<25}"
        )


# ======================================================================
# models
# ======================================================================


@cli.command()
@click.option("--endpoint", "-e", default=None, help="Podstack gateway endpoint.")
@click.option("--api-key", "-k", default=None, help="API key for authentication.")
def models(endpoint: str | None, api_key: str | None) -> None:
    """List all available models on the gateway."""
    from .client import PodstackClient

    try:
        client = PodstackClient(base_url=endpoint, api_key=api_key)
        response = client.list_models()

        if not response.data:
            click.echo("No models available.")
            return

        click.echo(f"{'MODEL ID':<50} {'OWNED BY':<15}")
        click.echo("-" * 65)
        for model in response.data:
            click.echo(f"{model.id:<50} {model.owned_by:<15}")

    except Exception as exc:
        click.echo(f"Failed to list models: {exc}", err=True)
        sys.exit(1)


# ======================================================================
# init
# ======================================================================


@cli.command()
@click.argument("name")
@click.option("--model", "-m", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model to deploy.")
@click.option("--runtime", "-r", default="vllm", help="Inference runtime.")
def init(name: str, model: str, runtime: str) -> None:
    """Generate a starter app.py file for a new model deployment."""
    template = f'''"""Podstack model deployment: {name}"""
from podstack import App, GPU, Scaling

app = App("{name}")

@app.model(
    name="{name}",
    model="{model}",
    runtime="{runtime}",
    gpu=GPU(count=1, type="l40s", memory_mb=48000),
    scaling=Scaling(
        min_replicas=0,
        max_replicas=5,
        idle_timeout=300,
        standby_pool=1,
    ),
    snapshot=True,
    model_type="llm",
    model_source="huggingface",
)
class Model:
    pass

if __name__ == "__main__":
    app.deploy(dry_run=True)
'''
    output_path = Path("app.py")
    if output_path.exists():
        click.confirm("app.py already exists. Overwrite?", abort=True)

    output_path.write_text(template)
    click.echo(f"Created app.py for '{name}'")
    click.echo(f"  Model:   {model}")
    click.echo(f"  Runtime: {runtime}")
    click.echo(f"\nDeploy with: podstack deploy app.py")


if __name__ == "__main__":
    cli()
