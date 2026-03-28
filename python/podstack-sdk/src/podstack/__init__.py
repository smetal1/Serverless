"""Podstack Inference OS SDK.

Deploy and manage GPU inference models with a Python-native DSL::

    from podstack import App, GPU, Scaling

    app = App("my-stack")

    @app.model(
        name="llama3-70b",
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        gpu=GPU(count=4, type="a100"),
        scaling=Scaling(min_replicas=0, max_replicas=10),
    )
    class Llama3: pass

    app.deploy()
"""

__version__ = "0.1.0"

from .app import App, GPU, Scaling
from .client import PodstackClient

__all__ = [
    "App",
    "GPU",
    "Scaling",
    "PodstackClient",
]
