"""Launch a JupyterLab server on Modal.com with an A100 GPU.

Usage:
    modal serve deploy/modal/launch_jupyter.py

This starts a cloud JupyterLab instance with an A100 GPU and prints a URL
that you can open directly in your web browser.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None  # type: ignore[assignment]

if modal is not None:
    app = modal.App("aeitron-jupyter-server")
    volume = modal.Volume.from_name("aeitron-training-volume", create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install(
            "jupyterlab>=4.0.0",
            "torch>=2.3.0",
            "tokenizers>=0.15.2",
            "matplotlib>=3.8.0",
            "pydantic>=2.7.0",
            "httpx>=0.27.0",
            "tqdm>=4.66.0",
            "pyyaml>=6.0.1",
        )
        .add_local_python_source("src", "config", "deploy")
    )

    @app.function(
        gpu="A100-80GB",
        timeout=7200,  # 2 hours
        volumes={"/vol": volume},
        image=image,
    )
    @modal.web_server(port=8888, startup_timeout=60)
    def run_jupyter():
        """Launch JupyterLab listening on port 8888."""
        cmd = [
            "jupyter",
            "lab",
            "--no-browser",
            "--port=8888",
            "--ip=0.0.0.0",
            "--allow-root",
            "--NotebookApp.token=''",
            "--NotebookApp.password=''",
            "--ServerApp.disable_check_xsrf=True",
        ]
        subprocess.Popen(cmd)
else:
    if __name__ == "__main__":
        print("Please install modal: pip install modal")
