"""Public Python SDK surface for the Craftly Training Workspace.

The implementation remains canonical in ``src.aeitron.training_client`` so the
installed CLI, notebooks, and the control-plane repository use identical code.
"""

from src.aeitron.training_client import TrainingRun, Workspace

__all__ = ["TrainingRun", "Workspace"]
