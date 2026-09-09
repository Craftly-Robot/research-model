"""Continual Learning and Catastrophic Forgetting Mitigation for Craftly Models.

Provides full-parameter anti-forgetting techniques for scratch-origin models:
1. Experience Replay Buffer (interleaved pretraining code replay)
2. Reference Model KL-Divergence Anchor (policy drift penalty)
3. Elastic Weight Consolidation (EWC) - parameter importance penalty
4. Weight Space Merging (Spherical Linear Interpolation / SLERP and Task Vectors)
5. Scientific Catastrophic Forgetting Evaluation Benchmark
"""

from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from src.craftly.model_ops.torch_decoder import (
    load_trusted_checkpoint,
    require_torch,
    save_trusted_checkpoint,
)
from src.craftly.shared.schemas import StrictModel

if TYPE_CHECKING:
    import torch
    import torch.nn.functional as F
else:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:  # pragma: no cover
        torch = None  # type: ignore[assignment]
        F = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 1. Experience Replay: Pretraining Distribution Anchoring
# ---------------------------------------------------------------------------

CANONICAL_PRETRAINING_REPLAY_SAMPLES: list[dict[str, str]] = [
    {
        "id": "py-algo-quicksort",
        "category": "general_coding",
        "text": (
            "def quicksort(arr: list[int]) -> list[int]:\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    pivot = arr[len(arr) // 2]\n"
            "    left = [x for x in arr if x < pivot]\n"
            "    middle = [x for x in arr if x == pivot]\n"
            "    right = [x for x in arr if x > pivot]\n"
            "    return quicksort(left) + middle + quicksort(right)\n"
        ),
    },
    {
        "id": "py-algo-binary-search",
        "category": "general_coding",
        "text": (
            "def binary_search(elements: list[int], target: int) -> int:\n"
            "    low, high = 0, len(elements) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if elements[mid] == target:\n"
            "            return mid\n"
            "        elif elements[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1\n"
        ),
    },
    {
        "id": "py-core-lru-cache",
        "category": "general_coding",
        "text": (
            "from collections import OrderedDict\n\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity: int):\n"
            "        self.capacity = capacity\n"
            "        self.cache: OrderedDict[int, int] = OrderedDict()\n\n"
            "    def get(self, key: int) -> int:\n"
            "        if key not in self.cache:\n"
            "            return -1\n"
            "        self.cache.move_to_end(key)\n"
            "        return self.cache[key]\n\n"
            "    def put(self, key: int, value: int) -> None:\n"
            "        if key in self.cache:\n"
            "            self.cache.move_to_end(key)\n"
            "        self.cache[key] = value\n"
            "        if len(self.cache) > self.capacity:\n"
            "            self.cache.popitem(last=False)\n"
        ),
    },
    {
        "id": "py-network-http-server",
        "category": "general_coding",
        "text": (
            "import http.server\n"
            "import socketserver\n\n"
            "PORT = 8080\n\n"
            "class StatusHandler(http.server.SimpleHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        self.send_response(200)\n"
            "        self.send_header('Content-type', 'application/json')\n"
            "        self.end_headers()\n"
            '        self.wfile.write(b\'{"status": "healthy"}\')\n'
        ),
    },
    {
        "id": "py-data-json-parser",
        "category": "general_coding",
        "text": (
            "import json\n"
            "from pathlib import Path\n\n"
            "def read_json_manifest(filepath: str | Path) -> dict:\n"
            "    path = Path(filepath)\n"
            "    if not path.is_file():\n"
            "        raise FileNotFoundError(f'Missing file: {path}')\n"
            "    with path.open('r', encoding='utf-8') as f:\n"
            "        return json.load(f)\n"
        ),
    },
]


class ExperienceReplayBuffer:
    """Manages pretraining and general coding memory to interleave with SFT batches."""

    def __init__(
        self,
        replay_samples: list[str] | None = None,
        replay_ratio: float = 0.20,
        seed: int = 1337,
    ) -> None:
        self.replay_ratio = max(0.0, min(0.9, replay_ratio))
        self.rng = random.Random(seed)
        if replay_samples is not None and len(replay_samples) > 0:
            self.samples = list(replay_samples)
        else:
            self.samples = [item["text"] for item in CANONICAL_PRETRAINING_REPLAY_SAMPLES]

    def add_sample(self, text: str) -> None:
        clean = text.strip()
        if clean and clean not in self.samples:
            self.samples.append(clean)

    def sample_batch(
        self,
        count: int,
        tokenizer: Any,
        max_sequence_length: int = 1024,
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Produce encoded input_ids and autoregressive labels for replay sequences."""
        batch_input_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for _ in range(count):
            text = self.rng.choice(self.samples)
            token_ids = tokenizer.encode(text).ids
            if not token_ids:
                token_ids = [0]
            if len(token_ids) > max_sequence_length:
                token_ids = token_ids[:max_sequence_length]

            # Autoregressive next-token prediction targets
            inp = list(token_ids)
            lbl = list(token_ids)

            # Pad to uniform length
            pad_len = max_sequence_length - len(inp)
            if pad_len > 0:
                inp.extend([pad_token_id] * pad_len)
                lbl.extend([ignore_index] * pad_len)

            batch_input_ids.append(inp)
            batch_labels.append(lbl)

        return batch_input_ids, batch_labels


# ---------------------------------------------------------------------------
# 2. KL Divergence Penalty: Reference Policy Constraint
# ---------------------------------------------------------------------------


class ReferenceModelKLLoss:
    """Computes token-level KL divergence penalty between active model and frozen base model."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(0.1, temperature)

    def compute(
        self,
        active_logits: Any,
        ref_logits: Any,
        mask: Any | None = None,
    ) -> Any:
        """Calculate forward KL(P_active || P_ref) over valid tokens.

        Args:
            active_logits: (batch_size, seq_len, vocab_size)
            ref_logits: (batch_size, seq_len, vocab_size)
            mask: (batch_size, seq_len) boolean or int mask where 1 = compute KL
        """
        require_torch()
        t = self.temperature
        # P = active distribution, Q = reference base distribution
        log_p = F.log_softmax(active_logits / t, dim=-1)
        p = torch.exp(log_p)
        log_q = F.log_softmax(ref_logits / t, dim=-1)

        # KL(P || Q) = sum P * (log P - log Q)
        kl_per_token = torch.sum(p * (log_p - log_q), dim=-1) * (t**2)

        if mask is not None:
            active_mask = (mask != -100).float()
            denom = active_mask.sum() + 1e-8
            return (kl_per_token * active_mask).sum() / denom

        return kl_per_token.mean()


# ---------------------------------------------------------------------------
# 3. Elastic Weight Consolidation (EWC): Parameter Importance Penalty
# ---------------------------------------------------------------------------

# WHAT IS EWC?
# ============
# When a neural network learns a new task, it tends to overwrite the weights
# that were important for old tasks. This is called "catastrophic forgetting."
#
# EWC prevents this by:
# 1. Measuring how important each parameter is for the OLD task (Fisher Information)
# 2. During NEW task training, penalizing changes to important parameters
#
# The penalty works like a spring: important parameters are pulled back toward
# their old values, while unimportant parameters are free to change.
#
# Mathematically:
#   total_loss = new_task_loss + lambda * sum(F_i * (theta_i - theta_old_i)^2)
#
# Where:
#   - F_i = Fisher Information for parameter i (how important it is)
#   - theta_i = current value of parameter i
#   - theta_old_i = value of parameter i after old task training
#   - lambda = strength of the penalty (higher = less forgetting, but less learning)


class ElasticWeightConsolidation:
    """EWC: Penalizes changes to parameters important for previous tasks.

    Usage:
        # After finishing task A:
        ewc = ElasticWeightConsolidation(model, device="cuda")
        ewc.compute_fisher(dataloader_for_task_a)

        # While training on task B, add ewc.penalty() to your loss:
        loss = task_b_loss + ewc.penalty(model)
    """

    def __init__(self, model: Any, device: str = "cpu", damping: float = 1.0) -> None:
        """Initialize EWC with a reference model.

        Args:
            model: The trained model (weights will be snapshotted as "old" weights)
            device: Device to compute on
            damping: Lambda parameter - higher means stronger penalty against forgetting.
                     Typical values: 1000-10000 for strong protection, 1-100 for mild.
        """
        require_torch()
        self.device = device
        self.damping = damping  # lambda in the EWC formula

        # Store the current model weights as "old" weights (theta_old)
        # These are the weights we want to protect from large changes
        self.old_weights: dict[str, Any] = {}
        for name, param in model.named_parameters():
            self.old_weights[name] = param.data.clone().detach()

        # Fisher Information will be computed later via compute_fisher()
        # Fisher is a diagonal approximation of the Hessian matrix
        # It measures: "if I change this parameter slightly, how much does the loss change?"
        # High Fisher = parameter is important for the old task
        self.fisher: dict[str, Any] = {}

    def compute_fisher(self, dataloader: Any, model: Any, num_samples: int = 1000) -> None:
        """Compute Fisher Information Matrix (diagonal approximation).

        The Fisher Information tells us which parameters are important for a task.
        It's computed by:
        1. Running the model on old task data
        2. Computing gradients of the loss w.r.t. each parameter
        3. Squaring those gradients (Fisher ≈ E[gradient^2])

        Args:
            dataloader: Iterator yielding (input_ids, labels) tuples from the OLD task
            model: The model to compute Fisher for
            num_samples: How many samples to use (more = more accurate, but slower)
        """
        require_torch()

        # Initialize Fisher to zero for each parameter
        self.fisher = {}
        for name, param in model.named_parameters():
            self.fisher[name] = torch.zeros_like(param.data)

        model.train()
        sample_count = 0

        for batch in dataloader:
            if sample_count >= num_samples:
                break

            # Unpack batch - handle both tuple and dict formats
            if isinstance(batch, (list, tuple)):
                input_ids, labels = batch[0].to(self.device), batch[1].to(self.device)
            else:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

            # Forward pass: compute loss
            model.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            # Backward pass: compute gradients
            loss.backward()

            # Accumulate squared gradients (this is the Fisher Information)
            # We use running average to handle batches
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Fisher = E[gradient^2]
                    # We accumulate and average at the end
                    self.fisher[name] += param.grad.data**2

            sample_count += input_ids.shape[0]

        # Average over all samples
        if sample_count > 0:
            for name in self.fisher:
                self.fisher[name] /= sample_count

    def penalty(self, model: Any) -> Any:
        """Compute the EWC penalty term.

        This returns: lambda * sum(F_i * (theta_i - theta_old_i)^2)

        Add this to your training loss:
            total_loss = new_task_loss + ewc.penalty(model)
        """
        require_torch()
        penalty_loss = torch.tensor(0.0, device=self.device)

        for name, param in model.named_parameters():
            if name in self.fisher and name in self.old_weights:
                # How much has this parameter changed from the old task?
                weight_diff = param - self.old_weights[name]

                # How important is this parameter? (Fisher Information)
                fisher_importance = self.fisher[name]

                # Penalty = importance * squared_change
                # Important parameters that changed a lot get a high penalty
                penalty_loss += (fisher_importance * weight_diff**2).sum()

        return self.damping * penalty_loss


# ---------------------------------------------------------------------------
# 4. Weight Space Merging: SLERP & Task Vector Arithmetic
# ---------------------------------------------------------------------------


class WeightMerger:
    """Merges foundation base weights and specialized SFT weights without external checkpoints."""

    @staticmethod
    def slerp(
        v0: Any,
        v1: Any,
        t: float = 0.5,
        eps: float = 1e-7,
    ) -> Any:
        """Spherical Linear Interpolation between two parameter tensors.

        Formula:
            Omega = arccos( <v0, v1> / (||v0|| * ||v1||) )
            slerp(v0, v1; t) = (sin((1-t)*Omega)/sin(Omega))*v0 + (sin(t*Omega)/sin(Omega))*v1
        """
        require_torch()
        orig_shape = v0.shape
        t0 = v0.flatten().float()
        t1 = v1.flatten().float()

        norm_0 = torch.norm(t0)
        norm_1 = torch.norm(t1)

        if norm_0 < eps or norm_1 < eps:
            return ((1.0 - t) * v0 + t * v1).to(v0.dtype)

        u0 = t0 / norm_0
        u1 = t1 / norm_1

        dot = torch.clamp(torch.sum(u0 * u1), -1.0 + eps, 1.0 - eps)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)

        # Collinear fallback (angle close to 0)
        if sin_omega.abs() < eps:
            return ((1.0 - t) * v0 + t * v1).to(v0.dtype)

        scale_0 = torch.sin((1.0 - t) * omega) / sin_omega
        scale_1 = torch.sin(t * omega) / sin_omega

        res = scale_0 * t0 + scale_1 * t1
        return res.view(orig_shape).to(v0.dtype)

    @staticmethod
    def task_arithmetic(
        base_tensor: Any,
        sft_tensor: Any,
        alpha: float = 0.5,
    ) -> Any:
        """Linear task vector scaling: theta_merged = theta_base + alpha * (theta_sft - theta_base)."""
        require_torch()
        diff = sft_tensor.float() - base_tensor.float()
        res = base_tensor.float() + (alpha * diff)
        return res.to(base_tensor.dtype)

    @classmethod
    def merge_state_dicts(
        cls,
        base_state: dict[str, Any],
        sft_state: dict[str, Any],
        method: Literal["slerp", "linear"] = "slerp",
        t: float = 0.3,
    ) -> dict[str, Any]:
        """Apply parameter-wise SLERP or linear merging across two state dicts."""
        require_torch()
        merged: dict[str, Any] = {}
        for key, base_val in base_state.items():
            if key not in sft_state:
                merged[key] = copy.deepcopy(base_val)
                continue

            sft_val = sft_state[key]
            if not isinstance(base_val, torch.Tensor) or not isinstance(sft_val, torch.Tensor):
                merged[key] = copy.deepcopy(sft_val)
                continue

            if base_val.shape != sft_val.shape:
                raise ValueError(
                    f"Shape mismatch for parameter '{key}': {base_val.shape} vs {sft_val.shape}"
                )

            # 1D biases and layer norms are typically linearly averaged
            if base_val.dim() <= 1 or method == "linear":
                merged[key] = ((1.0 - t) * base_val.float() + t * sft_val.float()).to(
                    base_val.dtype
                )
            else:
                merged[key] = cls.slerp(base_val, sft_val, t=t)

        return merged

    @classmethod
    def merge_checkpoint_files(
        cls,
        base_path: str | Path,
        sft_path: str | Path,
        output_path: str | Path,
        method: Literal["slerp", "linear"] = "slerp",
        t: float = 0.3,
    ) -> Path:
        """Load base and SFT checkpoint archives, merge parameters, and write clean package."""
        require_torch()
        base_file = Path(base_path).resolve()
        sft_file = Path(sft_path).resolve()
        out_file = Path(output_path).resolve()

        if not base_file.exists():
            raise FileNotFoundError(f"Base checkpoint not found: {base_file}")
        if not sft_file.exists():
            raise FileNotFoundError(f"SFT checkpoint not found: {sft_file}")

        base_payload = cls._extract_or_load(base_file)
        sft_payload = cls._extract_or_load(sft_file)

        base_state = base_payload.get("model", base_payload)
        sft_state = sft_payload.get("model", sft_payload)

        merged_state = cls.merge_state_dicts(base_state, sft_state, method=method, t=t)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "model": merged_state,
            "config": sft_payload.get("config", base_payload.get("config", {})),
            "stage": f"continual_merged_{method}_t{int(t * 100)}",
            "merged_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "provenance": {
                "base_source": base_file.name,
                "sft_source": sft_file.name,
                "method": method,
                "interpolation_weight": t,
            },
        }

        save_trusted_checkpoint(meta, out_file)
        return out_file

    @staticmethod
    def _extract_or_load(path: Path) -> dict[str, Any]:
        if path.suffix == ".zip":
            import tempfile
            import zipfile

            with tempfile.TemporaryDirectory(prefix="craftly_merge_") as td:
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(td)
                model_pt = Path(td) / "model.pt"
                if not model_pt.exists():
                    raise FileNotFoundError(f"Archive {path} does not contain model.pt")
                return load_trusted_checkpoint(model_pt, map_location="cpu")
        return load_trusted_checkpoint(path, map_location="cpu")


# ---------------------------------------------------------------------------
# 5. Scientific Catastrophic Forgetting Benchmark Suite
# ---------------------------------------------------------------------------


class ForgettingTask(StrictModel):
    task_id: str
    category: str
    prompt: str
    expected_patterns: list[str] = Field(default_factory=list)


CANONICAL_FORGETTING_SUITE: list[ForgettingTask] = [
    ForgettingTask(
        task_id="base-algo-fibonacci",
        category="general_coding",
        prompt="Write a Python function to return the nth Fibonacci number efficiently:\n\ndef fib(n: int) -> int:\n",
        expected_patterns=["def fib", "return", "if n"],
    ),
    ForgettingTask(
        task_id="base-algo-quicksort",
        category="general_coding",
        prompt="Write quicksort in Python using list comprehension:\n\ndef quicksort(arr):\n",
        expected_patterns=["quicksort", "pivot", "return"],
    ),
    ForgettingTask(
        task_id="base-data-invert-dict",
        category="general_coding",
        prompt="Write a Python function to invert a dictionary mapping keys to values:\n\ndef invert_dict(d: dict) -> dict:\n",
        expected_patterns=["def invert_dict", "return", "items()"],
    ),
    ForgettingTask(
        task_id="defensive-cwe-89",
        category="defensive_security",
        prompt="Fix the following SQL injection vulnerability using parameterized queries:\ncursor.execute(f'SELECT * FROM users WHERE id={user_id}')\n",
        expected_patterns=["SELECT", "%s", "cursor.execute"],
    ),
    ForgettingTask(
        task_id="defensive-cwe-22",
        category="defensive_security",
        prompt="Fix path traversal vulnerability when opening files in export directory:\n\ndef read_file(name: str):\n",
        expected_patterns=["Path", "resolve", "is_relative_to"],
    ),
    ForgettingTask(
        task_id="defensive-cwe-208",
        category="defensive_security",
        prompt="Harden secret comparison against side-channel timing attacks in Python:\n\ndef check_key(a, b):\n",
        expected_patterns=["hmac", "compare_digest"],
    ),
]


class ForgettingReport(StrictModel):
    model_name: str
    general_coding_loss: float
    defensive_security_loss: float
    perplexity_general: float
    perplexity_defensive: float
    general_retention_score: float
    evaluated_tasks_count: int
    created_at_utc: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    )


def evaluate_continual_perplexity(
    model: Any,
    tokenizer: Any,
    tasks: list[ForgettingTask] | None = None,
    device: str = "cpu",
) -> ForgettingReport:
    """Calculate cross-entropy loss and retention scores over general vs defensive tasks."""
    require_torch()
    tasks_to_eval = tasks or CANONICAL_FORGETTING_SUITE
    model.eval()

    gen_losses: list[float] = []
    def_losses: list[float] = []

    with torch.no_grad():
        for t in tasks_to_eval:
            tokens = tokenizer.encode(t.prompt).ids
            if len(tokens) < 2:
                continue
            inp = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            target = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            out = model(inp, labels=target)
            loss_val = float(out.loss.item())

            if t.category == "defensive_security":
                def_losses.append(loss_val)
            else:
                gen_losses.append(loss_val)

    avg_gen = sum(gen_losses) / len(gen_losses) if gen_losses else 0.0
    avg_def = sum(def_losses) / len(def_losses) if def_losses else 0.0

    ppl_gen = math.exp(min(20.0, avg_gen)) if avg_gen > 0 else 1.0
    ppl_def = math.exp(min(20.0, avg_def)) if avg_def > 0 else 1.0

    retention_score = max(0.0, min(100.0, 100.0 / (1.0 + avg_gen)))

    return ForgettingReport(
        model_name=getattr(model, "name", "craftly-model"),
        general_coding_loss=round(avg_gen, 4),
        defensive_security_loss=round(avg_def, 4),
        perplexity_general=round(ppl_gen, 2),
        perplexity_defensive=round(ppl_def, 2),
        general_retention_score=round(retention_score, 2),
        evaluated_tasks_count=len(tasks_to_eval),
    )
