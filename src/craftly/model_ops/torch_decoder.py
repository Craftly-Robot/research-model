"""PyTorch Craftly decoder-only language model.

This is the numerical reference implementation for Craftly-owned scratch
weights. Canonical architecture contracts and profiles live in ``foundation``;
this module owns the executable dense/GQA and small-scale MLA/MoE kernels used
for parity, checkpoint qualification, and single-node validation.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.craftly.model_ops.foundation import (
    ScratchDecoderConfig,
    model_profile,
    tiny_smoke_config,
)

try:
    import torch
    import torch.utils.checkpoint
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only on missing torch installs.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass
class DecoderForwardOutput:
    logits: "torch.Tensor"
    loss: "torch.Tensor | None" = None
    past_key_values: tuple[tuple["torch.Tensor", "torch.Tensor"], ...] | None = None
    language_model_loss: "torch.Tensor | None" = None
    mtp_loss: "torch.Tensor | None" = None
    router_metrics: tuple[dict[str, float], ...] = ()


def require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("torch is required for Craftly scratch decoder execution")


def select_torch_device(requested: str) -> "torch.device":
    """Resolve a requested PyTorch device with consistent CUDA fail-fast behavior."""
    require_torch()
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    try:
        return torch.device(normalized)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid torch device: {requested!r}") from exc


def save_trusted_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    """Save a Craftly-owned local training checkpoint.

    The training checkpoint includes optimizer/scheduler state, so PyTorch's
    native format is still required for resumable training. External/untrusted
    checkpoints must not be loaded through this path.
    """
    require_torch()
    torch.save(payload, path)  # nosec B614 # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch


def load_trusted_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    """Load a Craftly-owned checkpoint with PyTorch's restricted unpickler."""
    require_torch()
    checkpoint_path = Path(path).resolve()
    if checkpoint_path.suffix != ".pt":
        raise ValueError(f"unsupported checkpoint suffix: {checkpoint_path.suffix}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(  # nosec B614 # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(payload, dict) or "model" not in payload or "config" not in payload:
        raise ValueError("checkpoint payload must contain model and config dictionaries")
    return payload


class RMSNorm(nn.Module):  # type: ignore[misc]
    def __init__(self, hidden_size: int, eps: float) -> None:
        require_torch()
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype=self.weight.dtype)


def rotate_half(x: "torch.Tensor") -> "torch.Tensor":
    left, right = x[..., ::2], x[..., 1::2]
    return torch.stack((-right, left), dim=-1).flatten(-2)


def apply_rope(x: "torch.Tensor", cos: "torch.Tensor", sin: "torch.Tensor") -> "torch.Tensor":
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):  # type: ignore[misc]
    """RoPE without a sequence-length-sized persistent allocation.

    A 1M context profile must not allocate multi-gigabyte cosine/sine tables at
    model construction time. Frequencies are generated for the active span and
    remain non-persistent, which also keeps checkpoint formats topology neutral.
    """

    def __init__(
        self,
        dim: int,
        max_position: int,
        theta: float,
        *,
        scaling_type: str = "none",
        scaling_factor: float = 1.0,
    ) -> None:
        require_torch()
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position = max_position
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor

    def forward(self, q: "torch.Tensor", k: "torch.Tensor", *, position_offset: int = 0) -> tuple["torch.Tensor", "torch.Tensor"]:
        seq_len = q.size(-2)
        end = position_offset + seq_len
        if end > self.max_position:
            raise ValueError(f"sequence position {end} exceeds configured context {self.max_position}")
        positions = torch.arange(position_offset, end, device=q.device, dtype=torch.float32)
        if self.scaling_type != "none":
            positions = positions / self.scaling_factor
        freqs = torch.outer(positions, self.inv_freq.to(device=q.device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :].to(dtype=q.dtype)
        sin = emb.sin()[None, None, :, :].to(dtype=q.dtype)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


class CausalSelfAttention(nn.Module):  # type: ignore[misc]
    def __init__(self, config: ScratchDecoderConfig) -> None:
        require_torch()
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.max_sequence_length,
            (
                config.rope_theta * config.rope_scaling_factor
                if config.rope_scaling_type == "none" and config.rope_scaling_factor > 1.0
                else config.rope_theta
            ),
            scaling_type=config.rope_scaling_type,
            scaling_factor=config.rope_scaling_factor,
        )
        self.dropout = config.dropout

    def build_attention_mask(
        self,
        *,
        query_len: int,
        key_len: int,
        past_len: int,
        device: "torch.device",
    ) -> "torch.Tensor | None":
        if self.config.attention_window is None and past_len == 0:
            return None
        query_positions = torch.arange(past_len, past_len + query_len, device=device)[:, None]
        key_positions = torch.arange(0, key_len, device=device)[None, :]
        mask = key_positions <= query_positions
        if self.config.attention_window is not None:
            mask = mask & (key_positions >= query_positions - self.config.attention_window + 1)
        return mask[None, None, :, :]

    def eager_attention(
        self,
        q: "torch.Tensor",
        k: "torch.Tensor",
        v: "torch.Tensor",
        *,
        mask: "torch.Tensor | None",
    ) -> "torch.Tensor":
        scale = 1.0 / math.sqrt(self.config.head_dim)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        else:
            causal = torch.ones(scores.size(-2), scores.size(-1), device=scores.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal[None, None, :, :], torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1).to(dtype=q.dtype)
        if self.training and self.dropout > 0:
            probs = F.dropout(probs, p=self.dropout)
        return torch.matmul(probs, v)

    def forward(
        self,
        x: "torch.Tensor",
        *,
        past_key_value: tuple["torch.Tensor", "torch.Tensor"] | None = None,
        use_cache: bool = False,
    ) -> tuple["torch.Tensor", tuple["torch.Tensor", "torch.Tensor"] | None]:
        batch, seq_len, _ = x.shape
        past_len = 0 if past_key_value is None else int(past_key_value[0].size(-2))
        q = self.q_proj(x).view(batch, seq_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, position_offset=past_len)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        present = (k, v) if use_cache else None
        repeat = self.config.num_attention_heads // self.config.num_key_value_heads
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        mask = self.build_attention_mask(query_len=seq_len, key_len=k.size(-2), past_len=past_len, device=x.device)
        use_eager = self.config.attention_impl == "eager" or not hasattr(F, "scaled_dot_product_attention")
        if use_eager:
            attn = self.eager_attention(q, k, v, mask=mask)
        else:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=mask is None and past_len == 0,
            )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.config.hidden_size)
        return self.o_proj(attn), present


class MultiLatentAttention(nn.Module):  # type: ignore[misc]
    """DeepSeek-style latent attention with a compressed inference cache.

    The cache stores the latent KV representation and the decoupled rotary key,
    not expanded per-head keys and values. This is the reference algorithm;
    trillion-scale execution is delegated to the validated Megatron backend.
    """

    def __init__(self, config: ScratchDecoderConfig) -> None:
        require_torch()
        super().__init__()
        if config.attention_architecture != "mla":
            raise ValueError("MultiLatentAttention requires an MLA config")
        self.config = config
        self.q_rank = int(config.q_lora_rank or 0)
        self.kv_rank = int(config.kv_lora_rank or 0)
        self.nope_dim = int(config.qk_nope_head_dim or 0)
        self.rope_dim = int(config.qk_rope_head_dim or 0)
        self.value_dim = int(config.v_head_dim or 0)
        self.q_down = nn.Linear(config.hidden_size, self.q_rank, bias=False)
        self.q_norm = RMSNorm(self.q_rank, config.norm_eps)
        self.q_up = nn.Linear(
            self.q_rank,
            config.num_attention_heads * (self.nope_dim + self.rope_dim),
            bias=False,
        )
        self.kv_down = nn.Linear(config.hidden_size, self.kv_rank + self.rope_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_rank, config.norm_eps)
        self.kv_up = nn.Linear(
            self.kv_rank,
            config.num_attention_heads * (self.nope_dim + self.value_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(config.num_attention_heads * self.value_dim, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(
            self.rope_dim,
            config.max_sequence_length,
            config.rope_theta,
            scaling_type=config.rope_scaling_type,
            scaling_factor=config.rope_scaling_factor,
        )
        self.dropout = config.dropout

    def _validate_cache(
        self,
        cache: tuple["torch.Tensor", "torch.Tensor"],
        *,
        x: "torch.Tensor",
    ) -> int:
        if not isinstance(cache, tuple) or len(cache) != 2:
            raise ValueError("MLA cache must be a (kv_latent, rotary_key) tuple")
        latent, rotary = cache
        expected_latent = (x.size(0), self.kv_rank)
        expected_rotary = (x.size(0), 1, self.rope_dim)
        if latent.ndim != 3 or (latent.size(0), latent.size(2)) != expected_latent:
            raise ValueError(
                "MLA latent cache must have shape "
                f"[{x.size(0)}, past_sequence, {self.kv_rank}]"
            )
        if rotary.ndim != 4 or (rotary.size(0), rotary.size(1), rotary.size(3)) != expected_rotary:
            raise ValueError(
                "MLA rotary cache must have shape "
                f"[{x.size(0)}, 1, past_sequence, {self.rope_dim}]"
            )
        if latent.size(1) != rotary.size(2):
            raise ValueError("MLA latent and rotary cache sequence lengths differ")
        if latent.device != x.device or rotary.device != x.device:
            raise ValueError("MLA cache and hidden states must be on the same device")
        if latent.dtype != x.dtype or rotary.dtype != x.dtype:
            raise ValueError("MLA cache and hidden states must have the same dtype")
        past_len = int(latent.size(1))
        if past_len + x.size(1) > self.config.max_sequence_length:
            raise ValueError("MLA cache plus input exceeds max_sequence_length")
        return past_len

    def _mask(self, *, query_len: int, key_len: int, past_len: int, device: "torch.device") -> "torch.Tensor | None":
        if self.config.attention_window is None and past_len == 0:
            return None
        query_positions = torch.arange(past_len, past_len + query_len, device=device)[:, None]
        key_positions = torch.arange(key_len, device=device)[None, :]
        mask = key_positions <= query_positions
        if self.config.attention_window is not None:
            mask &= key_positions >= query_positions - self.config.attention_window + 1
        return mask[None, None, :, :]

    def _eager(
        self,
        q: "torch.Tensor",
        k: "torch.Tensor",
        v: "torch.Tensor",
        *,
        mask: "torch.Tensor | None",
    ) -> "torch.Tensor":
        scale = 1.0 / math.sqrt(self.nope_dim + self.rope_dim)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        if mask is None:
            causal = torch.ones(scores.size(-2), scores.size(-1), device=scores.device, dtype=torch.bool).tril()
            mask = causal[None, None, :, :]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1).to(dtype=q.dtype)
        if self.training and self.dropout:
            probabilities = F.dropout(probabilities, p=self.dropout)
        return torch.matmul(probabilities, v)

    def forward(
        self,
        x: "torch.Tensor",
        *,
        past_key_value: tuple["torch.Tensor", "torch.Tensor"] | None = None,
        use_cache: bool = False,
    ) -> tuple["torch.Tensor", tuple["torch.Tensor", "torch.Tensor"] | None]:
        batch, sequence, _ = x.shape
        past_len = 0 if past_key_value is None else self._validate_cache(past_key_value, x=x)
        q = self.q_up(self.q_norm(self.q_down(x))).view(
            batch,
            sequence,
            self.config.num_attention_heads,
            self.nope_dim + self.rope_dim,
        ).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.nope_dim, self.rope_dim], dim=-1)

        compressed = self.kv_down(x)
        kv_latent, k_rope = torch.split(compressed, [self.kv_rank, self.rope_dim], dim=-1)
        kv_latent = self.kv_norm(kv_latent)
        k_rope_heads = k_rope[:, None, :, :]
        q_rope, k_rope_heads = self.rope(q_rope, k_rope_heads, position_offset=past_len)

        if past_key_value is not None:
            past_latent, past_rope = past_key_value
            kv_latent = torch.cat([past_latent, kv_latent], dim=1)
            k_rope_heads = torch.cat([past_rope, k_rope_heads], dim=-2)
        present = (kv_latent, k_rope_heads) if use_cache else None

        expanded = self.kv_up(kv_latent).view(
            batch,
            kv_latent.size(1),
            self.config.num_attention_heads,
            self.nope_dim + self.value_dim,
        ).transpose(1, 2)
        k_nope, values = torch.split(expanded, [self.nope_dim, self.value_dim], dim=-1)
        repeated_rope = k_rope_heads.expand(-1, self.config.num_attention_heads, -1, -1)
        keys = torch.cat([k_nope, repeated_rope], dim=-1)
        queries = torch.cat([q_nope, q_rope], dim=-1)
        mask = self._mask(
            query_len=sequence,
            key_len=keys.size(-2),
            past_len=past_len,
            device=x.device,
        )
        use_eager = self.config.attention_impl == "eager" or not hasattr(F, "scaled_dot_product_attention")
        if use_eager:
            attended = self._eager(queries, keys, values, mask=mask)
        else:
            attended = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=mask is None and past_len == 0,
            )
        attended = attended.transpose(1, 2).contiguous().view(
            batch,
            sequence,
            self.config.num_attention_heads * self.value_dim,
        )
        return self.o_proj(attended), present


class SwiGLU(nn.Module):  # type: ignore[misc]
    def __init__(self, config: ScratchDecoderConfig, *, intermediate_size: int | None = None) -> None:
        require_torch()
        super().__init__()
        width = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.down_proj = nn.Linear(width, config.hidden_size, bias=False)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DroplessMixtureOfExperts(nn.Module):  # type: ignore[misc]
    """Top-k routed MoE that never discards tokens.

    This reference dispatcher intentionally favors transparent correctness over
    throughput. Megatron-Core provides the grouped-GEMM/all-to-all production
    implementation. Selection bias is updated without contributing an
    auxiliary loss to the language-model objective.
    """

    def __init__(self, config: ScratchDecoderConfig) -> None:
        require_torch()
        super().__init__()
        if config.feed_forward_architecture != "moe":
            raise ValueError("DroplessMixtureOfExperts requires an MoE config")
        width = int(config.moe_intermediate_size or 0)
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.num_routed_experts, bias=False)
        self.routed_experts = nn.ModuleList(
            [SwiGLU(config, intermediate_size=width) for _ in range(config.num_routed_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [SwiGLU(config, intermediate_size=width) for _ in range(config.num_shared_experts)]
        )
        self.register_buffer("selection_bias", torch.zeros(config.num_routed_experts), persistent=True)
        self.last_metrics: dict[str, float] = {}

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", dict[str, float]]:
        original_shape = x.shape
        flat = x.reshape(-1, self.config.hidden_size)
        raw_logits = self.router(flat).float()
        router_probabilities = torch.sigmoid(raw_logits)
        selection_scores = router_probabilities + self.selection_bias[None, :]
        selected = torch.topk(selection_scores, k=self.config.experts_per_token, dim=-1).indices
        selected_probabilities = router_probabilities.gather(1, selected)
        weights = (
            selected_probabilities
            / selected_probabilities.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
        ).to(dtype=x.dtype)
        output = torch.zeros_like(flat)
        counts = torch.bincount(selected.reshape(-1), minlength=self.config.num_routed_experts)
        for expert_index, expert in enumerate(self.routed_experts):
            token_indices, slots = torch.where(selected == expert_index)
            if token_indices.numel() == 0:
                continue
            expert_output = expert(flat.index_select(0, token_indices))
            output.index_add_(
                0,
                token_indices,
                expert_output * weights[token_indices, slots, None],
            )
        for shared in self.shared_experts:
            output = output + shared(flat)

        expected = flat.size(0) * self.config.experts_per_token
        routed = int(counts.sum().item())
        if routed != expected:
            raise RuntimeError(f"MoE token routing invariant violated: routed={routed}, expected={expected}")
        mean_load = float(counts.float().mean().item())
        p99_load = float(torch.quantile(counts.float(), 0.99).item())
        load_ratio = p99_load / max(mean_load, 1.0)
        metrics = {
            "tokens": float(flat.size(0)),
            "assignments": float(routed),
            "dropped_tokens": 0.0,
            "mean_expert_load": mean_load,
            "p99_expert_load": p99_load,
            "p99_to_mean_load": load_ratio,
            "load_limit": self.config.router_load_limit,
            "load_within_limit": float(load_ratio <= self.config.router_load_limit),
        }
        self.last_metrics = metrics
        if self.training and self.config.router_bias_update_rate > 0:
            with torch.no_grad():
                direction = torch.sign(counts.float().mean() - counts.float())
                self.selection_bias.add_(direction * self.config.router_bias_update_rate)
                self.selection_bias.sub_(self.selection_bias.mean())
        return output.view(original_shape), metrics


class DecoderBlock(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        config: ScratchDecoderConfig,
        *,
        layer_index: int = 0,
        force_dense: bool = False,
    ) -> None:
        require_torch()
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attention = (
            MultiLatentAttention(config)
            if config.attention_architecture == "mla"
            else CausalSelfAttention(config)
        )
        self.post_attention_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.is_moe = (
            not force_dense
            and config.feed_forward_architecture == "moe"
            and layer_index >= config.dense_layer_count
        )
        self.mlp = DroplessMixtureOfExperts(config) if self.is_moe else SwiGLU(config)

    def forward(
        self,
        x: "torch.Tensor",
        *,
        past_key_value: tuple["torch.Tensor", "torch.Tensor"] | None = None,
        use_cache: bool = False,
    ) -> tuple[
        "torch.Tensor",
        tuple["torch.Tensor", "torch.Tensor"] | None,
        dict[str, float] | None,
    ]:
        attn, present = self.attention(self.input_norm(x), past_key_value=past_key_value, use_cache=use_cache)
        x = x + attn
        normalized = self.post_attention_norm(x)
        if self.is_moe:
            mlp_output, router_metrics = self.mlp(normalized)
        else:
            mlp_output = self.mlp(normalized)
            router_metrics = None
        x = x + mlp_output
        return x, present, router_metrics


class MultiTokenPredictionLayer(nn.Module):  # type: ignore[misc]
    """Training-only next-next-token prediction transformer.

    For position ``t`` the layer combines the main decoder state at ``t`` with
    the observed token embedding at ``t + 1`` and predicts the token at
    ``t + 2``. The prediction block is dense even when the main decoder is MoE;
    this keeps the auxiliary route deterministic and matches the canonical
    parameter accounting in ``ScratchDecoderConfig``.
    """

    def __init__(self, config: ScratchDecoderConfig) -> None:
        require_torch()
        super().__init__()
        self.hidden_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.next_embedding_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.fusion = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.block = DecoderBlock(config, layer_index=0, force_dense=True)
        self.output_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self,
        hidden_states: "torch.Tensor",
        next_token_embeddings: "torch.Tensor",
    ) -> "torch.Tensor":
        if hidden_states.shape != next_token_embeddings.shape:
            raise ValueError("MTP hidden states and next-token embeddings must have identical shapes")
        fused = self.fusion(
            torch.cat(
                [
                    self.hidden_norm(hidden_states),
                    self.next_embedding_norm(next_token_embeddings),
                ],
                dim=-1,
            )
        )
        predicted, present, router_metrics = self.block(fused, use_cache=False)
        if present is not None or router_metrics is not None:
            raise RuntimeError("MTP dense prediction block produced unexpected cache or router state")
        return self.output_norm(predicted)


class CraftlyDecoderLM(nn.Module):  # type: ignore[misc]
    def __init__(self, config: ScratchDecoderConfig) -> None:
        require_torch()
        super().__init__()
        if config.runtime_backend != "native_reference":
            raise RuntimeError(
                f"{config.name} requires runtime_backend={config.runtime_backend}; "
                "the native reference runtime must not materialize production-scale profiles"
            )
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(config, layer_index=index) for index in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp_head = (
            MultiTokenPredictionLayer(config)
            if config.mtp_num_layers
            else None
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)
        self.gradient_checkpointing = config.gradient_checkpointing

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: "torch.Tensor",
        labels: "torch.Tensor | None" = None,
        past_key_values: tuple[tuple["torch.Tensor", "torch.Tensor"], ...] | None = None,
        use_cache: bool | None = None,
    ) -> DecoderForwardOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, sequence]")
        if input_ids.size(0) < 1 or input_ids.size(1) < 1:
            raise ValueError("input_ids cannot contain an empty batch or sequence")
        if input_ids.dtype not in {torch.int32, torch.int64}:
            raise ValueError("input_ids must use an integer tensor dtype")
        if int(input_ids.min().item()) < 0 or int(input_ids.max().item()) >= self.config.vocab_size:
            raise ValueError("input_ids contain a token outside the configured vocabulary")
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if labels.dtype not in {torch.int32, torch.int64}:
                raise ValueError("labels must use an integer tensor dtype")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain exactly one cache tuple per decoder layer")
        past_len = 0 if not past_key_values else int(past_key_values[0][0].size(-2))
        if past_len + input_ids.size(1) > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds max_sequence_length")
        if labels is not None and past_key_values is not None:
            raise ValueError("labels cannot be used with past_key_values")
        active_use_cache = self.config.use_cache if use_cache is None else use_cache
        if self.gradient_checkpointing and self.training and active_use_cache:
            active_use_cache = False
        x = self.embed_tokens(input_ids)
        presents: list[tuple["torch.Tensor", "torch.Tensor"]] = []
        router_metrics: list[dict[str, float]] = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            if self.gradient_checkpointing and self.training and not active_use_cache:
                def custom_forward(hidden: "torch.Tensor", active_layer: DecoderBlock = layer) -> "torch.Tensor":
                    output, _present, _router = active_layer(hidden, use_cache=False)
                    return output

                x = torch.utils.checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
                present = None
            else:
                x, present, layer_router_metrics = layer(
                    x,
                    past_key_value=past,
                    use_cache=active_use_cache,
                )
                if layer_router_metrics is not None:
                    router_metrics.append(layer_router_metrics)
            if present is not None:
                presents.append(present)
        normalized = self.norm(x)
        logits = self.lm_head(normalized)
        if self.config.logits_soft_cap is not None:
            cap = self.config.logits_soft_cap
            logits = torch.tanh(logits / cap) * cap
        language_model_loss = None
        mtp_loss = None
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            language_model_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss = language_model_loss
            if self.mtp_head is not None and labels.size(1) >= 3:
                next_token_embeddings = self.embed_tokens(input_ids[:, 1:-1])
                mtp_hidden = self.mtp_head(normalized[:, :-2, :], next_token_embeddings)
                mtp_logits = self.lm_head(mtp_hidden)
                mtp_targets = labels[:, 2:].contiguous()
                mtp_loss = F.cross_entropy(
                    mtp_logits.reshape(-1, mtp_logits.size(-1)),
                    mtp_targets.reshape(-1),
                )
                loss = language_model_loss + self.config.mtp_loss_weight * mtp_loss
        return DecoderForwardOutput(
            logits=logits,
            loss=loss,
            past_key_values=tuple(presents) if presents else None,
            language_model_loss=language_model_loss,
            mtp_loss=mtp_loss,
            router_metrics=tuple(router_metrics),
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: "torch.Tensor",
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> "torch.Tensor":
        if max_new_tokens < 1:
            return input_ids
        self.eval()
        generated = input_ids
        past: tuple[tuple["torch.Tensor", "torch.Tensor"], ...] | None = None
        next_input = input_ids
        for _ in range(max_new_tokens):
            output = self(next_input, past_key_values=past, use_cache=True)
            past = output.past_key_values
            logits = output.logits[:, -1, :]
            if temperature and temperature > 0:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    values, _indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], torch.finfo(logits.dtype).min)
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            next_input = next_token
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return generated

    def export_checkpoint(self, output_dir: str | Path, *, dtype: str = "float32") -> Path:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(self.config.model_dump_json(indent=2), encoding="utf-8")
        save_trusted_checkpoint({"model": self.state_dict(), "config": self.config.model_dump(), "format": "craftly_decoder_v2", "dtype": dtype}, target / "model.pt")
        serving_contract = {
            "format": "craftly_decoder_v2",
            "scratch_only": True,
            "dtype": dtype,
            "architecture": {
                "decoder_only": True,
                "rms_norm": True,
                "swiglu": True,
                "rope": True,
                "attention_architecture": self.config.attention_architecture,
                "feed_forward_architecture": self.config.feed_forward_architecture,
                "gqa": self.config.attention_architecture == "gqa",
                "mla": self.config.attention_architecture == "mla",
                "moe": self.config.feed_forward_architecture == "moe",
                "mtp_num_layers": self.config.mtp_num_layers,
                "num_attention_heads": self.config.num_attention_heads,
                "num_key_value_heads": self.config.num_key_value_heads,
                "num_routed_experts": self.config.num_routed_experts,
                "num_shared_experts": self.config.num_shared_experts,
                "experts_per_token": self.config.experts_per_token,
                "hidden_size": self.config.hidden_size,
                "max_sequence_length": self.config.max_sequence_length,
                "effective_context_length": self.config.effective_context_length,
                "vocab_size": self.config.vocab_size,
            },
            "runtime_features": {
                "kv_cache": self.config.use_cache,
                "attention_impl": self.config.attention_impl,
                "attention_window": self.config.attention_window,
                "gradient_checkpointing": self.config.gradient_checkpointing,
                "logits_soft_cap": self.config.logits_soft_cap,
                "compressed_mla_cache": self.config.attention_architecture == "mla",
            },
            "serving_targets": {
                "native_craftly": "supported",
                "vllm": "requires Craftly model adapter or conversion to a Hugging Face compatible module",
                "tensorrt_llm": "requires a Craftly decoder weight conversion plugin",
            },
            "required_conversion_checks": [
                "tokenizer vocab size matches config.vocab_size",
                "tied embeddings preserved when tie_word_embeddings=true",
                "RoPE scaling and theta preserved",
                "attention projection and cache layout preserved",
                "MoE router and expert placement preserved",
                "KV-cache decode numerically matches native Craftly decode on a fixed prompt",
            ],
        }
        (target / "serving_compatibility.json").write_text(json.dumps(serving_contract, indent=2, sort_keys=True), encoding="utf-8")
        (target / "generation_config.json").write_text(
            json.dumps(
                {
                    "do_sample": False,
                    "temperature": 0.0,
                    "top_k": None,
                    "max_new_tokens": 256,
                    "eos_token_id": None,
                    "format": "craftly_generation_config_v1",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return target

