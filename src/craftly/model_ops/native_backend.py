"""Native in-memory Craftly model backend for agentic reasoning and execution.

Provides an authoritative ModelBackend implementation that directly executes
Craftly scratch foundation and SFT checkpoints (model.pt + tokenizer.json)
without requiring an external HTTP service or fallback to mock models.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from src.craftly.model_ops.backends import ModelBackend
from src.craftly.model_ops.checkpoint_compare import GenerationConfig, generate_text
from src.craftly.model_ops.sft_loop import resolve_base_model_assets
from src.craftly.model_ops.tokenizer_pipeline import load_tokenizer
from src.craftly.model_ops.torch_decoder import (
    CraftlyDecoderLM,
    ScratchDecoderConfig,
    require_torch,
    select_torch_device,
)


class NativeCraftlyModelBackend(ModelBackend):
    """Authoritative direct PyTorch execution backend for Craftly models."""

    name: str = "native_craftly"

    def __init__(
        self,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        require_torch()

        self.device = select_torch_device(device)
        self.raw_checkpoint_path = Path(checkpoint_path).resolve()

        # 1. Resolve weights and config
        model_file, state_dict, config_dict = resolve_base_model_assets(
            self.raw_checkpoint_path
        )
        self.config = ScratchDecoderConfig(**config_dict)

        # 2. Instantiate and load weights
        self.model = CraftlyDecoderLM(self.config).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 3. Resolve and load tokenizer
        if tokenizer_path:
            tok_p = Path(tokenizer_path).resolve()
        else:
            # Check sibling or extracted directory
            parent = model_file.parent
            if (parent / "tokenizer.json").exists():
                tok_p = parent / "tokenizer.json"
            else:
                tok_p = Path(
                    "artifacts/craftly/train_run/tokenizer/tokenizer.json"
                ).resolve()

        if not tok_p.exists():
            raise FileNotFoundError(f"Tokenizer not found at: {tok_p}")

        self.tokenizer = load_tokenizer(tok_p)
        self.tokenizer_path = tok_p

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_k: int = 20,
        stop_tokens: list[str] | None = None,
    ) -> str:
        """Run token generation asynchronously in an executor thread."""
        stops = stop_tokens or ["<|tool_end|>", "<|patch_end|>", "<|thought_end|>"]
        cfg = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=stops,
        )

        def _sync_generate() -> str:
            output_text, _ = generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                device=self.device,
                config=cfg,
            )
            return output_text

        return await asyncio.to_thread(_sync_generate)

    async def identity(self) -> dict[str, Any]:
        """Return backend metadata and model provenance."""
        return {
            "backend": self.name,
            "architecture": self.config.name,
            "device": str(self.device),
            "layers": self.config.num_layers,
            "hidden_size": self.config.hidden_size,
            "vocab_size": self.config.vocab_size,
            "checkpoint": str(self.raw_checkpoint_path),
            "tokenizer": str(self.tokenizer_path),
        }
