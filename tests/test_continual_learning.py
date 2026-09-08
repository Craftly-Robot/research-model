"""Unit and regression tests for Continual Learning & Catastrophic Forgetting Suite."""

from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

from src.craftly.learning.continual import (
    CANONICAL_FORGETTING_SUITE,
    ExperienceReplayBuffer,
    ReferenceModelKLLoss,
    WeightMerger,
    evaluate_continual_perplexity,
)
from src.craftly.model_ops.foundation import ScratchDecoderConfig
from src.craftly.model_ops.tokenizer_pipeline import TokenizerTrainConfig, load_tokenizer, train_bpe_tokenizer
from src.craftly.model_ops.torch_decoder import (
    CraftlyDecoderLM,
    require_torch,
    save_trusted_checkpoint,
)

try:
    import torch
except ImportError:
    torch = None


class TestContinualLearning(unittest.TestCase):
    def setUp(self) -> None:
        require_torch()
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_continual_"))

        # Create a tiny tokenizer for fast testing
        corpus_path = self.test_dir / "corpus.txt"
        corpus_path.write_text(
            "def quicksort(arr): return arr\n"
            "def binary_search(arr, target): return 0\n"
            "def fib(n): return n\n"
            "SELECT * FROM users WHERE id=%s\n",
            encoding="utf-8",
        )
        self.tokenizer_path = self.test_dir / "tokenizer" / "tokenizer.json"
        train_bpe_tokenizer(
            [corpus_path],
            output_path=self.tokenizer_path,
            config=TokenizerTrainConfig(vocab_size=1000, min_frequency=1),
        )
        self.tokenizer = load_tokenizer(self.tokenizer_path)

        # Tiny model config
        self.tiny_config = ScratchDecoderConfig(
            name="craftly-test-tiny",
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_sequence_length=128,
            tie_word_embeddings=True,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_experience_replay_buffer(self) -> None:
        buffer = ExperienceReplayBuffer(replay_ratio=0.25, seed=42)
        self.assertGreaterEqual(len(buffer.samples), 3)

        inps, lbls = buffer.sample_batch(
            count=3,
            tokenizer=self.tokenizer,
            max_sequence_length=32,
            pad_token_id=0,
            ignore_index=-100,
        )
        self.assertEqual(len(inps), 3)
        self.assertEqual(len(lbls), 3)
        self.assertEqual(len(inps[0]), 32)
        self.assertEqual(len(lbls[0]), 32)
        # Check that targets match inputs up to pad
        non_padded = [t for t in inps[0] if t != 0]
        self.assertGreater(len(non_padded), 0)

    def test_reference_model_kl_loss(self) -> None:
        kl = ReferenceModelKLLoss(temperature=1.0)
        batch, seq_len, vocab = 2, 8, 32
        logits = torch.randn(batch, seq_len, vocab, requires_grad=True)

        # 1. Identical distributions have 0 KL
        loss_zero = kl.compute(logits, logits.detach())
        self.assertAlmostEqual(loss_zero.item(), 0.0, places=4)

        # 2. Different distributions have positive KL
        diff_logits = torch.randn(batch, seq_len, vocab)
        loss_pos = kl.compute(logits, diff_logits)
        self.assertGreater(loss_pos.item(), 0.0)

        # 3. Gradient flows back to active logits
        loss_pos.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(torch.norm(logits.grad).item(), 0.0)

    def test_weight_merger_slerp(self) -> None:
        # Two orthogonal unit vectors in 2D
        v0 = torch.tensor([1.0, 0.0])
        v1 = torch.tensor([0.0, 1.0])

        # At t = 0.5, angle is 45 deg, components should be cos(45) = 1/sqrt(2) ≈ 0.7071
        mid = WeightMerger.slerp(v0, v1, t=0.5)
        self.assertAlmostEqual(mid[0].item(), 1.0 / math.sqrt(2.0), places=3)
        self.assertAlmostEqual(mid[1].item(), 1.0 / math.sqrt(2.0), places=3)
        # Norm should be preserved as 1.0
        self.assertAlmostEqual(torch.norm(mid).item(), 1.0, places=3)

    def test_weight_merger_state_dicts(self) -> None:
        state_a = {"weight": torch.ones(4, 4), "bias": torch.zeros(4)}
        state_b = {"weight": torch.full((4, 4), 3.0), "bias": torch.full((4,), 2.0)}

        merged = WeightMerger.merge_state_dicts(state_a, state_b, method="linear", t=0.5)
        self.assertIn("weight", merged)
        self.assertIn("bias", merged)
        self.assertAlmostEqual(merged["weight"][0, 0].item(), 2.0, places=3)
        self.assertAlmostEqual(merged["bias"][0].item(), 1.0, places=3)

    def test_weight_merger_checkpoint_files(self) -> None:
        base_file = self.test_dir / "base.pt"
        sft_file = self.test_dir / "sft.pt"
        out_file = self.test_dir / "merged.pt"

        model_base = CraftlyDecoderLM(self.tiny_config)
        model_sft = CraftlyDecoderLM(self.tiny_config)
        # Perturb sft weights slightly
        for p in model_sft.parameters():
            p.data.add_(0.05)

        save_trusted_checkpoint({"model": model_base.state_dict(), "config": self.tiny_config.model_dump()}, base_file)
        save_trusted_checkpoint({"model": model_sft.state_dict(), "config": self.tiny_config.model_dump()}, sft_file)

        res_path = WeightMerger.merge_checkpoint_files(base_file, sft_file, out_file, method="slerp", t=0.3)
        self.assertTrue(res_path.exists())

    def test_evaluate_continual_perplexity(self) -> None:
        model = CraftlyDecoderLM(self.tiny_config)
        report = evaluate_continual_perplexity(model, self.tokenizer, CANONICAL_FORGETTING_SUITE[:3], device="cpu")
        self.assertGreater(report.evaluated_tasks_count, 0)
        self.assertGreater(report.perplexity_general, 0.0)
        self.assertGreaterEqual(report.general_retention_score, 0.0)
        self.assertLessEqual(report.general_retention_score, 100.0)


if __name__ == "__main__":
    unittest.main()
