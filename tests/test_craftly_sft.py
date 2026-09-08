"""Unit tests for Craftly Phase 2 Defensive SFT pipeline."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.craftly.learning.sft_dataset import SFTDatasetEngine, SFTRecord, encode_sft_record
from src.craftly.model_ops.foundation import ScratchDecoderConfig, tiny_smoke_config
from src.craftly.model_ops.sft_loop import SFTRunConfig, run_sft
from src.craftly.model_ops.tokenizer_pipeline import TokenizerTrainConfig, train_bpe_tokenizer
from src.craftly.model_ops.torch_decoder import CraftlyDecoderLM, require_torch, save_trusted_checkpoint

try:
    import torch
except ImportError:
    torch = None


class TestCraftlySFT(unittest.TestCase):
    """Test suite covering SFT dataset, loss masking, and training loop execution."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="craftly_sft_test_"))
        self.corpus_path = self.temp_dir / "corpus.txt"
        self.corpus_path.write_text(
            "def sanitize(val): return val.strip()\n"
            "User:\nFix SQL injection\nAssistant:\n<|thought_start|>\nAnalyze\n<|thought_end|>\n<|patch_start|>\nquery\n<|patch_end|><|document_end|>\n",
            encoding="utf-8",
        )
        self.tokenizer_path = self.temp_dir / "tokenizer" / "tokenizer.json"
        train_bpe_tokenizer(
            [self.corpus_path],
            output_path=self.tokenizer_path,
            config=TokenizerTrainConfig(vocab_size=1000, min_frequency=1),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sft_record_rendering(self) -> None:
        record = SFTRecord(
            record_id="test-cwe-89",
            category="sql_injection",
            cwe_id="CWE-89",
            instruction="Fix SQL injection in query",
            thought="Use parameterized queries",
            response="Use query parameters: cursor.execute(q, (user,))",
        )

        prompt = record.render_prompt()
        self.assertIn("System:\nYou are Craftly", prompt)
        self.assertIn("User:\nFix SQL injection", prompt)
        self.assertTrue(prompt.endswith("Assistant:\n"))

        completion = record.render_completion()
        self.assertIn("<|thought_start|>\nUse parameterized queries\n<|thought_end|>", completion)
        self.assertIn("cursor.execute", completion)
        self.assertTrue(completion.endswith("<|document_end|>"))

    def test_encode_sft_record_prompt_masking(self) -> None:
        from src.craftly.model_ops.tokenizer_pipeline import load_tokenizer

        tokenizer = load_tokenizer(self.tokenizer_path)
        record = SFTRecord(
            record_id="test-masking",
            category="test",
            instruction="Hello world",
            response="Defensive response here",
        )

        max_len = 256
        input_ids, labels, attention_mask = encode_sft_record(
            record,
            tokenizer,
            max_sequence_length=max_len,
            ignore_index=-100,
        )

        self.assertEqual(len(input_ids), max_len)
        self.assertEqual(len(labels), max_len)
        self.assertEqual(len(attention_mask), max_len)

        # Check that prompt tokens are masked with -100
        prompt_len = len(tokenizer.encode(record.render_prompt()).ids)
        for i in range(min(prompt_len, max_len)):
            self.assertEqual(labels[i], -100, f"Prompt token at {i} was not masked with -100")

        # Check that completion tokens have valid targets
        completion_ids = tokenizer.encode(record.render_completion()).ids
        for j in range(prompt_len, min(prompt_len + len(completion_ids), max_len)):
            self.assertNotEqual(labels[j], -100, f"Completion token at {j} was incorrectly masked")

    def test_sft_dataset_engine_expansion_and_partition(self) -> None:
        engine = SFTDatasetEngine()
        self.assertGreaterEqual(len(engine.records), 5)

        expanded = engine.expand_dataset(target_count=20, seed=42)
        self.assertEqual(len(expanded), 20)

        train, val = engine.partition(val_ratio=0.2, seed=42)
        self.assertEqual(len(train) + len(val), 20)
        self.assertGreater(len(train), len(val))

        jsonl_path = self.temp_dir / "test_sft.jsonl"
        engine.save_jsonl(jsonl_path)
        self.assertTrue(jsonl_path.exists())

        loaded_engine = SFTDatasetEngine.load_jsonl(jsonl_path)
        self.assertEqual(len(loaded_engine.records), 20)

    def test_run_sft_smoke_execution(self) -> None:
        if torch is None:
            self.skipTest("PyTorch is not available")

        # 1. Create a tiny base checkpoint
        config = tiny_smoke_config()
        model = CraftlyDecoderLM(config)
        base_dir = self.temp_dir / "base_model"
        base_dir.mkdir(parents=True, exist_ok=True)
        save_trusted_checkpoint(
            {
                "model": model.state_dict(),
                "config": config.model_dump(),
                "step": 0,
                "trained_tokens": 0,
            },
            base_dir / "model.pt",
        )
        (base_dir / "config.json").write_text(json.dumps(config.model_dump(), indent=2), encoding="utf-8")

        # 2. Run small 2-step SFT
        sft_out = self.temp_dir / "sft_out"
        run_cfg = SFTRunConfig(
            base_checkpoint=str(base_dir / "model.pt"),
            tokenizer_path=str(self.tokenizer_path),
            output_dir=str(sft_out),
            steps=2,
            batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            warmup_steps=0,
            validate_every=1,
            checkpoint_every=2,
            max_sequence_length=128,
            dtype="fp32",
        )

        summary = run_sft(run_cfg)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["steps_completed"], 2)
        self.assertTrue(Path(summary["checkpoint_manifest"]).exists())
        self.assertTrue((Path(summary["latest_checkpoint_dir"]) / "model.pt").exists())


if __name__ == "__main__":
    unittest.main()
