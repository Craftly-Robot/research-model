from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from src.craftly.model_ops.gpu_smoke import run_scratch_gpu_smoke
from src.craftly.model_ops.foundation import ScratchDecoderConfig
from src.craftly.model_ops.torch_decoder import CraftlyDecoderLM, model_profile, require_torch, tiny_smoke_config

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "torch is not installed")
class CraftlyScratchDecoderTest(unittest.TestCase):
    def test_tiny_decoder_forward_backward(self) -> None:
        require_torch()
        config = tiny_smoke_config()
        model = CraftlyDecoderLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        output = model(input_ids, labels=input_ids)
        self.assertEqual(tuple(output.logits.shape), (2, 16, config.vocab_size))
        self.assertIsNotNone(output.loss)
        output.loss.backward()
        grad_count = sum(1 for parameter in model.parameters() if parameter.grad is not None)
        self.assertGreater(grad_count, 0)

    def test_decoder_kv_cache_generation_eager_and_export(self) -> None:
        require_torch()
        config = tiny_smoke_config().model_copy(update={"attention_impl": "eager", "max_sequence_length": 64})
        model = CraftlyDecoderLM(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        first = model(input_ids[:, :4], use_cache=True)
        self.assertIsNotNone(first.past_key_values)
        second = model(input_ids[:, 4:5], past_key_values=first.past_key_values, use_cache=True)
        self.assertEqual(tuple(second.logits.shape), (1, 1, config.vocab_size))
        generated = model.generate(input_ids[:, :3], max_new_tokens=2)
        self.assertEqual(tuple(generated.shape), (1, 5))
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = model.export_checkpoint(temp_dir)
            self.assertTrue(Path(export_dir, "model.pt").exists())
            serving = json.loads(Path(export_dir, "serving_compatibility.json").read_text(encoding="utf-8"))
            self.assertEqual(serving["format"], "craftly_decoder_v2")
            self.assertEqual(serving["serving_targets"]["native_craftly"], "supported")
            self.assertTrue(serving["runtime_features"]["kv_cache"])
            self.assertTrue(Path(export_dir, "generation_config.json").exists())

    def test_large_profile_contracts_are_shape_valid_without_instantiation(self) -> None:
        profile = model_profile("62b")
        self.assertEqual(profile.name, "craftly-62b-scratch")
        self.assertEqual(profile.max_sequence_length, 262_144)
        self.assertEqual(profile.attention_impl, "auto")
        self.assertTrue(profile.gradient_checkpointing)
        self.assertGreater(profile.parameter_estimate(), 50_000_000_000)

    def test_4t_profile_parameter_accounting_and_materialization_guard(self) -> None:
        profile = model_profile("4t_moe")
        report = profile.parameter_report()
        self.assertTrue(report["total_target_passed"], report)
        self.assertTrue(report["active_target_passed"], report)
        self.assertLessEqual(abs(report["total_target_relative_delta"]), 0.005)
        self.assertLessEqual(abs(report["active_target_relative_delta"]), 0.05)
        self.assertEqual(profile.runtime_backend, "megatron_core")
        with self.assertRaisesRegex(RuntimeError, "must not materialize"):
            CraftlyDecoderLM(profile)
        topology = profile.distributed_topology_report(
            tensor_parallel=8,
            pipeline_parallel=12,
            data_parallel=32,
            context_parallel=8,
            expert_parallel=32,
            gpus_per_node=8,
        )
        self.assertTrue(topology["passed"], topology)
        self.assertFalse(topology["cluster_proven"])
        self.assertEqual(topology["routed_experts_per_expert_rank"], 8)
        self.assertEqual(topology["world_size"], 24_576)

    def test_1b_dense_moe_ab_profiles_have_iso_active_compute(self) -> None:
        dense = model_profile("1b").parameter_report()
        sparse = model_profile("1b_moe").parameter_report()
        relative_delta = abs(sparse["active"] - dense["active"]) / dense["active"]
        self.assertLessEqual(relative_delta, 0.05)
        self.assertGreater(sparse["total"], dense["total"] * 3)

    def test_mla_moe_forward_backward_cache_parity_and_zero_drop(self) -> None:
        config = model_profile("tiny_moe").model_copy(
            update={"attention_impl": "eager", "router_load_limit": 4.0}
        )
        model = CraftlyDecoderLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 12))
        output = model(input_ids, labels=input_ids, use_cache=False)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(torch.isfinite(output.mtp_loss))
        self.assertEqual(len(output.router_metrics), config.moe_layer_count)
        self.assertTrue(all(metric["dropped_tokens"] == 0 for metric in output.router_metrics))
        self.assertTrue(
            all(
                metric["assignments"] == metric["tokens"] * config.experts_per_token
                for metric in output.router_metrics
            )
        )
        output.loss.backward()
        self.assertIsNotNone(model.mtp_head.fusion.weight.grad)
        self.assertIsNotNone(model.mtp_head.block.attention.q_down.weight.grad)
        self.assertTrue(any(expert.gate_proj.weight.grad is not None for expert in model.layers[-1].mlp.routed_experts))

        model.eval()
        prefix = input_ids[:1, :5]
        continuation = input_ids[:1, 5:6]
        with torch.no_grad():
            full = model(torch.cat([prefix, continuation], dim=1), use_cache=False)
            cached = model(prefix, use_cache=True)
            decoded = model(continuation, past_key_values=cached.past_key_values, use_cache=True)
        self.assertTrue(
            torch.allclose(full.logits[:, -1], decoded.logits[:, -1], atol=2e-5, rtol=2e-4)
        )
        latent, rotary = cached.past_key_values[0]
        self.assertEqual(latent.size(-1), config.kv_lora_rank)
        self.assertEqual(rotary.size(-1), config.qk_rope_head_dim)

        actual_parameters = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(actual_parameters, config.parameter_report()["total"])

    def test_mla_cache_and_input_contracts_fail_closed(self) -> None:
        config = model_profile("tiny_moe").model_copy(update={"attention_impl": "eager"})
        model = CraftlyDecoderLM(config).eval()
        valid = torch.randint(0, config.vocab_size, (1, 4))
        cached = model(valid, use_cache=True)
        self.assertIsNotNone(cached.past_key_values)

        malformed = list(cached.past_key_values)
        latent, rotary = malformed[0]
        malformed[0] = (latent[:, :, :-1], rotary)
        with self.assertRaisesRegex(ValueError, "latent cache"):
            model(valid[:, :1], past_key_values=tuple(malformed), use_cache=True)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model(valid[:, :1], past_key_values=cached.past_key_values[:-1], use_cache=True)
        with self.assertRaisesRegex(ValueError, "outside"):
            model(torch.tensor([[config.vocab_size]], dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "same shape"):
            model(valid, labels=valid[:, :-1])

    def test_invalid_moe_contract_rejects_token_dropping(self) -> None:
        with self.assertRaises(ValueError):
            ScratchDecoderConfig(
                feed_forward_architecture="moe",
                num_layers=2,
                num_dense_layers=1,
                num_routed_experts=4,
                num_shared_experts=1,
                experts_per_token=2,
                moe_intermediate_size=128,
                router_drop_tokens=True,
            )

    def test_scratch_gpu_smoke_runs_on_cpu_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_scratch_gpu_smoke(
                device="cpu",
                output_dir=temp_dir,
                batch_size=1,
                sequence_length=16,
                steps=1,
                dtype="fp32",
            )
            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["scratch_only"])
            self.assertFalse(report["borrowed_model_used"])
            self.assertTrue(Path(report["checkpoint_manifest"]).exists())
            self.assertTrue(Path(report["checkpoint_dir"], "model.pt").exists())


if __name__ == "__main__":
    unittest.main()

