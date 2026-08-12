from __future__ import annotations

import importlib.util
import math
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "vendor" / "comfyui_custom_nodes" / "ComfyUI-ARP" / "ltx_video_only_patch.py"


class SparseGuideAttentionTests(unittest.TestCase):
    def test_video_only_pruning_removes_only_audio_block_modules(self) -> None:
        spec = importlib.util.spec_from_file_location("arp_video_only_patch_test", PATCH_PATH)
        assert spec is not None and spec.loader is not None
        patch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch_module)

        class Block:
            def __init__(self):
                self.attn1 = object()
                self.ff = object()
                for attribute in patch_module._LTXAV_AUDIO_BLOCK_ATTRIBUTES:
                    setattr(self, attribute, object())
                    setattr(self, f"{attribute}_comfy_model_dtype", torch.bfloat16)

        diffusion_model = types.SimpleNamespace(transformer_blocks=[Block(), Block()])
        shared_model = types.SimpleNamespace(
            diffusion_model=diffusion_model,
            memory_usage_factor=0.077,
        )

        class Patcher:
            def __init__(self):
                self.model = shared_model
                self.size = 123
                self.patches = {
                    "diffusion_model.transformer_blocks.0.attn1.weight": ["video"],
                    "diffusion_model.transformer_blocks.0.audio_ff.weight": ["audio"],
                }

            def clone(self):
                clone = Patcher.__new__(Patcher)
                clone.model = self.model
                clone.size = self.size
                clone.patches = self.patches.copy()
                return clone

        original = Patcher()
        pruned = patch_module.prune_ltxav_audio_transformer_blocks(original)

        self.assertIsNot(pruned, original)
        self.assertEqual(pruned.size, 0)
        self.assertEqual(
            pruned.model.memory_usage_factor,
            patch_module.DEFAULT_VIDEO_ONLY_MEMORY_USAGE_FACTOR,
        )
        self.assertIn("diffusion_model.transformer_blocks.0.attn1.weight", pruned.patches)
        self.assertNotIn("diffusion_model.transformer_blocks.0.audio_ff.weight", pruned.patches)
        for block in pruned.model.diffusion_model.transformer_blocks:
            self.assertTrue(hasattr(block, "attn1"))
            self.assertTrue(hasattr(block, "ff"))
            for attribute in patch_module._LTXAV_AUDIO_BLOCK_ATTRIBUTES:
                self.assertFalse(hasattr(block, attribute))
                self.assertFalse(hasattr(block, f"{attribute}_comfy_model_dtype"))
        # The cached/shared source model remains structurally intact for a
        # later non-ARP or audio-capable workflow.
        for block in diffusion_model.transformer_blocks:
            for attribute in patch_module._LTXAV_AUDIO_BLOCK_ATTRIBUTES:
                self.assertTrue(hasattr(block, attribute))
                self.assertTrue(hasattr(block, f"{attribute}_comfy_model_dtype"))

    def test_video_only_pruning_rejects_unknown_registered_audio_state(self) -> None:
        spec = importlib.util.spec_from_file_location("arp_video_only_guard_test", PATCH_PATH)
        assert spec is not None and spec.loader is not None
        patch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch_module)

        class Block:
            def __init__(self):
                self.attn1 = object()
                self.ff = object()
                self._modules = {"future_audio_branch": object()}
                self._parameters = {}
                self._buffers = {}
                for attribute in patch_module._LTXAV_AUDIO_BLOCK_ATTRIBUTES:
                    setattr(self, attribute, object())

        shared_model = types.SimpleNamespace(
            diffusion_model=types.SimpleNamespace(transformer_blocks=[Block()]),
            memory_usage_factor=0.077,
        )
        patcher = types.SimpleNamespace(
            model=shared_model,
            size=1,
            patches={},
            clone=lambda: types.SimpleNamespace(
                model=shared_model,
                size=1,
                patches={},
            ),
        )
        with self.assertRaisesRegex(
            patch_module.ARPLTXCompatibilityError, "future_audio_branch"
        ):
            patch_module.prune_ltxav_audio_transformer_blocks(patcher)

    def test_partitioned_xformers_preserves_exact_guide_mask_results(self) -> None:
        spec = importlib.util.spec_from_file_location("arp_guide_attention_patch_test", PATCH_PATH)
        assert spec is not None and spec.loader is not None
        patch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch_module)

        comfy = types.ModuleType("comfy")
        ldm = types.ModuleType("comfy.ldm")
        lightricks = types.ModuleType("comfy.ldm.lightricks")
        model = types.ModuleType("comfy.ldm.lightricks.model")

        def reference_attention(q, k, v, _heads, mask=None, **_kwargs):
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            if mask is not None:
                scores = scores + mask.squeeze(1)
            return torch.matmul(torch.softmax(scores, dim=-1), v)

        attention_calls: list[tuple[str, bool]] = []

        def unmasked_attention(*args, **kwargs):
            attention_calls.append(("unmasked", kwargs.get("mask") is not None))
            return reference_attention(*args, **kwargs)

        def masked_attention(*args, **kwargs):
            attention_calls.append(("masked", kwargs.get("mask") is not None))
            return reference_attention(*args, **kwargs)

        partition_calls: list[int] = []

        xformers = types.ModuleType("xformers")
        xformers_ops = types.ModuleType("xformers.ops")

        def partition_attention(q, k, v):
            partition_calls.append(k.shape[1])
            scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
            lse = torch.logsumexp(scores, dim=-1)
            out = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, dim=-1), v)
            return out, lse

        xformers_ops.memory_efficient_attention_forward_requires_grad = partition_attention
        xformers.ops = xformers_ops

        attention = types.SimpleNamespace(
            optimized_attention=unmasked_attention,
            attention_pytorch=masked_attention,
        )
        ldm.modules = types.SimpleNamespace(attention=attention)
        ldm.lightricks = lightricks
        lightricks.model = model
        comfy.ldm = ldm
        model.comfy = comfy
        class OriginalGuideAttentionMask:
            __slots__ = ("guide_start", "tracked_count", "noisy_mask", "tracked_mask")

            def __init__(self, total_tokens, guide_start, tracked_count, tracked_weights):
                flat = tracked_weights.reshape(-1)
                finfo = torch.finfo(flat.dtype)
                positive = flat > 0
                logs = torch.full_like(flat, finfo.min)
                logs[positive] = flat[positive].log()
                self.guide_start = guide_start
                self.tracked_count = tracked_count
                self.noisy_mask = torch.zeros((1, 1, 1, total_tokens), dtype=flat.dtype)
                self.noisy_mask[:, :, :, guide_start:guide_start + tracked_count] = logs.view(1, 1, 1, -1)
                self.tracked_mask = torch.zeros((1, 1, tracked_count, total_tokens), dtype=flat.dtype)
                self.tracked_mask[:, :, :, :guide_start] = logs.view(1, 1, -1, 1)

        model.GuideAttentionMask = OriginalGuideAttentionMask
        def original_attention(q, k, v, heads, guide_mask, attn_precision, transformer_options):
            return None

        model._attention_with_guide_mask = original_attention
        feed_forward_calls: list[int] = []

        class FakeQuantizedLinear:
            def __init__(self, weight, bias):
                self.weight = weight
                self.bias = bias
                self.cast_calls = 0

            def is_ggml_quantized(self):
                return True

            def cast_bias_weight(self, _input):
                self.cast_calls += 1
                return self.weight, self.bias

            def __call__(self, x):
                weight, bias = self.cast_bias_weight(x)
                return torch.nn.functional.linear(x, weight, bias)

        class GELU_approx:
            def __init__(self, projection):
                self.proj = projection

            def __call__(self, x):
                return torch.nn.functional.gelu(self.proj(x), approximate="tanh")

        class FakeFeedForward:
            def __init__(self, quantized=False):
                self.net = None
                if quantized:
                    weight_in = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4) / 100
                    bias_in = torch.arange(8, dtype=torch.float32) / 50
                    weight_out = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8) / 80
                    bias_out = torch.arange(4, dtype=torch.float32) / 40
                    self.net = [
                        GELU_approx(FakeQuantizedLinear(weight_in, bias_in)),
                        torch.nn.Identity(),
                        FakeQuantizedLinear(weight_out, bias_out),
                    ]

            def forward(self, x):
                if self.net is not None:
                    hidden = self.net[0](x)
                    hidden = self.net[1](hidden)
                    return self.net[2](hidden)
                feed_forward_calls.append(x.shape[-2])
                return x.square() + 0.25

        model.FeedForward = FakeFeedForward

        class FakeCrossAttention:
            def __init__(self):
                self.to_q = torch.nn.Identity()
                self.to_k = torch.nn.Identity()
                self.to_v = torch.nn.Identity()
                self.q_norm = torch.nn.Identity()
                self.k_norm = torch.nn.Identity()
                self.heads = 1
                self.attn_precision = None
                self.to_gate_logits = None
                self.dim_head = 4
                self.to_out = torch.nn.Identity()

            def forward(
                self, x, context=None, mask=None, pe=None, k_pe=None,
                transformer_options=None,
            ):
                active_context = x if context is None else context
                return reference_attention(x, active_context, active_context, 1, mask=mask)

        model.CrossAttention = FakeCrossAttention
        model.apply_rotary_emb_qk = lambda q, k, _pe: (q, k)
        model.apply_rotary_emb = lambda value, _pe: value

        class FakeBlock:
            def __init__(self, quantized=False):
                self.attn1 = FakeCrossAttention()
                self.ff = FakeFeedForward(quantized=quantized)
                for attribute in patch_module._LTXAV_AUDIO_BLOCK_ATTRIBUTES:
                    setattr(self, attribute, object())

        class FakeDiffusion:
            def __init__(self):
                self.transformer_blocks = [FakeBlock(quantized=True), FakeBlock()]

            def _build_guide_self_attention_mask(self, x, transformer_options, merged_args):
                weights = merged_args["weights"]
                guide_start = int(merged_args["guide_start"])
                return OriginalGuideAttentionMask(
                    x.shape[1], guide_start, weights.numel(), weights
                )

        shared_model = types.SimpleNamespace(
            diffusion_model=FakeDiffusion(),
            memory_usage_factor=0.077,
        )

        class Patcher:
            def __init__(self):
                self.model = shared_model
                self.size = 123
                self.patches = {}

            def clone(self):
                clone = Patcher.__new__(Patcher)
                clone.model = self.model
                clone.size = self.size
                clone.patches = self.patches.copy()
                return clone

        fake_modules = {
            "comfy": comfy,
            "comfy.ldm": ldm,
            "comfy.ldm.lightricks": lightricks,
            "comfy.ldm.lightricks.model": model,
            "xformers": xformers,
            "xformers.ops": xformers_ops,
        }
        with (
            mock.patch.dict(sys.modules, fake_modules),
            mock.patch.dict(os.environ, {"ARP_LTX_FF_CHUNK_TOKENS": "256"}),
        ):
            original_patcher = Patcher()
            quantized_ff = original_patcher.model.diffusion_model.transformer_blocks[0].ff
            quantized_input = torch.arange(600 * 4, dtype=torch.float32).reshape(1, 600, 4) / 100
            with torch.inference_mode():
                quantized_expected = quantized_ff.forward(quantized_input)
            quantized_ff.net[0].proj.cast_calls = 0
            quantized_ff.net[2].cast_calls = 0
            self.assertTrue(patch_module.install_sparse_guide_attention_patch())
            video_only = patch_module.prune_ltxav_audio_transformer_blocks(original_patcher)
            patched = patch_module.install_sparse_guide_attention_patch(video_only)

        patched_diffusion = patched.model.diffusion_model
        patched_quantized_ff = patched_diffusion.transformer_blocks[0].ff
        with torch.inference_mode():
            quantized_actual = patched_quantized_ff.forward(quantized_input)
        self.assertTrue(torch.equal(quantized_actual, quantized_expected))
        self.assertEqual(quantized_ff.net[0].proj.cast_calls, 1)
        self.assertEqual(quantized_ff.net[2].cast_calls, 1)
        self.assertIs(model.GuideAttentionMask, OriginalGuideAttentionMask)
        self.assertIs(model._attention_with_guide_mask, original_attention)
        self.assertFalse(hasattr(quantized_ff, "_arp_chunked_feed_forward"))

        ff_input = torch.arange(600 * 4, dtype=torch.float32).reshape(1, 600, 4) / 100
        with torch.inference_mode():
            ff_actual = patched_diffusion.transformer_blocks[1].ff.forward(ff_input)
        self.assertTrue(torch.equal(ff_actual, ff_input.square() + 0.25))
        self.assertEqual(feed_forward_calls, [256, 256, 88])

        weights = torch.tensor([[1.0, 1.0, 0.95, 0.0]], dtype=torch.float32)
        generator = torch.Generator().manual_seed(42)
        q = torch.randn((1, 8, 4), generator=generator)
        guide_mask = patched_diffusion._build_guide_self_attention_mask(
            q, {}, {"weights": weights, "guide_start": 3}
        )
        self.assertEqual(guide_mask.full_query_indices.tolist(), [3, 4])
        self.assertEqual(guide_mask.soft_query_indices.tolist(), [5, 6])
        self.assertEqual(guide_mask.soft_mask.numel(), 2 * 8)
        self.assertAlmostEqual(float(guide_mask.tracked_log_weights[2]), math.log(0.95), places=6)

        log_weights = torch.full_like(weights.reshape(-1), torch.finfo(torch.float32).min)
        positive = weights.reshape(-1) > 0
        log_weights[positive] = torch.log(weights.reshape(-1)[positive])
        dense_mask = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        dense_mask[:, :, :3, 3:7] = log_weights.view(1, 1, 1, -1)
        dense_mask[:, :, 3:7, :3] = log_weights.view(1, 1, -1, 1)

        expected = reference_attention(q, q, q, 1, mask=dense_mask)
        with mock.patch.dict(
            sys.modules,
            {"xformers": xformers, "xformers.ops": xformers_ops},
        ):
            actual = patched_diffusion.transformer_blocks[0].attn1.forward(
                q, mask=guide_mask
            )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        self.assertEqual(
            attention_calls,
            [("unmasked", False), ("unmasked", False)],
        )
        self.assertEqual(partition_calls, [5, 1, 1, 3, 5])

        # Arbitrary spatial masks can produce many short weight runs. They
        # retain the broadcast-safe PyTorch fallback instead of launching an
        # excessive number of xFormers partitions.
        attention_calls.clear()
        partition_calls.clear()
        spatial_weights = torch.tensor(
            [[0.5 if index % 2 else 0.75 for index in range(34)]],
            dtype=torch.float32,
        )
        q2 = torch.randn((1, 40, 4), generator=generator)
        spatial_mask = patched_diffusion._build_guide_self_attention_mask(
            q2, {}, {"weights": spatial_weights, "guide_start": 2}
        )
        self.assertGreater(
            len(spatial_mask.noisy_partitions),
            patch_module.MAX_PARTITIONED_ATTENTION_RUNS,
        )
        dense2 = torch.zeros((1, 1, 40, 40), dtype=torch.float32)
        log2 = spatial_weights.log()
        dense2[:, :, :2, 2:36] = log2.view(1, 1, 1, -1)
        dense2[:, :, 2:36, :2] = log2.view(1, 1, -1, 1)
        expected2 = reference_attention(q2, q2, q2, 1, mask=dense2)
        actual2 = patched_diffusion.transformer_blocks[0].attn1.forward(
            q2, mask=spatial_mask
        )
        self.assertTrue(torch.allclose(actual2, expected2, atol=1e-6, rtol=1e-6))
        self.assertEqual(
            attention_calls,
            [("masked", True), ("masked", True), ("unmasked", False)],
        )
        self.assertEqual(partition_calls, [])


if __name__ == "__main__":
    unittest.main()
