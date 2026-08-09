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
PATCH_PATH = ROOT / "vendor" / "comfyui_custom_nodes" / "ComfyUI-LTXVideo" / "guide_attention_patch.py"


class SparseGuideAttentionTests(unittest.TestCase):
    def test_full_strength_rows_are_unmasked_without_changing_results(self) -> None:
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

        attention = types.SimpleNamespace(optimized_attention=reference_attention)
        ldm.modules = types.SimpleNamespace(attention=attention)
        ldm.lightricks = lightricks
        lightricks.model = model
        comfy.ldm = ldm
        model.comfy = comfy
        model.GuideAttentionMask = type("OriginalGuideAttentionMask", (), {})
        model._attention_with_guide_mask = lambda *_args, **_kwargs: None
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

        class FakeGelu:
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
                        FakeGelu(FakeQuantizedLinear(weight_in, bias_in)),
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

        fake_modules = {
            "comfy": comfy,
            "comfy.ldm": ldm,
            "comfy.ldm.lightricks": lightricks,
            "comfy.ldm.lightricks.model": model,
        }
        with (
            mock.patch.dict(sys.modules, fake_modules),
            mock.patch.dict(os.environ, {"ARP_LTX_FF_CHUNK_TOKENS": "256"}),
        ):
            quantized_ff = model.FeedForward(quantized=True)
            quantized_input = torch.arange(600 * 4, dtype=torch.float32).reshape(1, 600, 4) / 100
            with torch.inference_mode():
                quantized_expected = quantized_ff.forward(quantized_input)
            quantized_ff.net[0].proj.cast_calls = 0
            quantized_ff.net[2].cast_calls = 0
            self.assertTrue(patch_module.install_sparse_guide_attention_patch())

        with torch.inference_mode():
            quantized_actual = quantized_ff.forward(quantized_input)
        self.assertTrue(torch.equal(quantized_actual, quantized_expected))
        self.assertEqual(quantized_ff.net[0].proj.cast_calls, 1)
        self.assertEqual(quantized_ff.net[2].cast_calls, 1)

        ff_input = torch.arange(600 * 4, dtype=torch.float32).reshape(1, 600, 4) / 100
        with torch.inference_mode():
            ff_actual = model.FeedForward().forward(ff_input)
        self.assertTrue(torch.equal(ff_actual, ff_input.square() + 0.25))
        self.assertEqual(feed_forward_calls, [256, 256, 88])

        weights = torch.tensor([[1.0, 1.0, 0.5, 0.0]], dtype=torch.float32)
        guide_mask = model.GuideAttentionMask(8, 3, 4, weights)
        self.assertEqual(guide_mask.full_query_indices.tolist(), [3, 4])
        self.assertEqual(guide_mask.soft_query_indices.tolist(), [5, 6])
        self.assertEqual(guide_mask.soft_mask.numel(), 2 * 8)

        generator = torch.Generator().manual_seed(42)
        q = torch.randn((1, 8, 4), generator=generator)
        k = torch.randn((1, 8, 4), generator=generator)
        v = torch.randn((1, 8, 4), generator=generator)

        log_weights = torch.full_like(weights.reshape(-1), torch.finfo(torch.float32).min)
        positive = weights.reshape(-1) > 0
        log_weights[positive] = torch.log(weights.reshape(-1)[positive])
        dense_mask = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        dense_mask[:, :, :3, 3:7] = log_weights.view(1, 1, 1, -1)
        dense_mask[:, :, 3:7, :3] = log_weights.view(1, 1, -1, 1)

        expected = reference_attention(q, k, v, 1, mask=dense_mask)
        actual = model._attention_with_guide_mask(q, k, v, 1, guide_mask, None, {})
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
