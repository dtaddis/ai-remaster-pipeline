from __future__ import annotations

import importlib.util
import math
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

        fake_modules = {
            "comfy": comfy,
            "comfy.ldm": ldm,
            "comfy.ldm.lightricks": lightricks,
            "comfy.ldm.lightricks.model": model,
        }
        with mock.patch.dict(sys.modules, fake_modules):
            self.assertTrue(patch_module.install_sparse_guide_attention_patch())

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
