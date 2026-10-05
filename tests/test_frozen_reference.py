from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "vendor" / "comfyui_custom_nodes" / "ComfyUI-ARP" / "ltx_video_only_patch.py"


def load_patch_module(name: str):
    spec = importlib.util.spec_from_file_location(name, PATCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sdpa_attention(q, k, v, heads, mask=None, attn_precision=None, transformer_options=None, **_kwargs):
    batch, tokens, width = q.shape
    dim = width // heads
    split = lambda t: t.view(batch, t.shape[1], heads, dim).transpose(1, 2)  # noqa: E731
    out = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=mask)
    return out.transpose(1, 2).reshape(batch, tokens, width)


def comfy_attention_modules():
    attention = types.ModuleType("comfy.ldm.modules.attention")
    attention.optimized_attention = sdpa_attention
    modules = {name: types.ModuleType(name) for name in ("comfy", "comfy.ldm", "comfy.ldm.modules")}
    modules["comfy.ldm.modules.attention"] = attention
    modules["comfy"].ldm = modules["comfy.ldm"]
    modules["comfy.ldm"].modules = modules["comfy.ldm.modules"]
    modules["comfy.ldm.modules"].attention = attention
    return modules


class FakeAttention(torch.nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = width // heads
        self.attn_precision = None
        self.to_q = torch.nn.Linear(width, width)
        self.to_k = torch.nn.Linear(width, width)
        self.to_v = torch.nn.Linear(width, width)
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()
        self.to_gate_logits = torch.nn.Linear(width, heads)
        self.to_out = torch.nn.Linear(width, width)


class FrozenReferenceAttentionTests(unittest.TestCase):
    def run_attention(self, patch, attention, x, call):
        with mock.patch.dict(sys.modules, comfy_attention_modules()):
            return patch._frozen_reference_self_attention(
                attention, x, None, None, {}, call, 0, types.SimpleNamespace(), torch
            )

    def check_replay_matches_record(self, storage: str, tolerance: float) -> None:
        patch = load_patch_module(f"arp_frozen_reference_{storage}_test")
        torch.manual_seed(0)
        attention = FakeAttention(width=32, heads=4)
        live, reference = 20, 12
        x = torch.randn(1, live + reference, 32)
        cache = patch._FrozenReferenceCache("key", 1, storage, 1.0)

        with torch.no_grad():
            recorded = self.run_attention(
                patch, attention, x, patch._FrozenCall("record", live, reference, live + reference, cache)
            )
            replayed = self.run_attention(
                patch, attention, x[:, :live], patch._FrozenCall("replay", live, reference, live, cache)
            )
            # The reference queries see only reference keys: their result equals running
            # the same attention on the reference tokens alone.
            reference_alone = self.run_attention(
                patch, attention, x[:, live:], patch._FrozenCall("record", 0, reference, reference, None)
            )

        self.assertTrue(cache.filled())
        torch.testing.assert_close(replayed, recorded[:, :live], atol=tolerance, rtol=0)
        torch.testing.assert_close(recorded[:, live:], reference_alone, atol=1e-6, rtol=0)

    def test_bf16_replay_matches_record_for_live_tokens(self) -> None:
        self.check_replay_matches_record("bf16", 2e-2)

    def test_int8_replay_matches_record_for_live_tokens(self) -> None:
        self.check_replay_matches_record("int8", 2e-2)

    def test_token_count_mismatch_fails_loudly(self) -> None:
        patch = load_patch_module("arp_frozen_reference_mismatch_test")
        attention = FakeAttention(width=16, heads=2)
        with self.assertRaisesRegex(RuntimeError, "expected 10 tokens"):
            self.run_attention(patch, attention, torch.randn(1, 12, 16), patch._FrozenCall("replay", 4, 2, 10, None))


class FrozenReferencePlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.patch = load_patch_module("arp_frozen_reference_plan_test")
        self.model = types.SimpleNamespace(patchifier=types.SimpleNamespace(patch_size=(1, 1, 1)))
        # 3 generated frames + 2 reference frames of 2x3 latent tokens.
        self.vx = torch.randn(1, 4, 5, 2, 3)
        self.keyframes = torch.zeros(1, 3, 12, 2)
        self.mask = torch.ones(1, 1, 5, 2, 3)
        self.mask[:, :, 3:] = 0.0

    def plan(self, mask, entries=()):
        return self.patch._plan_frozen_reference(
            torch, self.model, [self.vx], self.keyframes, mask, {"guide_attention_entries": list(entries)}
        )

    def test_clean_reference_frames_are_frozen(self) -> None:
        plan = self.plan(self.mask)
        self.assertEqual((plan["frame_start"], plan["frame_end"]), (3, 5))
        self.assertEqual((plan["start"], plan["count"], plan["total"]), (18, 12, 30))

    def test_downscaled_reference_holes_are_not_counted(self) -> None:
        mask = self.mask.clone()
        mask[:, :, 3:, :, 1:] = -1.0  # dilated half-res reference: holes are dropped tokens
        plan = self.plan(mask)
        self.assertEqual((plan["start"], plan["count"], plan["total"]), (18, 4, 22))

    def test_soft_guides_and_attention_strengths_disable_freezing(self) -> None:
        soft = self.mask.clone()
        soft[:, :, 3:] = 0.3
        self.assertIsNone(self.plan(soft))
        self.assertIsNone(self.plan(self.mask, [{"strength": 0.5, "pre_filter_count": 12}]))

    def test_cache_key_ignores_dropped_reference_holes(self) -> None:
        mask = self.mask.clone()
        mask[:, :, 3:, :, 1:] = -1.0
        key = self.plan(mask)["key"]
        self.vx[:, :, 3:, :, 1:] += torch.randn(1, 4, 2, 2, 2)  # sampler noise in dropped holes
        self.assertEqual(key, self.plan(mask)["key"])
        self.vx[:, :, 3:, :, :1] += 1.0
        self.assertNotEqual(key, self.plan(mask)["key"])

    def test_cache_key_follows_reference_content(self) -> None:
        key = self.plan(self.mask)["key"]
        self.assertEqual(key, self.plan(self.mask.clone())["key"])
        self.vx[:, :, 4] += 1.0
        self.assertNotEqual(key, self.plan(self.mask)["key"])


if __name__ == "__main__":
    unittest.main()
