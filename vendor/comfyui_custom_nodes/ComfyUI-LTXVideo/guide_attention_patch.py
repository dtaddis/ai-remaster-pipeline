"""Memory-efficient, exact-semantics LTX guide attention for ARP.

ComfyUI's grouped guide mask still allocates one row for every tracked guide
token.  An IC-LoRA video guide can contain tens of thousands of full-strength
tokens, even though those rows have an all-zero mask and therefore need normal
unmasked attention.  Keep only the genuinely attenuated rows in the large
mask and evaluate the full-strength rows separately.
"""

from __future__ import annotations

import logging


LOGGER = logging.getLogger(__name__)


def install_sparse_guide_attention_patch() -> bool:
    """Patch current ComfyUI's LTX guide attention without changing its mask semantics."""
    try:
        import torch
        import comfy.ldm.lightricks.model as ltx_model
    except Exception as exc:  # ComfyUI reports custom-node import failures separately.
        LOGGER.warning("ARP could not import the LTX guide-attention implementation: %s", exc)
        return False

    current_mask = getattr(ltx_model, "GuideAttentionMask", None)
    current_attention = getattr(ltx_model, "_attention_with_guide_mask", None)
    if current_mask is None or current_attention is None:
        LOGGER.warning("ARP sparse guide attention is unsupported by this ComfyUI version")
        return False
    if getattr(current_mask, "_arp_sparse_full_strength_rows", False):
        return True

    class SparseGuideAttentionMask:
        """Masks noisy queries plus only the soft-strength guide query rows."""

        _arp_sparse_full_strength_rows = True
        __slots__ = (
            "guide_start",
            "tracked_count",
            "noisy_mask",
            "full_query_indices",
            "soft_query_indices",
            "soft_mask",
        )

        def __init__(self, total_tokens, guide_start, tracked_count, tracked_weights):
            device = tracked_weights.device
            dtype = tracked_weights.dtype
            finfo = torch.finfo(dtype)
            flat_weights = tracked_weights.reshape(-1)

            positive = flat_weights > 0
            log_weights = torch.full_like(flat_weights, finfo.min)
            log_weights[positive] = torch.log(flat_weights[positive].clamp(min=finfo.tiny))

            self.guide_start = int(guide_start)
            self.tracked_count = int(tracked_count)
            self.noisy_mask = torch.zeros((1, 1, 1, total_tokens), device=device, dtype=dtype)
            self.noisy_mask[:, :, :, guide_start:guide_start + tracked_count] = log_weights.view(1, 1, 1, -1)

            full_rows = flat_weights == 1
            full_relative = torch.nonzero(full_rows, as_tuple=False).flatten()
            soft_relative = torch.nonzero(~full_rows, as_tuple=False).flatten()
            self.full_query_indices = full_relative + guide_start
            self.soft_query_indices = soft_relative + guide_start

            self.soft_mask = torch.zeros(
                (1, 1, soft_relative.numel(), total_tokens), device=device, dtype=dtype
            )
            if soft_relative.numel():
                # This matches ComfyUI's tracked mask exactly: guide query rows
                # attenuate only the original noisy/generated prefix. Guide-to-guide
                # attention remains unmasked.
                self.soft_mask[:, :, :, :guide_start] = log_weights[soft_relative].view(1, 1, -1, 1)

    def sparse_attention_with_guide_mask(
        q, k, v, heads, guide_mask, attn_precision, transformer_options
    ):
        """Evaluate noisy, full-guide, and soft-guide queries with their exact masks."""
        optimized_attention = ltx_model.comfy.ldm.modules.attention.optimized_attention
        guide_start = guide_mask.guide_start
        tracked_end = guide_start + guide_mask.tracked_count
        out = torch.empty_like(q)

        if guide_start > 0:
            noisy_out = optimized_attention(
                q[:, :guide_start, :],
                k,
                v,
                heads,
                mask=guide_mask.noisy_mask,
                attn_precision=attn_precision,
                transformer_options=transformer_options,
                low_precision_attention=False,
            )
            out[:, :guide_start, :] = noisy_out

        if guide_mask.full_query_indices.numel():
            full_out = optimized_attention(
                q.index_select(1, guide_mask.full_query_indices),
                k,
                v,
                heads,
                attn_precision=attn_precision,
                transformer_options=transformer_options,
            )
            out.index_copy_(1, guide_mask.full_query_indices, full_out)

        if guide_mask.soft_query_indices.numel():
            soft_out = optimized_attention(
                q.index_select(1, guide_mask.soft_query_indices),
                k,
                v,
                heads,
                mask=guide_mask.soft_mask,
                attn_precision=attn_precision,
                transformer_options=transformer_options,
                low_precision_attention=False,
            )
            out.index_copy_(1, guide_mask.soft_query_indices, soft_out)

        if tracked_end < q.shape[1]:
            out[:, tracked_end:, :] = optimized_attention(
                q[:, tracked_end:, :],
                k,
                v,
                heads,
                attn_precision=attn_precision,
                transformer_options=transformer_options,
            )
        return out

    ltx_model.GuideAttentionMask = SparseGuideAttentionMask
    ltx_model._attention_with_guide_mask = sparse_attention_with_guide_mask
    LOGGER.info("ARP sparse exact-semantics LTX guide attention enabled")
    return True
