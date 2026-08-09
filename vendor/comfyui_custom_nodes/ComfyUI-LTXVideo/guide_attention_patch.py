"""Memory-efficient, exact-semantics LTX guide execution for ARP.

ComfyUI's grouped guide mask still allocates one row for every tracked guide
token.  An IC-LoRA video guide can contain tens of thousands of full-strength
tokens, even though those rows have an all-zero mask and therefore need normal
unmasked attention.  Keep only the genuinely attenuated rows in the large
mask and evaluate the full-strength rows separately.

The LTX feed-forward MLP is token-independent but normally expands the whole
guided sequence to four times its hidden width.  Evaluate that operation in
token slices to bound its transient GELU allocation without changing context,
resolution, or guide behavior.
"""

from __future__ import annotations

import logging
import os


LOGGER = logging.getLogger(__name__)
DEFAULT_FEED_FORWARD_CHUNK_TOKENS = 4096


def _feed_forward_chunk_tokens() -> int:
    try:
        configured = os.environ.get(
            "ARP_LTX_FF_CHUNK_TOKENS", DEFAULT_FEED_FORWARD_CHUNK_TOKENS
        )
        return max(256, int(configured))
    except (TypeError, ValueError):
        return DEFAULT_FEED_FORWARD_CHUNK_TOKENS


def _install_chunked_feed_forward(torch, ltx_model) -> bool:
    """Bound LTX MLP activation memory by slicing its independent token axis."""
    feed_forward = getattr(ltx_model, "FeedForward", None)
    if feed_forward is None:
        LOGGER.warning("ARP chunked LTX feed-forward is unsupported by this ComfyUI version")
        return False

    current_forward = feed_forward.forward
    if getattr(current_forward, "_arp_chunked_feed_forward", False):
        return True

    chunk_tokens = _feed_forward_chunk_tokens()

    def chunked_forward(self, x):
        # LTX feed-forward layers operate independently on every token. Splitting
        # dimension -2 therefore preserves the operation while avoiding the full
        # sequence_length x (4 * hidden_size) GELU allocation.
        token_count = x.shape[-2]
        if torch.is_grad_enabled() or token_count <= chunk_tokens:
            return current_forward(self, x)

        first_count = min(chunk_tokens, token_count)
        first = current_forward(self, x.narrow(-2, 0, first_count))
        output_shape = list(x.shape)
        output_shape[-1] = first.shape[-1]
        output = torch.empty(output_shape, device=first.device, dtype=first.dtype)
        output.narrow(-2, 0, first_count).copy_(first)
        del first

        for start in range(first_count, token_count, chunk_tokens):
            count = min(chunk_tokens, token_count - start)
            chunk = current_forward(self, x.narrow(-2, start, count))
            output.narrow(-2, start, count).copy_(chunk)
            del chunk
        return output

    chunked_forward._arp_chunked_feed_forward = True
    feed_forward.forward = chunked_forward
    LOGGER.info("ARP exact-semantics chunked LTX feed-forward enabled (%d tokens)", chunk_tokens)
    return True


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
        return _install_chunked_feed_forward(torch, ltx_model)

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
    return _install_chunked_feed_forward(torch, ltx_model)
