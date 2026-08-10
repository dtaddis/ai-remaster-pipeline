"""Memory-efficient, exact-semantics LTX guide execution for ARP.

ComfyUI's grouped guide mask still allocates one row for every tracked guide
token. An IC-LoRA video guide can contain tens of thousands of full-strength
tokens, while a few soft guide rows force the much larger generated-query
group through pathological masked attention. Decompose scalar guide biases
into unmasked key partitions and merge their softmax normalizers exactly.

The LTX feed-forward MLP is token-independent but normally expands the whole
guided sequence to four times its hidden width.  Evaluate that operation in
token slices to bound its transient GELU allocation without changing context,
resolution, or guide behavior.
"""

from __future__ import annotations

import logging
import os


LOGGER = logging.getLogger(__name__)
DEFAULT_FEED_FORWARD_CHUNK_TOKENS = 8192
MAX_PARTITIONED_ATTENTION_RUNS = 32
_LTXAV_AUDIO_BLOCK_ATTRIBUTES = (
    "audio_attn1",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
    "audio_ff",
    "audio_scale_shift_table",
    "audio_prompt_scale_shift_table",
    "scale_shift_table_a2v_ca_audio",
    "scale_shift_table_a2v_ca_video",
)


def prune_ltxav_audio_transformer_blocks(model_patcher):
    """Remove LTXAV block modules that cannot run for a video-only latent."""
    pruned = model_patcher.clone()
    diffusion_model = getattr(getattr(pruned, "model", None), "diffusion_model", None)
    blocks = getattr(diffusion_model, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("ARP video-only mode requires an LTXAV diffusion model")
    if getattr(diffusion_model, "_arp_video_only_audio_pruned", False):
        pruned.size = 0
        return pruned

    removed = 0
    for block in blocks:
        for attribute in _LTXAV_AUDIO_BLOCK_ATTRIBUTES:
            if hasattr(block, attribute):
                delattr(block, attribute)
                removed += 1

    if removed == 0:
        raise ValueError("ARP video-only mode found no LTXAV audio transformer modules")

    # LoRA patches are attached lazily. Discard patches targeting modules that
    # have just been removed so the GGUF patcher never tries to resolve them.
    patches = getattr(pruned, "patches", None)
    if isinstance(patches, dict):
        pruned.patches = {
            key: value
            for key, value in patches.items()
            if not any(
                f".{attribute}." in key or key.endswith(f".{attribute}")
                for attribute in _LTXAV_AUDIO_BLOCK_ATTRIBUTES
            )
        }

    diffusion_model._arp_video_only_audio_pruned = True
    # ModelPatcher caches model size. Force the model manager to measure the
    # smaller module tree before deciding how much of the GGUF fits in VRAM.
    pruned.size = 0
    LOGGER.info(
        "ARP video-only LTXAV model removed %d unused audio transformer modules",
        removed,
    )
    return pruned


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

        # ComfyUI-GGUF normally dequantizes a Linear's weight inside every
        # forward call. Calling the complete MLP once per token slice would
        # therefore dequantize both very large weights dozens of times. Hold
        # each dequantized weight for this one MLP invocation and reuse it
        # across slices. LoRA patches are already applied by cast_bias_weight.
        network = getattr(self, "net", None)
        project_in = getattr(network[0], "proj", None) if network is not None else None
        project_out = network[-1] if network is not None else None
        is_quantized_in = callable(
            getattr(project_in, "is_ggml_quantized", None)
        ) and project_in.is_ggml_quantized()
        is_quantized_out = callable(
            getattr(project_out, "is_ggml_quantized", None)
        ) and project_out.is_ggml_quantized()
        if is_quantized_in and is_quantized_out:
            weight_in, bias_in = project_in.cast_bias_weight(x)
            weight_out, bias_out = project_out.cast_bias_weight(x)
            middle_layers = tuple(network[1:-1])

            def run_chunk(chunk):
                hidden = torch.nn.functional.linear(chunk, weight_in, bias_in)
                hidden = torch.nn.functional.gelu(hidden, approximate="tanh")
                for layer in middle_layers:
                    hidden = layer(hidden)
                return torch.nn.functional.linear(hidden, weight_out, bias_out)

            if not chunked_forward._arp_reported_gguf_reuse:
                LOGGER.info("ARP chunked LTX feed-forward is reusing dequantized GGUF weights")
                chunked_forward._arp_reported_gguf_reuse = True
        else:
            def run_chunk(chunk):
                return current_forward(self, chunk)

        first_count = min(chunk_tokens, token_count)
        first = run_chunk(x.narrow(-2, 0, first_count))
        output_shape = list(x.shape)
        output_shape[-1] = first.shape[-1]
        output = torch.empty(output_shape, device=first.device, dtype=first.dtype)
        output.narrow(-2, 0, first_count).copy_(first)
        del first

        for start in range(first_count, token_count, chunk_tokens):
            count = min(chunk_tokens, token_count - start)
            chunk = run_chunk(x.narrow(-2, start, count))
            output.narrow(-2, start, count).copy_(chunk)
            del chunk
        return output

    chunked_forward._arp_chunked_feed_forward = True
    chunked_forward._arp_reported_gguf_reuse = False
    feed_forward.forward = chunked_forward
    LOGGER.info("ARP exact-semantics chunked LTX feed-forward enabled (%d tokens)", chunk_tokens)
    return True


def _constant_runs(torch, values):
    """Return contiguous ``(start, count, value)`` runs for a 1-D tensor."""
    if values.numel() == 0:
        return []
    changes = torch.ones(values.numel(), dtype=torch.bool, device=values.device)
    if values.numel() > 1:
        changes[1:] = values[1:] != values[:-1]
    starts = torch.nonzero(changes, as_tuple=False).flatten().tolist()
    ends = starts[1:] + [values.numel()]
    return [
        (start, end - start, float(values[start].item()))
        for start, end in zip(starts, ends)
    ]


def _merge_adjacent_partitions(partitions):
    """Coalesce adjacent key spans carrying the same scalar logit bias."""
    merged = []
    for start, count, bias in partitions:
        if count <= 0:
            continue
        if (
            merged
            and merged[-1][0] + merged[-1][1] == start
            and merged[-1][2] == bias
        ):
            previous_start, previous_count, previous_bias = merged[-1]
            merged[-1] = (previous_start, previous_count + count, previous_bias)
        else:
            merged.append((start, count, bias))
    return merged


def _xformers_partitioned_attention(torch, q, k, v, heads, partitions):
    """Evaluate additive per-partition biases using unmasked xFormers kernels.

    Each partition is ``(key_start, key_count, logit_bias)``. ``logit_bias``
    may be a scalar or one value per query row.  xFormers returns the
    log-sum-exp normalizer for every partition; combining outputs according to
    those normalizers is algebraically identical to one masked softmax.
    """
    import xformers.ops as xops

    batch, query_count, inner_dim = q.shape
    dim_head = inner_dim // heads
    q_heads = q.reshape(batch, query_count, heads, dim_head)
    k_heads = k.reshape(batch, k.shape[1], heads, dim_head)
    v_heads = v.reshape(batch, v.shape[1], heads, dim_head)

    merged_out = None
    merged_lse = None
    for key_start, key_count, logit_bias in partitions:
        if key_count <= 0:
            continue
        if not torch.is_tensor(logit_bias) and logit_bias < -1.0e30:
            # ComfyUI represents zero guide strength with finfo.min. Its
            # contribution underflows to zero, so avoid doing useless work.
            continue
        part_out, part_lse = xops.memory_efficient_attention_forward_requires_grad(
            q_heads,
            k_heads.narrow(1, key_start, key_count),
            v_heads.narrow(1, key_start, key_count),
        )
        # xFormers pads this axis (normally to 32 queries) for its backward API.
        part_lse = part_lse[..., :query_count]
        if torch.is_tensor(logit_bias):
            part_lse = part_lse + logit_bias.reshape(1, 1, query_count)
        elif logit_bias:
            part_lse = part_lse + logit_bias

        if merged_out is None:
            merged_out = part_out
            merged_lse = part_lse
            continue

        combined_lse = torch.logaddexp(merged_lse, part_lse)
        previous_weight = (
            (merged_lse - combined_lse)
            .exp()
            .transpose(1, 2)
            .unsqueeze(-1)
            .to(merged_out.dtype)
        )
        part_weight = (
            (part_lse - combined_lse)
            .exp()
            .transpose(1, 2)
            .unsqueeze(-1)
            .to(part_out.dtype)
        )
        merged_out.mul_(previous_weight).addcmul_(part_out, part_weight)
        merged_lse = combined_lse

    if merged_out is None:
        return torch.zeros_like(q)
    return merged_out.reshape(batch, query_count, inner_dim)


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
            "tracked_log_weights",
            "noisy_partitions",
            "noisy_mask",
            "full_query_indices",
            "full_query_slice",
            "soft_query_indices",
            "soft_query_slice",
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
            self.tracked_log_weights = log_weights
            self.noisy_mask = torch.zeros((1, 1, 1, total_tokens), device=device, dtype=dtype)
            self.noisy_mask[:, :, :, guide_start:guide_start + tracked_count] = log_weights.view(1, 1, 1, -1)

            tracked_end = guide_start + tracked_count
            key_partitions = [(0, guide_start, 0.0)]
            key_partitions.extend(
                (guide_start + start, count, bias)
                for start, count, bias in _constant_runs(torch, log_weights)
            )
            key_partitions.append((tracked_end, total_tokens - tracked_end, 0.0))
            self.noisy_partitions = _merge_adjacent_partitions(key_partitions)

            full_rows = flat_weights == 1
            full_relative = torch.nonzero(full_rows, as_tuple=False).flatten()
            soft_relative = torch.nonzero(~full_rows, as_tuple=False).flatten()
            self.full_query_indices = full_relative + guide_start
            self.soft_query_indices = soft_relative + guide_start

            self.full_query_slice = None
            if full_relative.numel() and int(
                (full_relative[-1] - full_relative[0]).item()
            ) == full_relative.numel() - 1:
                self.full_query_slice = (
                    int(full_relative[0].item()) + guide_start,
                    full_relative.numel(),
                )
            self.soft_query_slice = None
            if soft_relative.numel() and int(
                (soft_relative[-1] - soft_relative[0]).item()
            ) == soft_relative.numel() - 1:
                self.soft_query_slice = (
                    int(soft_relative[0].item()) + guide_start,
                    soft_relative.numel(),
                )

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
        attention_module = ltx_model.comfy.ldm.modules.attention
        unmasked_attention = attention_module.optimized_attention
        # xFormers expands even a broadcast (1, 1, 1, K) tensor mask to
        # (1, 1, Q, K), which defeats this patch and can allocate many GB.
        # PyTorch SDPA keeps these masks broadcast and selects cuDNN attention
        # on the managed Windows runtime. Reserve xFormers for the genuinely
        # unmasked rows where it is faster at long-video sequence lengths.
        masked_attention = getattr(
            attention_module, "attention_pytorch", unmasked_attention
        )
        guide_start = guide_mask.guide_start
        tracked_end = guide_start + guide_mask.tracked_count
        out = torch.empty_like(q)

        # The mask used by LTX guide conditioning is separable: noisy query
        # rows add one scalar bias per guide key group, while soft guide query
        # rows add one scalar bias to the noisy key prefix. Evaluate those key
        # partitions with fast unmasked xFormers kernels and merge them with
        # their log-sum-exp normalizers. This preserves the exact softmax while
        # avoiding the pathological long-sequence masked SDPA kernel.
        use_partitioned_xformers = len(guide_mask.noisy_partitions) <= MAX_PARTITIONED_ATTENTION_RUNS
        if use_partitioned_xformers:
            try:
                import xformers.ops  # noqa: F401
            except Exception:
                use_partitioned_xformers = False

        if use_partitioned_xformers and not sparse_attention_with_guide_mask._arp_reported_partitioned:
            LOGGER.info(
                "ARP exact partitioned xFormers guide attention: %d total, %d generated, "
                "%d full-guide, %d soft-guide tokens, %d key partitions",
                q.shape[1],
                guide_start,
                guide_mask.full_query_indices.numel(),
                guide_mask.soft_query_indices.numel(),
                len(guide_mask.noisy_partitions),
            )
            sparse_attention_with_guide_mask._arp_reported_partitioned = True

        if guide_start > 0:
            if use_partitioned_xformers:
                noisy_out = _xformers_partitioned_attention(
                    torch,
                    q[:, :guide_start, :],
                    k,
                    v,
                    heads,
                    guide_mask.noisy_partitions,
                )
            else:
                noisy_out = masked_attention(
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
            if guide_mask.full_query_slice is not None:
                full_start, full_count = guide_mask.full_query_slice
                full_q = q.narrow(1, full_start, full_count)
            else:
                full_q = q.index_select(1, guide_mask.full_query_indices)
            full_out = unmasked_attention(
                full_q,
                k,
                v,
                heads,
                attn_precision=attn_precision,
                transformer_options=transformer_options,
            )
            if guide_mask.full_query_slice is not None:
                out.narrow(1, full_start, full_count).copy_(full_out)
            else:
                out.index_copy_(1, guide_mask.full_query_indices, full_out)

        if guide_mask.soft_query_indices.numel():
            if guide_mask.soft_query_slice is not None:
                soft_start, soft_count = guide_mask.soft_query_slice
                soft_q = q.narrow(1, soft_start, soft_count)
            else:
                soft_q = q.index_select(1, guide_mask.soft_query_indices)
            if use_partitioned_xformers:
                if guide_mask.soft_query_slice is not None:
                    soft_bias = guide_mask.tracked_log_weights.narrow(
                        0, soft_start - guide_start, soft_count
                    )
                else:
                    soft_relative = guide_mask.soft_query_indices - guide_start
                    soft_bias = guide_mask.tracked_log_weights.index_select(0, soft_relative)
                soft_out = _xformers_partitioned_attention(
                    torch,
                    soft_q,
                    k,
                    v,
                    heads,
                    [
                        (0, guide_start, soft_bias),
                        (guide_start, k.shape[1] - guide_start, 0.0),
                    ],
                )
            else:
                soft_out = masked_attention(
                    soft_q,
                    k,
                    v,
                    heads,
                    mask=guide_mask.soft_mask,
                    attn_precision=attn_precision,
                    transformer_options=transformer_options,
                    low_precision_attention=False,
                )
            if guide_mask.soft_query_slice is not None:
                out.narrow(1, soft_start, soft_count).copy_(soft_out)
            else:
                out.index_copy_(1, guide_mask.soft_query_indices, soft_out)

        if tracked_end < q.shape[1]:
            out[:, tracked_end:, :] = unmasked_attention(
                q[:, tracked_end:, :],
                k,
                v,
                heads,
                attn_precision=attn_precision,
                transformer_options=transformer_options,
            )
        return out

    sparse_attention_with_guide_mask._arp_reported_partitioned = False

    ltx_model.GuideAttentionMask = SparseGuideAttentionMask
    ltx_model._attention_with_guide_mask = sparse_attention_with_guide_mask
    LOGGER.info("ARP sparse exact-semantics LTX guide attention enabled")
    return _install_chunked_feed_forward(torch, ltx_model)
