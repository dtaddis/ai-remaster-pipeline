"""Lazy, memory-efficient, exact-semantics LTX guide execution for ARP.

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

import copy
import inspect
import logging
import os
import time
import types


LOGGER = logging.getLogger(__name__)
DEFAULT_FEED_FORWARD_CHUNK_TOKENS = 4096
DEFAULT_ATTENTION_QUERY_CHUNK_TOKENS = 4096
DEFAULT_VIDEO_ONLY_MEMORY_USAGE_FACTOR = 3.0
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


def _copy_module_structure(module):
    """Copy a torch module's registries while sharing its tensor/module values."""
    cloned = copy.copy(module)
    for registry in ("_modules", "_parameters", "_buffers"):
        values = getattr(module, registry, None)
        if isinstance(values, dict):
            setattr(cloned, registry, values.copy())
    return cloned


def _video_only_model_structure(base_model):
    """Privatize only containers that ARP mutates; never duplicate model weights."""
    diffusion_model = getattr(base_model, "diffusion_model", None)
    blocks = getattr(diffusion_model, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("ARP video-only mode requires an LTXAV diffusion model")

    private_base = _copy_module_structure(base_model)
    private_diffusion = _copy_module_structure(diffusion_model)
    if hasattr(blocks, "_modules"):
        private_blocks = _copy_module_structure(blocks)
        for index, block in enumerate(blocks):
            private_block = _copy_module_structure(block)
            # These are the only video modules whose forward methods ARP
            # specializes. Privatizing them keeps every override on this model
            # clone instead of mutating ComfyUI's global classes.
            for attribute in ("attn1", "ff"):
                child = getattr(block, attribute, None)
                if child is not None:
                    setattr(private_block, attribute, _copy_module_structure(child))
            private_blocks._modules[str(index)] = private_block
    else:
        private_blocks = []
        for block in blocks:
            private_block = _copy_module_structure(block)
            for attribute in ("attn1", "ff"):
                child = getattr(block, attribute, None)
                if child is not None:
                    setattr(private_block, attribute, _copy_module_structure(child))
            private_blocks.append(private_block)
    setattr(private_diffusion, "transformer_blocks", private_blocks)
    setattr(private_base, "diffusion_model", private_diffusion)
    return private_base, private_diffusion, private_blocks


def prune_ltxav_audio_transformer_blocks(model_patcher):
    """Remove LTXAV block modules that cannot run for a video-only latent."""
    pruned = model_patcher.clone()
    original_base = getattr(pruned, "model", None)
    diffusion_model = getattr(original_base, "diffusion_model", None)
    if getattr(diffusion_model, "_arp_video_only_audio_pruned", False):
        pruned.size = 0
        return pruned
    base_model, diffusion_model, blocks = _video_only_model_structure(original_base)
    pruned.model = base_model

    # ComfyUI 0.30's LTXAV config uses a placeholder memory_usage_factor of
    # 0.077, versus roughly 11 for the same-width LTXV model. That causes its
    # model manager to reserve only ~100 MiB for activations and fully load the
    # model, even though a full-length IC-LoRA guide plus keyframes needs about
    # 17 GiB of working memory. The CUDA async allocator then oversubscribes a
    # 24 GiB card by many GiB and silently pages through system RAM.
    #
    # Raise the estimate only on ARP's explicitly video-only clone. Chunked
    # attention and feed-forward execution bound the large transient tensors,
    # so a factor of 3 reserves useful activation headroom while keeping the
    # quantized transformer and IC-LoRA resident on a 24 GiB card.
    current_factor = float(getattr(base_model, "memory_usage_factor", 0.0))
    configured_text = os.environ.get("ARP_LTX_VIDEO_ONLY_MEMORY_USAGE_FACTOR")
    try:
        configured_factor = (
            float(configured_text)
            if configured_text is not None
            else DEFAULT_VIDEO_ONLY_MEMORY_USAGE_FACTOR
        )
    except (TypeError, ValueError):
        configured_factor = DEFAULT_VIDEO_ONLY_MEMORY_USAGE_FACTOR
    # Prefer a future upstream LTXAV estimate once it is no longer an obvious
    # placeholder. The ARP policy only substitutes for today's sub-1 value,
    # unless the user explicitly requests a larger hardware-specific reserve.
    if current_factor < 1.0 or configured_text is not None:
        base_model.memory_usage_factor = max(current_factor, configured_factor)

    removed = 0
    for block in blocks:
        for attribute in _LTXAV_AUDIO_BLOCK_ATTRIBUTES:
            if hasattr(block, attribute):
                delattr(block, attribute)
                removed += 1
            # ComfyUI records the original dtype of every parameter/buffer as
            # ordinary Python metadata during model loading. Removing the
            # registered tensor does not automatically remove that harmless
            # companion value, so clean it up explicitly as part of pruning.
            dtype_metadata = f"{attribute}_comfy_model_dtype"
            if hasattr(block, dtype_metadata):
                delattr(block, dtype_metadata)

        # Unknown executable/model state is meaningful only when it remains in
        # a torch module registry. Plain ``*_comfy_model_dtype`` attributes are
        # bookkeeping, not weights or branches, and must not trip this guard.
        registered_names: set[str] = set()
        for registry_name in ("_modules", "_parameters", "_buffers"):
            registry = getattr(block, registry_name, None)
            if isinstance(registry, dict):
                registered_names.update(registry)
        residual_audio = sorted(
            name
            for name in registered_names
            if any(marker in name.lower() for marker in ("audio", "a2v", "v2a"))
        )
        if residual_audio:
            raise ARPLTXCompatibilityError(
                "ComfyUI's LTXAV block contains unrecognized audio-only state: "
                f"{', '.join(residual_audio)}. Update ARP before rendering."
            )

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
        "ARP video-only LTXAV model removed %d unused audio transformer modules; "
        "activation memory factor %.2f",
        removed,
        base_model.memory_usage_factor,
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


def _attention_query_chunk_tokens() -> int:
    try:
        configured = os.environ.get(
            "ARP_LTX_ATTN_QUERY_CHUNK_TOKENS",
            DEFAULT_ATTENTION_QUERY_CHUNK_TOKENS,
        )
        return max(256, int(configured))
    except (TypeError, ValueError):
        return DEFAULT_ATTENTION_QUERY_CHUNK_TOKENS


def _cuda_memory_summary(torch, device) -> str:
    if not getattr(device, "type", None) == "cuda":
        return "non-CUDA"
    free, total = torch.cuda.mem_get_info(device)
    mib = 1024 ** 2
    return (
        f"allocated={torch.cuda.memory_allocated(device) / mib:.0f} MiB, "
        f"reserved={torch.cuda.memory_reserved(device) / mib:.0f} MiB, "
        f"device_free={free / mib:.0f} MiB/{total / mib:.0f} MiB"
    )


def _install_chunked_feed_forward(torch, feed_forward, diagnostics: dict[str, bool]) -> None:
    """Bound one model clone's MLP memory without changing ComfyUI's class."""
    if getattr(feed_forward, "_arp_chunked_feed_forward", False):
        return

    current_forward = feed_forward.forward

    chunk_tokens = _feed_forward_chunk_tokens()

    def chunked_forward(self, x):
        # LTX feed-forward layers operate independently on every token. Splitting
        # dimension -2 therefore preserves the operation while avoiding the full
        # sequence_length x (4 * hidden_size) GELU allocation.
        token_count = x.shape[-2]
        if torch.is_grad_enabled() or token_count <= chunk_tokens:
            return current_forward(x)

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
        diagnose_first = not diagnostics["first_timing"]
        if diagnose_first and x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
            LOGGER.info(
                "ARP first block reached feed-forward input (%s)",
                _cuda_memory_summary(torch, x.device),
            )
        known_gelu_projection = (
            network is not None
            and len(network) >= 2
            and type(network[0]).__name__ == "GELU_approx"
        )
        if is_quantized_in and is_quantized_out and known_gelu_projection:
            dequant_start = time.perf_counter()
            weight_in, bias_in = project_in.cast_bias_weight(x)
            weight_out, bias_out = project_out.cast_bias_weight(x)
            if diagnose_first and x.device.type == "cuda":
                torch.cuda.synchronize(x.device)
                LOGGER.info(
                    "ARP first feed-forward GGUF dequantization completed in %.2fs (%s)",
                    time.perf_counter() - dequant_start,
                    _cuda_memory_summary(torch, x.device),
                )
            middle_layers = tuple(network[1:-1])

            def run_chunk(chunk):
                hidden = torch.nn.functional.linear(chunk, weight_in, bias_in)
                hidden = torch.nn.functional.gelu(hidden, approximate="tanh")
                for layer in middle_layers:
                    hidden = layer(hidden)
                return torch.nn.functional.linear(hidden, weight_out, bias_out)

            if not diagnostics["gguf_reuse"]:
                LOGGER.info("ARP chunked LTX feed-forward is reusing dequantized GGUF weights")
                diagnostics["gguf_reuse"] = True
        else:
            if is_quantized_in and is_quantized_out and not diagnostics.get("gguf_fallback"):
                LOGGER.warning(
                    "ARP found a newer LTX feed-forward layout; preserving upstream "
                    "semantics with slower per-chunk GGUF dequantization"
                )
                diagnostics["gguf_fallback"] = True
            def run_chunk(chunk):
                return current_forward(chunk)

        first_count = min(chunk_tokens, token_count)
        first_chunk_start = time.perf_counter()
        first = run_chunk(x.narrow(-2, 0, first_count))
        if diagnose_first and x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
            LOGGER.info(
                "ARP first feed-forward chunk completed in %.2fs (%d/%d tokens; %s)",
                time.perf_counter() - first_chunk_start,
                first_count,
                token_count,
                _cuda_memory_summary(torch, x.device),
            )
            diagnostics["first_timing"] = True
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

    feed_forward.forward = types.MethodType(chunked_forward, feed_forward)
    feed_forward._arp_chunked_feed_forward = True


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


class ARPLTXCompatibilityError(RuntimeError):
    """The installed ComfyUI/LTX implementation is incompatible with ARP's patch."""


def _require_parameters(callable_object, required: set[str], label: str) -> None:
    try:
        available = set(inspect.signature(callable_object).parameters)
    except (TypeError, ValueError) as exc:
        raise ARPLTXCompatibilityError(
            f"ARP could not inspect {label}; the installed ComfyUI version is unsupported."
        ) from exc
    missing = sorted(required - available)
    if missing:
        raise ARPLTXCompatibilityError(
            f"ARP's video-only LTX patch is incompatible with this ComfyUI update: "
            f"{label} is missing {', '.join(missing)}. Update ARP before rendering."
        )


def install_sparse_guide_attention_patch(model_patcher=None):
    """Validate the runtime and optimize only an ARP-owned model clone.

    Passing ``None`` performs the loader's cheap compatibility preflight. A
    model patcher must already have private diffusion/block containers (as
    produced by :func:`prune_ltxav_audio_transformer_blocks`) before the
    instance-local execution overrides are attached.
    """
    try:
        import torch
        import comfy.ldm.lightricks.model as ltx_model
    except Exception as exc:
        raise ARPLTXCompatibilityError(
            "ARP could not import ComfyUI's LTX guide-attention implementation. "
            "Re-run the ARP installer or update ARP before rendering."
        ) from exc

    current_mask = getattr(ltx_model, "GuideAttentionMask", None)
    current_attention = getattr(ltx_model, "_attention_with_guide_mask", None)
    if current_mask is None or current_attention is None:
        raise ARPLTXCompatibilityError(
            "ARP's video-only LTX patch is incompatible with this ComfyUI update: "
            "guide-attention hooks are missing. Update ARP before rendering."
        )
    _require_parameters(
        current_attention,
        {"q", "k", "v", "heads", "guide_mask", "attn_precision", "transformer_options"},
        "ComfyUI LTX guide attention",
    )
    _require_parameters(
        current_mask,
        {"total_tokens", "guide_start", "tracked_count", "tracked_weights"},
        "ComfyUI LTX guide mask",
    )
    feed_forward = getattr(ltx_model, "FeedForward", None)
    if feed_forward is None:
        raise ARPLTXCompatibilityError(
            "ARP's video-only LTX patch is incompatible with this ComfyUI update: "
            "the LTX FeedForward class is missing. Update ARP before rendering."
        )
    _require_parameters(feed_forward.forward, {"self", "x"}, "ComfyUI LTX feed-forward")
    try:
        import xformers.ops as xops
    except Exception as exc:
        raise ARPLTXCompatibilityError(
            "ARP video-only LTX requires xFormers. Re-run the ARP installer before rendering."
        ) from exc
    if not callable(getattr(xops, "memory_efficient_attention_forward_requires_grad", None)):
        raise ARPLTXCompatibilityError(
            "The installed xFormers build lacks the attention API required by ARP. "
            "Re-run the ARP installer before rendering."
        )
    cross_attention = getattr(ltx_model, "CrossAttention", None)
    if cross_attention is None:
        raise ARPLTXCompatibilityError(
            "ARP's video-only LTX patch is incompatible with this ComfyUI update: "
            "the LTX CrossAttention class is missing. Update ARP before rendering."
        )
    _require_parameters(
        cross_attention.forward,
        {"self", "x", "context", "mask", "pe", "k_pe", "transformer_options"},
        "ComfyUI LTX cross-attention",
    )
    if model_patcher is None:
        return True

    diffusion_model = getattr(getattr(model_patcher, "model", None), "diffusion_model", None)
    blocks = getattr(diffusion_model, "transformer_blocks", None)
    if diffusion_model is None or blocks is None:
        raise ARPLTXCompatibilityError(
            "ARP video-only execution requires a private LTX diffusion model clone."
        )
    if getattr(diffusion_model, "_arp_model_scoped_execution", False):
        return model_patcher

    class SparseGuideAttentionMask(current_mask):
        """Masks noisy queries plus only the soft-strength guide query rows."""

        _arp_sparse_full_strength_rows = True
        __slots__ = (
            "tracked_log_weights",
            "noisy_partitions",
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
        query_chunk_tokens = _attention_query_chunk_tokens()

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

        diagnose_first = (
            use_partitioned_xformers
            and not sparse_attention_with_guide_mask._arp_reported_partitioned
        )
        if diagnose_first and q.device.type == "cuda":
            # Separate Q/K/V projection and RoPE work from the attention timing.
            torch.cuda.synchronize(q.device)
        attention_start = time.perf_counter()
        if diagnose_first:
            LOGGER.info(
                "ARP exact partitioned xFormers guide attention: %d total, %d generated, "
                "%d full-guide, %d soft-guide tokens, %d key partitions, %d-query chunks (%s)",
                q.shape[1],
                guide_start,
                guide_mask.full_query_indices.numel(),
                guide_mask.soft_query_indices.numel(),
                len(guide_mask.noisy_partitions),
                query_chunk_tokens,
                _cuda_memory_summary(torch, q.device),
            )

        if guide_start > 0:
            if use_partitioned_xformers:
                for query_start in range(0, guide_start, query_chunk_tokens):
                    query_count = min(query_chunk_tokens, guide_start - query_start)
                    noisy_out = _xformers_partitioned_attention(
                        torch,
                        q.narrow(1, query_start, query_count),
                        k,
                        v,
                        heads,
                        guide_mask.noisy_partitions,
                    )
                    out.narrow(1, query_start, query_count).copy_(noisy_out)
                    del noisy_out
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
                del noisy_out

        if guide_mask.full_query_indices.numel():
            if guide_mask.full_query_slice is not None:
                full_start, full_count = guide_mask.full_query_slice
                for relative_start in range(0, full_count, query_chunk_tokens):
                    query_count = min(query_chunk_tokens, full_count - relative_start)
                    query_start = full_start + relative_start
                    full_out = unmasked_attention(
                        q.narrow(1, query_start, query_count),
                        k,
                        v,
                        heads,
                        attn_precision=attn_precision,
                        transformer_options=transformer_options,
                    )
                    out.narrow(1, query_start, query_count).copy_(full_out)
                    del full_out
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
                out.index_copy_(1, guide_mask.full_query_indices, full_out)
                del full_q, full_out

        if guide_mask.soft_query_indices.numel():
            if use_partitioned_xformers and guide_mask.soft_query_slice is not None:
                soft_start, soft_count = guide_mask.soft_query_slice
                for relative_start in range(0, soft_count, query_chunk_tokens):
                    query_count = min(query_chunk_tokens, soft_count - relative_start)
                    query_start = soft_start + relative_start
                    soft_bias = guide_mask.tracked_log_weights.narrow(
                        0,
                        query_start - guide_start,
                        query_count,
                    )
                    soft_out = _xformers_partitioned_attention(
                        torch,
                        q.narrow(1, query_start, query_count),
                        k,
                        v,
                        heads,
                        [
                            (0, guide_start, soft_bias),
                            (guide_start, k.shape[1] - guide_start, 0.0),
                        ],
                    )
                    out.narrow(1, query_start, query_count).copy_(soft_out)
                    del soft_out
            elif use_partitioned_xformers:
                soft_relative = guide_mask.soft_query_indices - guide_start
                soft_q = q.index_select(1, guide_mask.soft_query_indices)
                soft_bias = guide_mask.tracked_log_weights.index_select(0, soft_relative)
                soft_out = _xformers_partitioned_attention(
                    torch, soft_q, k, v, heads,
                    [(0, guide_start, soft_bias), (guide_start, k.shape[1] - guide_start, 0.0)],
                )
                out.index_copy_(1, guide_mask.soft_query_indices, soft_out)
                del soft_q, soft_out
            else:
                soft_q = q.index_select(1, guide_mask.soft_query_indices)
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
                out.index_copy_(1, guide_mask.soft_query_indices, soft_out)
                del soft_q, soft_out

        if tracked_end < q.shape[1]:
            for query_start in range(tracked_end, q.shape[1], query_chunk_tokens):
                query_count = min(query_chunk_tokens, q.shape[1] - query_start)
                tail_out = unmasked_attention(
                    q.narrow(1, query_start, query_count), k, v, heads,
                    attn_precision=attn_precision,
                    transformer_options=transformer_options,
                )
                out.narrow(1, query_start, query_count).copy_(tail_out)
                del tail_out
        if diagnose_first:
            if q.device.type == "cuda":
                torch.cuda.synchronize(q.device)
            LOGGER.info(
                "ARP first exact guide self-attention completed in %.2fs (%s)",
                time.perf_counter() - attention_start,
                _cuda_memory_summary(torch, q.device),
            )
            sparse_attention_with_guide_mask._arp_reported_partitioned = True
        return out

    sparse_attention_with_guide_mask._arp_reported_partitioned = False

    original_builder = getattr(type(diffusion_model), "_build_guide_self_attention_mask", None)
    if original_builder is None:
        raise ARPLTXCompatibilityError(
            "ARP's video-only LTX patch is incompatible with this ComfyUI update: "
            "the guide-mask builder is missing. Update ARP before rendering."
        )
    _require_parameters(
        original_builder,
        {"self", "x", "transformer_options", "merged_args"},
        "ComfyUI LTX guide-mask builder",
    )

    def build_sparse_guide_mask(self, x, transformer_options, merged_args):
        original_mask = original_builder(self, x, transformer_options, merged_args)
        if original_mask is None or isinstance(original_mask, SparseGuideAttentionMask):
            return original_mask
        try:
            guide_start = int(original_mask.guide_start)
            tracked_count = int(original_mask.tracked_count)
            total_tokens = int(original_mask.noisy_mask.shape[-1])
            tracked_logs = original_mask.noisy_mask[
                0, 0, 0, guide_start:guide_start + tracked_count
            ]
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ARPLTXCompatibilityError(
                "ComfyUI returned an unsupported LTX guide-mask structure. "
                "Update ARP before rendering."
            ) from exc
        # exp(finfo.min) is exactly zero; every other entry recovers the
        # original scalar guide strength without rounding soft guides to one.
        tracked_weights = tracked_logs.exp().reshape(1, -1)
        return SparseGuideAttentionMask(
            total_tokens, guide_start, tracked_count, tracked_weights
        )

    diffusion_model._build_guide_self_attention_mask = types.MethodType(
        build_sparse_guide_mask,
        diffusion_model,
    )

    def install_model_scoped_attention(attention) -> None:
        if getattr(attention, "_arp_sparse_guide_attention", False):
            return
        original_forward = attention.forward
        required_attributes = (
            "to_q", "to_k", "to_v", "q_norm", "k_norm", "heads",
            "attn_precision", "to_gate_logits", "dim_head", "to_out",
        )
        missing = [name for name in required_attributes if not hasattr(attention, name)]
        if missing:
            raise ARPLTXCompatibilityError(
                "ARP's video-only LTX patch is incompatible with this ComfyUI update: "
                f"cross-attention is missing {', '.join(missing)}. Update ARP before rendering."
            )

        def model_scoped_forward(
            self,
            x,
            context=None,
            mask=None,
            pe=None,
            k_pe=None,
            transformer_options=None,
        ):
            options = transformer_options or {}
            if not isinstance(mask, SparseGuideAttentionMask):
                return original_forward(
                    x,
                    context=context,
                    mask=mask,
                    pe=pe,
                    k_pe=k_pe,
                    transformer_options=options,
                )

            q = self.to_q(x)
            active_context = x if context is None else context
            k = self.to_k(active_context)
            v = self.to_v(active_context)
            q = self.q_norm(q)
            k = self.k_norm(k)
            if pe is not None:
                if k_pe is None and q.shape == k.shape:
                    q, k = ltx_model.apply_rotary_emb_qk(q, k, pe)
                else:
                    q = ltx_model.apply_rotary_emb(q, pe)
                    k = ltx_model.apply_rotary_emb(k, pe if k_pe is None else k_pe)

            out = sparse_attention_with_guide_mask(
                q,
                k,
                v,
                self.heads,
                mask,
                attn_precision=self.attn_precision,
                transformer_options=options,
            )
            if self.to_gate_logits is not None:
                gate_logits = self.to_gate_logits(x)
                batch, tokens, _ = out.shape
                out = out.view(batch, tokens, self.heads, self.dim_head)
                gates = 2.0 * torch.sigmoid(gate_logits)
                out = (out * gates.unsqueeze(-1)).view(
                    batch, tokens, self.heads * self.dim_head
                )
            return self.to_out(out)

        attention.forward = types.MethodType(model_scoped_forward, attention)
        attention._arp_sparse_guide_attention = True

    diagnostics = {
        "gguf_reuse": False,
        "gguf_fallback": False,
        "first_timing": False,
    }
    patched_blocks = 0
    for block in blocks:
        attention = getattr(block, "attn1", None)
        feed_forward = getattr(block, "ff", None)
        if attention is None or feed_forward is None:
            raise ARPLTXCompatibilityError(
                "ARP's video-only LTX patch found a transformer block without "
                "video self-attention or feed-forward modules. Update ARP before rendering."
            )
        install_model_scoped_attention(attention)
        _install_chunked_feed_forward(torch, feed_forward, diagnostics)
        patched_blocks += 1

    diffusion_model._arp_model_scoped_execution = True
    LOGGER.info(
        "ARP enabled model-scoped exact guide attention and chunked feed-forward "
        "on %d LTX blocks",
        patched_blocks,
    )
    return model_patcher


def ltx_runtime_capabilities() -> dict[str, object]:
    """Validate the live Comfy/LTX internals without loading model weights."""

    install_sparse_guide_attention_patch()
    return {
        "compatible": True,
        "adapter": "model_scoped_ltx_guide_v1",
        "global_comfy_mutation": False,
        "exact_soft_guide_weights": True,
        "chunked_feed_forward": True,
        "video_only_audio_pruning": True,
    }
