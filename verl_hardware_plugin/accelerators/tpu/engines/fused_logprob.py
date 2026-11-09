# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Per-token LM-head log-probs on TPU without materializing the ``[T, V]`` logits.

Forward: Tokamax's Mosaic-TPU kernel (``reduction="none"``) through torchtitan's torch op. Backward: chunked PyTorch,
because torchtitan's op only backpropagates a reduced loss and tokamax 0.0.14's Mosaic backward returns a dW ~7% off.
"""

import importlib
from typing import Any

import torch
import torch.nn.functional as F

BACKWARD_CHUNK = 2048  # rows per backward chunk: FP32 logits take rows x vocab x 4 B (1.2 GiB for Qwen3)


def _loss_module() -> Any:
    """torchtitan's TPU loss-kernel module (it imports tokamax, working around tokamax#240)."""
    return importlib.import_module("torchtitan.experiments.tpu.kernels.linear_softmax_cross_entropy_loss")


def require_tokamax() -> None:
    """Raises a ``RuntimeError`` if torchtitan's TPU loss kernels, tokamax or torch_tpu's Pallas bridge is missing."""
    try:
        _loss_module()
        importlib.import_module("torch_tpu._internal.pallas.pallas")
    except Exception as e:  # ImportError, or a jax / tokamax version mismatch at import time
        raise RuntimeError(
            f"the fused LM-head log-prob needs torchtitan's TPU kernels, tokamax and torch_tpu's Pallas bridge: {e!r}"
        ) from e


_PALLAS_FAKE_LIB: Any = None
_VERIFIED_FORWARD_OPS: set[str] = set()


def _ensure_per_row_fake(fwd_name: str, op: Any, b_block: int, h_block: int, v_block: int) -> None:
    """Ensure ``pallas::<fwd_name>`` reports shape ``(rows,)`` on meta tensors for ``reduction="none"``.

    TorchTitan's ``linear_softmax_cross_entropy_loss`` registers a ``(1,)`` fake loss for every reduction, which
    eager runs never call, whereas ``make_fx`` tracing (``trainer_mode=graph_trainer``) would silently broadcast a
    ``(1,)`` log-prob across the batch. Override the op's fake registration with the per-row shape when needed.
    """
    global _PALLAS_FAKE_LIB
    if fwd_name in _VERIFIED_FORWARD_OPS:
        return
    with torch._C._DisableTorchDispatch():
        x = torch.empty(b_block, h_block, dtype=torch.bfloat16, device="meta")
        labels = torch.empty(b_block, dtype=torch.int32, device="meta")
        weight_t = torch.empty(h_block, v_block, dtype=torch.bfloat16, device="meta")
        loss, _ = op(x, labels, weight_t)
    if tuple(loss.shape) != (b_block,):
        qualname = f"pallas::{fwd_name}"
        if torch._C._dispatch_has_kernel_for_dispatch_key(qualname, "Meta"):
            if _PALLAS_FAKE_LIB is None:
                _PALLAS_FAKE_LIB = torch.library.Library("pallas", "IMPL")
            torch.library.register_fake(
                qualname,
                lambda x, labels, weights: (
                    torch.empty(x.shape[0], dtype=torch.float32, device=x.device),
                    torch.empty(x.shape[0], dtype=torch.float32, device=x.device),
                ),
                lib=_PALLAS_FAKE_LIB,
            )
            with torch._C._DisableTorchDispatch():
                loss, _ = op(x, labels, weight_t)
        if tuple(loss.shape) != (b_block,):
            raise RuntimeError(
                "torchtitan's fused linear-softmax-CE forward op reports a fake loss of shape "
                f"{tuple(loss.shape)} instead of ({b_block},) for reduction='none', so a graph_trainer trace of "
                "the fused log-prob would be wrong; set use_fused_linear_cross_entropy=False."
            )
    _VERIFIED_FORWARD_OPS.add(fwd_name)


def _forward_op(b_block: int, h_block: int, v_block: int) -> Any:
    """torchtitan's forward custom op ``(x, labels, weight_t) -> (loss, lse)`` for ``reduction="none"``."""
    fwd_name, _ = _loss_module()._get_loss_op_names("none", "mosaic_tpu", b_block, h_block, v_block)
    op = getattr(torch.ops.pallas, fwd_name)
    _ensure_per_row_fake(fwd_name, op, b_block, h_block, v_block)
    return op


class _FusedLogprob(torch.autograd.Function):
    """``log p(label)`` per token: the Pallas forward kernel and a chunked PyTorch backward.

    TODO(torchtitan): upstream per-token (``reduction="none"``) backward support in torchtitan's
    ``linear_softmax_cross_entropy_loss`` (its ``_my_bwd`` currently reshapes ``g_loss`` to a scalar ``()``
    and tokamax 0.0.14's Mosaic-TPU backward has ~7% dW error), then call ``linear_softmax_cross_entropy_loss``
    directly and remove this custom ``autograd.Function``.
    """

    @staticmethod
    def forward(ctx: Any, x, labels, weight_t, blocks):  # type: ignore[override]
        loss, lse = _forward_op(*blocks)(x, labels, weight_t)
        ctx.save_for_backward(x, labels, weight_t, lse)
        return -loss

    @staticmethod
    def backward(ctx: Any, grad):  # type: ignore[override]
        # d log p(label) / d logits = onehot(label) - softmax(logits); one chunk of FP32 logits at a time.
        # On torch_tpu, FP32 matmuls already default to single-pass BF16 MXU with FP32 accumulation; honor the
        # process-wide float32_matmul_precision rather than mutating it here.
        x, labels, weight_t, lse = ctx.saved_tensors
        w32 = weight_t.float()
        grad_x = torch.empty_like(x)
        grad_w = torch.zeros_like(w32)
        for start in range(0, x.shape[0], BACKWARD_CHUNK):
            rows = slice(start, start + BACKWARD_CHUNK)
            x32 = x[rows].float()
            g = -torch.exp(x32 @ w32 - lse[rows].unsqueeze(-1))
            g.scatter_add_(-1, labels[rows].long().unsqueeze(-1), torch.ones_like(g[:, :1]))
            g *= grad[rows].float().unsqueeze(-1)
            grad_x[rows] = (g @ w32.t()).to(x.dtype)
            grad_w += x32.t() @ g
        return grad_x, None, grad_w.to(weight_t.dtype), None


class _FusedLogprobFunctional(_FusedLogprob):
    """``_FusedLogprob`` with an out-of-place backward, for whole-step tracing (``trainer_mode=graph_trainer``).

    TorchTitan's graph trainer traces the backward with ``make_fx`` without functionalization, so the in-place chunk
    writes above (``grad_x[rows] = ...``, ``grad_w += ...``) would land in the graph as mutations of intermediates.
    Same math and chunking; the chunks are concatenated / summed instead. The trace unrolls the chunk loop.
    """

    @staticmethod
    def backward(ctx: Any, grad):  # type: ignore[override]
        # Same math as _FusedLogprob.backward; honor the process-wide float32_matmul_precision without mutating it.
        x, labels, weight_t, lse = ctx.saved_tensors
        w32 = weight_t.float()
        grad_x_chunks = []
        grad_w = None
        for start in range(0, x.shape[0], BACKWARD_CHUNK):
            rows = slice(start, start + BACKWARD_CHUNK)
            x32 = x[rows].float()
            g = -torch.exp(x32 @ w32 - lse[rows].unsqueeze(-1))
            g = g.scatter_add(-1, labels[rows].long().unsqueeze(-1), torch.ones_like(g[:, :1]))
            g = g * grad[rows].float().unsqueeze(-1)
            grad_x_chunks.append((g @ w32.t()).to(x.dtype))
            chunk_grad_w = x32.t() @ g
            grad_w = chunk_grad_w if grad_w is None else grad_w + chunk_grad_w
        assert grad_w is not None
        return torch.cat(grad_x_chunks, dim=0), None, grad_w.to(weight_t.dtype), None


def fused_linear_logprobs_tpu(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight_t: torch.Tensor,
    *,
    temperature: float = 1.0,
    b_block_size: int = 1024,
    h_block_size: int = 512,
    v_block_size: int = 2048,
    functional_backward: bool = False,
) -> torch.Tensor:
    """Per-token ``log_softmax(hidden_states @ weight_t / temperature)[labels]`` in FP32, shape ``(T,)``.

    ``hidden_states`` ``(..., H)`` and ``labels`` ``(...)`` are flattened to ``T`` tokens; ``weight_t`` is the
    ``(H, V)`` transposed LM-head weight (BF16), passed as is. ``T`` is zero-padded to whole ``b_block_size`` row
    blocks for the kernel; padded rows get no gradient. ``functional_backward`` selects the out-of-place backward
    for whole-step tracing (see ``_FusedLogprobFunctional``).
    """
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    if temperature != 1.0:
        x = x * (1.0 / float(temperature))  # logits / t == (x / t) @ W
    x, labels = x.to(weight_t.dtype), labels.reshape(-1).to(torch.int32)
    num_tokens, pad = x.shape[0], -x.shape[0] % b_block_size
    if pad:
        x, labels = F.pad(x, (0, 0, 0, pad)), F.pad(labels, (0, pad))
    blocks = (b_block_size, h_block_size, v_block_size)
    fn = _FusedLogprobFunctional if functional_backward else _FusedLogprob
    return fn.apply(x.contiguous(), labels.contiguous(), weight_t, blocks)[:num_tokens]


def check_fused_logprob_traceable(b_block_size: int = 1024, h_block_size: int = 512, v_block_size: int = 2048) -> None:
    """Verifies (fixing TorchTitan's ``(1,)`` fake registration if needed) that the forward op traces per row."""
    rows = b_block_size
    op = _forward_op(b_block_size, h_block_size, v_block_size)
    with torch._C._DisableTorchDispatch():
        x = torch.empty(rows, h_block_size, dtype=torch.bfloat16, device="meta")
        labels = torch.empty(rows, dtype=torch.int32, device="meta")
        weight_t = torch.empty(h_block_size, v_block_size, dtype=torch.bfloat16, device="meta")
        loss, _ = op(x, labels, weight_t)
    if tuple(loss.shape) != (rows,):
        raise RuntimeError(
            "torchtitan's fused linear-softmax-CE forward op reports a loss of shape "
            f"{tuple(loss.shape)} instead of ({rows},) for reduction='none': its fake output is wrong, so a "
            "graph_trainer trace of the fused log-prob would be silently wrong; set "
            "use_fused_linear_cross_entropy=False."
        )
