# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""P1 parity check of ``engine_kwargs.tpu.trainer_mode=graph_trainer`` against the eager engine, on TPU.

Builds the real TorchTitan TPU engine twice on one TPU host -- ``eager_blocks`` (per-block ``torch.compile``, eager
loss and backward) and ``graph_trainer`` (TorchTitan's ``TpuGraphTrainer``: forward + fused log-prob + PPO loss +
backward traced and compiled as one program) -- on the same Qwen3 MoE model (Qwen3-30B-A3B cut to its first
``PARITY_NUM_LAYERS`` layers, loaded from the full HF checkpoint) and the same synthetic GRPO micro-batch per rank
(bypass mode, ``PARITY_MICRO_BATCHES`` micro-batches per rank, ``static_micro_batch_len``), and compares:

1. the verl loss and metrics of the update micro-batch,
2. the per-token log-probs of the response tokens,
3. the global gradient norm and every parameter's gradient norm,
4. after one optimizer step, the loss and log-probs of a second update on the same micro-batch.

``old_log_probs`` are the eager engine's own (forward-only) log-probs plus small noise, so the PPO ratio stays near
one and most tokens contribute a gradient. Not a pytest module (the plugin CI runs on CPU only). Submit it as a Ray
job to a cluster with one TPU host (e.g. v5p 2x2x1, or one v5p host of a larger slice; tpu7x 2x2x1), with verl, the
plugin and TorchTitan importable::

    python3 tests/accelerators/tpu/e2e/graphtrainer_parity_check.py

Environment (defaults in brackets):
    PARITY_MODEL_PATH        HF checkpoint [Qwen/Qwen3-30B-A3B]
    PARITY_NUM_LAYERS        transformer layers kept [4]
    PARITY_STATIC_LEN        static_micro_batch_len = token budget of the micro-batch [4096]
    PARITY_NUM_SEQS          sequences per rank and micro-batch [6]
    PARITY_MICRO_BATCHES     micro-batches per rank (gradient accumulation) [1]
    PARITY_CHIPS             TPU processes on the host [all TPU chips of the first TPU node]
    PARITY_FUSED_LCE         1: fused Tokamax log-prob (as the recipe), 0: plain LM head (the fallback) [1]
    PARITY_GRAPH_OPTIONS     extra engine_kwargs.tpu options of the graph run, JSON or k=v;k=v [{}]
                             (e.g. {"graph_trainer_memory_policy": "tpu_selective",
                             "graph_trainer_save_regions": "moe.routing,attention.inner_attention"})
    PARITY_MODES             runs, comma-separated [eager_blocks,graph_trainer]
    PARITY_LOGP_ATOL         max |delta log-prob| over response tokens [0.05]
    PARITY_LOSS_RTOL         loss / post-step loss relative tolerance [0.02]
    PARITY_GRAD_RTOL         global grad-norm relative tolerance [0.01]
    PARITY_OUT               JSON summary path [/tmp/graphtrainer_parity.json]

Prints ``[graphtrainer parity] PASSED`` / ``FAILED`` and exits 0 / 1.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
import time
import traceback
from typing import Optional

import ray
import torch
from tensordict import TensorDict

import verl_hardware_plugin  # noqa: F401  (registers the TPU platform and engines)
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import tensordict_utils as tu
from verl.workers.config import HFModelConfig, TorchtitanEngineConfig, TorchtitanOptimizerConfig, TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

MODEL_PATH = os.environ.get("PARITY_MODEL_PATH", "Qwen/Qwen3-30B-A3B")
NUM_LAYERS = int(os.environ.get("PARITY_NUM_LAYERS", "4"))
STATIC_LEN = int(os.environ.get("PARITY_STATIC_LEN", "4096"))
NUM_SEQS = int(os.environ.get("PARITY_NUM_SEQS", "6"))
MICRO_BATCHES = int(os.environ.get("PARITY_MICRO_BATCHES", "1"))
FUSED_LCE = os.environ.get("PARITY_FUSED_LCE", "1") == "1"
# Draw the batch's tokens from embedding rows with norm >= this x the median (0: the whole vocabulary).
MIN_EMB_NORM_FRAC = float(os.environ.get("PARITY_MIN_EMB_NORM_FRAC", "0.25"))


def _parse_options(raw: str) -> dict:
    """JSON, or ``key=value;key=value`` (no quotes, easier to pass through job launchers)."""
    raw = raw.strip()
    if not raw or raw.startswith("{"):
        return json.loads(raw or "{}")
    options = {}
    for item in filter(None, (part.strip() for part in raw.split(";"))):
        key, _, value = item.partition("=")
        options[key.strip()] = {"true": True, "false": False}.get(value.strip().lower(), value.strip())
    return options


GRAPH_OPTIONS = _parse_options(os.environ.get("PARITY_GRAPH_OPTIONS", ""))
MODES = [m.strip() for m in os.environ.get("PARITY_MODES", "eager_blocks,graph_trainer").split(",") if m.strip()]
LOGP_ATOL = float(os.environ.get("PARITY_LOGP_ATOL", "0.05"))
LOSS_RTOL = float(os.environ.get("PARITY_LOSS_RTOL", "0.02"))
LOSS_ATOL = float(os.environ.get("PARITY_LOSS_ATOL", "0.005"))
GRAD_COSINE = float(os.environ.get("PARITY_GRAD_COSINE", "0.99"))
GRAD_SAMPLE = 1 << 18
UNCOMPILED_FORWARD = os.environ.get("PARITY_UNCOMPILED_FORWARD", "1") == "1"
GRAD_RTOL = float(os.environ.get("PARITY_GRAD_RTOL", "0.01"))
OUT_PATH = os.environ.get("PARITY_OUT", "/tmp/graphtrainer_parity.json")
MAX_PROMPT, MAX_RESPONSE, SEQ_BUCKET = 256, 512, 1024
SEED = 1234
TPU_SLICE_RESOURCE_PREFIX = "tpu-group-"


def log(msg: str) -> None:
    print(f"[graphtrainer parity] {msg}", flush=True)


class _ActorConfig(dict):
    """The fields of verl's actor config that ``ppo_loss`` reads, as in the Qwen3-30B-A3B TPU recipe."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _recipe_actor_config() -> _ActorConfig:
    return _ActorConfig(
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        loss_agg_mode="token-mean",
        use_kl_loss=False,
        entropy_coeff=0.0,
        loss_scale_factor=None,
        global_batch_info={},
        policy_loss=_ActorConfig(
            loss_mode="bypass_mode",
            rollout_correction=_ActorConfig(
                loss_type="ppo_clip", rollout_is="token", rollout_is_threshold=3.0, rollout_rs=None
            ),
        ),
    )


def _truncate_qwen3_layers(num_layers: int) -> None:
    """Make the engine build only the first ``num_layers`` layers (the HF load then reads only those weights).

    ``derive_torchtitan_name_and_flavor`` calls ``model_registry(flavor)`` to match the HF config (48 layers), the
    engine calls it with ``seq_len=...``: only the latter is cut.
    """
    import torchtitan.models.qwen3 as qwen3

    registry = qwen3.model_registry
    if getattr(registry, "_parity_truncated", False):
        return

    @functools.wraps(registry)
    def model_registry(flavor, *args, **kwargs):
        spec = registry(flavor, *args, **kwargs)
        if "seq_len" in kwargs and len(spec.model.layers) > num_layers:
            spec.model.layers = list(spec.model.layers[:num_layers])
        return spec

    model_registry._parity_truncated = True
    qwen3.model_registry = model_registry


def build_batch(rank: int, dp_size: int, vocab_size: int, token_pool: Optional[torch.Tensor] = None) -> TensorDict:
    """A GRPO-like update batch for ``rank`` as verl hands it to the engine (after ``left_right_2_no_padding``).

    Token ids are drawn uniformly from ``token_pool`` (default: the whole vocabulary).
    """
    gen = torch.Generator().manual_seed(SEED + rank)
    # MICRO_BATCHES > 1: about that many static-length micro-batches of whole sequences (use_dynamic_bsz with
    # max_token_len_per_gpu = STATIC_LEN), to exercise gradient accumulation.
    budget = STATIC_LEN * MICRO_BATCHES - 1 - (MICRO_BATCHES - 1) * (MAX_PROMPT + MAX_RESPONSE)
    prompt_lens, response_lens = [], []
    for _ in range(NUM_SEQS * MICRO_BATCHES):
        p = int(torch.randint(16, MAX_PROMPT + 1, (1,), generator=gen))
        r = int(torch.randint(16, MAX_RESPONSE + 1, (1,), generator=gen))
        if sum(prompt_lens) + sum(response_lens) + p + r > budget:
            break
        prompt_lens.append(p)
        response_lens.append(r)
    bsz = len(prompt_lens)
    input_ids = torch.zeros(bsz, MAX_PROMPT + MAX_RESPONSE, dtype=torch.int64)
    attention_mask = torch.zeros_like(input_ids)
    for i, (p, r) in enumerate(zip(prompt_lens, response_lens, strict=True)):
        lo, hi = MAX_PROMPT - p, MAX_PROMPT + r  # left-padded prompt, right-padded response
        if token_pool is None:
            input_ids[i, lo:hi] = torch.randint(0, vocab_size, (p + r,), generator=gen)
        else:
            input_ids[i, lo:hi] = token_pool[torch.randint(0, token_pool.numel(), (p + r,), generator=gen)]
        attention_mask[i, lo:hi] = 1
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0) * attention_mask
    response_mask = attention_mask[:, MAX_PROMPT:].clone()
    # GRPO: one advantage per sequence; mixed signs exercise both PPO clip branches.
    advantages = torch.randn(bsz, 1, generator=gen).expand(bsz, MAX_RESPONSE) * response_mask
    td = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompts": input_ids[:, :MAX_PROMPT].clone(),
            "responses": input_ids[:, MAX_PROMPT:].clone(),
            "response_mask": response_mask,
            "advantages": advantages.contiguous().float(),
            "old_log_probs": torch.zeros(bsz, MAX_RESPONSE),
        },
        batch_size=[bsz],
    )
    td = left_right_2_no_padding(td)
    tu.assign_non_tensor(
        td,
        temperature=1.0,
        calculate_entropy=False,
        global_batch_size=bsz * dp_size,
        use_remove_padding=True,
        use_dynamic_bsz=True,
        max_token_len_per_gpu=STATIC_LEN,
        micro_batch_size_per_gpu=1,
        use_fused_kernels=False,
    )
    return td


def _response_values(log_probs, data: TensorDict) -> torch.Tensor:
    """Model-output log-probs (packed jagged) -> ``[bsz, max_response_len]`` response layout, as ``ppo_loss`` does."""
    return no_padding_2_padding(log_probs, data).float().cpu()


class ParityWorker(TrainingWorker):
    """The TPU engine on one rank, driven directly (``forward_backward_batch`` / ``optimizer_step``)."""

    def __init__(self, config: TrainingWorkerConfig) -> None:
        _truncate_qwen3_layers(NUM_LAYERS)
        super().__init__(config)
        self.loss_fn = functools.partial(_ppo_loss(), config=_recipe_actor_config())

    def _batch(self, old_log_probs=None) -> TensorDict:
        rank = torch.distributed.get_rank()
        vocab_size = self.model_config.hf_config.vocab_size
        data = build_batch(rank, self.engine.get_data_parallel_size(), vocab_size, self._token_pool())
        if old_log_probs is not None:
            data["old_log_probs"] = old_log_probs[rank]["old_log_probs"]
        return data

    def _token_pool(self) -> Optional[torch.Tensor]:
        """Token ids whose embedding row norm is at least ``MIN_EMB_NORM_FRAC`` x the median (cached).

        Qwen3-30B-A3B has untrained rows (norm ~3e-5; rms ~1e-6 < the RMSNorm eps): their input gradient is ~1000x
        amplified by the first RMSNorm and differs by 10-15% between any two compilations (v7x P1: one such token
        was 98% of the embedding-gradient norm). Real data never contains them, so the parity batch leaves them out.
        The pool depends only on the weights, so both modes' worker groups draw the same batch.
        """
        if MIN_EMB_NORM_FRAC <= 0:
            return None
        if getattr(self, "_cached_token_pool", None) is None:
            for module in self.engine.module:
                for name, param in module.named_parameters():
                    if _canonical_name(name) == "tok_embeddings.weight":
                        weight = param.full_tensor() if hasattr(param, "full_tensor") else param
                        norms = weight.detach().float().norm(dim=-1).cpu()
                        if norms.numel() != self.model_config.hf_config.vocab_size:  # not the full table
                            log(f"token pool: embedding has {norms.numel()} rows here; using the whole vocabulary")
                            self._cached_token_pool = torch.arange(self.model_config.hf_config.vocab_size)
                            break
                        self._cached_token_pool = torch.nonzero(norms >= MIN_EMB_NORM_FRAC * norms.median()).flatten()
                        if torch.distributed.get_rank() == 0:
                            pool_count = self._cached_token_pool.numel()
                            log(
                                f"token pool: {pool_count}/{norms.numel()} rows with embedding norm "
                                f">= {MIN_EMB_NORM_FRAC} x median {float(norms.median()):.4f}"
                            )
        return getattr(self, "_cached_token_pool", None)

    def _grad_norms(self) -> dict[str, float]:
        """Global L2 norm of every parameter's gradient, by canonical FQN (see ``_canonical_name``)."""
        norms = {}
        for module in self.engine.module:
            for name, param in module.named_parameters():
                if param.grad is None:
                    continue
                norm = torch.nn.utils.get_total_norm([param.grad], norm_type=2.0, foreach=False)
                if hasattr(norm, "full_tensor"):
                    norm = norm.full_tensor()
                norms[_canonical_name(name)] = float(norm.item())
        return norms

    def _grad_samples(self) -> dict[str, torch.Tensor]:
        """A strided sample (<= GRAD_SAMPLE elements) of this rank's local gradient shard of every parameter.

        Both modes shard every parameter the same way, so the samples line up element for element and their cosine
        similarity measures whether the two gradients point the same way.
        """
        samples = {}
        for module in self.engine.module:
            for name, param in module.named_parameters():
                if param.grad is None:
                    continue
                grad = param.grad
                local = grad.to_local() if hasattr(grad, "to_local") else grad
                flat = local.detach().reshape(-1)
                stride = max(1, flat.numel() // GRAD_SAMPLE)
                samples[_canonical_name(name)] = flat[::stride][:GRAD_SAMPLE].float().cpu()
        return samples

    def _embedding_rows(self) -> Optional[dict[str, torch.Tensor]]:
        """Diagnostic: this rank's local shard (dim-0 rows) of the token-embedding gradient: row norms + 64 columns.

        The batch's token ids are random over the vocabulary, so almost every row is the input gradient of one token:
        the driver maps rows back to (rank, sequence, position) to see where the graph's gradient differs.
        """
        for module in self.engine.module:
            for name, param in module.named_parameters():
                if _canonical_name(name) == "tok_embeddings.weight" and param.grad is not None:
                    grad = param.grad
                    local = (grad.to_local() if hasattr(grad, "to_local") else grad).detach().float()
                    stride = max(1, local.shape[1] // 64)
                    return {"norms": local.norm(dim=-1).cpu(), "cols": local[:, ::stride].cpu()}
        return None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def on_policy_old_log_probs(self, noise: float = 0.05) -> dict:
        """Forward-only log-probs of the batch (eager engine) + noise, in the response layout, keyed by rank."""
        data = self._batch()
        with self.engine.eval_mode():
            output = self.engine.infer_batch(data.copy(), loss_function=None)
        response_mask = data["response_mask"].float()
        logp = _response_values(output["model_output"]["log_probs"], data) * response_mask
        gen = torch.Generator().manual_seed(SEED + 7 + torch.distributed.get_rank())
        old = (logp + noise * torch.randn(logp.shape, generator=gen)) * response_mask
        # The forward-only log-probs also give the noise floor: the same eager engine, a different XLA program.
        return {torch.distributed.get_rank(): {"old_log_probs": old, "infer_log_probs": logp}}

    def _uncompiled_forward_log_probs(self, data: TensorDict) -> Optional[torch.Tensor]:
        """Diagnostic: forward-only log-probs through the engine's eager path on the same module and weights.

        In graph_trainer mode no block is compiled, so this is the plain eager model math (DEFER_AND_FUSE), against
        which the traced graph's log-probs can be compared without the per-block compile of the eager engine.
        """
        if not getattr(self.engine, "_graph_trainer_mode", False) or not UNCOMPILED_FORWARD:
            return None
        self.engine._graph_trainer_mode = False
        try:
            with self.engine.eval_mode():
                output = self.engine.infer_batch(data.copy(), loss_function=None)
            return _response_values(output["model_output"]["log_probs"], data) * data["response_mask"].float()
        except Exception as e:  # diagnostics only
            log(f"uncompiled forward failed: {e!r}")
            return None
        finally:
            self.engine._graph_trainer_mode = True

    def _uncompiled_grad_norms(self, data: TensorDict) -> Optional[dict[str, float]]:
        """Diagnostic: per-parameter grad norms of one update through the eager path on the graph-mode module.

        Nothing is compiled per block in graph mode, so this is the plain eager forward + backward of the same
        weights: a third opinion when graph and eager_blocks gradients disagree.
        """
        if not getattr(self.engine, "_graph_trainer_mode", False) or not UNCOMPILED_FORWARD:
            return None
        self.engine._graph_trainer_mode = False
        try:
            with self.engine.train_mode(disable_auto_offload=True):
                self.engine.optimizer_zero_grad()
                self.engine.forward_backward_batch(data.copy(), self.loss_fn, forward_only=False)
                norms = self._grad_norms()
                self._uncompiled_embedding_rows = self._embedding_rows()
                self.engine.optimizer_zero_grad()
            return norms
        except Exception as e:  # diagnostics only
            log(f"uncompiled backward failed: {e!r}")
            return None
        finally:
            self.engine._graph_trainer_mode = True

    def _graph_summary(self) -> dict:
        """Diagnostic: op and dtype histogram of the traced graph (graph_trainer mode, rank 0)."""
        traced = getattr(getattr(self.engine, "trainer", None), "_traced_step", None)
        if traced is None:
            return {}
        from collections import Counter

        ops, dtypes = Counter(), Counter()
        for node in traced.gm.graph.nodes:
            if node.op != "call_function":
                continue
            ops[str(node.target)] += 1
            val = node.meta.get("val")
            target = str(node.target)
            watched = ("mm", "pallas", "embedding", "index_put", "scatter", "rms_norm", "sum.dim")
            if any(w in target for w in watched):
                vals = val if isinstance(val, tuple | list) else (val,)
                ins = [a.meta.get("val") for a in node.args if isinstance(a, torch.fx.Node)]
                in_dtypes = ",".join(str(t.dtype).replace("torch.", "") for t in ins if isinstance(t, torch.Tensor))
                out_dtypes = ",".join(str(t.dtype).replace("torch.", "") for t in vals if isinstance(t, torch.Tensor))
                dtypes[f"{target}:{in_dtypes}->{out_dtypes}"] += 1
        return {"num_ops": sum(ops.values()), "top_ops": ops.most_common(40), "matmul_dtypes": dict(dtypes)}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def run_parity_step(self, old_log_probs: dict) -> dict:
        data = self._batch(old_log_probs)
        response_mask = data["response_mask"].float()
        result = {
            "rank": torch.distributed.get_rank(),
            "num_response_tokens": float(response_mask.sum()),
            "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        }
        result["uncompiled_log_probs"] = self._uncompiled_forward_log_probs(data)
        self._uncompiled_embedding_rows = None
        result["uncompiled_param_grad_norms"] = self._uncompiled_grad_norms(data)
        result["uncompiled_embedding_rows"] = self._uncompiled_embedding_rows
        input_ids = data["input_ids"]
        result["token_ids"] = (input_ids.values() if input_ids.is_nested else input_ids.reshape(-1)).cpu()
        result["seq_lens"] = (
            input_ids.offsets().diff().cpu() if input_ids.is_nested else torch.tensor([input_ids.numel()])
        )
        result["vocab_size"] = self.model_config.hf_config.vocab_size
        with self.engine.train_mode(disable_auto_offload=True):
            self.engine.optimizer_zero_grad()
            t0 = time.perf_counter()
            output = self.engine.forward_backward_batch(data.copy(), self.loss_fn, forward_only=False)
            result["fwd_bwd_s"] = time.perf_counter() - t0
            result["graph_summary"] = self._graph_summary() if result["rank"] == 0 else {}
            result["loss"] = float(sum(output["loss"])) if isinstance(output["loss"], list) else float(output["loss"])
            result["metrics"] = _plain_metrics(output["metrics"])
            result["log_probs"] = _response_values(output["model_output"]["log_probs"], data) * response_mask
            result["param_grad_norms"] = self._grad_norms()
            result["embedding_rows"] = self._embedding_rows()
            result["grad_samples"] = self._grad_samples() if result["rank"] == 0 else {}
            result["grad_norm"] = float(self.engine.optimizer_step())
            self.engine.optimizer_zero_grad()
            # Second update on the same micro-batch after one optimizer step (also re-runs the traced graph).
            t0 = time.perf_counter()
            output = self.engine.forward_backward_batch(data.copy(), self.loss_fn, forward_only=False)
            result["fwd_bwd_2_s"] = time.perf_counter() - t0
            result["post_step_loss"] = (
                float(sum(output["loss"])) if isinstance(output["loss"], list) else float(output["loss"])
            )
            result["post_step_log_probs"] = _response_values(output["model_output"]["log_probs"], data) * response_mask
            self.engine.optimizer_zero_grad()
        return result


def _ppo_loss():
    from verl.workers.utils.losses import ppo_loss

    return ppo_loss


def _plain_metrics(metrics) -> dict:
    out = {}
    for key, value in (metrics or {}).items():
        values = value if isinstance(value, list) else [value]
        plain = []
        for v in values:
            if isinstance(getattr(v, "values", None), list):  # verl.utils.metric.Metric
                inner = [float(x.detach().float().mean()) if isinstance(x, torch.Tensor) else x for x in v.values]
                v = sum(inner) / len(inner) if inner else None
            if isinstance(v, torch.Tensor):
                v = v.detach().float().mean().item()
            try:
                plain.append(float(v))
            except (TypeError, ValueError):
                pass
        if plain:
            out[key] = sum(plain) / len(plain)
    return out


def _canonical_name(name: str) -> str:
    """Parameter FQN without the eager activation-checkpoint wrapper (graph mode runs without it)."""
    return name.replace("_checkpoint_wrapped_module.", "")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm() * b.norm())
    return float((a * b).sum()) / denom if denom > 0 else (1.0 if float(a.norm() + b.norm()) == 0 else 0.0)


def discover_tpu_host() -> tuple[str, str, int]:
    """(node id, TPU slice resource or "", chips) of the first alive node with TPUs."""
    for node in ray.nodes():
        resources = node.get("Resources", {})
        if not node.get("Alive") or not resources.get("TPU"):
            continue
        names = [r for r in resources if r.startswith(TPU_SLICE_RESOURCE_PREFIX)]
        return node["NodeID"], (names[0] if names else ""), int(resources["TPU"])
    raise RuntimeError("no alive Ray node with TPU resources")


def create_worker_group(mode: str, chips: int, slice_name: str) -> tuple[RayWorkerGroup, RayResourcePool]:
    tpu = {
        "use_splash_attention": True,
        "tpu_eager_mode": "DEFER_AND_FUSE",
        "use_simple_fsdp": True,
        "seq_bucket_size": SEQ_BUCKET,
        "splash_max_block_size": 1024,
        "moe_paired_grouped_mm": True,
        "moe_native_weight_layout": True,
        "static_micro_batch_len": STATIC_LEN,
    }
    if not FUSED_LCE:
        tpu["use_fused_linear_cross_entropy"] = False
    if mode == "graph_trainer":
        tpu.update({"trainer_mode": "graph_trainer", **GRAPH_OPTIONS})
    engine_config = TorchtitanEngineConfig(
        forward_only=False,
        data_parallel_shard_size=chips,
        tensor_parallel_size=1,
        expert_parallel_size=1,
        pipeline_parallel_size=1,
        use_torch_compile=True,
        attn_type="varlen",
        activation_checkpoint="full",
        reshard_after_forward="always",
        max_seq_len=MAX_PROMPT + MAX_RESPONSE,
        pad_to_length=True,
        pad_to_length_bucket=512,
        seed=SEED,
        engine_kwargs={"tpu": tpu},
    )
    optimizer_config = TorchtitanOptimizerConfig(lr=1e-5, total_training_steps=10, clip_grad=1.0, lr_warmup_steps=0)
    config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=HFModelConfig(path=MODEL_PATH, use_remove_padding=True),
        engine_config=engine_config,
        optimizer_config=optimizer_config,
    )
    pool_kwargs = {"accelerator_type": slice_name} if slice_name else {}
    pool = RayResourcePool(
        process_on_nodes=[chips], use_gpu=True, name_prefix=f"graphtrainer_parity_{mode}", **pool_kwargs
    )
    wg = RayWorkerGroup(
        resource_pool=pool,
        ray_cls_with_init=RayClassWithInitArgs(cls=ray.remote(ParityWorker), config=config),
        device_name="tpu",
    )
    t0 = time.perf_counter()
    wg.reset()  # HF load
    log(f"{mode}: engine ready on {chips} chips in {time.perf_counter() - t0:.1f}s")
    return wg, pool


def _kill(wg: RayWorkerGroup, pool: RayResourcePool) -> None:
    """Kill the workers and free their placement groups, so the next mode's pool can claim the same chips."""
    for worker in getattr(wg, "_workers", []):
        ray.kill(worker)
    for pg in pool.pgs or []:
        ray.util.remove_placement_group(pg)
    time.sleep(15)  # let the TPU runtime release the chips before the next worker group


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def _max_mean(diff: torch.Tensor) -> tuple[float, float]:
    return (float(diff.max()), float(diff.mean())) if diff.numel() else (0.0, 0.0)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= LOSS_RTOL * max(abs(a), abs(b)) + LOSS_ATOL


def embedding_breakdown(eager: list[dict], graph: list[dict]) -> Optional[dict]:
    """Diagnostic: where the token-embedding gradients of the two modes (and the uncompiled reference) differ.

    The batch's tokens are random over the vocabulary, so a row hit by exactly one token is that token's input
    gradient. Rows are bucketed by the token's rank, relative position in its sequence and first / last position;
    per bucket: ``sqrt(sum graph^2 / sum eager^2)`` (``ratio_graph``), the same for the uncompiled reference and the
    cosine over 64 sampled columns.
    """
    if any(r.get("embedding_rows") is None for r in eager + graph):
        return None

    def cat(results, key, field):
        return torch.cat([r[key][field] for r in sorted(results, key=lambda r: r["rank"])])

    e_norm, g_norm = cat(eager, "embedding_rows", "norms"), cat(graph, "embedding_rows", "norms")
    e_cols, g_cols = cat(eager, "embedding_rows", "cols"), cat(graph, "embedding_rows", "cols")
    has_unc = all(r.get("uncompiled_embedding_rows") is not None for r in graph)
    u_norm = cat(graph, "uncompiled_embedding_rows", "norms") if has_unc else None
    num_rows = e_norm.numel()
    rows, ranks, rel_pos, first, last = [], [], [], [], []
    for r in sorted(eager, key=lambda r: r["rank"]):
        start = 0
        for n in r["seq_lens"].tolist():
            ids = r["token_ids"][start : start + n]
            rows.append(ids)
            ranks.append(torch.full((n,), r["rank"]))
            rel_pos.append(torch.arange(n, dtype=torch.float32) / max(n - 1, 1))
            first.append(torch.arange(n) == 0)
            last.append(torch.arange(n) == n - 1)
            start += n
    rows, ranks, rel_pos = torch.cat(rows).long(), torch.cat(ranks), torch.cat(rel_pos)
    first, last = torch.cat(first), torch.cat(last)
    count = torch.bincount(rows, minlength=num_rows)[:num_rows]
    single = count[rows] == 1

    def stats(row_idx: torch.Tensor) -> dict:
        if row_idx.numel() == 0:
            return {"n": 0}
        e2, g2 = (e_norm[row_idx] ** 2).sum(), (g_norm[row_idx] ** 2).sum()
        out = {
            "n": int(row_idx.numel()),
            "eager_norm": float(e2.sqrt()),
            "ratio_graph": float((g2 / e2.clamp(min=1e-30)).sqrt()),
            "cos_graph": _cosine(e_cols[row_idx].reshape(-1), g_cols[row_idx].reshape(-1)),
        }
        if u_norm is not None:
            out["ratio_uncompiled"] = float(((u_norm[row_idx] ** 2).sum() / e2.clamp(min=1e-30)).sqrt())
        return out

    buckets = {"all_hit_rows": stats(torch.nonzero(count > 0).flatten())}
    buckets["unhit_rows"] = stats(torch.nonzero(count == 0).flatten())  # should be ~0: checks the shard layout
    buckets["multi_hit_rows"] = stats(torch.nonzero(count > 1).flatten())
    for k in sorted(set(ranks.tolist())):
        buckets[f"rank{k}"] = stats(rows[single & (ranks == k)])
    shard_rows = [r["embedding_rows"]["norms"].numel() for r in sorted(eager, key=lambda r: r["rank"])]
    bounds = torch.tensor([0] + shard_rows).cumsum(0).tolist()
    hit_rows = torch.nonzero(count > 0).flatten()
    for k in range(len(shard_rows)):  # the destination shard (reduce-scatter output) of the row
        buckets[f"shard{k}"] = stats(hit_rows[(hit_rows >= bounds[k]) & (hit_rows < bounds[k + 1])])
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        buckets[f"pos_{lo:.1f}-{min(hi, 1.0):.1f}"] = stats(rows[single & (rel_pos >= lo) & (rel_pos < hi)])
    buckets["first_token"] = stats(rows[single & first])
    buckets["last_token"] = stats(rows[single & last])
    hit = rows[single]
    big = hit[e_norm[hit] > e_norm[hit].median()]
    ratio = g_norm[big] / e_norm[big].clamp(min=1e-30)
    quantiles = torch.quantile(ratio, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])).tolist() if big.numel() else []
    worst = torch.argsort((e_norm[hit] - g_norm[hit]).abs(), descending=True)[:10]
    token_of_row = {int(t): i for i, t in enumerate(rows.tolist())}
    worst_rows = []
    for w in worst.tolist():
        row = int(hit[w])
        i = token_of_row[row]
        worst_rows.append(
            {
                "row": row,
                "rank": int(ranks[i]),
                "rel_pos": round(float(rel_pos[i]), 3),
                "eager": float(e_norm[row]),
                "graph": float(g_norm[row]),
                "uncompiled": float(u_norm[row]) if u_norm is not None else None,
            }
        )
    return {
        "num_rows": num_rows,
        "buckets": buckets,
        "per_row_ratio_graph_quantiles_5_25_50_75_95": quantiles,
        "worst_rows": worst_rows,
    }


def compare(eager: list[dict], graph: list[dict], old_log_probs: dict) -> dict:
    """graph_trainer vs eager_blocks, per rank, against the eager engine's own program-to-program noise floor.

    Noise floor: the eager engine's forward-only log-probs (no grad, no activation checkpointing: another XLA program
    over the same weights) vs its training-step log-probs. bf16 programs that fuse differently differ by about that
    much, and MoE routing flips make the max large; the graph may differ from eager by a small multiple of it.
    """
    summary: dict = {"ranks": [], "passed": True}
    for e, g in zip(eager, graph, strict=True):
        assert e["rank"] == g["rank"]
        mask = e["log_probs"] != 0
        infer = old_log_probs[e["rank"]]["infer_log_probs"]
        logp = _max_mean((e["log_probs"] - g["log_probs"]).abs()[mask])
        floor = _max_mean((e["log_probs"] - infer).abs()[mask])
        graph_vs_infer = _max_mean((g["log_probs"] - infer).abs()[mask])
        uncompiled = g.get("uncompiled_log_probs")
        if uncompiled is not None:
            graph_vs_uncompiled = _max_mean((g["log_probs"] - uncompiled).abs()[mask])
            eager_vs_uncompiled = _max_mean((e["log_probs"] - uncompiled).abs()[mask])
        else:
            graph_vs_uncompiled = eager_vs_uncompiled = None
        post = _max_mean((e["post_step_log_probs"] - g["post_step_log_probs"]).abs()[mask])
        names = sorted(set(e["param_grad_norms"]) | set(g["param_grad_norms"]))
        missing = sorted(set(e["param_grad_norms"]) ^ set(g["param_grad_norms"]))
        param_rel = {
            n: _rel(e["param_grad_norms"].get(n, 0.0), g["param_grad_norms"].get(n, 0.0))
            for n in names
            if max(e["param_grad_norms"].get(n, 0.0), g["param_grad_norms"].get(n, 0.0)) > 1e-8
        }
        uncompiled_norms = g.get("uncompiled_param_grad_norms") or {}
        top = [n for n, _ in sorted(param_rel.items(), key=lambda kv: -kv[1])[:10]]
        rank_summary_norms = {
            n: (e["param_grad_norms"].get(n), uncompiled_norms.get(n), g["param_grad_norms"].get(n)) for n in top
        }
        rank_summary = {
            "rank": e["rank"],
            "libtpu_init_args": (e.get("libtpu_init_args"), g.get("libtpu_init_args")) if e["rank"] == 0 else None,
            "param_grad_norms_eager_uncompiled_graph": rank_summary_norms if uncompiled_norms else None,
            "uncompiled_total_grad_norm": (sum(v * v for v in uncompiled_norms.values()) ** 0.5)
            if uncompiled_norms
            else None,
            "response_tokens": e["num_response_tokens"],
            "loss": (e["loss"], g["loss"]),
            "post_step_loss": (e["post_step_loss"], g["post_step_loss"]),
            "grad_norm": (e["grad_norm"], g["grad_norm"], _rel(e["grad_norm"], g["grad_norm"])),
            "logp_max_mean_abs": logp,
            "noise_floor_logp_max_mean_abs": floor,
            "graph_vs_eager_infer_logp_max_mean_abs": graph_vs_infer,
            "graph_vs_uncompiled_logp_max_mean_abs": graph_vs_uncompiled,
            "eager_vs_uncompiled_logp_max_mean_abs": eager_vs_uncompiled,
            "graph_summary": g.get("graph_summary") or None,
            "post_step_logp_max_mean_abs": post,
            "param_grad_norm_worst_rel": sorted(param_rel.items(), key=lambda kv: -kv[1])[:10],
            "params_with_grad_in_one_mode_only": missing,
            "metrics": {k: (e["metrics"][k], g["metrics"][k]) for k in sorted(set(e["metrics"]) & set(g["metrics"]))},
            "metric_keys_only_eager": sorted(set(e["metrics"]) - set(g["metrics"])),
            "metric_keys_only_graph": sorted(set(g["metrics"]) - set(e["metrics"])),
            "fwd_bwd_s": (e["fwd_bwd_s"], g["fwd_bwd_s"]),
            "fwd_bwd_2_s": (e["fwd_bwd_2_s"], g["fwd_bwd_2_s"]),
        }
        # The reference noise: eager_blocks vs the uncompiled eager forward of the same weights (two compilations of
        # the same bf16 model math). The graph may differ from eager_blocks by about as much, not more.
        reference = eager_vs_uncompiled if eager_vs_uncompiled is not None else floor
        rank_summary["reference_noise_logp_max_mean_abs"] = reference
        checks = {
            "loss": _close(e["loss"], g["loss"]),
            "post_step_loss": _close(e["post_step_loss"], g["post_step_loss"]),
            "logp_mean": logp[1] <= max(LOGP_ATOL / 5, 1.5 * reference[1]),
            # after one optimizer step the weights also differ (slightly different gradients): twice the noise.
            "post_step_logp_mean": post[1] <= max(LOGP_ATOL / 5, 3.0 * reference[1]),
            "same_params_with_grad": not missing,
            "metric_keys": not rank_summary["metric_keys_only_eager"] and not rank_summary["metric_keys_only_graph"],
        }
        if e.get("grad_samples") and g.get("grad_samples"):
            cos = {
                n: _cosine(e["grad_samples"][n], g["grad_samples"][n])
                for n in e["grad_samples"]
                if n in g["grad_samples"]
            }
            all_e = torch.cat([e["grad_samples"][n] for n in cos])
            all_g = torch.cat([g["grad_samples"][n] for n in cos])
            rank_summary["grad_cosine_all_params"] = _cosine(all_e, all_g)
            rank_summary["grad_cosine_lowest"] = sorted(cos.items(), key=lambda kv: kv[1])[:10]
            checks["grad_cosine"] = rank_summary["grad_cosine_all_params"] >= GRAD_COSINE
        rank_summary["strict"] = {
            "grad_norm_rel<=grad_rtol": rank_summary["grad_norm"][2] <= GRAD_RTOL,
            "logp_max<=logp_atol": logp[0] <= LOGP_ATOL,
        }
        rank_summary["checks"] = checks
        summary["passed"] &= all(checks.values())
        summary["ranks"].append(rank_summary)
    return summary


def main() -> int:
    from verl.plugin.platform import get_platform

    logging.basicConfig(format="%(levelname)s:%(asctime)s:%(name)s: %(message)s")
    logging.getLogger("verl_hardware_plugin.accelerators.tpu.engines.graph_trainer_mode").setLevel(logging.INFO)
    ray.init(**get_platform().get_ray_init_kwargs())
    _, slice_name, host_chips = discover_tpu_host()
    chips = int(os.environ.get("PARITY_CHIPS", host_chips))
    log(
        f"model={MODEL_PATH} layers={NUM_LAYERS} static_len={STATIC_LEN} seqs/rank<={NUM_SEQS * MICRO_BATCHES} "
        f"micro_batches~{MICRO_BATCHES} chips={chips} "
        f"slice={slice_name or '-'} fused_lce={FUSED_LCE} graph_options={GRAPH_OPTIONS} modes={MODES}"
    )

    results: dict[str, list[dict]] = {}
    old_log_probs = None
    for mode in MODES:
        wg = pool = None
        try:
            wg, pool = create_worker_group(mode, chips, slice_name)
            if old_log_probs is None:
                if mode != "eager_blocks":
                    raise RuntimeError("the first run must be eager_blocks: it computes the on-policy old_log_probs")
                old_log_probs = {}
                for part in wg.on_policy_old_log_probs():
                    old_log_probs.update(part)
            results[mode] = sorted(wg.run_parity_step(old_log_probs), key=lambda r: r["rank"])
            r0 = results[mode][0]
            log(
                f"{mode}: rank0 loss={r0['loss']:.6g} grad_norm={r0['grad_norm']:.6g} "
                f"post_step_loss={r0['post_step_loss']:.6g} fwd_bwd={r0['fwd_bwd_s']:.1f}s/{r0['fwd_bwd_2_s']:.1f}s"
            )
        except Exception:
            traceback.print_exc()
            log(f"FAILED: {mode} run raised")
            return 1
        finally:
            if wg is not None:
                _kill(wg, pool)

    if len(results) < 2:
        log(f"ran {list(results)} only; nothing to compare")
        return 0
    summary = compare(results["eager_blocks"], results["graph_trainer"], old_log_probs)
    try:
        summary["embedding_breakdown"] = embedding_breakdown(results["eager_blocks"], results["graph_trainer"])
    except Exception as e:  # diagnostics only
        log(f"embedding breakdown failed: {e!r}")
    summary["config"] = {
        "model": MODEL_PATH,
        "layers": NUM_LAYERS,
        "static_len": STATIC_LEN,
        "chips": chips,
        "fused_lce": FUSED_LCE,
        "graph_options": GRAPH_OPTIONS,
        "tolerances": {
            "logp_atol": LOGP_ATOL,
            "loss_rtol": LOSS_RTOL,
            "loss_atol": LOSS_ATOL,
            "grad_rtol": GRAD_RTOL,
            "grad_cosine": GRAD_COSINE,
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(json.dumps(summary, indent=2, default=str))
    log(f"summary written to {OUT_PATH}")
    log("PASSED" if summary["passed"] else "FAILED")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
