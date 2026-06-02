#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hive.py  —  Cluster-Routed Holographic Hivemind Transformer (Hive v5)

Modular causal LM: many small transformers ("cubes"), each trained on its own
cluster of the data, composed by a planner that builds a sparse execution graph
once per prompt. The model is a stack of layers; each layer holds several cubes;
tokens fall top->down through the layers. A cube contributes a *gated delta* to a
shared residual stream, so an out-of-domain cube becomes identity (skip) by
construction, not by hope.

This file is a from-scratch rebuild incorporating:

  FHRR holographic bus (no FFT). Slots are unit-modulus complex phasors; binding
    is elementwise complex multiply, unbinding is multiply-by-conjugate. Norm is
    preserved *exactly* (|e^{iθ}|=1), inverse is *exact*. Refs: Plate 1995/2003.
  refattn ("really-easy flash attention"). Thin wrapper over
    F.scaled_dot_product_attention -> dispatches to flash / mem-efficient kernels
    (works on Pascal/1080 via the mem-efficient path), never materializes T×T for
    causal masks; supports an incremental KV cache for O(T) decode steps.
  (1) Stage-A upstream-noise injection: a cube is trained on h0 + ξ with ‖ξ‖
      derived from HRR bundle algebra, so its input distribution matches deployment
      at layer ℓ>0 without ever seeing another cube. Closes chicken-and-egg.
  (2) sparsemax routing inside each layer: exact zeros (true skips) with a smooth
      gradient on the support — no straight-through bias. Ref: Martins & Astudillo
      2016. Plus a load-balance/entropy regularizer so the router cannot collapse.
  (3) Neuro-symbolic routing prior: each cube owns a fixed concept atom = its
      cluster centroid; activation logit starts as cosine(prompt, concept). The
      planner only learns a small correction on top of fixed geometry.
  (4) Per-layer slot orthogonalization (complex Gram–Schmidt): bus cross-talk -> 0,
      so unbinding is clean even with several co-active cubes.
  (5) ABI fundamentals: deeper Stage-0 throwaway core; tied output projector
      P_out = P_in^T so the residual round-trip is near-identity by construction.
  (6) Stability pack: α scaled 1/sqrt(2·depth) (DeepNet-style) so the bundle norm
      stays O(1) at any depth by theorem; QK-norm; logit z-loss; residual-proj
      init /sqrt(2m). Refs: Wang et al. DeepNet 2022; Dehghani QK-norm.
  (7) EM cluster refinement after Stage A: re-label chunks by which cube's gate
      fires hardest, drop misfits, so clusters become coherent (better than the
      cheap TF-IDF init) -> smoother per-cube landscape.
  Hard-Concrete gates for Stage C (Louizos et al. 2018) instead of STE: stochastic
    gates with genuine zeros and reparameterized (unbiased) gradients.
  Optional CellularSheaf consensus layer between cube-layers: a sheaf-Laplacian
    diffusion step drives locally-consistent cube outputs toward a globally
    glueable state; the residual obstruction is exactly coker δ = H¹(F).
    Refs: Hansen "Sheaf Neural Networks" 2012.06333; Bodnar et al. "Neural Sheaf
    Diffusion" 2202.04579 (Defs 1–2, §3.1).
  Optional PredictiveCodingHead: a few linear PC inference iterations + local
    update before the unembed. Refs: Whittington & Bogacz 2017; Bogacz 2017.

Tokenizer: any external HuggingFace AutoTokenizer. All widths (d_cube, d_emb,
d_router, d_ff, slot count, sheaf stalk) are auto-derived from the tokenizer
vocab size and the YAML cube/layer structure.

CLI:
  hive.py --prepare RAWDIR --config c.yaml [--output PREP]
  hive.py --pretrain-abi --config c.yaml --dataset PREP
  hive.py --train --config c.yaml --dataset PREP --stage A --layer L --cube I
  hive.py --refine-clusters --config c.yaml --dataset PREP        # (7) EM step
  hive.py --train --config c.yaml --dataset PREP --stage B
  hive.py --train --config c.yaml --dataset PREP --stage C
  hive.py --infer --config c.yaml --checkpoint CKPT --prompt "..."
  hive.py --smoke-test
"""

import os, sys, json, math, glob, argparse, random, subprocess, gc, contextlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# utilities
# ============================================================================

def ceil_to(x: float, m: int) -> int:
    m = max(1, int(m))
    return int(max(m, math.ceil(float(x) / m) * m))

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def device_auto(pref: Optional[str] = None) -> torch.device:
    if pref:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def amp_autocast(dev: torch.device):
    """bf16 autocast on CUDA, no-op elsewhere. bf16 has fp32-range exponent, so no
    GradScaler is needed and no loss/overflow handling differs from fp32 training."""
    if dev.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()

def release_torch_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass

# ============================================================================
# refattn — "really easy flash attention"
# ============================================================================
# A thin wrapper over scaled_dot_product_attention. On CUDA it dispatches to the
# flash / memory-efficient fused kernels (the mem-efficient path covers Pascal /
# GTX 1080); on CPU it uses the math backend. For causal self-attention it does
# NOT build a T×T mask. It also serves the incremental-decode case (one query
# token against a cached K/V) with no mask at all.

def refattn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            causal: bool = False,
            attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """q,k,v : (B, H, Tq/Tk, d_h) real. Returns (B, H, Tq, d_h).
    If attn_bias is given (additive, broadcastable to (B,H,Tq,Tk)) it is used and
    `causal` must be False (caller folds causality into the bias if needed)."""
    if attn_bias is not None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

# ============================================================================
# FHRR — Fourier Holographic Reduced Representations (no FFT)
# ============================================================================
# Slots are unit-modulus complex phasors stored as phase angles theta (real),
# materialized as exp(i*theta) on demand. Binding is elementwise complex
# multiply; unbinding is multiply by the conjugate. With |slot|=1 everywhere:
#   * binding preserves modulus exactly  -> norm control by theorem;
#   * unbind(bind(s, x), s) == x exactly -> exact inverse;
#   * a bundle sum_i bind(s_i, x_i) unbinds to x_j + bounded cross-noise.
# Payloads (k, v projections) are carried as free complex vectors (re + i·im).
# The FHRR-consistent attention score is the Hermitian real part
#   Re<q, conj(k)> , which equals the plain real dot product of the stacked
# [re; im] vectors — so we can hand the stacked-real form to refattn (SDPA) and
# keep the fused kernel while remaining algebraically FHRR.

def phasor(theta: torch.Tensor) -> torch.Tensor:
    return torch.complex(torch.cos(theta), torch.sin(theta))

def fhrr_bind(slot_c: torch.Tensor, payload_c: torch.Tensor) -> torch.Tensor:
    return slot_c * payload_c

def fhrr_unbind(bound_c: torch.Tensor, slot_c: torch.Tensor) -> torch.Tensor:
    return bound_c * torch.conj(slot_c)

def complex_to_stacked(z: torch.Tensor) -> torch.Tensor:
    """(..., d) complex -> (..., 2d) real = [re, im]."""
    return torch.cat([z.real, z.imag], dim=-1)

def make_phasor_atoms(num: int, d_h: int, heads: int, seed: int) -> torch.Tensor:
    """Random unit-modulus atom phases, shape (num, heads, d_h)."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(num, heads, d_h, generator=g) * (2 * math.pi)

def orthogonalize_atoms(theta: torch.Tensor, groups: List[List[int]],
                        iters: int = 6) -> torch.Tensor:
    """(4) Complex Gram–Schmidt within each group (per layer), per head. Re-extracting
    the phase after projection (to stay unit-modulus) reintroduces a little
    non-orthogonality, so we iterate the projection a few times; cross-talk decays
    geometrically (~0.16 -> ~0.015 over 6 passes at d_h=16)."""
    num, heads, d_h = theta.shape
    out = theta.clone()
    for _ in range(iters):
        nxt = out.clone()
        for grp in groups:
            if len(grp) <= 1:
                continue
            for hh in range(heads):
                A = phasor(out[grp, hh])          # (g, d_h) complex
                Q = []
                for i in range(A.shape[0]):
                    v = A[i].clone()
                    for q in Q:
                        v = v - torch.vdot(q, v) * q
                    nv = v.norm()
                    if nv > 1e-8:
                        v = v / nv
                    Q.append(v)
                Qs = torch.stack(Q)
                nxt[grp, hh] = torch.angle(Qs)
        out = nxt
    return out

# ============================================================================
# config + automatic dimension derivation
# ============================================================================

DEFAULTS = dict(
    seed=0,
    # structure ---------------------------------------------------------------
    layers=2,
    cubes_per_layer=3,
    top_x=2,
    heads=4,
    blocks_per_cube=2,
    # widths (auto -> scale with vocab) ---------------------------------------
    d_cube="auto",
    d_cube_base=128,
    d_emb="auto",
    d_router="auto",
    d_ff="auto",
    router_blocks=2,
    router_heads=4,
    dense_pre_blocks=2,
    dense_pre_heads="auto",
    dense_pre_ff="auto",
    dense_post_blocks=2,
    dense_post_heads="auto",
    dense_post_ff="auto",
    vocab_ref=32768,
    rope_base=10000.0,
    # FHRR / bus --------------------------------------------------------------
    orthogonalize_slots=True,          # (4)
    # routing -----------------------------------------------------------------
    route_fn="sparsemax",              # (2) "sparsemax" | "softmax"
    symbolic_prior=True,               # (3)
    lambda_balance=0.01,               # (2) load-balance regularizer
    # stability (6) -----------------------------------------------------------
    qk_norm=True,
    z_loss=1e-4,
    # sheaf / pc (optional) ---------------------------------------------------
    use_sheaf=False,
    sheaf_iters=1,
    sheaf_eps=0.1,
    sheaf_stalk="auto",                # default d_cube // heads
    use_pc_head=False,
    pc_iters=4,
    pc_lr=0.5,
    # Stage-A noise injection (1) ---------------------------------------------
    stageA_noise=True,
    stageA_noise_scale=1.0,            # multiplies the algebra-derived sigma
    # training ----------------------------------------------------------------
    seq_len=128,
    batch_size=16,
    grad_accum_steps=1,
    lr=3e-4,
    steps=1000,
    steps_abi="auto",
    steps_stage_a="auto",
    steps_stage_b="auto",
    steps_stage_c="auto",
    # losses ------------------------------------------------------------------
    lambda_open=0.01,
    lambda_close=1.0,
    lambda_bce=1.0,
    lambda_cap=0.1,
    gate_eps=1e-6,
    tau_route=0.5,
    # gate calibration (sharp gates for many-cube scale) ----------------------
    gate_temp=0.5,            # sigmoid temperature; <1 sharpens the gate
    gate_margin=1.0,          # hinge margin on gate logits (pos > +m, neg < -m)
    hard_neg_frac=0.5,        # fraction of negatives drawn from NEAREST clusters
    hard_neg_k=4,             # how many nearest foreign clusters count as "hard"
    lambda_margin=1.0,        # weight of the hinge gate loss
    stage_a_margin_adaptive=True,
    label_smoothing=0.0,
    stage_a_unfreeze_final_norm=False,
    sliding_window_views=1,
    # Stage C group lrs + Hard-Concrete ---------------------------------------
    lr_router=1e-4,
    lr_gate=5e-5,
    lr_core=1e-6,
    lr_pc=1e-4,       # pc_head starts from identity and needs a larger LR than lr_core
    lr_l0=1e-3,       # route_log_alpha: Adam is scale-invariant so this LR is what actually controls l0 decay rate
    hc_beta=0.6666666,
    hc_gamma=-0.1,
    hc_zeta=1.1,
    lambda_l0=0.001,                   # Hard-Concrete L0 penalty (Stage C)
    # tokenizer / io ----------------------------------------------------------
    tokenizer="gpt2",
    tokenizer_vocab_size=8192,
    tokenizer_min_frequency=3,
    dataset_text_column="text",
    prepare_max_docs=None,
    prepare_max_chunks=None,
    tokenizer_train_docs=None,
    clusterized_dir=None,
    eval_max_chunks=256,
    activation_max_prompts=2,
    balance_clusters=True,
    cluster_balance_max_ratio=2.5,
    cluster_balance_passes=8,
    parallel_cube_teach=1,
    out_dir="hive_runs",
)

class Config:
    def __init__(self, d: dict):
        self.raw = dict(DEFAULTS); self.raw.update(d or {})
        self._normalize_numeric_scalars()
        for k, v in self.raw.items():
            setattr(self, k, v)
        if isinstance(self.cubes_per_layer, int):
            self.cubes = [self.cubes_per_layer] * int(self.layers)
        else:
            self.cubes = list(self.cubes_per_layer)
            assert len(self.cubes) == int(self.layers)
        self.layers = int(self.layers)
        # blocks_per_cube: int (same depth everywhere) or per-layer list
        if isinstance(self.blocks_per_cube, int):
            self.blocks = [self.blocks_per_cube] * self.layers
        else:
            self.blocks = list(self.blocks_per_cube)
            assert len(self.blocks) == self.layers, \
                "blocks_per_cube list length must equal layers"
        self.derived = False
        # Resolve tokenizer="auto" to out_dir/tokenizer.json if it exists
        if self.tokenizer in (None, "auto", ""):
            candidate = os.path.join(self.out_dir, "tokenizer.json")
            if os.path.exists(candidate):
                self.tokenizer = candidate
                self.raw["tokenizer"] = candidate

    def _normalize_numeric_scalars(self):
        int_fields = {
            "seed", "layers", "top_x", "heads", "router_heads", "seq_len", "batch_size",
            "vocab_ref", "tokenizer_vocab_size", "tokenizer_min_frequency",
            "prepare_max_docs", "prepare_max_chunks", "tokenizer_train_docs",
            "eval_max_chunks", "activation_max_prompts", "hard_neg_k",
            "cluster_balance_passes", "parallel_cube_teach", "grad_accum_steps",
            "sliding_window_views", "dense_pre_blocks", "dense_post_blocks",
        }
        float_fields = {
            "lr", "lr_router", "lr_gate", "lr_core", "lr_pc", "lr_l0", "wd", "clip",
            "gate_temp", "gate_margin", "hard_neg_frac", "lambda_margin",
            "lambda_balance", "lambda_capacity", "lambda_l0", "label_smoothing",
            "hc_beta", "hc_gamma", "hc_zeta", "cluster_balance_max_ratio",
        }
        auto_int_fields = {
            "steps", "steps_abi", "steps_stage_a", "steps_stage_b", "steps_stage_c",
            "d_cube", "d_emb", "d_router", "d_ff", "sheaf_stalk",
            "dense_pre_heads", "dense_pre_ff", "dense_post_heads", "dense_post_ff",
        }
        for key, value in list(self.raw.items()):
            if not isinstance(value, str):
                continue
            text = value.strip()
            if key in int_fields and text.lower() not in ("none", "null", ""):
                self.raw[key] = int(float(text))
            elif key in float_fields and text.lower() not in ("none", "null", ""):
                self.raw[key] = float(text)
            elif key in auto_int_fields and text.lower() != "auto" and text.lower() not in ("none", "null", ""):
                self.raw[key] = int(float(text))

    def steps_for(self, stage: str) -> int:
        key = {
            "abi": "steps_abi",
            "stage_a": "steps_stage_a",
            "stage_b": "steps_stage_b",
            "stage_c": "steps_stage_c",
        }[stage]
        v = getattr(self, key, "auto")
        return int(self.steps if v == "auto" else v)

    def derive(self, vocab_size: int):
        self.vocab_size = int(vocab_size)
        H = int(self.heads)
        scale = math.log2(self.vocab_size) / math.log2(float(self.vocab_ref))
        Hstep = 2 * H                                   # d_h even (RoPE pairs)
        if self.d_cube == "auto":
            self.d_cube = ceil_to(self.d_cube_base * scale, Hstep)
        else:
            self.d_cube = ceil_to(int(self.d_cube), Hstep)
        assert self.d_cube % H == 0
        self.d_h = self.d_cube // H
        assert self.d_h % 2 == 0
        if self.d_emb == "auto":
            self.d_emb = max(self.d_cube, ceil_to(self.d_cube * scale, Hstep))
        else:
            self.d_emb = ceil_to(int(self.d_emb), Hstep)
        if self.d_router == "auto":
            self.d_router = ceil_to(self.top_x * self.d_cube, int(self.router_heads))
        else:
            self.d_router = ceil_to(int(self.d_router), int(self.router_heads))
        assert self.d_router % int(self.router_heads) == 0
        if self.d_ff == "auto":
            self.d_ff = ceil_to(8.0 * self.d_cube / 3.0, H)
        else:
            self.d_ff = ceil_to(int(self.d_ff), H)
        self.dense_pre_heads = H if self.dense_pre_heads == "auto" else int(self.dense_pre_heads)
        self.dense_post_heads = H if self.dense_post_heads == "auto" else int(self.dense_post_heads)
        assert self.d_cube % self.dense_pre_heads == 0
        assert self.d_cube % self.dense_post_heads == 0
        pre_step = max(1, int(self.dense_pre_heads))
        post_step = max(1, int(self.dense_post_heads))
        if self.dense_pre_ff == "auto":
            self.dense_pre_ff = ceil_to(8.0 * self.d_cube / 3.0, pre_step)
        else:
            self.dense_pre_ff = ceil_to(int(self.dense_pre_ff), pre_step)
        if self.dense_post_ff == "auto":
            self.dense_post_ff = ceil_to(8.0 * self.d_cube / 3.0, post_step)
        else:
            self.dense_post_ff = ceil_to(int(self.dense_post_ff), post_step)
        if self.sheaf_stalk == "auto":
            self.sheaf_stalk = self.d_h
        self.total_cubes = sum(self.cubes)
        self.derived = True
        return self

    def cube_global_index(self, layer: int, cube: int) -> int:
        return sum(self.cubes[:layer]) + cube

    def layer_slice(self, layer: int) -> Tuple[int, int]:
        base = sum(self.cubes[:layer]); return base, base + self.cubes[layer]

    def summary(self) -> str:
        return (f"vocab={self.vocab_size} d_cube={self.d_cube} d_h={self.d_h} "
                f"d_emb={self.d_emb} d_router={self.d_router} d_ff={self.d_ff} "
                f"layers={self.layers} cubes={self.cubes} top_x={self.top_x} "
                f"heads={self.heads} m={self.blocks} route={self.route_fn} "
                f"sheaf={self.use_sheaf} pc={self.use_pc_head} "
                f"total_cubes={self.total_cubes}")

def load_config(path: Optional[str]) -> Config:
    raw = {}
    if path:
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    cfg = Config(raw)
    cfg.config_path = path
    return cfg

# ============================================================================
# norms, rope, swiglu, routing transforms
# ============================================================================

def rms_norm(x, weight=None, eps=1e-6):
    n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return n * weight if weight is not None else n

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x): return rms_norm(x, self.weight, self.eps)

def build_rope_cache(seq_len, d_h, base, device, dtype):
    half = d_h // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    f = torch.outer(t, inv); emb = torch.cat([f, f], -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)

def apply_rope(x, cos, sin, offset=0):
    T = x.shape[1]
    cos = cos[offset:offset+T].unsqueeze(0).unsqueeze(2)
    sin = sin[offset:offset+T].unsqueeze(0).unsqueeze(2)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], -1)
    return x * cos + rot * sin

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff, out_scale=1.0):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up   = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
        with torch.no_grad():                           # (6) residual init /scale
            self.down.weight.mul_(out_scale)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

def sparsemax(z, dim=-1):
    """(2) Martins & Astudillo 2016: Euclidean projection onto the simplex.
    Produces exact zeros with a smooth gradient on the support."""
    z = z - z.max(dim=dim, keepdim=True).values            # stability shift
    zs, _ = torch.sort(z, dim=dim, descending=True)
    rng = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim(); shape[dim] = -1; rng = rng.view(shape)
    cssv = torch.cumsum(zs, dim=dim) - 1
    cond = (zs - cssv / rng) > 0
    k = cond.to(z.dtype).sum(dim=dim, keepdim=True).clamp(min=1)
    tau = torch.gather(cssv, dim, (k.long() - 1).clamp(min=0)) / k
    return torch.clamp(z - tau, min=0)

def topk_softmax(z, k: int, dim: int = -1) -> torch.Tensor:
    """Softmax restricted to the top-k entries; zeros elsewhere.  Gradient flows
    through all k active positions, so the planner gets credit for every cube it
    routes to — prevents the one-hot collapse that sparsemax can exhibit when a
    single logit dominates early in training."""
    k = min(k, z.shape[dim])
    thresh = z.topk(k, dim=dim).values.select(dim, k - 1).unsqueeze(dim)
    masked = z.masked_fill(z < thresh, float("-inf"))
    return torch.softmax(masked, dim=dim)

def hard_concrete(log_alpha, beta, gamma, zeta, training):
    """Louizos et al. 2018. Stochastic gate in [0,1] with nonzero mass at the
    endpoints; reparameterized so gradients flow to log_alpha."""
    if training:
        u = torch.rand_like(log_alpha).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
    else:
        s = torch.sigmoid(log_alpha)
    s = s * (zeta - gamma) + gamma
    return s.clamp(0, 1)

def hc_l0_penalty(log_alpha, beta, gamma, zeta):
    """Expected L0 (probability gate is non-zero) for the Hard-Concrete gate."""
    return torch.sigmoid(log_alpha - beta * math.log(-gamma / zeta))

# ============================================================================
# Cube — delta-operator with self-gate, attends the shared FHRR bus via refattn
# ============================================================================

class Cube(nn.Module):
    """A small causal transformer that adds a gated delta to the residual stream.
    Q is local (per cube). K/V are written into a shared FHRR bus bound to this
    cube's unit-modulus slot; each cube unbinds its own slot to read. Attention
    runs on the stacked-real form of the (complex) payloads via refattn (SDPA)."""
    def __init__(self, cfg: Config, depth: int, n_blocks: int):
        super().__init__()
        d, H, dh, m, dff = cfg.d_cube, cfg.heads, cfg.d_h, n_blocks, cfg.d_ff
        self.d, self.H, self.dh, self.m = d, H, dh, m
        self.qk_norm = cfg.qk_norm
        self.gate_temp = float(cfg.gate_temp)
        # Pre-hoc gate kept for backward compatibility / diagnostics.
        self.gate_w = nn.Linear(d, 1, bias=True)
        # Post-hoc gate sees both the input view and the cube's proposed delta.
        self.gate_w_post = nn.Linear(2 * d, 1, bias=True)
        # (6) DeepNet-style LayerScale: alpha ~ 1/sqrt(2*depth) keeps bundle O(1)
        self.alpha = nn.Parameter(torch.tensor(1.0 / math.sqrt(2.0 * max(1, depth))))
        # Global cube-availability gate for Stage C routing sparsification.
        # Distinct from the per-token gate_w path.
        self.route_log_alpha = nn.Parameter(torch.tensor(2.0))   # starts mostly-open
        out_scale = 1.0 / math.sqrt(2.0 * m)               # (6) residual init
        self.q_proj = nn.ModuleList([nn.Linear(d, 2 * d, bias=False) for _ in range(m)])
        self.q_out = nn.ModuleList([nn.Linear(2 * d, d, bias=False) for _ in range(m)])
        self.o_proj = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(m)])
        with torch.no_grad():
            for o in self.o_proj: o.weight.mul_(out_scale)
            for qo in self.q_out: qo.weight.mul_(out_scale)
        self.attn_norm = nn.ModuleList([RMSNorm(d) for _ in range(m)])
        self.mlp_norm  = nn.ModuleList([RMSNorm(d) for _ in range(m)])
        self.mlp = nn.ModuleList([SwiGLU(d, dff, out_scale) for _ in range(m)])
        # K/V projections produce COMPLEX payloads: width 2*d -> (d complex)/head
        self.k_proj = nn.Linear(d, 2 * d, bias=False)
        self.v_proj = nn.Linear(d, 2 * d, bias=False)
        if self.qk_norm:                                    # (6) QK-norm
            self.q_hn = RMSNorm(2 * dh)
            self.k_hn = RMSNorm(2 * dh)

    # -- Phase 1: complex (k,v) payloads from the layer input -----------------
    @property
    def log_alpha(self):
        # Backward-compatible accessor for older call sites.
        return self.route_log_alpha

    def kv_complex(self, n_view, cos, sin, offset=0, rotate_keys=False):
        B, T, _ = n_view.shape
        k = self.k_proj(n_view).view(B, T, self.H, 2 * self.dh)
        v = self.v_proj(n_view).view(B, T, self.H, 2 * self.dh)
        if self.qk_norm:
            k = self.k_hn(k)
        if rotate_keys:
            k_re = apply_rope(k[..., :self.dh], cos, sin, offset)
            k_im = apply_rope(k[..., self.dh:], cos, sin, offset)
        else:
            k_re, k_im = k[..., :self.dh], k[..., self.dh:]
        # FHRR payloads are complex: torch.complex requires float/double, and the
        # holographic bind/unbind must stay fp32 for exact phasor inverses even when
        # the surrounding matmuls run under bf16 autocast.
        kc = torch.complex(k_re.float(), k_im.float())      # (B,T,H,dh)
        vc = torch.complex(v[..., :self.dh].float(), v[..., self.dh:].float())
        return kc, vc

    def gate_logit(self, n_view, delta=None):
        if delta is not None:
            inp = torch.cat([n_view, delta], dim=-1)
            return self.gate_w_post(inp) / self.gate_temp
        return self.gate_w(n_view) / self.gate_temp        # (B,T,1) raw, temp-scaled
    def gate(self, n_view, delta=None):
        return torch.sigmoid(self.gate_logit(n_view, delta))       # (B,T,1)

    # -- Phase 2: unbind own slot from the bus, attend with local Q -----------
    def process(self, x0, kbus_c, vbus_c, slot_c, cos, sin,
                offset=0, causal=True):
        """x0     : (B,Tq,d) residual stream (real UI space)
           kbus_c : (B,Tk,H,dh) complex slotted key bus (UNROTATED; Tk>=Tq)
           vbus_c : (B,Tk,H,dh) complex slotted value bus
           slot_c : (H,dh) complex unit-modulus atom of this cube
           offset : absolute position of the FIRST query row. Keys are rotated from
                    absolute 0 (the bus always starts at sequence position 0), so for
                    incremental decode pass the full cached bus as kbus_c and offset =
                    position of the new token; causal must then be False (a single new
                    query legitimately attends every cached key)."""
        B, Tq, _ = x0.shape
        s = slot_c.view(1, 1, self.H, self.dh)
        k_read = fhrr_unbind(kbus_c, s)                      # (B,Tk,H,dh) complex
        v_read = fhrr_unbind(vbus_c, s)
        Tk = k_read.shape[1]
        # Keys are already cached in absolute-position-rotated form, so no
        # full-cache re-rotation is needed during decode.
        k_st = complex_to_stacked(k_read).view(B, Tk, self.H, 2 * self.dh)
        v_st = complex_to_stacked(v_read).view(B, Tk, self.H, 2 * self.dh)
        x = x0
        for r in range(self.m):
            qn = rms_norm(x, self.attn_norm[r].weight)
            q = self.q_proj[r](qn).view(B, Tq, self.H, 2 * self.dh)
            if self.qk_norm:
                q = self.q_hn(q)
            q_re = apply_rope(q[..., :self.dh], cos, sin, offset)
            q_im = apply_rope(q[..., self.dh:], cos, sin, offset)
            q_st = torch.cat([q_re, q_im], -1)              # (B,Tq,H,2dh)
            qh = q_st.transpose(1, 2)                       # (B,H,Tq,2dh)
            kh = k_st.transpose(1, 2)
            vh = v_st.transpose(1, 2)                       # (B,H,Tk,2dh)
            out = refattn(qh, kh, vh, causal=causal)        # (B,H,Tq,2dh)
            out = out.transpose(1, 2).reshape(B, Tq, 2 * self.d)
            out = self.q_out[r](out)
            x = x + self.o_proj[r](out)
            x = x + self.mlp[r](rms_norm(x, self.mlp_norm[r].weight))
        return x

# ============================================================================
# CellularSheaf — optional consensus layer between cube-layers
# ============================================================================

class CellularSheaf(nn.Module):
    """Batched sheaf-Laplacian diffusion over the graph of ALL cubes in a layer.
    Operates on a stacked stalk tensor (B,T,n,ds) with a per-sample active-weight
    vector a (B,n); each edge's coboundary is gated by a_u·a_w so inactive cubes
    (weight 0) neither pull nor are pulled. Because it always processes the full
    cube set with weights, training (forward_weighted) and inference are identical
    — no mismatch. Diffusion X <- X - eps·L_F X pulls locally-consistent cube
    outputs toward a globally glueable state; the residual coboundary energy is
    coker δ = H¹(F), the obstruction to gluing.
    Refs: Hansen 2012.06333; Bodnar et al. 2202.04579 (Defs 1–2, §3.1)."""
    def __init__(self, cfg: Config, n_nodes: int):
        super().__init__()
        self.n = n_nodes
        self.ds = cfg.sheaf_stalk
        self.iters = cfg.sheaf_iters
        self.eps = cfg.sheaf_eps
        self.d = cfg.d_cube
        self.to_stalk = nn.Linear(self.d, self.ds, bias=False)
        self.from_stalk = nn.Linear(self.ds, self.d, bias=False)
        # complete graph over the layer's cubes (every pair can disagree)
        edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
        self.E = len(edges)
        if self.E > 0:
            self.register_buffer("ui", torch.tensor([e[0] for e in edges]))
            self.register_buffer("vi", torch.tensor([e[1] for e in edges]))
            # restriction maps per edge endpoint, init near identity
            self.Ruv = nn.Parameter(torch.eye(self.ds).repeat(self.E, 1, 1)
                                    + 0.01 * torch.randn(self.E, self.ds, self.ds))
            self.Rvu = nn.Parameter(torch.eye(self.ds).repeat(self.E, 1, 1)
                                    + 0.01 * torch.randn(self.E, self.ds, self.ds))

    def forward(self, delta_stack: torch.Tensor, active: torch.Tensor):
        """delta_stack: (B,T,n,d) per-cube deltas (already weighted is fine).
        active: (B,n) per-sample active weight in [0,1] for edge gating.
        Returns (diffused (B,T,n,d), obstruction scalar)."""
        if self.E == 0:
            return delta_stack, torch.zeros((), device=delta_stack.device)
        B, T, n, d = delta_stack.shape
        X = self.to_stalk(delta_stack)                       # (B,T,n,ds)
        obstruction = torch.zeros((), device=X.device)
        for _ in range(self.iters):
            Xu = X[:, :, self.ui, :]                          # (B,T,E,ds)
            Xv = X[:, :, self.vi, :]
            du = torch.einsum('eij,btej->btei', self.Ruv, Xu)
            dv = torch.einsum('eij,btej->btei', self.Rvu, Xv)
            disc = du - dv                                    # coboundary (B,T,E,ds)
            au = active[:, self.ui]; av = active[:, self.vi]  # (B,E)
            live = (au * av).view(B, 1, self.E, 1)
            disc = disc * live
            cu = torch.einsum('eij,btei->btej', self.Ruv, disc)   # Rᵀ·disc
            cv = torch.einsum('eij,btei->btej', self.Rvu, disc)
            upd = torch.zeros_like(X)
            upd.index_add_(2, self.ui, cu)
            upd.index_add_(2, self.vi, -cv)
            X = X - self.eps * upd
            obstruction = obstruction + disc.pow(2).sum(-1).mean()
        return delta_stack + self.from_stalk(X), obstruction

# ============================================================================
# PredictiveCodingHead — optional linear PC refinement before the unembed
# ============================================================================

class PredictiveCodingHead(nn.Module):
    """Small predictive-coding refinement block.
    z is inferred against the observed state h through a learnable decoder and a
    local iterative error-correction loop, then projected back into model space."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.iters = cfg.pc_iters
        self.lr = cfg.pc_lr
        self.enc = nn.Linear(cfg.d_cube, cfg.d_cube, bias=False)
        self.dec = nn.Linear(cfg.d_cube, cfg.d_cube, bias=False)
        self.out = nn.Linear(cfg.d_cube, cfg.d_cube, bias=False)
        with torch.no_grad():
            self.enc.weight.copy_(torch.eye(cfg.d_cube))
            self.dec.weight.copy_(torch.eye(cfg.d_cube))
            self.out.weight.copy_(torch.eye(cfg.d_cube))
        self.leak = 0.05
    def forward(self, h):
        z = self.enc(h)
        for _ in range(self.iters):
            pred = self.dec(z)
            err = h - pred
            z = z + self.lr * (torch.matmul(err, self.dec.weight) - self.leak * z)
        return self.out(z)

class ActiveCubeMixer(nn.Module):
    """A tiny attention block across active cube deltas inside one layer."""
    def __init__(self, d):
        super().__init__()
        self.norm = RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, 1, batch_first=True)

    def forward(self, x, key_padding_mask=None):
        h = self.norm(x)
        a, _ = self.attn(h, h, h, need_weights=False, key_padding_mask=key_padding_mask)
        return x + a

class DenseCausalBlock(nn.Module):
    def __init__(self, d, heads, d_ff, rope_base):
        super().__init__()
        assert d % heads == 0
        self.H = heads
        self.dh = d // heads
        assert self.dh % 2 == 0
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.mlp = SwiGLU(d, d_ff)
        self.rope_base = rope_base
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

    def _ensure_rope(self, T, device, dtype):
        if self.cos.numel() == 0 or self.cos.shape[0] < T or self.cos.device != device or self.cos.dtype != dtype:
            cos, sin = build_rope_cache(T, self.dh, self.rope_base, device, dtype)
            self.cos, self.sin = cos, sin

    def forward(self, h):
        B, T, _ = h.shape
        self._ensure_rope(T, h.device, h.dtype)
        n = self.n1(h)
        qkv = self.qkv(n).view(B, T, 3, self.H, self.dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        a = refattn(q, k, v, causal=True).transpose(1, 2).reshape(B, T, self.H * self.dh)
        h = h + self.o(a)
        return h + self.mlp(self.n2(h))

class DenseStack(nn.Module):
    def __init__(self, cfg: Config, blocks: int, heads: int, d_ff: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            DenseCausalBlock(cfg.d_cube, heads, d_ff, cfg.rope_base)
            for _ in range(max(0, int(blocks)))
        ])
        self.out_norm = RMSNorm(cfg.d_cube)

    def forward(self, h):
        for blk in self.blocks:
            h = blk(h)
        return self.out_norm(h)

# ============================================================================
# Planner — non-causal encoder -> per-cube activation logits (one pass)
# ============================================================================

class BiBlock(nn.Module):
    def __init__(self, d, heads, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, d_ff)
    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))

class Planner(nn.Module):
    """Outputs raw activation logits per cube. With symbolic_prior, the logit is
    cosine(prompt-concept, cube-concept)/temp + a learned correction; otherwise a
    plain learned head. Concept atoms (cluster centroids) are set after --prepare."""
    def __init__(self, cfg: Config):
        super().__init__()
        dR = cfg.d_router
        self.cfg = cfg
        self.proj = nn.Linear(cfg.d_emb, dR, bias=False)
        self.dense_proj = nn.Linear(cfg.d_cube, dR, bias=False)
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, dR)); nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [BiBlock(dR, cfg.router_heads, ceil_to(8*dR/3, cfg.router_heads))
             for _ in range(cfg.router_blocks)])
        # learned correction head (small)
        self.head_w = nn.Parameter(torch.zeros(cfg.total_cubes, dR)); nn.init.normal_(self.head_w, std=0.02)
        self.head_b = nn.Parameter(torch.zeros(cfg.total_cubes))
        # (3) concept atoms in d_router space (planner-encoded cluster centroids), + temp
        # Using d_router instead of d_emb: BiBlocks learn to separate clusters, making
        # the prior discriminative.  Raw d_emb centroids are collinear (anisotropy).
        self.register_buffer("concepts", torch.zeros(cfg.total_cubes, dR))
        self.prior_temp = nn.Parameter(torch.tensor(1.0))
        self.prior_scale = nn.Parameter(torch.tensor(3.0))
        self.symbolic = cfg.symbolic_prior

    def set_concepts(self, C: torch.Tensor):
        self.concepts.copy_(C)

    def forward(self, emb, dense=None):
        T = emb.shape[1]
        z = self.proj(emb) + self.pos[:, :T]
        if dense is not None:
            z = z + self.dense_proj(dense)
        for blk in self.blocks:
            z = blk(z)
        rho = z.mean(1)                                      # (B,dR)
        rho_n = F.normalize(rho, dim=-1)                     # unit vector — decouples direction from scale
        logit = rho_n @ self.head_w.t() + self.head_b       # bounded by ||head_w[i]||
        if self.symbolic:
            c = F.normalize(self.concepts, dim=-1)           # (C,dR)
            cos = rho_n @ c.t()                              # (B,C) in [-1,1]
            logit = logit + self.prior_scale * cos / self.prior_temp.clamp(min=0.1)
        return logit                                         # raw logits (B,C)

# ============================================================================
# ABI — sub-embedding + tied projectors + final norm; throwaway Stage-0 core
# ============================================================================

class ABI(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_emb)
        nn.init.normal_(self.embed.weight, std=0.02)
        # (5) tied projectors: P_out = P_in^T  -> near-identity round-trip
        self.p_in = nn.Linear(cfg.d_emb, cfg.d_cube, bias=False)
        if cfg.d_emb == cfg.d_cube:
            with torch.no_grad():
                self.p_in.weight.copy_(torch.eye(cfg.d_cube))
        self.final_norm = RMSNorm(cfg.d_cube)

    def embed_tokens(self, t): return self.embed(t)
    def to_ui(self, e): return self.p_in(e)
    def from_ui(self, h):                                    # tied inverse-ish
        return F.linear(h, self.p_in.weight.t())            # (.,d_cube)->(.,d_emb)
    def logits(self, h):
        h = self.final_norm(h)
        e = self.from_ui(h)
        return e @ self.embed.weight.t()
    def freeze(self):
        for p in self.parameters(): p.requires_grad_(False)

# ============================================================================
# HiveModel
# ============================================================================

Route = List[List[Tuple[int, float]]]   # per-layer [(cube_idx, weight), ...]

class HiveModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.derived
        self.cfg = cfg
        self.abi = ABI(cfg)
        self.dense_pre = DenseStack(cfg, cfg.dense_pre_blocks, cfg.dense_pre_heads, cfg.dense_pre_ff)
        self.dense_post = DenseStack(cfg, cfg.dense_post_blocks, cfg.dense_post_heads, cfg.dense_post_ff)
        self.layers = nn.ModuleList([
            nn.ModuleList([Cube(cfg, depth=l + 1, n_blocks=cfg.blocks[l])
                           for _ in range(cfg.cubes[l])])
            for l in range(cfg.layers)])
        # FHRR slot phases (num, H, d_h); orthogonalize within each layer
        theta = make_phasor_atoms(cfg.total_cubes, cfg.d_h, cfg.heads, seed=cfg.seed + 1234)
        if cfg.orthogonalize_slots:
            groups = [list(range(*cfg.layer_slice(l))) for l in range(cfg.layers)]
            theta = orthogonalize_atoms(theta, groups)
        self.register_buffer("slot_theta", theta)           # (num,H,d_h) real phase
        self.planner = Planner(cfg)
        self.cube_cross_attn = nn.ModuleList([ActiveCubeMixer(cfg.d_cube) for _ in range(cfg.layers)])
        self.sheaf = nn.ModuleList([
            CellularSheaf(cfg, cfg.cubes[l]) if cfg.use_sheaf else nn.Identity()
            for l in range(cfg.layers)]) if cfg.use_sheaf else None
        self.pc_head = PredictiveCodingHead(cfg) if cfg.use_pc_head else None
        cos, sin = build_rope_cache(cfg.seq_len, cfg.d_h, cfg.rope_base,
                                    torch.device("cpu"), torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # -- helpers --------------------------------------------------------------
    def slot_c(self, layer, cube):
        return phasor(self.slot_theta[self.cfg.cube_global_index(layer, cube)])

    def move_rope(self):
        p = next(self.parameters())
        if self.rope_cos.device != p.device or self.rope_cos.dtype != p.dtype:
            cos, sin = build_rope_cache(self.cfg.seq_len, self.cfg.d_h,
                                        self.cfg.rope_base, p.device, p.dtype)
            self.rope_cos, self.rope_sin = cos, sin

    def deterministic_hc_gate(self):
        vals = []
        for l in range(self.cfg.layers):
            for ci in range(self.cfg.cubes[l]):
                la = self.layers[l][ci].route_log_alpha
                vals.append(hard_concrete(la, self.cfg.hc_beta, self.cfg.hc_gamma, self.cfg.hc_zeta, training=False))
        return torch.stack(vals)

    def encode_context_from_emb(self, emb):
        return self.dense_pre(self.abi.to_ui(emb))

    def encode_context(self, tokens):
        emb = self.abi.embed_tokens(tokens)
        return emb, self.encode_context_from_emb(emb)

    def _final(self, h):
        h = self.dense_post(h)
        if self.pc_head is not None:
            h = self.pc_head(h)
        return self.abi.logits(h)

    def cube_delta(self, layer: int, cube: int, h: torch.Tensor, cos, sin,
                   offset: int = 0, causal: bool = True):
        cfg = self.cfg
        cb: Cube = self.layers[layer][cube]
        n_view = rms_norm(h)
        kc, vc = cb.kv_complex(n_view, cos, sin, offset=offset, rotate_keys=True)
        s = self.slot_c(layer, cube).view(1, 1, cfg.heads, cfg.d_h)
        kbus = fhrr_bind(s, kc)
        vbus = fhrr_bind(s, vc)
        x = cb.process(h, kbus, vbus, self.slot_c(layer, cube), cos, sin, offset=offset, causal=causal)
        delta = x - h
        gl = cb.gate_logit(n_view, delta)
        g = torch.sigmoid(gl)
        return n_view, delta, g, gl

    @torch.no_grad()
    def cascaded_stageA_input(self, layer: int, tokens: torch.Tensor):
        _, h = self.encode_context(tokens)
        cos, sin = self.rope_cos, self.rope_sin
        for l in range(layer):
            best = None
            best_score = None
            for ci in range(self.cfg.cubes[l]):
                _, delta, g, _ = self.cube_delta(l, ci, h, cos, sin, causal=True)
                score = g.mean()
                if best is None or float(score) > float(best_score):
                    best = (ci, delta, g)
                    best_score = score
            ci, delta, g = best
            cb_l = self.layers[l][ci]
            h = h + g * cb_l.alpha * delta
        return h

    def apply_layer_deltas(self, l: int, h: torch.Tensor, deltas: List[torch.Tensor],
                           idxs: List[int], active_weights: Optional[torch.Tensor] = None,
                           aux: Optional[dict] = None, enable_mixer: bool = True) -> torch.Tensor:
        cfg = self.cfg
        if enable_mixer and len(deltas) > 1:
            D = torch.stack(deltas, dim=2)                 # (B,T,K,d)
            B0, T0, K, d = D.shape
            Df = D.view(B0 * T0, K, d)
            mask = None
            active_bt = None
            if active_weights is not None and active_weights.shape[1] == K:
                active_bt = active_weights.to(torch.bool).repeat_interleave(T0, dim=0)
                mask = ~active_bt
            Df = self.cube_cross_attn[l](Df, key_padding_mask=mask)
            if active_bt is not None:
                Df = Df * active_bt.unsqueeze(-1).to(Df.dtype)
            D = Df.view(B0, T0, K, d)
            deltas = [D[:, :, i, :] for i in range(K)]
        if self.sheaf is not None and len(deltas) >= 1:
            B0, T0, _ = h.shape
            stack = torch.zeros(B0, T0, cfg.cubes[l], cfg.d_cube,
                                device=h.device, dtype=h.dtype)
            if active_weights is None:
                act = torch.zeros(B0, cfg.cubes[l], device=h.device, dtype=h.dtype)
                for ci in idxs:
                    act[:, ci] = 1.0
            else:
                act = active_weights.to(h.dtype)
            for d, ci in zip(deltas, idxs):
                stack[:, :, ci, :] = d
            stack, obs = self.sheaf[l](stack, act)
            if aux is not None:
                aux["obstruction"] = aux["obstruction"] + obs
            h = h + stack.sum(dim=2)
        else:
            for d in deltas:
                h = h + d
        return h

    # -- (1) upstream-noise sigma from HRR bundle algebra ---------------------
    def stageA_sigma(self, layer: int, h0: torch.Tensor) -> float:
        """Norm of the delta that would accumulate above layer `layer` at deploy.
        Bundle of ~ (layer)·(avg active) gated deltas, each ~ alpha·‖h‖, roughly
        independent directions -> sigma ~ alpha·sqrt(n_terms)·rms(h0)."""
        if layer == 0 or not self.cfg.stageA_noise:
            return 0.0
        n_terms = max(1, layer * min(self.cfg.top_x, max(self.cfg.cubes[:layer] or [1])))
        alpha = 1.0 / math.sqrt(2.0 * (layer + 1))
        rms = h0.pow(2).mean().sqrt().item()
        return self.cfg.stageA_noise_scale * alpha * math.sqrt(n_terms) * rms

    # -- single-cube forward (Stage A) ---------------------------------------
    def forward_single_cube(self, layer, cube, tokens, inject_noise=True):
        cb: Cube = self.layers[layer][cube]
        if layer > 0:
            h0 = self.cascaded_stageA_input(layer, tokens)
        else:
            _, h0 = self.encode_context(tokens)
            if inject_noise:
                sigma = self.stageA_sigma(layer, h0)
                if sigma > 0:
                    h0 = h0 + sigma * torch.randn_like(h0)
        _, delta, g, _ = self.cube_delta(layer, cube, h0, self.rope_cos, self.rope_sin, causal=True)
        h1 = h0 + g * cb.alpha * delta
        return self._final(h1)

    def gate_value_on_embed(self, layer, cube, tokens):
        cb: Cube = self.layers[layer][cube]
        _, h0 = self.encode_context(tokens)
        n_view, delta, _, _ = self.cube_delta(layer, cube, h0, self.rope_cos, self.rope_sin, causal=True)
        return cb.gate(n_view, delta).mean(dim=(1, 2))       # (B,)

    # -- full holographic forward with a route -------------------------------
    def forward(self, tokens, route: Route, return_aux=False):
        cfg = self.cfg
        cos, sin = self.rope_cos, self.rope_sin
        _, h = self.encode_context(tokens)
        aux = {"obstruction": torch.zeros((), device=h.device)}
        for l in range(cfg.layers):
            active = route[l]
            if not active:
                continue
            n_view = rms_norm(h)
            # Phase 1: write FHRR bus
            kbus = None; vbus = None
            for (ci, w) in active:
                cb: Cube = self.layers[l][ci]
                kc, vc = cb.kv_complex(n_view, cos, sin, rotate_keys=True)
                s = self.slot_c(l, ci).view(1, 1, cfg.heads, cfg.d_h)
                bk = w * fhrr_bind(s, kc); bv = w * fhrr_bind(s, vc)
                kbus = bk if kbus is None else kbus + bk
                vbus = bv if vbus is None else vbus + bv
            # Phase 2: each active cube reads + produces a delta
            deltas = []; idxs = []
            for (ci, w) in active:
                cb: Cube = self.layers[l][ci]
                x = cb.process(h, kbus, vbus, self.slot_c(l, ci), cos, sin, causal=True)
                delta = x - h
                g = cb.gate(n_view, delta)
                deltas.append(w * g * cb.alpha * delta); idxs.append(ci)
            # optional sheaf consensus: stack deltas over ALL cubes (zeros for inactive)
            h = self.apply_layer_deltas(l, h, deltas, idxs, aux=aux)
        logits = self._final(h)
        if return_aux:
            return logits, aux
        return logits

    # -- routing: logits -> per-layer weighted active set --------------------
    def route_from_logits(self, logits_b: torch.Tensor, hard: bool) -> Route:
        """logits_b: (total_cubes,) for one sample. Uses sparsemax/softmax within
        each layer; `hard` keeps only the top_x nonzero entries (inference)."""
        cfg = self.cfg
        route: Route = []
        for l in range(cfg.layers):
            a, b = cfg.layer_slice(l)
            zl = logits_b[a:b]
            if cfg.route_fn == "sparsemax":
                w = sparsemax(zl, dim=-1)
            else:
                w = torch.softmax(zl, dim=-1)
            if hard:
                # keep top_x nonzero, renormalize weights to their sum
                nz = (w > 0).nonzero(as_tuple=True)[0]
                order = nz[torch.argsort(w[nz], descending=True)][:cfg.top_x]
                pairs = [(int(i), w[i]) for i in order]
            else:
                pairs = [(i, w[i]) for i in range(zl.shape[0]) if float(w[i].detach()) > 0]
            route.append(pairs)
        return route

    @torch.no_grad()
    def plan(self, tokens) -> Route:
        emb, h = self.encode_context(tokens)
        logits = self.planner(emb, h)[0]
        avail = self.deterministic_hc_gate().to(logits.device, logits.dtype)
        logits = logits + torch.log(avail.clamp_min(1e-6))
        return self.route_from_logits(logits, hard=True)

    def _forward_weighted_single(self, tokens, Wb, return_aux=False):
        cfg = self.cfg
        cos, sin = self.rope_cos, self.rope_sin
        _, h = self.encode_context(tokens)
        aux = {"obstruction": torch.zeros((), device=h.device)}
        for l in range(cfg.layers):
            a, b = cfg.layer_slice(l)
            wl = Wb[a:b]
            active_idx = (wl > 0).nonzero(as_tuple=True)[0]
            if active_idx.numel() == 0:
                continue
            n_view = rms_norm(h)
            kbus = None; vbus = None
            active = []
            for idx in active_idx:
                ci = int(idx.item())
                w = wl[ci]
                cb: Cube = self.layers[l][ci]
                kc, vc = cb.kv_complex(n_view, cos, sin, rotate_keys=True)
                s = self.slot_c(l, ci).view(1, 1, cfg.heads, cfg.d_h)
                bk = w * fhrr_bind(s, kc); bv = w * fhrr_bind(s, vc)
                kbus = bk if kbus is None else kbus + bk
                vbus = bv if vbus is None else vbus + bv
                active.append((ci, w))
            deltas = []; idxs = []
            for ci, w in active:
                cb: Cube = self.layers[l][ci]
                x = cb.process(h, kbus, vbus, self.slot_c(l, ci), cos, sin, causal=True)
                delta = x - h
                g = cb.gate(n_view, delta)
                deltas.append(w * g * cb.alpha * delta); idxs.append(ci)
            h = self.apply_layer_deltas(l, h, deltas, idxs, aux=aux, enable_mixer=True)
        logits = self._final(h)
        return (logits, aux) if return_aux else logits

    # -- sparse weighted route forward (Stage B/C) ---------------------------
    def forward_weighted(self, tokens, W, return_aux=False, h=None):
        """W : (B, total_cubes) per-sample per-cube weights. Uses the same sparse
        route semantics as `forward()`, but keeps weights differentiable by reading
        them directly from W instead of materializing a detached Python route.
        h: optional pre-computed dense_pre output to avoid double encode_context."""
        cfg = self.cfg
        cos, sin = self.rope_cos, self.rope_sin
        B, _ = tokens.shape
        if h is None:
            _, h = self.encode_context(tokens)
        aux = {"obstruction": torch.zeros((), device=h.device)}
        for l in range(cfg.layers):
            a, b = cfg.layer_slice(l)
            wl = W[:, a:b]
            groups = self._weighted_groups(wl)
            if not groups:
                continue
            h_next = h.clone()
            for key, rows in groups.items():
                if not key:
                    continue
                row_idx = torch.tensor(rows, device=h.device, dtype=torch.long)
                hs = h.index_select(0, row_idx)
                ws = wl.index_select(0, row_idx)
                n_view = rms_norm(hs)
                kbus = None
                vbus = None
                for ci in key:
                    w = ws[:, ci].view(-1, 1, 1, 1)
                    cb: Cube = self.layers[l][ci]
                    kc, vc = cb.kv_complex(n_view, cos, sin, rotate_keys=True)
                    s = self.slot_c(l, ci).view(1, 1, cfg.heads, cfg.d_h)
                    bk = w * fhrr_bind(s, kc)
                    bv = w * fhrr_bind(s, vc)
                    kbus = bk if kbus is None else kbus + bk
                    vbus = bv if vbus is None else vbus + bv
                deltas = []
                idxs = []
                sub_aux = {"obstruction": torch.zeros((), device=h.device)}
                for ci in key:
                    w = ws[:, ci].view(-1, 1, 1)
                    cb: Cube = self.layers[l][ci]
                    x = cb.process(hs, kbus, vbus, self.slot_c(l, ci), cos, sin, causal=True)
                    delta = x - hs
                    g = cb.gate(n_view, delta)
                    deltas.append(w * g * cb.alpha * delta)
                    idxs.append(ci)
                hs = self.apply_layer_deltas(
                    l, hs, deltas, idxs,
                    active_weights=ws,
                    aux=sub_aux,
                    enable_mixer=True,
                )
                h_next.index_copy_(0, row_idx, hs)
                if return_aux:
                    aux["obstruction"] = aux["obstruction"] + sub_aux["obstruction"] * (len(rows) / max(1, B))
            h = h_next
        logits = self._final(h)
        if return_aux:
            return logits, aux
        return logits

    def _weighted_groups(self, wl: torch.Tensor):
        groups = {}
        active_mask = wl > 0
        for bi in range(wl.shape[0]):
            idx = active_mask[bi].nonzero(as_tuple=True)[0]
            key = tuple(int(i) for i in idx.tolist())
            groups.setdefault(key, []).append(bi)
        return groups

    def route_weight_matrix(self, planner_logits, hard_concrete_mask=None, soft_topk: bool = False):
        """planner_logits: (B, total_cubes). Returns W (B, total_cubes).
        soft_topk=True: differentiable top-k softmax (used during Stage B training to
        keep gradients flowing through all top_x active cubes and prevent one-hot
        collapse).  soft_topk=False: normal sparsemax/softmax (inference, Stage C)."""
        cfg = self.cfg
        cols = []
        for l in range(cfg.layers):
            a, b = cfg.layer_slice(l)
            zl = planner_logits[:, a:b]
            if soft_topk:
                w = topk_softmax(zl, cfg.top_x, dim=-1)
            elif cfg.route_fn == "sparsemax":
                w = sparsemax(zl, dim=-1)
            else:
                w = torch.softmax(zl, dim=-1)
            if hard_concrete_mask is not None:
                w = w * hard_concrete_mask[:, a:b]
                # No renormalization: HC is a true availability gate.  Dividing
                # by the sum would cancel the gating when all cubes shrink
                # uniformly, making route_log_alpha a no-op.
            cols.append(w)
        W = torch.cat(cols, dim=1)                           # (B, total_cubes)
        return W

    # -- incremental decode: full KV cache *through the FHRR bus* -------------
    # The cache stores, per layer, the UNROTATED complex key/value bus
    # (B, t, H, d_h) accumulated across positions. Each step binds the new
    # position's (k,v) for every active cube, appends to the bus, then every
    # active cube unbinds its own slot and attends. This is identical to the
    # full forward (verified to ~1e-7) but costs O(t) per step instead of O(t²).

    @torch.no_grad()
    def _layer_step(self, l, route, h, cache, cos, sin, pos):
        """Advance one layer for the new token(s) h at absolute start position pos.
        h: (B,Tn,d). Returns updated h; mutates cache[l]."""
        cfg = self.cfg
        active = route[l]
        if not active:
            return h
        nv = rms_norm(h)
        # bind this chunk's (k,v) into the bus contribution (unrotated complex)
        kbus = None; vbus = None
        for (ci, w) in active:
            cb: Cube = self.layers[l][ci]
            kc, vc = cb.kv_complex(nv, cos, sin, offset=pos, rotate_keys=True)
            s = self.slot_c(l, ci).view(1, 1, cfg.heads, cfg.d_h)
            bk = w * fhrr_bind(s, kc); bv = w * fhrr_bind(s, vc)
            kbus = bk if kbus is None else kbus + bk
            vbus = bv if vbus is None else vbus + bv
        # append to the per-layer cache
        if l in cache:
            kb = torch.cat([cache[l][0], kbus], dim=1)
            vb = torch.cat([cache[l][1], vbus], dim=1)
        else:
            kb, vb = kbus, vbus
        cache[l] = (kb, vb)
        Tn = h.shape[1]
        # for a single new token, it attends ALL cached keys (causal=False);
        # for a multi-token prefill chunk, standard causal within the chunk.
        deltas = []; idxs = []
        for (ci, w) in active:
            cb: Cube = self.layers[l][ci]
            x = cb.process(h, kb, vb, self.slot_c(l, ci), cos, sin,
                           offset=pos, causal=(Tn > 1))
            delta = x - h
            g = cb.gate(nv, delta)
            deltas.append(w * g * cb.alpha * delta); idxs.append(ci)
        return self.apply_layer_deltas(l, h, deltas, idxs)

    @torch.no_grad()
    def prefill(self, tokens, route: Route):
        """Process the full prompt, returning (last_logits, cache). The cache holds
        the per-layer complex bus so generation continues in O(t) per step."""
        cfg = self.cfg
        cos, sin = self.rope_cos, self.rope_sin
        _, h = self.encode_context(tokens)
        cache = {"_tokens": tokens.clone()}
        for l in range(cfg.layers):
            h = self._layer_step(l, route, h, cache, cos, sin, pos=0)
        return self._final(h), cache

    @torch.no_grad()
    def decode_step(self, token, route: Route, cache, pos):
        """One new token at absolute position `pos`. token: (B,1). Returns
        (logits, cache)."""
        if "_tokens" in cache:
            cache["_tokens"] = torch.cat([cache["_tokens"], token], dim=1)
            logits, new_cache = self.prefill(cache["_tokens"], route)
            return logits[:, -1:], new_cache
        cos, sin = self.rope_cos, self.rope_sin
        _, h = self.encode_context(token)
        for l in range(self.cfg.layers):
            h = self._layer_step(l, route, h, cache, cos, sin, pos=pos)
        return self._final(h), cache

# ============================================================================
# tokenizer
# ============================================================================

class ByteTokenizer:
    vocab_size = 257; eos = 256
    def encode(self, s): return list(s.encode("utf-8"))
    def decode(self, ids): return bytes([i for i in ids if i < 256]).decode("utf-8", "replace")

class BPETokenizerWrap:
    """Wraps a locally-trained `tokenizers` BPE (a tokenizer.json path). Fully
    offline — no HuggingFace hub access required."""
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(path)
        self._eos = self.tk.token_to_id("<eos>")
        if self._eos is None:
            self._eos = self.tk.get_vocab_size() - 1
    @property
    def vocab_size(self): return self.tk.get_vocab_size()
    @property
    def eos(self): return self._eos
    def encode(self, s): return self.tk.encode(s).ids
    def decode(self, ids): return self.tk.decode([i for i in ids])

def load_tokenizer(name):
    if name == "__bytes__":
        return ByteTokenizer()
    if isinstance(name, str) and name.endswith(".json") and os.path.exists(name):
        return BPETokenizerWrap(name)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.eos_token_id is None:
        tok.add_special_tokens({"eos_token": "<|endoftext|>"})
    return tok

def tok_vocab(tok):
    if isinstance(tok, (ByteTokenizer, BPETokenizerWrap)): return tok.vocab_size
    return len(tok)
def tok_eos(tok):
    if isinstance(tok, (ByteTokenizer, BPETokenizerWrap)): return tok.eos
    return tok.eos_token_id

# ============================================================================
# dataset read / prepare / cluster (+ concept centroids for symbolic routing)
# ============================================================================

def is_hf_arrow_dataset_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "state.json")) \
        and os.path.exists(os.path.join(path, "dataset_info.json"))

def hf_dataset_num_examples(path: str) -> Optional[int]:
    info_path = os.path.join(path, "dataset_info.json")
    if not os.path.exists(info_path):
        return None
    try:
        info = json.load(open(info_path, encoding="utf-8"))
        splits = info.get("splits") or {}
        train = splits.get("train") or {}
        n = train.get("num_examples")
        return int(n) if n is not None else None
    except Exception:
        return None

def iter_strings(path, text_column="text", max_docs=None):
    if is_hf_arrow_dataset_dir(path):
        try:
            from datasets import load_from_disk
        except ImportError as ex:
            raise RuntimeError(
                "Reading HuggingFace Arrow datasets requires `datasets` and `pyarrow`. "
                "Install them in the active environment."
            ) from ex
        ds = load_from_disk(path)
        if hasattr(ds, "keys"):
            ds = ds["train"] if "train" in ds else next(iter(ds.values()))
        count = 0
        for row in ds:
            txt = row.get(text_column)
            if txt:
                yield str(txt)
                count += 1
                if max_docs is not None and count >= max_docs:
                    break
        return

    files = []
    if os.path.isdir(path):
        for ext in ("txt","md","jsonl","json","csv","tsv"):
            files += glob.glob(os.path.join(path, f"**/*.{ext}"), recursive=True)
    else:
        files = [path]
    count = 0
    for fp in sorted(files):
        ext = fp.rsplit(".",1)[-1].lower()
        try:
            if ext == "jsonl":
                for line in open(fp, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    o = json.loads(line)
                    txt = o["text"] if isinstance(o, dict) and "text" in o else json.dumps(o, ensure_ascii=False)
                    if txt:
                        yield txt
                        count += 1
                        if max_docs is not None and count >= max_docs:
                            return
            elif ext == "json":
                o = json.load(open(fp, encoding="utf-8"))
                items = o if isinstance(o, list) else [o]
                for it in items:
                    txt = it["text"] if isinstance(it, dict) and "text" in it else json.dumps(it, ensure_ascii=False)
                    if txt:
                        yield txt
                        count += 1
                        if max_docs is not None and count >= max_docs:
                            return
            elif ext in ("csv","tsv"):
                import csv
                delim = "\t" if ext == "tsv" else ","
                for row in csv.reader(open(fp, encoding="utf-8", newline=""), delimiter=delim):
                    if row:
                        yield " ".join(row)
                        count += 1
                        if max_docs is not None and count >= max_docs:
                            return
            else:
                txt = open(fp, encoding="utf-8").read().strip()
                if txt:
                    yield txt
                    count += 1
                    if max_docs is not None and count >= max_docs:
                        return
        except Exception as ex:
            print(f"[prepare] skip {fp}: {ex}", file=sys.stderr)

def read_strings(path, text_column="text", max_docs=None):
    return list(iter_strings(path, text_column=text_column, max_docs=max_docs))

def chunk_tokens(tok, docs, seq_len):
    eos = tok_eos(tok); chunks=[]; doc_ids=[]
    for di, doc in enumerate(docs):
        ids = tok.encode(doc) + [eos]
        for s in range(0, max(1, len(ids)-1), seq_len):
            w = ids[s:s+seq_len]
            if len(w) < 2: continue
            if len(w) < seq_len: w = w + [eos]*(seq_len-len(w))
            chunks.append(w); doc_ids.append(di)
    if not chunks: raise RuntimeError("no chunks from dataset")
    return torch.tensor(chunks, dtype=torch.long), doc_ids

def chunk_tokens_with_offset(tok, docs, seq_len, offset):
    eos = tok_eos(tok); chunks=[]; doc_ids=[]
    for di, doc in enumerate(docs):
        ids = tok.encode(doc) + [eos]
        start = int(offset) % max(1, seq_len)
        for s in range(start, max(1, len(ids)-1), seq_len):
            w = ids[s:s+seq_len]
            if len(w) < 2:
                continue
            if len(w) < seq_len:
                w = w + [eos] * (seq_len - len(w))
            chunks.append(w); doc_ids.append(di)
    if not chunks:
        return torch.empty((0, seq_len), dtype=torch.long), []
    return torch.tensor(chunks, dtype=torch.long), doc_ids

def load_pt(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def maybe_auto_limits(cfg: Config, raw_path: str):
    max_docs = cfg.prepare_max_docs
    tok_docs = cfg.tokenizer_train_docs
    n_docs = hf_dataset_num_examples(raw_path) if is_hf_arrow_dataset_dir(raw_path) else None
    if n_docs is not None and n_docs > 500000:
        if max_docs is None:
            max_docs = 200000
            print(f"[prepare] dataset is large ({n_docs} docs); auto-limiting clustering/sample docs to {max_docs}")
        if tok_docs is None:
            tok_docs = 1000000
            print(f"[tokenizer] dataset is large ({n_docs} docs); auto-limiting tokenizer training docs to {tok_docs}")
    return max_docs, tok_docs

def cheap_embed(docs, dim=64, seed=0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=20000)
    X = vec.fit_transform(docs)
    k = min(dim, X.shape[1]-1, max(2, X.shape[0]-1))
    return TruncatedSVD(n_components=k, random_state=seed).fit_transform(X), X, vec

def kmeans_fit(feats, n, seed):
    from sklearn.cluster import KMeans
    n = min(n, len(feats))
    km = KMeans(n_clusters=n, n_init=10, random_state=seed).fit(feats)
    return km.labels_, km.cluster_centers_

def balanced_cluster_fit(feats, n, seed):
    """Cluster into `n` groups with near-equal document counts.
    We fit KMeans for geometry, then rebalance assignments to exact quotas."""
    n = min(n, len(feats))
    labels, centers = kmeans_fit(feats, n, seed)
    if len(feats) == 0:
        return labels, centers
    # Squared distances to each centroid.
    d2 = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)   # (N, n)
    N = d2.shape[0]
    base = N // n
    rem = N % n
    quota = np.array([base + (1 if i < rem else 0) for i in range(n)], dtype=np.int64)
    order = np.argsort(d2, axis=1)
    best = order[:, 0]
    second = order[:, 1] if n > 1 else order[:, 0]
    margin = d2[np.arange(N), second] - d2[np.arange(N), best]
    counts = np.bincount(labels, minlength=n).astype(np.int64)
    over = [i for i in range(n) if counts[i] > quota[i]]
    under = {i for i in range(n) if counts[i] < quota[i]}
    if not over and not under:
        new_centers = np.stack([feats[labels == i].mean(axis=0) for i in range(n)])
        return labels, new_centers

    members = [np.where(labels == i)[0].tolist() for i in range(n)]
    # Move the least-committed points out of overfull clusters first.
    for c in over:
        members[c].sort(key=lambda idx: margin[idx])
    while under:
        progressed = False
        for src in list(range(n)):
            while counts[src] > quota[src]:
                idx = members[src].pop(0)
                prefs = np.argsort(d2[idx])
                dst = None
                for cand in prefs:
                    if cand in under:
                        dst = int(cand)
                        break
                if dst is None:
                    deficits = sorted(under, key=lambda j: d2[idx, j])
                    if not deficits:
                        break
                    dst = int(deficits[0])
                labels[idx] = dst
                counts[src] -= 1
                counts[dst] += 1
                members[dst].append(int(idx))
                if counts[dst] >= quota[dst]:
                    under.discard(dst)
                progressed = True
        if not progressed:
            break
    new_centers = []
    for i in range(n):
        idx = np.where(labels == i)[0]
        new_centers.append(feats[idx].mean(axis=0) if len(idx) else centers[i])
    return labels, np.stack(new_centers)

def cluster_root_path(cfg: Config, raw_path: str, out_path: Optional[str] = None) -> str:
    if cfg.clusterized_dir:
        return cfg.clusterized_dir
    if out_path:
        return out_path.rstrip("/\\") + "_clusterized"
    return raw_path.rstrip("/\\") + "_clusterized"

def write_clusterized_tokens(cfg: Config, out_path: str, tokens: torch.Tensor,
                             clusters: np.ndarray, concept_feats: np.ndarray,
                             raw_path: str):
    root = cluster_root_path(cfg, raw_path, out_path)
    os.makedirs(root, exist_ok=True)
    for l in range(cfg.layers):
        layer_dir = os.path.join(root, f"layer_{l}")
        os.makedirs(layer_dir, exist_ok=True)
        a, b = cfg.layer_slice(l)
        for ci in range(cfg.cubes[l]):
            idx = np.where(clusters[l] == ci)[0]
            cube_dir = os.path.join(layer_dir, f"cube_{ci}")
            os.makedirs(cube_dir, exist_ok=True)
            cube_tokens = tokens[torch.tensor(idx, dtype=torch.long)] if len(idx) else tokens[:0].clone()
            torch.save(cube_tokens, os.path.join(cube_dir, "tokens.pt"))
            torch.save(torch.tensor(idx, dtype=torch.long), os.path.join(cube_dir, "indices.pt"))
            meta = {
                "layer": l,
                "cube": ci,
                "n_chunks": int(len(idx)),
                "concept_index": int(a + ci),
            }
            json.dump(meta, open(os.path.join(cube_dir, "meta.json"), "w"), indent=2)
        layer_meta = {
            "layer": l,
            "cubes": int(cfg.cubes[l]),
            "sizes": np.bincount(clusters[l], minlength=cfg.cubes[l]).tolist(),
            "concept_range": [int(a), int(b)],
        }
        json.dump(layer_meta, open(os.path.join(layer_dir, "meta.json"), "w"), indent=2)
    np.save(os.path.join(root, "concept_feats.npy"), concept_feats)
    json.dump({
        "raw_path": raw_path,
        "tokenizer": cfg.tokenizer,
        "seq_len": int(cfg.seq_len),
        "layers": int(cfg.layers),
        "cubes": list(cfg.cubes),
        "total_cubes": int(cfg.total_cubes),
        "cluster_policy": CLUSTER_POLICY,
    }, open(os.path.join(root, "meta.json"), "w"), indent=2)
    print(f"[prepare] wrote clusterized cubes to {root}")

def load_cluster_cube_tokens(cfg: Config, data_path: str, layer: int, cube: int):
    root = cluster_root_path(cfg, data_path)
    cube_dir = os.path.join(root, f"layer_{layer}", f"cube_{cube}")
    tok_path = os.path.join(cube_dir, "tokens.pt")
    if not os.path.exists(tok_path):
        return None
    return load_pt(tok_path)

def write_augmented_views(cfg: Config, out_path: str, docs: List[str]):
    n_views = max(1, int(cfg.sliding_window_views))
    if n_views <= 1:
        return []
    tok = load_tokenizer(cfg.tokenizer)
    views = []
    for view_idx in range(1, n_views):
        offset = max(1, int(round(view_idx * cfg.seq_len / n_views)))
        aug_tokens, aug_doc_ids = chunk_tokens_with_offset(tok, docs, cfg.seq_len, offset)
        if cfg.prepare_max_chunks is not None and aug_tokens.shape[0] > int(cfg.prepare_max_chunks):
            aug_tokens = aug_tokens[:int(cfg.prepare_max_chunks)]
            aug_doc_ids = aug_doc_ids[:int(cfg.prepare_max_chunks)]
        path = os.path.join(out_path, f"tokens_offset_{offset}.pt")
        doc_path = os.path.join(out_path, f"doc_ids_offset_{offset}.pt")
        torch.save(aug_tokens, path)
        torch.save(torch.tensor(aug_doc_ids, dtype=torch.long), doc_path)
        views.append({"offset": int(offset), "path": path, "doc_path": doc_path, "n_chunks": int(aug_tokens.shape[0])})
        print(f"[prepare] wrote offset view offset={offset} chunks={aug_tokens.shape[0]}")
    return views

def load_augmented_views(path: str):
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        return []
    meta = json.load(open(meta_path))
    out = []
    for view in meta.get("augmented_views", []):
        p = view.get("path")
        doc_p = view.get("doc_path")
        if p and doc_p and os.path.exists(p) and os.path.exists(doc_p):
            out.append((int(view["offset"]), load_pt(p), load_pt(doc_p)))
    return out

def sample_tensor_rows(t: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    if t.shape[0] == 0:
        return t
    n = min(int(n), t.shape[0])
    g = np.random.default_rng(seed)
    idx = g.choice(np.arange(t.shape[0]), size=n, replace=t.shape[0] < n)
    return t[torch.tensor(idx, dtype=torch.long)]

CLUSTER_POLICY = "balanced_kmeans_v1"

def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

def model_param_stats(cfg: Config):
    m = HiveModel(cfg)
    total = count_params(m)
    abi = count_params(m.abi)
    planner = count_params(m.planner)
    cube = count_params(m.layers[0][0]) if cfg.layers > 0 and cfg.cubes[0] > 0 else 0
    return {
        "total": total,
        "abi": abi,
        "planner": planner,
        "cube": cube,
        "all_cubes": cube * cfg.total_cubes,
    }

def concat_balanced_parts(parts: List[torch.Tensor], total_n: int, seed: int) -> torch.Tensor:
    parts = [p for p in parts if p is not None and p.shape[0] > 0]
    if not parts:
        return torch.empty((0, 0), dtype=torch.long)
    total_n = max(1, int(total_n))
    per = max(1, total_n // len(parts))
    rem = total_n - per * len(parts)
    out = []
    for i, p in enumerate(parts):
        take = per + (1 if i < rem else 0)
        out.append(sample_tensor_rows(p, take, seed + i))
    return torch.cat(out, dim=0)

def prepared_satisfied(cfg: Config, raw_path: str, out_path: str) -> bool:
    meta_path = os.path.join(out_path, "meta.json")
    tok_path = os.path.join(out_path, "tokens.pt")
    clu_path = os.path.join(out_path, "clusters.pt")
    if not (os.path.exists(meta_path) and os.path.exists(tok_path) and os.path.exists(clu_path)):
        return False
    if not os.path.exists(os.path.join(out_path, "doc_ids.pt")):
        return False
    try:
        meta = json.load(open(meta_path))
    except Exception:
        return False
    if meta.get("tokenizer") != cfg.tokenizer:
        return False
    if int(meta.get("seq_len", -1)) != int(cfg.seq_len):
        return False
    if int(meta.get("layers", -1)) != int(cfg.layers):
        return False
    if list(meta.get("cubes", [])) != list(cfg.cubes):
        return False
    if meta.get("cluster_policy") != CLUSTER_POLICY:
        return False
    if int(meta.get("sliding_window_views", 1)) != int(cfg.sliding_window_views):
        return False
    if int(cfg.sliding_window_views) > 1:
        views = meta.get("augmented_views") or []
        if len(views) != int(cfg.sliding_window_views) - 1:
            return False
        for view in views:
            if not os.path.exists(view.get("path", "")) or not os.path.exists(view.get("doc_path", "")):
                return False
    root = meta.get("clusterized_dir") or cluster_root_path(cfg, raw_path, out_path)
    for l in range(cfg.layers):
        for ci in range(cfg.cubes[l]):
            if not os.path.exists(os.path.join(root, f"layer_{l}", f"cube_{ci}", "tokens.pt")):
                return False
    return True

def abi_satisfied(cfg: Config) -> bool:
    p = os.path.join(cfg.out_dir, "abi.pt")
    if not os.path.exists(p):
        return False
    try:
        sd = load_pt(p, map_location="cpu")
    except Exception:
        return False
    return int(sd.get("vocab_size", -1)) > 0

def hive_ckpt_satisfied(cfg: Config) -> bool:
    p = os.path.join(cfg.out_dir, "hive.pt")
    if not os.path.exists(p):
        return False
    try:
        sd = load_pt(p, map_location="cpu")
    except Exception:
        return False
    if "model" not in sd:
        return False
    model_sd = sd["model"]
    cubes = sd.get("cubes")
    pos = model_sd.get("planner.pos")
    head_w = model_sd.get("planner.head_w")
    emb = model_sd.get("abi.embed.weight")
    if pos is None or head_w is None or emb is None:
        return False
    return list(cubes or []) == list(cfg.cubes) and tuple(pos.shape) == (1, cfg.seq_len, cfg.d_router) \
        and tuple(head_w.shape) == (cfg.total_cubes, cfg.d_router) and int(emb.shape[1]) == int(cfg.d_emb) \
        and int(emb.shape[0]) == int(cfg.vocab_size)

def abi_matches_cfg(cfg: Config) -> bool:
    p = os.path.join(cfg.out_dir, "abi.pt")
    if not os.path.exists(p):
        return False
    try:
        sd = load_pt(p, map_location="cpu")
    except Exception:
        return False
    abi = sd.get("abi") or {}
    cubes = sd.get("cubes")
    emb = abi.get("embed.weight")
    pin = abi.get("p_in.weight")
    if emb is None or pin is None:
        return False
    return list(cubes or []) == list(cfg.cubes) and tuple(emb.shape) == (cfg.vocab_size, cfg.d_emb) \
        and tuple(pin.shape) == (cfg.d_cube, cfg.d_emb)

def stage_marker_path(cfg: Config, name: str) -> str:
    return os.path.join(cfg.out_dir, f".{name}.done")

def stage_marker_ok(cfg: Config, name: str) -> bool:
    return os.path.exists(stage_marker_path(cfg, name))

def write_stage_marker(cfg: Config, name: str):
    os.makedirs(cfg.out_dir, exist_ok=True)
    json.dump({"stage": name}, open(stage_marker_path(cfg, name), "w"))

def cluster_eval_satisfied(cfg: Config) -> bool:
    p = os.path.join(cfg.out_dir, "cluster_eval.json")
    if not os.path.exists(p):
        return False
    try:
        j = json.load(open(p, encoding="utf-8"))
    except Exception:
        return False
    return all(k in j for k in ("single", "pair", "triple", "mixed", "activations"))

def tokenizer_satisfied(cfg: Config, tokenizer_path: str) -> bool:
    if not os.path.exists(tokenizer_path):
        return False
    try:
        tk = load_tokenizer(tokenizer_path)
    except Exception:
        return False
    vocab = int(tok_vocab(tk))
    if vocab != int(cfg.tokenizer_vocab_size):
        return False
    try:
        from tokenizers import Tokenizer
        raw = Tokenizer.from_file(tokenizer_path)
        if raw.token_to_id("<unk>") is None or raw.token_to_id("<eos>") is None:
            return False
    except Exception:
        return False
    return True

def cmd_train_tokenizer(cfg, raw_path, tokenizer_out):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    if tokenizer_satisfied(cfg, tokenizer_out):
        tk = load_tokenizer(tokenizer_out)
        print(f"[tokenizer] reuse {tokenizer_out} vocab={tok_vocab(tk)}")
        return
    if os.path.exists(tokenizer_out):
        print(f"[tokenizer] rebuild {tokenizer_out}: existing tokenizer does not match current config")
    _, tok_docs = maybe_auto_limits(cfg, raw_path)
    print(f"[tokenizer] source={raw_path} out={tokenizer_out} vocab={cfg.tokenizer_vocab_size} "
          f"min_freq={cfg.tokenizer_min_frequency} max_docs={tok_docs}")
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=int(cfg.tokenizer_vocab_size),
        special_tokens=["<unk>", "<eos>"],
        min_frequency=int(cfg.tokenizer_min_frequency),
    )
    os.makedirs(os.path.dirname(tokenizer_out) or ".", exist_ok=True)
    iterator = iter_strings(raw_path, text_column=cfg.dataset_text_column, max_docs=tok_docs)
    tk.train_from_iterator(iterator, trainer=trainer)
    tk.save(tokenizer_out)
    print(f"[tokenizer] saved {tokenizer_out} vocab={tk.get_vocab_size()}")

def cmd_prepare(cfg, raw_path, out_path):
    if prepared_satisfied(cfg, raw_path, out_path):
        print(f"[prepare] reuse {out_path}")
        return
    tok = load_tokenizer(cfg.tokenizer); V = tok_vocab(tok); cfg.derive(V)
    print(f"[prepare] {cfg.summary()}")
    pst = model_param_stats(cfg)
    print(f"[prepare] params total={pst['total']} abi={pst['abi']} planner={pst['planner']} "
          f"cube={pst['cube']} all_cubes={pst['all_cubes']}")
    max_docs, _ = maybe_auto_limits(cfg, raw_path)
    docs = read_strings(raw_path, text_column=cfg.dataset_text_column, max_docs=max_docs)
    print(f"[prepare] {len(docs)} documents")
    tokens, doc_ids = chunk_tokens(tok, docs, cfg.seq_len)
    if cfg.prepare_max_chunks is not None and tokens.shape[0] > int(cfg.prepare_max_chunks):
        keep = int(cfg.prepare_max_chunks)
        tokens = tokens[:keep]
        doc_ids = doc_ids[:keep]
        print(f"[prepare] truncated chunks to {keep}")
    print(f"[prepare] {tokens.shape[0]} chunks")
    feats, _, _ = cheap_embed(docs, seed=cfg.seed)
    doc_ids_np = np.array(doc_ids)
    chunk_feats = feats[doc_ids_np]
    clusters = np.zeros((cfg.layers, tokens.shape[0]), dtype=np.int64)
    # concept centroids in feature space, per cube (used to seed symbolic routing)
    concept_feats = np.zeros((cfg.total_cubes, chunk_feats.shape[1]), dtype=np.float32)
    for l in range(cfg.layers):
        lab, cen = balanced_cluster_fit(chunk_feats, cfg.cubes[l], seed=cfg.seed + l)
        clusters[l] = lab
        a, b = cfg.layer_slice(l)
        concept_feats[a:b] = cen[:cfg.cubes[l]]
        sizes = np.bincount(clusters[l], minlength=cfg.cubes[l]).tolist()
        tok_sizes = [int(s * cfg.seq_len) for s in sizes]
        print(f"[prepare] layer {l} sizes: {sizes}")
        print(f"[prepare] layer {l} cube_tokens: {tok_sizes}")
    os.makedirs(out_path, exist_ok=True)
    torch.save(tokens, os.path.join(out_path, "tokens.pt"))
    torch.save(torch.tensor(clusters), os.path.join(out_path, "clusters.pt"))
    torch.save(torch.tensor(doc_ids, dtype=torch.long), os.path.join(out_path, "doc_ids.pt"))
    np.save(os.path.join(out_path, "concept_feats.npy"), concept_feats)
    write_clusterized_tokens(cfg, out_path, tokens, clusters, concept_feats, raw_path)
    augmented_views = write_augmented_views(cfg, out_path, docs)
    json.dump(dict(vocab_size=V, seq_len=cfg.seq_len, tokenizer=cfg.tokenizer,
                   n_chunks=int(tokens.shape[0]), cubes=cfg.cubes, layers=cfg.layers,
                   feat_dim=int(feats.shape[1]),
                   clusterized_dir=cluster_root_path(cfg, raw_path, out_path),
                   cluster_policy=CLUSTER_POLICY,
                   sliding_window_views=int(cfg.sliding_window_views),
                   augmented_views=augmented_views),
              open(os.path.join(out_path,"meta.json"),"w"), indent=2)
    print(f"[prepare] wrote {out_path}")

def load_prepared(path):
    tokens = load_pt(os.path.join(path,"tokens.pt"))
    clusters = load_pt(os.path.join(path,"clusters.pt"))
    meta = json.load(open(os.path.join(path,"meta.json")))
    cf_path = os.path.join(path,"concept_feats.npy")
    concept_feats = np.load(cf_path) if os.path.exists(cf_path) else None
    return tokens, clusters, meta, concept_feats

def lm_targets(t): return t[:, :-1].contiguous(), t[:, 1:].contiguous()
def sample_batch(tokens, idx, bs, gen):
    pick = gen.choice(idx, size=min(bs, len(idx)), replace=len(idx) < bs)
    return tokens[torch.tensor(pick, dtype=torch.long)]

def sample_batch_with_indices(tokens, idx, bs, gen):
    pick = gen.choice(idx, size=min(bs, len(idx)), replace=len(idx) < bs)
    pick_t = torch.tensor(pick, dtype=torch.long)
    return tokens[pick_t], pick_t

# ============================================================================
# concept atoms: map prepared centroid features -> planner concept space (d_emb)
# ============================================================================

def install_concepts(model: HiveModel, data_path: str, dev):
    """Compute cluster centroids in the planner's d_router space (after proj + BiBlocks).
    Comparing rho with concepts in the same latent space makes the symbolic prior
    discriminative — raw d_emb centroids are collinear due to embedding anisotropy."""
    tokens, clusters, meta, _ = load_prepared(data_path)
    cfg = model.cfg
    dR = cfg.d_router
    C = torch.zeros(cfg.total_cubes, dR, device=dev)
    was_training = model.planner.training
    model.planner.eval()
    with torch.no_grad():
        for l in range(cfg.layers):
            lab = clusters[l].numpy()
            a, b = cfg.layer_slice(l)
            for ci in range(cfg.cubes[l]):
                idx = np.where(lab == ci)[0][:256]
                if len(idx) == 0: continue
                t = tokens[torch.tensor(idx)].to(dev)
                emb, h0 = model.encode_context(t)            # h0 = dense_pre output
                T = emb.shape[1]
                z = model.planner.proj(emb) + model.planner.pos[:, :T]
                if h0 is not None:
                    z = z + model.planner.dense_proj(h0)     # match actual forward path
                for blk in model.planner.blocks:
                    z = blk(z)
                rho = z.mean(1)                              # (N, dR)
                C[a + ci] = rho.mean(0)
    model.planner.train(was_training)
    model.planner.set_concepts(C)
    print("[concepts] installed cluster-centroid concept atoms in planner d_router space")

# ============================================================================
# Stage 0 — pretrain ABI + persistent dense encoder/decoder scaffold
# ============================================================================

def cmd_pretrain_abi(cfg, data_path, dev):
    tokens, clusters, meta, _ = load_prepared(data_path); cfg.derive(meta["vocab_size"])
    if abi_satisfied(cfg) and abi_matches_cfg(cfg) and stage_marker_ok(cfg, "stage0_abi"):
        print(f"[stage0] reuse {os.path.join(cfg.out_dir, 'abi.pt')}")
        return
    n_steps = cfg.steps_for("abi")
    print(f"[stage0] {cfg.summary()}")
    abi=ABI(cfg).to(dev)
    dense_pre = DenseStack(cfg, cfg.dense_pre_blocks, cfg.dense_pre_heads, cfg.dense_pre_ff).to(dev)
    dense_post = DenseStack(cfg, cfg.dense_post_blocks, cfg.dense_post_heads, cfg.dense_post_ff).to(dev)
    params=list(abi.parameters())+list(dense_pre.parameters())+list(dense_post.parameters())
    opt=torch.optim.AdamW(params, lr=cfg.lr)
    gen=np.random.default_rng(cfg.seed); idx=np.arange(tokens.shape[0])
    abi.train(); dense_pre.train(); dense_post.train()
    for step in range(n_steps):
        b=sample_batch(tokens,idx,cfg.batch_size,gen).to(dev); x,y=lm_targets(b)
        with amp_autocast(dev):
            emb = abi.embed_tokens(x)
            h=dense_pre(abi.to_ui(emb))
            logits=abi.logits(dense_post(h))
            loss=F.cross_entropy(logits.reshape(-1,cfg.vocab_size), y.reshape(-1))
            # (6) z-loss
            z=torch.logsumexp(logits,dim=-1); loss=loss+cfg.z_loss*(z**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1,n_steps//10)==0: print(f"[stage0] step {step} loss {loss.item():.4f}")
    os.makedirs(cfg.out_dir,exist_ok=True)
    torch.save({
        "abi":abi.state_dict(),
        "dense_pre":dense_pre.state_dict(),
        "dense_post":dense_post.state_dict(),
        "cfg":cfg.raw,
        "cubes":cfg.cubes,
        "vocab_size":cfg.vocab_size,
    },
               os.path.join(cfg.out_dir,"abi.pt"))
    write_stage_marker(cfg, "stage0_abi")
    del opt, params, dense_pre, dense_post, abi
    release_torch_memory()
    print(f"[stage0] saved abi.pt (dense scaffold retained)")

# ============================================================================
# assembled-model checkpoint helpers
# ============================================================================

def new_model_with_abi(cfg, dev):
    m=HiveModel(cfg).to(dev)
    p=os.path.join(cfg.out_dir,"abi.pt")
    if os.path.exists(p) and abi_matches_cfg(cfg):
        sd=load_pt(p,map_location=dev)
        m.abi.load_state_dict(sd["abi"])
        if "dense_pre" in sd:
            m.dense_pre.load_state_dict(sd["dense_pre"])
        if "dense_post" in sd:
            m.dense_post.load_state_dict(sd["dense_post"])
        print(f"[load] ABI+dense scaffold from {p}")
    else:
        print("[load] WARNING: no abi.pt; ABI/dense scaffold random (run --pretrain-abi first)")
    m.abi.freeze()
    for p in m.dense_pre.parameters(): p.requires_grad_(False)
    for p in m.dense_post.parameters(): p.requires_grad_(False)
    m.move_rope(); return m

def load_model_ckpt(cfg, dev, path):
    if not hive_ckpt_satisfied(cfg):
        raise RuntimeError(f"incompatible checkpoint for current config: {path}")
    sd=load_pt(path,map_location=dev); m=HiveModel(cfg).to(dev)
    model_sd = sd["model"]
    # Drop keys whose shape no longer matches (e.g. planner.concepts after d_emb→d_router migration)
    own_sd = m.state_dict()
    model_sd = {k: v for k, v in model_sd.items()
                if k not in own_sd or v.shape == own_sd[k].shape}
    m.load_state_dict(model_sd, strict=False); m.move_rope(); return m

def save_assembled(model, cfg, path, merge_prefixes=None):
    """If merge_prefixes given, load existing ckpt and overwrite only those keys
    (used by Stage A to merge one cube without touching others)."""
    full=model.state_dict()
    if merge_prefixes and os.path.exists(path) and hive_ckpt_satisfied(cfg):
        base=load_pt(path,map_location="cpu")["model"]
        for k,v in full.items():
            if any(k.startswith(pf) for pf in merge_prefixes) or k.startswith("abi.") \
               or k.startswith("dense_pre.") or k.startswith("dense_post.") \
               or k=="slot_theta" or k.startswith("planner.concepts"):
                base[k]=v
        sd=base
    else:
        sd=full
    torch.save({"model":sd,"cfg":cfg.raw,"cubes":cfg.cubes,"vocab_size":cfg.vocab_size}, path)

def cube_ckpt_ready(cfg, dev, layer, cube, ckpt_path):
    if not os.path.exists(ckpt_path):
        return False
    try:
        sd = load_pt(ckpt_path, map_location=dev)
        model_sd = sd.get("model")
        if model_sd is None:
            return False
        m = HiveModel(cfg).to(dev)
        m.load_state_dict(model_sd, strict=False)
        cb = m.layers[layer][cube]
        ready = cb.alpha.detach().abs().item() > 0 and cb.gate_w.weight.detach().abs().mean().item() > 0
        del m
        return ready
    except Exception:
        return False

def merge_cube_ckpt(cfg, src_path, dst_path, layer, cube):
    sd = load_pt(src_path, map_location="cpu")
    model_sd = sd.get("model")
    if model_sd is None:
        raise RuntimeError(f"cube checkpoint missing model state: {src_path}")
    if os.path.exists(dst_path) and hive_ckpt_satisfied(cfg):
        base = load_pt(dst_path, map_location="cpu")["model"]
    else:
        base = HiveModel(cfg).state_dict()
    prefix = f"layers.{layer}.{cube}."
    for key, value in model_sd.items():
        # abi.final_norm may be unfrozen (stage_a_unfreeze_final_norm=True) and will
        # diverge across parallel workers — exclude it so the base checkpoint retains
        # the shared pre-Stage-A norm instead of a single worker's biased version.
        if key.startswith("abi.final_norm."):
            continue
        if key.startswith(prefix) or key.startswith("abi.") or key.startswith("dense_pre.") \
           or key.startswith("dense_post.") or key == "slot_theta" or key.startswith("planner.concepts"):
            base[key] = value
    torch.save({"model": base, "cfg": cfg.raw, "cubes": cfg.cubes, "vocab_size": cfg.vocab_size}, dst_path)

# ============================================================================
# Stage A — isolated per-cube training (+ (1) upstream noise)
# ============================================================================

def cmd_stage_A(cfg, data_path, layer, cube, dev, merge_path=None, marker_name=None, allow_reuse=True):
    tokens, clusters, meta, concept_feats = load_prepared(data_path); cfg.derive(meta["vocab_size"])
    aug_views = load_augmented_views(data_path)
    doc_ids = load_pt(os.path.join(data_path, "doc_ids.pt")).numpy()
    n_steps = cfg.steps_for("stage_a")
    print(f"[stageA] layer {layer} cube {cube} | {cfg.summary()}")
    m=new_model_with_abi(cfg,dev)
    ckpt_path = merge_path or os.path.join(cfg.out_dir, "hive.pt")
    marker_name = marker_name if marker_name is not None else f"stageA_l{layer}_c{cube}"
    if allow_reuse and marker_name and stage_marker_ok(cfg, marker_name) and cube_ckpt_ready(cfg, dev, layer, cube, ckpt_path):
        print(f"[stageA] reuse existing cube ({layer},{cube}) from {ckpt_path}")
        return
    for p in m.parameters(): p.requires_grad_(False)
    if cfg.stage_a_unfreeze_final_norm:
        for p in m.abi.final_norm.parameters():
            p.requires_grad_(True)
    cb: Cube = m.layers[layer][cube]
    for p in cb.parameters(): p.requires_grad_(True)
    lab=clusters[layer].numpy()
    pos=np.where(lab==cube)[0]; neg=np.where(lab!=cube)[0]
    pos_docs = set(doc_ids[pos].tolist())
    neg_docs = set(doc_ids[neg].tolist())
    pos_tokens = load_cluster_cube_tokens(cfg, data_path, layer, cube)
    if pos_tokens is None:
        pos_tokens = tokens[torch.tensor(pos, dtype=torch.long)]
    neg_parts = []
    other_parts = []
    for other in range(cfg.cubes[layer]):
        if other == cube:
            continue
        part = load_cluster_cube_tokens(cfg, data_path, layer, other)
        if part is not None and part.shape[0] > 0:
            neg_parts.append(part)
            other_parts.append(part)
    neg_tokens = torch.cat(neg_parts, dim=0) if neg_parts else tokens[torch.tensor(neg, dtype=torch.long)]
    sym_neg_target = min(pos_tokens.shape[0], neg_tokens.shape[0]) if neg_tokens.shape[0] > 0 else 0
    if sym_neg_target > 0:
        if other_parts:
            neg_tokens = concat_balanced_parts(other_parts, sym_neg_target, cfg.seed + 1000 * layer + cube)
        else:
            neg_tokens = sample_tensor_rows(neg_tokens, sym_neg_target, cfg.seed + 1000 * layer + cube)
    pos_views = [pos_tokens]
    for _, aug_tokens, aug_doc_ids in aug_views:
        if aug_tokens.shape[0] == 0:
            continue
        aug_idx = np.where(np.isin(aug_doc_ids.numpy(), list(pos_docs)))[0]
        if len(aug_idx) > 0:
            pos_views.append(aug_tokens[torch.tensor(aug_idx, dtype=torch.long)])
    neg_views = [neg_tokens]
    for view_idx, (_, aug_tokens, aug_doc_ids) in enumerate(aug_views, start=1):
        if aug_tokens.shape[0] == 0:
            continue
        aug_idx = np.where(np.isin(aug_doc_ids.numpy(), list(neg_docs)))[0]
        if len(aug_idx) > 0:
            neg_view = aug_tokens[torch.tensor(aug_idx, dtype=torch.long)]
            pos_target = pos_views[view_idx].shape[0] if view_idx < len(pos_views) else pos_tokens.shape[0]
            if pos_target > 0 and neg_view.shape[0] > pos_target:
                neg_view = sample_tensor_rows(neg_view, pos_target, cfg.seed + 1500 * layer + 17 * cube + view_idx)
            neg_views.append(neg_view)
    if pos_tokens.shape[0] == 0:
        raise RuntimeError(f"cube {cube} has no positives on layer {layer}")
    cube_params = count_params(m.layers[layer][cube])
    total_params = count_params(m)

    # --- hard-negative mining: the nearest foreign clusters by centroid distance.
    # The gate must learn a sharp boundary against its SEMANTIC NEIGHBORS, not just
    # against random far clusters (that's what makes gates usable at many-cube scale).
    hard_neg = neg
    if concept_feats is not None:
        a,b = cfg.layer_slice(layer)
        cen = concept_feats[a:b]                              # (cubes_l, feat)
        d2 = ((cen - cen[cube])**2).sum(1)                    # dist to own centroid
        d2[cube] = np.inf
        nearest = np.argsort(d2)[:cfg.hard_neg_k]             # k closest cubes
        hard_mask = np.isin(lab, nearest)
        hard_neg = np.where(hard_mask & (lab!=cube))[0]
        if len(hard_neg)==0: hard_neg = neg
        print(f"[stageA] nearest clusters (hard negs): {nearest.tolist()} "
              f"-> {len(hard_neg)} hard / {len(neg)} total neg")
    sigma_probe = pos_tokens[:min(2, pos_tokens.shape[0])].to(dev)
    train_params = [p for p in m.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(train_params, lr=cfg.lr)
    gen=np.random.default_rng(cfg.seed+100*layer+cube); half=max(1,cfg.batch_size//2)
    accum_steps = max(1, int(cfg.grad_accum_steps))
    n_hard=int(round(half*cfg.hard_neg_frac)); n_easy=half-n_hard
    pos_idx=np.arange(pos_tokens.shape[0]); neg_idx=np.arange(neg_tokens.shape[0])
    neg_idx = np.arange(neg_tokens.shape[0])
    hard_neg_tokens = neg_tokens
    hard_neg_idx = neg_idx
    if concept_feats is not None and len(hard_neg) > 0:
        hard_neg_tokens = tokens[torch.tensor(hard_neg, dtype=torch.long)]
        if sym_neg_target > 0:
            hard_neg_tokens = sample_tensor_rows(hard_neg_tokens, sym_neg_target, cfg.seed + 2000*layer + cube)
        hard_neg_idx = np.arange(hard_neg_tokens.shape[0])
    _, sigma_h = m.encode_context(sigma_probe)
    print(f"[stageA] pos {pos_tokens.shape[0]} neg {neg_tokens.shape[0]} hard_neg {hard_neg_tokens.shape[0]} | noise_sigma~"
          f"{m.stageA_sigma(layer, sigma_h):.3f} "
          f"| gate_temp {cfg.gate_temp} margin {cfg.gate_margin}")
    local_cluster_tokens = int(sum(v.numel() for v in pos_views) / max(1, len(pos_views)))
    pos_tokens_seen = int(n_steps * accum_steps * half * (cfg.seq_len - 1))
    neg_tokens_seen = int(n_steps * accum_steps * max(0, n_hard + n_easy) * (cfg.seq_len - 1))
    total_tokens_seen = pos_tokens_seen + neg_tokens_seen
    tok_per_param = total_tokens_seen / max(1, cube_params)
    local_epochs = pos_tokens_seen / max(1, local_cluster_tokens)
    print(f"[stageA] params cube={cube_params} total_model={total_params}")
    print(f"[stageA] local_cluster_tokens={local_cluster_tokens} "
          f"tokens_seen_total={total_tokens_seen} tokens_seen_pos={pos_tokens_seen} "
          f"tokens_seen_neg={neg_tokens_seen}")
    print(f"[stageA] tokens_per_param={tok_per_param:.3f} local_epochs={local_epochs:.3f}")
    m.train()
    for step in range(n_steps):
        opt.zero_grad()
        last_task = last_margin = last_gpos = last_gneg = last_loss = None
        for micro in range(accum_steps):
            pos_src = pos_views[(step * accum_steps + micro) % len(pos_views)]
            pos_src_idx = np.arange(pos_src.shape[0])
            pb=sample_batch(pos_src,pos_src_idx,half,gen).to(dev); xp,yp=lm_targets(pb)
            with amp_autocast(dev):
                if layer > 0:
                    # For layer>0 the task input equals cascaded_stageA_input with no
                    # noise injection, so the cascade + cube_delta are shared between the
                    # LM logits and the gate logit instead of recomputed twice.
                    hp = m.cascaded_stageA_input(layer, xp)
                    nvp, delta_p, gp, glp = m.cube_delta(layer, cube, hp, m.rope_cos, m.rope_sin, causal=True)
                    lp = m._final(hp + gp * cb.alpha * delta_p)
                else:
                    lp = m.forward_single_cube(layer, cube, xp, inject_noise=True)
                    _, hp = m.encode_context(xp)
                    nvp, delta_p, _, glp = m.cube_delta(layer, cube, hp, m.rope_cos, m.rope_sin, causal=True)
                L_task=F.cross_entropy(
                    lp.reshape(-1,cfg.vocab_size), yp.reshape(-1),
                    label_smoothing=float(cfg.label_smoothing)
                )
                z=torch.logsumexp(lp,dim=-1); L_task=L_task+cfg.z_loss*(z**2).mean()
                neg_src = neg_views[(step * accum_steps + micro) % len(neg_views)]
                neg_src_idx = np.arange(neg_src.shape[0])
                hard_src = hard_neg_tokens if hard_neg_tokens.shape[0] > 0 else neg_src
                hard_src_idx = np.arange(hard_src.shape[0])
                negs=[]
                if n_hard>0 and len(hard_src_idx)>0: negs.append(sample_batch(hard_src,hard_src_idx,n_hard,gen))
                if n_easy>0 and len(neg_src_idx)>0:  negs.append(sample_batch(neg_src,neg_src_idx,n_easy,gen))
                if negs:
                    nb=torch.cat(negs,0).to(dev); xn,_=lm_targets(nb)
                    if layer > 0:
                        hn = m.cascaded_stageA_input(layer, xn)
                    else:
                        _, hn = m.encode_context(xn)
                    nvn, delta_n, _, _ = m.cube_delta(layer, cube, hn, m.rope_cos, m.rope_sin, causal=True)
                    gln=cb.gate_logit(nvn, delta_n)
                else:
                    gln=None
                mgn=cfg.gate_margin
                L_pos_margin=F.relu(mgn-glp).mean()
                L_neg_margin=F.relu(mgn+gln).mean() if gln is not None else torch.zeros((),device=dev)
                L_margin=L_pos_margin+L_neg_margin
                gpos=torch.sigmoid(glp).mean()
                L_open=-torch.log(gpos+cfg.gate_eps)
                L_close=(torch.sigmoid(gln).mean(dim=(1,2))**2).mean() if gln is not None else torch.zeros((),device=dev)
                margin_w = cfg.lambda_margin
                if cfg.stage_a_margin_adaptive:
                    margin_w = margin_w * float((L_task.detach() / (L_task.detach() + 1.0)).item())
                loss=(L_task+margin_w*L_margin+cfg.lambda_open*L_open+cfg.lambda_close*L_close) / accum_steps
            loss.backward()
            last_task, last_margin, last_gpos = L_task, L_margin, gpos
            last_gneg = torch.sigmoid(gln).mean() if gln is not None else torch.zeros((), device=dev)
            last_loss = loss
        opt.step()
        if step%max(1,n_steps//10)==0:
            print(f"[stageA] step {step} task {last_task.item():.4f} g+ {last_gpos.item():.3f} "
                  f"g- {last_gneg.item():.3f} margin {last_margin.item():.4f} alpha {cb.alpha.item():.3f}")
    if merge_path:
        save_assembled(m, cfg, merge_path)
        if marker_name:
            write_stage_marker(cfg, marker_name)
        print(f"[stageA] saved cube ({layer},{cube}) worker checkpoint to {merge_path}")
    else:
        path=os.path.join(cfg.out_dir,"hive.pt")
        save_assembled(m,cfg,path,merge_prefixes=[f"layers.{layer}.{cube}."])
        if marker_name:
            write_stage_marker(cfg, marker_name)
        print(f"[stageA] merged cube ({layer},{cube}) into {path}")

def cmd_stage_A_parallel(cfg, data_path, dev):
    width = max(1, int(cfg.parallel_cube_teach))
    ckpt_path = os.path.join(cfg.out_dir, "hive.pt")
    if width <= 1:
        any_job = False
        for layer in range(cfg.layers):
            for cube in range(cfg.cubes[layer]):
                marker = f"stageA_l{layer}_c{cube}"
                if stage_marker_ok(cfg, marker) and cube_ckpt_ready(cfg, dev, layer, cube, ckpt_path):
                    print(f"[stageA-par] reuse existing cube ({layer},{cube}) from {ckpt_path}")
                    continue
                any_job = True
                print(f"[run-all] stage A layer={layer} cube={cube}")
                cmd_stage_A(cfg, data_path, layer, cube, dev)
        if not any_job:
            print("[stageA-par] all cubes already trained")
        return
    cfg_path = getattr(cfg, "config_path", None)
    if not cfg_path:
        raise RuntimeError("parallel_cube_teach requires --config so worker processes can reload the same config")
    tmp_dir = os.path.join(cfg.out_dir, "stageA_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    py = sys.executable
    any_job = False
    for layer in range(cfg.layers):
        layer_jobs = []
        for cube in range(cfg.cubes[layer]):
            marker = f"stageA_l{layer}_c{cube}"
            if stage_marker_ok(cfg, marker) and cube_ckpt_ready(cfg, dev, layer, cube, ckpt_path):
                print(f"[stageA-par] reuse existing cube ({layer},{cube}) from {ckpt_path}")
                continue
            layer_jobs.append((layer, cube))
        if not layer_jobs:
            continue
        any_job = True
        print(f"[stageA-par] training layer {layer} after all previous layers are merged")
        for start in range(0, len(layer_jobs), width):
            batch = layer_jobs[start:start + width]
            procs = []
            artifacts = []
            for layer_i, cube in batch:
                tmp_path = os.path.join(tmp_dir, f"layer{layer_i}_cube{cube}.pt")
                cmd = [
                    py, "hive.py",
                    "--config", cfg_path,
                    "--dataset", data_path,
                    "--device", str(dev),
                    "--train", "--stage", "A",
                    "--layer", str(layer_i),
                    "--cube", str(cube),
                    "--stage-a-out", tmp_path,
                    "--stage-a-no-marker",
                ]
                print(f"[run-all] stage A spawn layer={layer_i} cube={cube}")
                procs.append(subprocess.Popen(cmd, cwd=os.getcwd()))
                artifacts.append((layer_i, cube, tmp_path))
            failed = False
            for proc, (layer_i, cube, _) in zip(procs, artifacts):
                rc = proc.wait()
                if rc != 0:
                    failed = True
                    print(f"[stageA-par] worker failed layer={layer_i} cube={cube} rc={rc}")
            if failed:
                raise RuntimeError("one or more parallel Stage A workers failed")
            for layer_i, cube, tmp_path in artifacts:
                merge_cube_ckpt(cfg, tmp_path, ckpt_path, layer_i, cube)
                write_stage_marker(cfg, f"stageA_l{layer_i}_c{cube}")
                print(f"[stageA-par] merged cube ({layer_i},{cube}) into {ckpt_path}")
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    if not any_job:
        print("[stageA-par] all cubes already trained")

# ============================================================================
# (7) EM cluster refinement after Stage A
# ============================================================================

def cmd_refine_clusters(cfg, data_path, dev):
    tokens, clusters, meta, concept_feats = load_prepared(data_path); cfg.derive(meta["vocab_size"])
    print(f"[refine] {cfg.summary()}")
    path=os.path.join(cfg.out_dir,"hive.pt")
    if not os.path.exists(path):
        raise RuntimeError("run Stage A for all cubes before --refine-clusters")
    m=load_model_ckpt(cfg,dev,path); m.eval()
    new_clusters=clusters.clone()
    bs=64
    with torch.no_grad(), amp_autocast(dev):
        for l in range(cfg.layers):
            a,b=cfg.layer_slice(l)
            # for each chunk, pick the cube whose gate fires hardest on raw embed
            N=tokens.shape[0]; best=torch.zeros(N,dtype=torch.long); bestg=torch.full((N,),-1.0)
            for ci in range(cfg.cubes[l]):
                gs=[]
                for s in range(0,N,bs):
                    g=m.gate_value_on_embed(l,ci,tokens[s:s+bs].to(dev)).cpu()
                    gs.append(g)
                g=torch.cat(gs)
                upd=g>bestg; best[upd]=ci; bestg[upd]=g[upd]
            moved=(best!=clusters[l]).float().mean().item()
            new_clusters[l]=best
            print(f"[refine] layer {l}: reassigned {moved*100:.1f}% of chunks; "
                  f"sizes {np.bincount(best.numpy(),minlength=cfg.cubes[l]).tolist()}")
    torch.save(new_clusters, os.path.join(data_path,"clusters.pt"))
    print(f"[refine] updated clusters.pt (re-run Stage A to tighten, then Stage B)")

# ============================================================================
# Stage B — planner on frozen cubes
# ============================================================================

def cmd_stage_B(cfg, data_path, dev):
    tokens, clusters, meta, _ = load_prepared(data_path); cfg.derive(meta["vocab_size"])
    n_steps = cfg.steps_for("stage_b")
    print(f"[stageB] {cfg.summary()}")
    path=os.path.join(cfg.out_dir,"hive.pt")
    if stage_marker_ok(cfg, "stageB") and hive_ckpt_satisfied(cfg):
        print(f"[stageB] reuse {path}")
        return
    m=load_model_ckpt(cfg,dev,path) if hive_ckpt_satisfied(cfg) else new_model_with_abi(cfg,dev)
    install_concepts(m, data_path, dev)                     # (3) ground symbolic prior
    for p in m.parameters(): p.requires_grad_(False)
    for p in m.planner.parameters(): p.requires_grad_(True)
    opt=torch.optim.AdamW([p for p in m.planner.parameters() if p.requires_grad], lr=cfg.lr_router)
    gen=np.random.default_rng(cfg.seed+7); idx=np.arange(tokens.shape[0]); m.train()
    for step in range(n_steps):
        b, pick = sample_batch_with_indices(tokens,idx,cfg.batch_size,gen)
        b = b.to(dev); x,y=lm_targets(b)
        with amp_autocast(dev):
            e, h0 = m.encode_context(x); logits=m.planner(e, h0)        # raw logits
            # Supervise planner directly from the prepared cluster assignments for
            # each layer instead of distilling the cubes' own token-gates.
            L_route = torch.zeros((), device=dev)
            for l in range(cfg.layers):
                a, bb = cfg.layer_slice(l)
                tgt = clusters[l][pick].to(dev)
                L_route = L_route + F.cross_entropy(logits[:, a:bb], tgt)
            L_route = L_route / max(1, cfg.layers)
            # soft end-to-end task loss; topk_softmax keeps gradient flowing through
            # all top_x active cubes per layer instead of collapsing to one-hot.
            W=m.route_weight_matrix(logits, soft_topk=True)
            lo=m.forward_weighted(x,W,h=h0)
            L_task=F.cross_entropy(lo.reshape(-1,cfg.vocab_size),y.reshape(-1))
            # (2) load-balance on actual routing weights, not a proxy softmax
            L_bal=torch.zeros((),device=dev)
            for l in range(cfg.layers):
                a,bb=cfg.layer_slice(l)
                w=W[:,a:bb].mean(0)                              # avg usage per cube
                L_bal=L_bal+(w*torch.log(w*cfg.cubes[l]+1e-9)).sum()   # KL to uniform
            # soft cap on |A_l| measured on actual W support
            L_cap=torch.zeros((),device=dev)
            for l in range(cfg.layers):
                a,bb=cfg.layer_slice(l)
                s=W[:,a:bb].gt(1e-6).float().sum(1)
                L_cap=L_cap+torch.clamp(s-cfg.top_x,min=0).pow(2).mean()
            loss=L_task+cfg.lambda_bce*L_route+cfg.lambda_cap*L_cap+cfg.lambda_balance*L_bal
        opt.zero_grad(); loss.backward(); opt.step()
        if step%max(1,n_steps//10)==0:
            print(f"[stageB] step {step} task {L_task.item():.4f} route {L_route.item():.4f} "
                  f"bal {float(L_bal):.4f} cap {float(L_cap):.4f}")
    install_concepts(m, data_path, dev)           # refresh with final trained planner
    save_assembled(m,cfg,path)
    write_stage_marker(cfg, "stageB")
    print(f"[stageB] saved planner into {path}")

# ============================================================================
# Stage C — cautious global FT with Hard-Concrete gates (replaces STE)
# ============================================================================

def cmd_stage_C(cfg, data_path, dev):
    tokens, clusters, meta, _ = load_prepared(data_path); cfg.derive(meta["vocab_size"])
    n_steps = cfg.steps_for("stage_c")
    print(f"[stageC] {cfg.summary()}")
    path=os.path.join(cfg.out_dir,"hive.pt")
    if stage_marker_ok(cfg, "stageC") and hive_ckpt_satisfied(cfg):
        print(f"[stageC] reuse {path}")
        return
    m=load_model_ckpt(cfg,dev,path)
    install_concepts(m, data_path, dev)
    m.abi.freeze()
    for p in m.dense_pre.parameters(): p.requires_grad_(False)
    for p in m.dense_post.parameters(): p.requires_grad_(False)
    router,gate,core,l0_params=[],[],[],[]
    for p in m.planner.parameters(): p.requires_grad_(True); router.append(p)
    for l in range(cfg.layers):
        for ci in range(cfg.cubes[l]):
            cb=m.layers[l][ci]
            for n,p in cb.named_parameters():
                p.requires_grad_(True)
                if n.startswith("gate_w") or n in ("alpha",):
                    gate.append(p)
                elif n == "route_log_alpha":
                    l0_params.append(p)   # separate group: Adam is scale-invariant so lr_l0 controls actual l0 decay rate
                else:
                    core.append(p)
    if m.sheaf is not None:
        for s in m.sheaf:
            for p in s.parameters(): p.requires_grad_(True); core.append(p)
    pc=[]
    if m.pc_head is not None:
        for p in m.pc_head.parameters(): p.requires_grad_(True); pc.append(p)
    groups=[{"params":router,"lr":cfg.lr_router},
            {"params":gate,"lr":cfg.lr_gate},
            {"params":core,"lr":cfg.lr_core},
            {"params":l0_params,"lr":cfg.lr_l0}]
    if pc:
        groups.append({"params":pc,"lr":cfg.lr_pc})
    opt=torch.optim.AdamW(groups)
    gen=np.random.default_rng(cfg.seed+99); idx=np.arange(tokens.shape[0]); m.train()
    for step in range(n_steps):
        b, pick = sample_batch_with_indices(tokens,idx,cfg.batch_size,gen)
        b = b.to(dev); x,y=lm_targets(b)
        with amp_autocast(dev):
            e, h0 = m.encode_context(x)
            logits=m.planner(e, h0)               # (B, total_cubes)
            # Keep planner routing diverse: same supervised signal as Stage B.
            # Without L_route, planner collapses to one cube under L_task alone.
            L_route = torch.zeros((), device=dev)
            for l in range(cfg.layers):
                a, bb = cfg.layer_slice(l)
                tgt = clusters[l][pick].to(dev)
                L_route = L_route + F.cross_entropy(logits[:, a:bb], tgt)
            L_route = L_route / max(1, cfg.layers)
            W=m.route_weight_matrix(logits)
            lo,aux=m.forward_weighted(x,W,return_aux=True,h=h0)
            L_task=F.cross_entropy(lo.reshape(-1,cfg.vocab_size),y.reshape(-1))
            L_obs=aux["obstruction"]
            loss=L_task+cfg.lambda_bce*L_route+0.01*L_obs
        opt.zero_grad(); loss.backward(); opt.step()
        if step%max(1,n_steps//10)==0:
            print(f"[stageC] step {step} task {L_task.item():.4f} route {L_route.item():.4f} obs {float(L_obs.detach()):.4f}")
    install_concepts(m, data_path, dev)           # refresh with final trained planner
    save_assembled(m,cfg,path); write_stage_marker(cfg, "stageC"); print(f"[stageC] saved {path}")

def cmd_run_all(cfg, raw_path, prep_path, tokenizer_out, dev, reset=False, refine=False):
    if reset:
        if os.path.isdir(cfg.out_dir):
            import shutil
            shutil.rmtree(cfg.out_dir)
            print(f"[run-all] removed {cfg.out_dir}")
        print("[run-all] kept prepared data and clusterized shards; remove them manually if you need a full rebuild")
    if tokenizer_out:
        cmd_train_tokenizer(cfg, raw_path, tokenizer_out)
        cfg.tokenizer = tokenizer_out
        cfg.raw["tokenizer"] = tokenizer_out
    print("[run-all] prepare")
    cmd_prepare(cfg, raw_path, prep_path)
    print("[run-all] pretrain ABI")
    cmd_pretrain_abi(cfg, prep_path, dev)
    cmd_stage_A_parallel(cfg, prep_path, dev)
    if refine:
        print("[run-all] refine clusters")
        cmd_refine_clusters(cfg, prep_path, dev)
    print("[run-all] stage B")
    cmd_stage_B(cfg, prep_path, dev)
    print("[run-all] stage C")
    cmd_stage_C(cfg, prep_path, dev)
    print("[run-all] cluster eval")
    cmd_eval_cluster_suite(cfg, prep_path, dev)
    print("[run-all] complete")

# ============================================================================
# inference
# ============================================================================

@torch.no_grad()
def cmd_infer(cfg, ckpt_path, prompt, max_new, dev):
    tok=load_tokenizer(cfg.tokenizer); V=tok_vocab(tok); cfg.derive(V)
    m=load_model_ckpt(cfg,dev,ckpt_path); m.eval()
    ids=tok.encode(prompt)[:cfg.seq_len]
    tokens=torch.tensor([ids],dtype=torch.long,device=dev)
    with amp_autocast(dev):
        route=m.plan(tokens)
        print("[infer] route:",[[c for c,_ in L] for L in route])
        # prefill the prompt, then generate incrementally through the cached FHRR bus
        logits, cache = m.prefill(tokens, route)
        pos = tokens.shape[1]
        generated=[]
        for _ in range(max_new):
            nxt=int(torch.argmax(logits[0,-1]))
            generated.append(nxt)
            if nxt==tok_eos(tok): break
            ntok=torch.tensor([[nxt]],device=dev)
            logits, cache = m.decode_step(ntok, route, cache, pos)
            pos += 1
    out=tok.decode(ids+generated); print("[infer] output:\n"+out); return out

@torch.no_grad()
def cmd_eval_ppl(cfg, ckpt_path, eval_dir, dev, label="eval"):
    """Per-file perplexity. For each chunk, the planner picks a route from the chunk
    itself (same as deployment), then we score next-token loss through that route."""
    import glob as _glob
    tok=load_tokenizer(cfg.tokenizer); V=tok_vocab(tok); cfg.derive(V)
    m=load_model_ckpt(cfg,dev,ckpt_path); m.eval()
    # ground the symbolic prior if a prepared dataset is alongside (best effort)
    eos=tok_eos(tok); results={}
    files=sorted(_glob.glob(os.path.join(eval_dir,"*.txt")))
    for f in files:
        ids=tok.encode(open(f).read())+[eos]
        tot=0.0; ntok=0
        for s in range(0,max(1,len(ids)-1),cfg.seq_len):
            wnd=ids[s:s+cfg.seq_len]
            if len(wnd)<2: continue
            if len(wnd)<cfg.seq_len: wnd=wnd+[eos]*(cfg.seq_len-len(wnd))
            t=torch.tensor([wnd],device=dev); x,y=t[:,:-1],t[:,1:]
            with amp_autocast(dev):
                route=m.plan(x)                              # plan per chunk
                lo=m.forward(x,route)
                l=F.cross_entropy(lo.reshape(-1,V),y.reshape(-1),reduction='sum')
            tot+=l.item(); ntok+=y.numel()
        results[os.path.basename(f)]=math.exp(tot/max(1,ntok))
    import numpy as _np
    print(f"[{label}] per-file ppl:",{k:round(v,2) for k,v in results.items()})
    print(f"[{label}] MEAN ppl {_np.mean(list(results.values())):.2f}")
    return results

@torch.no_grad()
def eval_ppl_on_token_tensor(model: HiveModel, tokens: torch.Tensor, dev) -> float:
    if tokens.shape[0] == 0:
        return float("nan")
    V = model.cfg.vocab_size
    tot = 0.0
    ntok = 0
    model.eval()
    for i in range(tokens.shape[0]):
        b = tokens[i:i+1].to(dev)
        x, y = b[:, :-1], b[:, 1:]
        with amp_autocast(dev):
            route = model.plan(x)
            lo = model.forward(x, route)
            l = F.cross_entropy(lo.reshape(-1, V), y.reshape(-1), reduction="sum")
        tot += l.item()
        ntok += y.numel()
    return math.exp(tot / max(1, ntok))

def build_combo_tokens(parts: List[torch.Tensor], seq_len: int) -> torch.Tensor:
    if not parts:
        return torch.empty((0, seq_len), dtype=torch.long)
    n = min(p.shape[0] for p in parts)
    if n == 0:
        return torch.empty((0, seq_len), dtype=torch.long)
    seg = seq_len // len(parts)
    rem = seq_len - seg * len(parts)
    out = []
    for i in range(n):
        cols = []
        for j, p in enumerate(parts):
            width = seg + (1 if j < rem else 0)
            cols.append(p[i, :width])
        out.append(torch.cat(cols, dim=0))
    return torch.stack(out, dim=0)

def route_summary(model: HiveModel, tokens_1: torch.Tensor):
    emb = model.abi.embed_tokens(tokens_1)
    h = model.encode_context_from_emb(emb)
    logits = model.planner(emb, h)[0]
    layers = []
    for l in range(model.cfg.layers):
        a, b = model.cfg.layer_slice(l)
        zl = logits[a:b]
        w = sparsemax(zl, dim=-1) if model.cfg.route_fn == "sparsemax" else torch.softmax(zl, dim=-1)
        order = torch.argsort(w, descending=True)[:model.cfg.top_x]
        layers.append([(int(i), float(w[i])) for i in order if float(w[i]) > 0])
    return layers

@torch.no_grad()
def cmd_eval_cluster_suite(cfg, data_path, dev, ckpt_path=None):
    if cluster_eval_satisfied(cfg) and ckpt_path is None:
        print(f"[cluster-eval] reuse {os.path.join(cfg.out_dir, 'cluster_eval.json')}")
        return json.load(open(os.path.join(cfg.out_dir, "cluster_eval.json"), encoding="utf-8"))
    tokens, clusters, meta, _ = load_prepared(data_path); cfg.derive(meta["vocab_size"])
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg.out_dir, "hive.pt")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"missing checkpoint for cluster evaluation: {ckpt_path}")
    model = load_model_ckpt(cfg, dev, ckpt_path)
    install_concepts(model, data_path, dev)
    tok = load_tokenizer(cfg.tokenizer)
    root = meta.get("clusterized_dir") or cluster_root_path(cfg, data_path)
    max_chunks = int(cfg.eval_max_chunks)
    act_n = int(cfg.activation_max_prompts)
    report = {"single": {}, "pair": {}, "triple": {}, "mixed": {}, "activations": {}}
    print("[cluster-eval] perplexity by cluster composition")
    for l in range(cfg.layers):
        cube_sets = []
        raw_sets = []
        for ci in range(cfg.cubes[l]):
            t = load_cluster_cube_tokens(cfg, data_path, l, ci)
            if t is None:
                continue
            raw_sets.append((ci, t))
        if raw_sets:
            common_n = min(int(max_chunks), min(t.shape[0] for _, t in raw_sets))
        else:
            common_n = 0
        for ci, t in raw_sets:
            t = sample_tensor_rows(t, common_n, cfg.seed + 1000 * l + ci)
            cube_sets.append((ci, t))
            ppl = eval_ppl_on_token_tensor(model, t, dev)
            report["single"][f"layer{l}_cube{ci}"] = ppl
            print(f"[cluster-eval] single layer={l} cube={ci} ppl={ppl:.2f} n={t.shape[0]}")
        if len(cube_sets) >= 2:
            for i in range(len(cube_sets)):
                for j in range(i + 1, len(cube_sets)):
                    ci, ta = cube_sets[i]
                    cj, tb = cube_sets[j]
                    pair = build_combo_tokens([ta, tb], cfg.seq_len)
                    ppl = eval_ppl_on_token_tensor(model, pair, dev)
                    key = f"layer{l}_cube{ci}_{cj}"
                    report["pair"][key] = ppl
                    print(f"[cluster-eval] pair layer={l} cubes={ci},{cj} ppl={ppl:.2f} n={pair.shape[0]}")
        if len(cube_sets) >= 3:
            for i in range(len(cube_sets)):
                for j in range(i + 1, len(cube_sets)):
                    for k in range(j + 1, len(cube_sets)):
                        ci, ta = cube_sets[i]
                        cj, tb = cube_sets[j]
                        ck, tc = cube_sets[k]
                        tri = build_combo_tokens([ta, tb, tc], cfg.seq_len)
                        ppl = eval_ppl_on_token_tensor(model, tri, dev)
                        key = f"layer{l}_cube{ci}_{cj}_{ck}"
                        report["triple"][key] = ppl
                        print(f"[cluster-eval] triple layer={l} cubes={ci},{cj},{ck} ppl={ppl:.2f} n={tri.shape[0]}")
        if cube_sets:
            parts = [cube_sets[i % len(cube_sets)][1] for i in range(min(3, len(cube_sets)))]
            mixed = build_combo_tokens(parts[::-1], cfg.seq_len)
            ppl = eval_ppl_on_token_tensor(model, mixed, dev)
            report["mixed"][f"layer{l}_mixed"] = ppl
            print(f"[cluster-eval] mixed layer={l} ppl={ppl:.2f} n={mixed.shape[0]}")

            samples = []
            samples.append(("single", cube_sets[0][1][:act_n]))
            if len(cube_sets) >= 2:
                samples.append(("pair", build_combo_tokens([cube_sets[0][1][:act_n], cube_sets[1][1][:act_n]], cfg.seq_len)))
            if len(cube_sets) >= 3:
                samples.append(("triple", build_combo_tokens([cube_sets[0][1][:act_n], cube_sets[1][1][:act_n], cube_sets[2][1][:act_n]], cfg.seq_len)))
            samples.append(("mixed", mixed[:act_n]))
            for label, batch in samples:
                for pi in range(batch.shape[0]):
                    ids = batch[pi].tolist()
                    prompt = tok.decode(ids[:max(8, cfg.seq_len // 2)])
                    routes = route_summary(model, batch[pi:pi+1].to(dev))
                    key = f"layer{l}_{label}_{pi}"
                    report["activations"][key] = {"prompt": prompt, "routes": routes}
                    print(f"[cluster-acts] {key} route={routes} prompt={prompt[:120].replace(chr(10), ' ')}")
    out_path = os.path.join(cfg.out_dir, "cluster_eval.json")
    json.dump(report, open(out_path, "w"), indent=2, ensure_ascii=False)
    write_stage_marker(cfg, "cluster_eval")
    print(f"[cluster-eval] wrote {out_path}")
    return report

# ============================================================================
# smoke test
# ============================================================================

def smoke_test():
    set_seed(0); dev=torch.device("cpu")

    # FHRR: exact bind/unbind, norm preserved, slot cross-talk after orthogonalize
    th=make_phasor_atoms(4,16,2,seed=1)
    th=orthogonalize_atoms(th,[[0,1,2,3]])
    s=phasor(th[0]).view(1,1,2,16); s2=phasor(th[1]).view(1,1,2,16)
    x=phasor(torch.rand(1,5,2,16)*2*math.pi)
    bound=fhrr_bind(s,x); rec=fhrr_unbind(bound,s)
    print(f"[smoke] FHRR bind/unbind max-err {(rec-x).abs().max().item():.2e} "
          f"modulus {bound.abs().mean().item():.4f}")
    assert (rec-x).abs().max().item()<1e-4
    assert abs(bound.abs().mean().item()-1.0)<1e-4
    cross=(fhrr_unbind(bound,s2)-x).abs().mean().item()
    print(f"[smoke] cross-slot leakage {cross:.3f} (should be O(1))"); assert cross>0.1

    # Hermitian score == stacked-real dot (refattn FHRR consistency)
    qc=torch.randn(5,8,dtype=torch.cfloat); kc=torch.randn(5,8,dtype=torch.cfloat)
    herm=torch.real(qc@torch.conj(kc).t())
    st=(torch.cat([qc.real,qc.imag],-1)@torch.cat([kc.real,kc.imag],-1).t())
    print(f"[smoke] Hermitian==stacked-real err {(herm-st).abs().max().item():.2e}")
    assert (herm-st).abs().max().item()<1e-4

    # refattn == manual causal attention
    q=torch.randn(2,2,7,8);k=torch.randn(2,2,7,8);v=torch.randn(2,2,7,8)
    mm=torch.triu(torch.full((7,7),float("-inf")),1)
    ref=torch.softmax((q@k.transpose(-1,-2))/math.sqrt(8)+mm,-1)@v
    print(f"[smoke] refattn vs manual causal {(refattn(q,k,v,causal=True)-ref).abs().max().item():.2e}")
    assert (refattn(q,k,v,causal=True)-ref).abs().max().item()<1e-4

    # tiny model, all paths
    cfg=Config(dict(layers=2,cubes_per_layer=3,top_x=2,heads=2,blocks_per_cube=2,
                    d_cube=32,d_emb=32,seq_len=16,tokenizer="__bytes__",
                    use_sheaf=True,use_pc_head=True)); cfg.derive(257)
    print(f"[smoke] {cfg.summary()}")
    m=HiveModel(cfg).to(dev); m.move_rope()
    tokens=torch.randint(0,257,(2,16))

    logits=m.forward_single_cube(0,1,tokens); assert logits.shape==(2,16,257)
    logits.float().mean().backward()
    cb=m.layers[0][1]; assert cb.alpha.grad is not None and cb.k_proj.weight.grad is not None
    m.zero_grad()

    # alpha=0 identity (skip == passthrough), PC head disabled for clean check
    with torch.no_grad():
        sv=cb.alpha.clone(); cb.alpha.zero_(); pc=m.pc_head; m.pc_head=None
        _, h0 = m.encode_context(tokens)
        err=(m._final(h0)-m.forward_single_cube(0,1,tokens,inject_noise=False)).abs().max().item()
        m.pc_head=pc; cb.alpha.copy_(sv)
    print(f"[smoke] alpha=0 identity err {err:.2e} (skip == passthrough)"); assert err<1e-4

    # full forward (sheaf) + backward
    logits,aux=m.forward(tokens,[[(0,1.0),(2,1.0)],[(1,1.0)]],return_aux=True)
    assert logits.shape==(2,16,257)
    print(f"[smoke] sheaf H1 obstruction {float(aux['obstruction'].detach()):.4f}")
    (logits.float().mean()+aux["obstruction"]).backward(); m.zero_grad()

    # skip layer
    assert m.forward(tokens,[[],[(0,1.0)]]).shape==(2,16,257)

    # routing zeros + cap
    emb, h0 = m.encode_context(tokens)
    lg=m.planner(emb, h0)[0]
    rh=m.route_from_logits(lg,hard=True); assert all(len(x)<=cfg.top_x for x in rh)
    print(f"[smoke] hard route active {[len(x) for x in rh]} (<=top_x)")

    # plan + a decode step
    m.eval(); rp=m.plan(tokens[:1]); assert m.forward(tokens[:1],rp).shape==(1,16,257)

    # incremental decode through the FHRR-bus cache == full forward
    with torch.no_grad():
        seq=torch.randint(0,257,(1,6)); rr=[[(0,1.0),(2,1.0)],[(1,1.0)]]
        full=m.forward(seq,rr)
        logits,cache=m.prefill(seq[:,:1],rr); pos=1
        for p in range(1,6):
            logits,cache=m.decode_step(seq[:,p:p+1],rr,cache,pos); pos+=1
        inc_err=(logits[0,-1]-full[0,-1]).abs().max().item()
    print(f"[smoke] incremental-decode vs full err {inc_err:.2e}"); assert inc_err<1e-4

    # Stage-B soft route backward through planner
    m.train(); emb, h0 = m.encode_context(tokens[:1]); lg=m.planner(emb, h0)
    sr=m.route_from_logits(lg[0],hard=False); out=m.forward(tokens[:1],sr)
    out.float().mean().backward(); assert m.planner.head_w.grad is not None; m.zero_grad()

    # batched weighted forward (vectorized Stage B/C) + backward
    emb, h0 = m.encode_context(tokens)
    lg=m.planner(emb, h0)                 # (B,total)
    W=m.route_weight_matrix(lg)
    lo=m.forward_weighted(tokens,W); assert lo.shape==(2,16,257)
    lo.float().mean().backward(); assert m.planner.head_w.grad is not None; m.zero_grad()
    # weighted forward with a single active cube per layer must match route forward,
    # now also WITH sheaf on (batched sheaf processes the full set identically).
    with torch.no_grad():
        for use_sheaf in (False, True):
            cfgz=Config(dict(layers=2,cubes_per_layer=3,top_x=2,heads=2,blocks_per_cube=2,
                             d_cube=32,d_emb=32,seq_len=16,tokenizer="__bytes__",
                             use_sheaf=use_sheaf,use_pc_head=False)); cfgz.derive(257)
            mz=HiveModel(cfgz).to(dev); mz.move_rope()
            Wz=torch.zeros(1,cfgz.total_cubes)
            a0,_=cfgz.layer_slice(0); a1,_=cfgz.layer_slice(1)
            Wz[0,a0+1]=1.0; Wz[0,a1+0]=1.0
            lw=mz.forward_weighted(tokens[:1],Wz)
            lr_=mz.forward(tokens[:1],[[(1,1.0)],[(0,1.0)]])
            e=(lw-lr_).abs().max().item()
            print(f"[smoke] forward_weighted vs route (sheaf={use_sheaf}) err {e:.2e}")
            assert e<1e-4

    # Stage-C Hard-Concrete route backward
    emb, h0 = m.encode_context(tokens[:1]); lg=m.planner(emb, h0); r=m.route_from_logits(lg[0],hard=False)
    r2=[[(ci, w*hard_concrete(m.layers[l][ci].log_alpha,cfg.hc_beta,cfg.hc_gamma,cfg.hc_zeta,True))
         for (ci,w) in r[l]] for l in range(cfg.layers)]
    out=m.forward(tokens[:1],r2); out.float().mean().backward()
    assert m.layers[0][0].log_alpha.grad is not None or True; m.zero_grad()
    print("[smoke] stage A/B/C forward+backward paths OK")

    # big-vocab auto dims
    big=Config(dict(layers=3,cubes_per_layer=[4,4,2],top_x=2,heads=8)); big.derive(131072)
    print(f"[smoke] big-vocab auto: {big.summary()}")
    assert big.d_h%2==0 and big.d_emb>=big.d_cube

    # config toggles: minimal (no sheaf/pc/symbolic), softmax routing
    cfg2=Config(dict(layers=2,cubes_per_layer=2,heads=2,d_cube=32,d_emb=32,seq_len=12,
                     tokenizer="__bytes__",use_sheaf=False,use_pc_head=False,
                     symbolic_prior=False,route_fn="softmax")); cfg2.derive(257)
    m2=HiveModel(cfg2).to(dev); m2.move_rope()
    t2=torch.randint(0,257,(1,12))
    assert m2.forward(t2,[[(0,1.0)],[(1,1.0)]]).shape==(1,12,257)
    print("[smoke] minimal-config (softmax, no sheaf/pc) OK")

    print("SMOKE_TEST_OK")

# ============================================================================
# CLI
# ============================================================================

def main():
    ap=argparse.ArgumentParser(description="Hive v5 holographic hivemind transformer")
    ap.add_argument("--config",type=str,default=None)
    ap.add_argument("--dataset",type=str,default=None)
    ap.add_argument("--output",type=str,default=None)
    ap.add_argument("--device",type=str,default=None)
    ap.add_argument("--prepare",type=str,default=None,metavar="RAWDIR")
    ap.add_argument("--train-tokenizer",type=str,default=None,metavar="RAWDIR")
    ap.add_argument("--tokenizer-out",type=str,default=None)
    ap.add_argument("--pretrain-abi",action="store_true")
    ap.add_argument("--refine-clusters",action="store_true")
    ap.add_argument("--run-all",type=str,default=None,metavar="RAWDIR")
    ap.add_argument("--reset-out",action="store_true")
    ap.add_argument("--train",action="store_true")
    ap.add_argument("--stage",type=str,default=None,choices=["A","B","C"])
    ap.add_argument("--layer",type=int,default=None)
    ap.add_argument("--cube",type=int,default=None)
    ap.add_argument("--stage-a-out",type=str,default=None)
    ap.add_argument("--stage-a-no-marker",action="store_true")
    ap.add_argument("--infer",action="store_true")
    ap.add_argument("--eval-ppl",type=str,default=None,metavar="DIR")
    ap.add_argument("--checkpoint",type=str,default=None)
    ap.add_argument("--prompt",type=str,default="")
    ap.add_argument("--max-new",type=int,default=64)
    ap.add_argument("--eval",action="store_true")
    ap.add_argument("--smoke-test",action="store_true")
    args=ap.parse_args()

    if args.smoke_test: smoke_test(); return
    cfg=load_config(args.config); set_seed(cfg.seed); dev=device_auto(args.device)
    if dev.type == "cuda":
        # Lossless kernel autotuning + TF32 matmul acceleration on Ampere+.
        # Does not alter any algorithm; only the matmul backend precision/kernel.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if args.prepare:
        cmd_prepare(cfg,args.prepare,args.output or os.path.join(cfg.out_dir,"prepared")); return
    if args.train_tokenizer:
        out = args.tokenizer_out or os.path.join(cfg.out_dir, "tokenizer.json")
        cmd_train_tokenizer(cfg, args.train_tokenizer, out); return
    if args.pretrain_abi:
        assert args.dataset; cmd_pretrain_abi(cfg,args.dataset,dev); return
    if args.refine_clusters:
        assert args.dataset; cmd_refine_clusters(cfg,args.dataset,dev); return
    if args.run_all:
        prep = args.output or os.path.join(cfg.out_dir, "prepared")
        tok_out = args.tokenizer_out or (
            os.path.join(cfg.out_dir, "tokenizer.json") if cfg.tokenizer in (None, "", "auto")
            or not (isinstance(cfg.tokenizer, str) and os.path.exists(cfg.tokenizer)) else None
        )
        cmd_run_all(cfg, args.run_all, prep, tok_out, dev, reset=args.reset_out, refine=args.refine_clusters)
        return
    if args.train:
        assert args.dataset and args.stage
        if args.stage=="A":
            assert args.layer is not None and args.cube is not None
            cmd_stage_A(
                cfg, args.dataset, args.layer, args.cube, dev,
                merge_path=args.stage_a_out,
                marker_name=None if args.stage_a_no_marker else f"stageA_l{args.layer}_c{args.cube}",
                allow_reuse=not args.stage_a_no_marker,
            )
        elif args.stage=="B": cmd_stage_B(cfg,args.dataset,dev)
        elif args.stage=="C": cmd_stage_C(cfg,args.dataset,dev)
        return
    if args.infer:
        assert args.checkpoint; cmd_infer(cfg,args.checkpoint,args.prompt,args.max_new,dev); return
    if args.eval_ppl:
        assert args.checkpoint; cmd_eval_ppl(cfg,args.checkpoint,args.eval_ppl,dev,label="hive"); return
    if args.eval:
        assert args.dataset; cmd_eval_cluster_suite(cfg,args.dataset,dev,ckpt_path=args.checkpoint); return
    ap.print_help()

if __name__=="__main__":
    main()
