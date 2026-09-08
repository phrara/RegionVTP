"""Model-agnostic QADP core (faithful, line-by-line port of `token_carve_llm_prune`).

This module is a **pure torch / numpy** extraction of the pruning math that lives in
``llava/model/llava_arch.py`` (``token_carve_llm_prune`` at :625-767 and its three
helpers at :168-184). It deliberately does **not** import ``transformers`` so it can be
imported from both:

- the LLaVA-1.5 environment (transformers 4.37.2), and
- the LLaVA-OneVision environment (transformers >= 4.45).

The LLaVA-1.5 path keeps calling its own in-file ``token_carve_llm_prune`` (this module
is NOT wired into it — that path stays byte-identical). The OneVision path calls
:func:`qadp_llm_prune` directly. The two helper copies are locked equivalent by
``tests/test_qadp_core_equiv.py``.

Abstractions vs. the original method:

* ``self.get_model().layers`` (Vicuna, 32 layers)  -> the ``layers`` argument (Qwen2, 28).
* ``self.get_visual_token_num()`` (=576)           -> ``cfg.rank`` / ``read_qadp_env``.
* ``IMAGE_TOKEN_INDEX`` / ``image_token_indices``  -> caller passes ``sys_length`` /
  ``image_length`` (position of the image block is located outside the core).
* module globals ``n_rank/n_sample/erank_log``     -> optional ``on_selected(erank)`` callback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch

# ---- constants (verbatim copies of llava_arch.py:164-165) ----
ERANK_AVG_REF = 90
TAU_MAX = 0.25

# Max block_len for which _rank_select_and_merge uses the Gram-matrix eigendecomposition
# instead of the direct SVD.  The eigh of an (N, N) matrix is O(N^3); it only beats LAPACK's
# tall-skinny gesdd while N is well below the feature dim (~3584).  Video frames are 196
# (fast path); M0 anyres multi-crop can be ~4724 (keeps the direct SVD).
_GRAM_EIGH_MAX = 1024


def calculate_adaptive_tau(order_i, erank_input, erank_avg=ERANK_AVG_REF, tau_max=TAU_MAX):
    """Eq. 6: tau_i = order_i * (erank_input / erank_avg * 0.01), capped at tau_max."""
    tau = order_i * (erank_input / erank_avg * 0.01)
    return min(tau, tau_max)


def effective_rank(features):  # fast Version
    X = features.float()
    X -= X.mean(dim=0, keepdim=True)
    C = torch.mm(X, X.T)  # (N, N)
    eigvals = torch.linalg.eigvalsh(C)
    S = torch.sqrt(torch.clamp(eigvals, min=1e-12))
    p = S / (S.sum() + 1e-12)
    H = -(p * torch.log(p + 1e-12)).sum()
    return torch.exp(H)


def select_diverse_tokens_by_attention_and_distance(
    image_attentions,
    d,
    erank_input,
    max_tokens=64,
    static_tau=None,
    erank_avg=ERANK_AVG_REF,
    tau_max=TAU_MAX,
):
    """Greedy attention-ranked diversity selector (AgilePruner Eq. 6).

    ``image_attentions`` is a ``(1, N)`` relevance/ranking score (the fused AV/SV score,
    not literal attention weights); ``d`` is an ``(N, N)`` cosine-distance matrix.
    """
    attention_scores = image_attentions[0]
    N = attention_scores.shape[0]
    device = attention_scores.device

    sorted_attention_indices = torch.argsort(attention_scores, descending=True)

    # order_i: each token's own 1-based rank in the attention-descending order.
    rank_of = torch.empty(N, dtype=torch.long, device=device)
    rank_of[sorted_attention_indices] = torch.arange(1, N + 1, device=device)

    alive = torch.ones(N, dtype=torch.bool, device=device)
    selected_indices = []

    for token_idx in sorted_attention_indices:
        i = token_idx.item()
        if not alive[i]:
            continue
        if len(selected_indices) >= max_tokens:
            break

        selected_indices.append(i)
        alive[i] = False

        if static_tau is not None:
            tau_i = static_tau
        else:
            tau_i = calculate_adaptive_tau(rank_of[i].item(), erank_input, erank_avg, tau_max)

        to_prune = (d[i] < tau_i) & alive
        alive[to_prune] = False

    # Guarantee exactly max_tokens: backfill with remaining highest-attention tokens.
    if len(selected_indices) < max_tokens:
        selected_set = set(selected_indices)
        for idx in sorted_attention_indices:
            if len(selected_indices) >= max_tokens:
                break
            i = idx.item()
            if i not in selected_set:
                selected_indices.append(i)
                selected_set.add(i)

    return torch.tensor(selected_indices, device=device)


@dataclass
class QADPConfig:
    """Runtime config mirroring llava_arch.py:643-652 (env defaults identical)."""

    work_layer: int = 2
    rank: int = 64
    merge_nums: int = 32
    sv_av_mode: int = 0  # 0=fuse, 1=SV only, 2=AV only
    sv_av_weight: float = 0.5
    qadp_diversity: int = 0  # 1 = QADP greedy cosine suppression
    qadp_tau: float = 0.2
    qadp_adaptive: int = 0  # 1 = erank-adaptive tau (Eq. 6)
    qadp_erank_ref: float = ERANK_AVG_REF
    qadp_tau_max: float = TAU_MAX


def read_qadp_env(visual_token_num: int) -> QADPConfig:
    """Mirror llava_arch.py:643-652 exactly (rank default = 1.5 * visual_token_num)."""
    return QADPConfig(
        work_layer=int(os.environ.get("TCARVE_LAYER", "2")),
        rank=int(os.environ.get("TCARVE_RANK", str(int(1.5 * visual_token_num)))),
        merge_nums=int(os.environ.get("TCARVE_MERGE", str(visual_token_num // 2))),
        sv_av_mode=int(os.environ.get("TCARVE_MODE", "0")),
        sv_av_weight=float(os.environ.get("TCARVE_WEIGHT", "0.5")),
        qadp_diversity=int(os.environ.get("QADP_DIVERSITY", "0")),
        qadp_tau=float(os.environ.get("QADP_TAU", "0.2")),
        qadp_adaptive=int(os.environ.get("QADP_ADAPTIVE", "0")),
        qadp_erank_ref=float(os.environ.get("QADP_ERANK_REF", str(ERANK_AVG_REF))),
        qadp_tau_max=float(os.environ.get("QADP_TAU_MAX", str(TAU_MAX))),
    )


def _rotate_half(x):
    """90° rotation of the trailing-dim feature pairs (GPT-NeoX RoPE half-rotation)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _last_token_attention(attn, hid_in, position_embeddings):
    """Head-averaged attention of the last query token to every key token (the AV term).

    Equivalent to the ``-1`` row of ``attn(hid_in, output_attentions=True)``'s weights
    averaged over heads, but computed with a single query — O(L) instead of O(L²) — and
    without forcing the SDPA→eager fallback (the QADP overhead source).  Head geometry is
    derived from the projection output widths and the RoPE ``cos`` shape (not from
    version-specific attribute names like ``num_heads``); ``position_embeddings`` is the
    ``(cos, sin)`` RoPE tuple (``(B, L, head_dim)`` each), exactly as the layer consumes it.
    """
    bsz, q_len, _ = hid_in.shape

    q = attn.q_proj(hid_in[:, -1:, :])              # (B, 1, num_heads * head_dim)
    k = attn.k_proj(hid_in)                         # (B, L, num_kv_heads * head_dim)
    if hasattr(attn, "q_norm"):                     # Qwen2.5+ applies q/k RMSNorm pre-RoPE
        q = attn.q_norm(q)
    if hasattr(attn, "k_norm"):
        k = attn.k_norm(k)

    cos, sin = position_embeddings                  # (B, L, head_dim) each
    head_dim = cos.shape[-1]
    num_heads = attn.q_proj.out_features // head_dim
    num_kv_heads = attn.k_proj.out_features // head_dim

    q = q.view(bsz, 1, num_heads, head_dim).transpose(1, 2)                    # (B, heads, 1, H)
    k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)             # (B, kv, L, H)

    cos_q = cos[:, -1:, :].unsqueeze(1)             # (B, 1, 1, H) — last position only
    sin_q = sin[:, -1:, :].unsqueeze(1)
    q = (q * cos_q) + (_rotate_half(q) * sin_q)
    cos_k = cos.unsqueeze(1)                        # (B, 1, L, H) — all key positions
    sin_k = sin.unsqueeze(1)
    k = (k * cos_k) + (_rotate_half(k) * sin_k)

    n_rep = num_heads // num_kv_heads
    k = k.repeat_interleave(n_rep, dim=1)           # (B, heads, L, H) — GQA broadcast

    scaling = head_dim ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scaling              # (B, heads, 1, L)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(hid_in.dtype)
    return weights.mean(dim=1)[0, 0, :]             # (L,) head-averaged last-row attention


def _partial_forward(embeds, position_ids, position_embeddings, layers, work_layer):
    """Run ``layers[:work_layer]`` over the full sequence; return ``(hid, q2img_full)``.

    ``q2img_full`` is the head-averaged last-row attention ``(seq_len,)`` used by the AV
    term.  When ``position_embeddings`` is provided (Qwen2, the OneVision path) we run the
    layers with SDPA (``output_attentions=False``) and reconstruct the single attention row
    in O(L) — avoiding the O(L²) eager fallback that ``output_attentions=True`` triggers.
    When it is None (Vicuna/Llama on transformers 4.37, which derive RoPE from
    ``position_ids`` internally) we keep the original eager ``output_attentions=True`` path.
    Set ``QADP_EAGER_ATTN=1`` to force the eager path even when ``position_embeddings`` is
    available (A/B to confirm the two AV terms rank identically).
    """
    bsz, seq_len, dim = embeds.shape
    dev = embeds.device
    dtype = embeds.dtype
    work = layers[:work_layer]

    if position_embeddings is not None and os.environ.get("QADP_EAGER_ATTN", "0") != "1":
        causal = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=dev), diagonal=1
        ).unsqueeze(0).unsqueeze(0)                 # (1, 1, L, L), True = masked (causal)
        hid = embeds
        hid_in = embeds
        for i, layer in enumerate(work):
            if i == len(work) - 1:
                hid_in = hid                        # input to the last layer -> AV term
            hid = layer(hid, attention_mask=causal, position_ids=position_ids,
                        use_cache=False, output_attentions=False,
                        position_embeddings=position_embeddings)[0]
        return hid, _last_token_attention(work[-1].self_attn, hid_in, position_embeddings)

    # Fallback: original eager path (float additive mask, full attention weights).
    causal = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=dev, dtype=dtype)
    causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)
    hid = embeds
    attn_w = None
    for layer in work:
        layer_kw = dict(attention_mask=causal, position_ids=position_ids,
                        use_cache=False, output_attentions=True)
        if position_embeddings is not None:
            layer_kw["position_embeddings"] = position_embeddings
        out = layer(hid, **layer_kw)
        hid = out[0]
        attn_w = out[1]
    return hid, attn_w.mean(dim=1)[0, -1, :]


def _sv_row_contrib(img_hid, block_len):
    """Row-wise |U·S| contribution of one block (SVD or Gram-eigh, per block shape).

    ``img_hid`` is ``(block_len, D)`` float32.  For tall-skinny blocks (video frames,
    block_len=196) the Gram-matrix eigendecomposition is mathematically identical to the SVD
    (|U·S| is sign/order-invariant) but replaces LAPACK's slow gesdd with a dense GEMM + a
    tiny eigh.
    """
    if block_len <= _GRAM_EIGH_MAX:
        C = img_hid @ img_hid.T  # (block_len, block_len)
        eigvals, U = torch.linalg.eigh(C)
        S = torch.sqrt(torch.clamp(eigvals, min=1e-12))
        return (U * S.unsqueeze(0)).abs().sum(dim=1)  # (block_len,)
    U, S, _ = torch.linalg.svd(img_hid, full_matrices=False)
    return (U * S.unsqueeze(0)).abs().sum(dim=1)  # (block_len,)


def _sv_row_contrib_batched(img_hid, block_len):
    """Batched ``_sv_row_contrib`` over a leading frame dim: ``img_hid`` is ``(F, N, D)``.

    One batched GEMM + eigh (or svd) over all frames amortizes the per-frame kernel /
    cuSOLVER launch overhead that dominates 64 sequential small SVD/eigh calls.
    """
    if block_len <= _GRAM_EIGH_MAX:
        C = img_hid @ img_hid.transpose(-1, -2)  # (F, N, N)
        eigvals, U = torch.linalg.eigh(C)
        S = torch.sqrt(torch.clamp(eigvals, min=1e-12))
        return (U * S.unsqueeze(-1)).abs().sum(dim=-1)  # (F, N)
    U, S, _ = torch.linalg.svd(img_hid, full_matrices=False)
    return (U * S.unsqueeze(-1)).abs().sum(dim=-1)  # (F, N)


def _rank_select_and_merge(q2img_full, hid, embeds, block_start, block_len, rank, cfg, row_contrib=None):
    """Select + prune-then-merge one contiguous block of visual tokens.

    Faithful extraction of ``qadp_llm_prune``'s per-block logic (the AV/SV rank fusion and
    the prune-then-merge fold), parameterised by ``block_start``/``block_len`` so it can be
    reused for the single-image block (M0) and for **each frame** in the per-frame video
    path.  ``rank`` must already be capped to ``[2, block_len]`` by the caller.

    ``q2img_full`` is the head-averaged attention of the last query token to every token
    (``(seq_len,)``); this function slices out the block's ``[block_start, block_start +
    block_len)`` portion for the AV term.

    Returns ``(final_keep_local, n_kept, erank)`` where ``final_keep_local`` holds
    block-relative indices (into ``[block_start, block_start + block_len)``).
    """
    dev = embeds.device
    dtype = embeds.dtype
    block_slice = slice(block_start, block_start + block_len)

    # AV: last input token's (question) attention to each block token, mean over heads.
    q2img = q2img_full[block_slice]  # (block_len,)
    _, av_order = torch.sort(q2img, descending=True)

    # SV: SVD row contribution of the contextualized block tokens.  ``row_contrib`` may be
    # pre-computed in a batched pass over all frames (the per-frame SVD/eigh has high GPU /
    # cuSOLVER launch overhead); when None it is computed here for this single block.
    img_hid = hid[0, block_slice, :].to(torch.float32)  # (block_len, D)
    if row_contrib is None:
        row_contrib = _sv_row_contrib(img_hid, block_len)
    _, sv_order = torch.sort(row_contrib, descending=True)

    # rank fusion: positional weighting (TokenCarve AV/SV fusion).
    m = block_len
    idx_w = torch.arange(m, 0, -1, device=dev, dtype=torch.float32)
    fused = torch.zeros(m, device=dev, dtype=torch.float32)
    fused[av_order] += idx_w * cfg.sv_av_weight
    fused[sv_order] += idx_w * (1.0 - cfg.sv_av_weight)
    if cfg.qadp_diversity == 1:
        # QADP: question-adaptive diversity coverage (select-only).
        norm_hid = img_hid / (img_hid.norm(dim=-1, keepdim=True) + 1e-8)
        d_hid = 1.0 - (norm_hid @ norm_hid.t())  # (m, m) cosine distance
        erank_llm = effective_rank(img_hid).item()
        if cfg.qadp_adaptive == 1:
            keep_local = select_diverse_tokens_by_attention_and_distance(
                fused.unsqueeze(0), d_hid, erank_llm,
                max_tokens=rank, static_tau=None,
                erank_avg=cfg.qadp_erank_ref, tau_max=cfg.qadp_tau_max,
            )
        else:
            keep_local = select_diverse_tokens_by_attention_and_distance(
                fused.unsqueeze(0), d_hid, erank_llm,
                max_tokens=rank, static_tau=(cfg.qadp_tau if cfg.qadp_tau > 0 else None),
            )
    elif cfg.sv_av_mode == 2:
        keep_local = av_order[:rank]
    elif cfg.sv_av_mode == 1:
        keep_local = sv_order[:rank]
    else:
        keep_local = torch.topk(fused, rank).indices
    keep_local = keep_local.to(device=dev, dtype=torch.long)  # (rank,) local block indices

    # prune-then-merge: fold the most-similar B (bottom half) into A (top half) via cosine.
    set_length = rank // 2
    A_local = keep_local[:set_length]
    B_local = keep_local[set_length:]
    img_emb = embeds[0, block_slice, :].float()  # (block_len, D)
    norm = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
    sim = norm[B_local] @ norm[A_local].t()  # (rank - set_length, set_length)
    sim_max, sim_arg = sim.max(dim=-1)
    reduce_n = min(set_length, cfg.merge_nums)
    order = torch.sort(sim_max, descending=True).indices
    merge_B = B_local[order[:reduce_n]]
    merge_A = A_local[sim_arg[order[:reduce_n]]]
    remaining_B = B_local[order[reduce_n:]]

    # merged embeddings: mean of each A and the B folded into it; order = A + remaining B.
    merged = img_emb.clone()
    counts = torch.ones(block_len, device=dev, dtype=torch.float32)
    merged.index_add_(0, merge_A, img_emb[merge_B])
    counts.index_add_(0, merge_A, torch.ones(reduce_n, device=dev, dtype=torch.float32))
    merged = merged / counts.unsqueeze(-1)
    final_keep_local = torch.cat([A_local, remaining_B])
    final_img_emb = merged[final_keep_local].to(dtype)  # (T_final, D)

    return final_keep_local, int(final_keep_local.numel()), effective_rank(final_img_emb).item()


def qadp_llm_prune(
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    sys_length: int,
    image_length: int,
    layers,
    cfg: QADPConfig,
    on_selected: Optional[Callable[[float], None]] = None,
    position_embeddings=None,
):
    """TokenCarve-style LLM-layer pruning (rank-fusion + prune-then-merge).

    Faithful port of ``llava_arch.py:token_carve_llm_prune`` (:625-767). Runs the first
    ``cfg.work_layer`` decoder layers over the FULL image-token sequence to get
    question-aware attention + contextualized hidden states, then re-selects image tokens
    by fusing (AV) the last input token's attention to each image token and (SV) each image
    token's SVD row contribution. The AV term needs only the last query's attention row, so
    on Qwen2 it is computed in O(L) via a single query (see ``_partial_forward``) instead of
    materializing the full ``output_attentions=True`` O(L²) weights (which forces the
    SDPA→eager fallback). Assumes batch_size=1 and a single contiguous image-token block at
    ``[sys_length, sys_length + image_length)``.

    Returns ``(new_embeds (1,L',D), new_attn_mask (1,L')|None, new_pos (1,L'),
    keep_global (L',), final_image_count int)``.
    """
    embeds = inputs_embeds
    bsz, seq_len, dim = embeds.shape
    dev = embeds.device
    dtype = embeds.dtype

    rank = max(2, min(cfg.rank, image_length))

    if position_ids is None:
        position_ids = torch.arange(seq_len, device=dev, dtype=torch.long).unsqueeze(0)

    # Transparent no-op: caller asked to keep >= all image tokens and merge nothing. Return
    # the inputs unchanged so the generate-override path can be diffed bit-identically against
    # the no-prune baseline (isolates embeds / attention_mask / position reconstruction bugs).
    if cfg.rank >= image_length and cfg.merge_nums == 0:
        keep_global = torch.arange(seq_len, device=dev, dtype=torch.long)
        return embeds, attention_mask, position_ids, keep_global, int(image_length)

    # pass 1: layers 0..work_layer-1 over the full sequence -> question-aware attention (AV)
    # + contextualized hidden states (SV).  The AV term needs only the LAST query's
    # attention row, so on Qwen2 it is reconstructed in O(L) (see _partial_forward) instead
    # of materializing the full O(L²) attention weights via output_attentions=True.
    hid, q2img_full = _partial_forward(
        embeds, position_ids, position_embeddings, layers, cfg.work_layer
    )

    final_keep_local, n_kept, er = _rank_select_and_merge(
        q2img_full, hid, embeds, sys_length, image_length, rank, cfg
    )

    # global keep indices: prefix + kept image tokens + suffix.
    keep_global = torch.cat([
        torch.arange(0, sys_length, device=dev, dtype=torch.long),
        sys_length + final_keep_local,
        torch.arange(sys_length + image_length, seq_len, device=dev, dtype=torch.long),
    ])
    new_embeds = embeds[:, keep_global, :]
    new_attn_mask = attention_mask[:, keep_global] if attention_mask is not None else None
    new_pos = torch.arange(new_embeds.shape[1], device=dev, dtype=torch.long).unsqueeze(0)

    # report erank of the FINAL selection.
    if on_selected is not None:
        on_selected(er)

    return new_embeds, new_attn_mask, new_pos, keep_global, n_kept


def qadp_llm_prune_frames(
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    sys_length: int,
    frame_len: int,
    num_frames: int,
    layers,
    cfg: QADPConfig,
    on_selected: Optional[Callable[[float], None]] = None,
    position_embeddings=None,
):
    """Per-frame QADP: prune each of ``num_frames`` contiguous ``frame_len``-token blocks.

    The video block is ``[sys_length, sys_length + num_frames * frame_len)``; each frame is
    pruned **independently** to ``rank`` tokens (``rank`` is capped to ``frame_len``), so the
    budget is per-frame — matching STC-Pruner's ``token_per_frame``.  A single partial
    forward over the full sequence produces the question-aware attention + contextualised
    hidden states reused by every frame's AV/SV rank fusion (no per-frame re-forward).

    Returns ``(new_embeds (1,L',D), new_attn_mask (1,L')|None, new_pos (1,L'),
    keep_global (L',), final_video_count int)`` where ``final_video_count`` is the total
    kept visual tokens across all frames.
    """
    embeds = inputs_embeds
    bsz, seq_len, dim = embeds.shape
    dev = embeds.device
    dtype = embeds.dtype

    rank = max(2, min(cfg.rank, frame_len))

    if position_ids is None:
        position_ids = torch.arange(seq_len, device=dev, dtype=torch.long).unsqueeze(0)

    # Transparent no-op (same contract as qadp_llm_prune): rank covers the whole frame and
    # nothing is merged -> return unchanged.
    if cfg.rank >= frame_len and cfg.merge_nums == 0:
        keep_global = torch.arange(seq_len, device=dev, dtype=torch.long)
        return embeds, attention_mask, position_ids, keep_global, int(num_frames * frame_len)

    # Partial forward over the full sequence (identical to qadp_llm_prune; see
    # _partial_forward for the single-row AV optimization that avoids the O(L²) eager fallback).
    hid, q2img_full = _partial_forward(
        embeds, position_ids, position_embeddings, layers, cfg.work_layer
    )

    # SV (the SVD/eigh) for ALL frames is computed in one batched pass first: 64 sequential
    # small SVD/eigh calls are dominated by per-call GPU/cuSOLVER launch overhead, so batching
    # fills the GPU and amortizes it.  The AV/SV fusion + merge stay per-frame (cheap).
    video_hid = hid[0, sys_length:sys_length + num_frames * frame_len, :].to(torch.float32)
    row_contrib_all = _sv_row_contrib_batched(video_hid.view(num_frames, frame_len, -1), frame_len)

    # Per-frame select + merge, preserving temporal order in the kept sequence.
    keep_video = []
    eranks = []
    for f in range(num_frames):
        block_start = sys_length + f * frame_len
        final_keep_local, n_f, er_f = _rank_select_and_merge(
            q2img_full, hid, embeds, block_start, frame_len, rank, cfg,
            row_contrib=row_contrib_all[f],
        )
        keep_video.append(block_start + final_keep_local)
        eranks.append(er_f)

    keep_video = torch.cat(keep_video)  # (total_kept,) global indices
    keep_global = torch.cat([
        torch.arange(0, sys_length, device=dev, dtype=torch.long),
        keep_video,
        torch.arange(sys_length + num_frames * frame_len, seq_len, device=dev, dtype=torch.long),
    ])
    new_embeds = embeds[:, keep_global, :]
    new_attn_mask = attention_mask[:, keep_global] if attention_mask is not None else None
    new_pos = torch.arange(new_embeds.shape[1], device=dev, dtype=torch.long).unsqueeze(0)

    if on_selected is not None:
        on_selected(float(sum(eranks) / max(1, len(eranks))))

    return new_embeds, new_attn_mask, new_pos, keep_global, int(keep_video.numel())
