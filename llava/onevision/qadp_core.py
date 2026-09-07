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
    ``cfg.work_layer`` decoder layers over the FULL image-token sequence with
    ``output_attentions=True`` to get question-aware attention + contextualized hidden
    states, then re-selects image tokens by fusing (AV) the last input token's attention
    to each image token and (SV) each image token's SVD row contribution. Assumes
    batch_size=1 and a single contiguous image-token block at ``[sys_length, sys_length
    + image_length)``.

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

    # 4D causal mask for the partial (prefill) forward over the full sequence.
    causal = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=dev, dtype=dtype)
    causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

    # pass 1: layers 0..work_layer-1 over the full sequence -> question-aware attn + hidden.
    hid = embeds
    attn_w = None
    for layer in layers[: cfg.work_layer]:
        layer_kw = dict(attention_mask=causal, position_ids=position_ids,
                        use_cache=False, output_attentions=True)
        if position_embeddings is not None:
            # transformers >= 4.46 Qwen2/Qwen3 layers require the precomputed RoPE
            # (cos, sin) tuple; Vicuna/Llama (4.37.2) computes RoPE from position_ids
            # internally and must NOT receive this kwarg.
            layer_kw["position_embeddings"] = position_embeddings
        out = layer(hid, **layer_kw)
        hid = out[0]
        attn_w = out[1]  # (B, heads, L, L)

    img_slice = slice(sys_length, sys_length + image_length)

    # AV: last input token's (question) attention to each image token, mean over heads.
    q2img = attn_w.mean(dim=1)[0, -1, img_slice]  # (image_length,)
    _, av_order = torch.sort(q2img, descending=True)

    # SV: SVD row contribution of the contextualized image tokens.
    img_hid = hid[0, img_slice, :].to(torch.float32)  # (image_length, D)
    U, S, _ = torch.linalg.svd(img_hid, full_matrices=False)
    row_contrib = (U * S.unsqueeze(0)).abs().sum(dim=1)  # (image_length,)
    _, sv_order = torch.sort(row_contrib, descending=True)

    # rank fusion: positional weighting (TokenCarve AV/SV fusion).
    m = image_length
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
    keep_local = keep_local.to(device=dev, dtype=torch.long)  # (rank,) local image indices

    # prune-then-merge: fold the most-similar B (bottom half) into A (top half) via cosine.
    set_length = rank // 2
    A_local = keep_local[:set_length]
    B_local = keep_local[set_length:]
    img_emb = embeds[0, img_slice, :].float()  # (image_length, D)
    norm = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
    sim = norm[B_local] @ norm[A_local].t()  # (set_length, set_length)
    sim_max, sim_arg = sim.max(dim=-1)
    reduce_n = min(set_length, cfg.merge_nums)
    order = torch.sort(sim_max, descending=True).indices
    merge_B = B_local[order[:reduce_n]]
    merge_A = A_local[sim_arg[order[:reduce_n]]]
    remaining_B = B_local[order[reduce_n:]]

    # merged image embeddings: mean of each A and the B folded into it; order = A + remaining B.
    merged = img_emb.clone()
    counts = torch.ones(image_length, device=dev, dtype=torch.float32)
    merged.index_add_(0, merge_A, img_emb[merge_B])
    counts.index_add_(0, merge_A, torch.ones(reduce_n, device=dev, dtype=torch.float32))
    merged = merged / counts.unsqueeze(-1)
    final_keep_local = torch.cat([A_local, remaining_B])
    final_img_emb = merged[final_keep_local].to(dtype)  # (T_final, D)

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
    er = effective_rank(final_img_emb)
    if on_selected is not None:
        on_selected(er.item())

    return new_embeds, new_attn_mask, new_pos, keep_global, int(final_keep_local.numel())
