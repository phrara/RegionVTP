#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
import time
import os

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
import math
n_rank = 0.0
n_sample = 0
erank_log = []

# --- [RegionVTP] CLIP text tower (question encoder) ---
_CLIP_TEXT = None
_CLIP_TOK = None


def _get_clip_text(model_name):
    """Load the CLIP text tower + projection heads from the SAME checkpoint the vision
    tower uses (clip-vit-large-patch14-336 is a full CLIP model: vision + text + both
    projections). Kept on CPU; only the ~1MB visual_projection is moved to GPU on demand."""
    global _CLIP_TEXT, _CLIP_TOK
    if _CLIP_TEXT is None:
        from transformers import CLIPModel, CLIPTokenizer
        _CLIP_TEXT = CLIPModel.from_pretrained(model_name).eval()
        for p in _CLIP_TEXT.parameters():
            p.requires_grad_(False)
        _CLIP_TOK = CLIPTokenizer.from_pretrained(model_name)
    return _CLIP_TEXT, _CLIP_TOK


def encode_query(text, model_name, device, dtype):
    """Raw question string -> 512-dim L2-normalized CLIP text embedding (joint space)."""
    clip, tok = _get_clip_text(model_name)
    with torch.no_grad():
        ids = tok(text, return_tensors="pt", truncation=True, max_length=77)
        q = clip.get_text_features(**ids)   # (1, 512), already text_projection + L2 norm
    return q[0].to(device=device, dtype=dtype)


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


ERANK_AVG_REF = 90
TAU_MAX = 0.25


def calculate_adaptive_tau(order_i, erank_input, erank_avg=ERANK_AVG_REF, tau_max=TAU_MAX):
    """Eq. 6: tau_i = order_i * (erank_input / erank_avg * 0.01), capped at tau_max."""
    tau = order_i * (erank_input / erank_avg * 0.01)
    return min(tau, tau_max)


def effective_rank(features): # fast Version
    X = features.float()
    X -= X.mean(dim=0, keepdim=True)
    C=torch.mm(X,X.T)# (N, N)
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
    """Section 4.3 "Empirically guided adaptive similarity thresholding":
    1. Sort tokens by attention score, descending.
    2. Take the highest-ranked surviving token; prune every surviving candidate whose
       cosine distance to it is smaller than the adaptive threshold tau_i (Eq. 6).
    3. Repeat with the next highest-ranked surviving token until max_tokens are selected.

    `static_tau`, when given, replaces tau_i with a single fixed threshold for every
    token (used by the DIST_THRESHOLD ablation in scripts/v1_5/eval/ablation_dist_threshold.sh,
    reproducing the constant-tau sweep in Table 12) instead of the per-token adaptive rule.
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

    # Guarantee exactly max_tokens: if aggressive pruning left too few survivors,
    # backfill with the remaining highest-attention tokens.
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


@torch.no_grad()
def merge_pruned_tokens(image_features, kept_indices):
    """ToMe-style token merging (Phase 1): fold each pruned token into its most similar
    kept token by count-averaging their features, instead of discarding it. Preserves the
    pruned information — the same "select + merge" recipe used by VScan / TokenCarve /
    VisionTrim to stay accurate under high compression. Kept-token positions are unchanged;
    only their features are enriched (pruned positions are ignored downstream).

    image_features: (B, N, D) projected features. kept_indices: (T,) global kept indices.
    """
    B, N, D = image_features.shape
    device = image_features.device
    kept_indices = kept_indices.to(device)

    kept_mask = torch.zeros(N, dtype=torch.bool, device=device)
    kept_mask[kept_indices] = True
    pruned_mask = ~kept_mask
    if pruned_mask.sum() == 0:
        return image_features

    kept_sorted = torch.nonzero(kept_mask, as_tuple=False).squeeze(-1)  # (K,) in index order
    out = image_features.float().clone()
    counts = torch.ones(B, N, device=device, dtype=torch.float32)
    for b in range(B):
        feats = out[b]  # (N, D) fp32
        fn = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        sim = fn @ fn.t()  # (N, N) cosine
        sim_to_kept = sim[pruned_mask][:, kept_mask]  # (P, K)
        nearest = kept_sorted[sim_to_kept.argmax(dim=1)]  # (P,) global kept idx
        src = feats[pruned_mask]  # (P, D)
        index = nearest.unsqueeze(1).expand_as(src)  # (P, D)
        out[b] = out[b].scatter_add(0, index, src)
        counts[b] = counts[b].scatter_add(0, nearest, torch.ones(src.shape[0], device=device, dtype=torch.float32))
    out = out / counts.unsqueeze(-1)
    return out.to(image_features.dtype)


def allocate_budget(weights, total, mode="waterfill"):
    """Distribute exactly `total` tokens across regions by `weights` (non-negative, sums to 1).

    waterfill: give every region a floor of 1 first (spatial coverage), then split the
               remaining budget by weight via largest-remainder rounding.
    softmax:   pure proportional split (largest-remainder rounding), regions may get 0.
    """
    n = weights.numel()
    device = weights.device
    if mode == "waterfill" and total >= n:
        base = torch.ones(n, dtype=torch.long, device=device)
        exact = weights * float(total - n)
        extra = exact.floor().long()
        rem = exact - extra.float()
        deficit = int(total - n - extra.sum().item())
        if deficit > 0:
            _, order = torch.sort(rem, descending=True)
            extra[order[:deficit]] += 1
        return base + extra
    else:
        exact = weights * float(total)
        budget = exact.floor().long()
        rem = exact - budget.float()
        deficit = int(total - budget.sum().item())
        if deficit > 0:
            _, order = torch.sort(rem, descending=True)
            budget[order[:deficit]] += 1
        return budget


def select_tokens_regionwise(
    ranking,
    d,
    image_features,
    max_tokens,
    side,
    R,
    gamma=1.0,
    budget_mode="waterfill",
    static_tau=None,
    erank_avg=ERANK_AVG_REF,
    tau_max=TAU_MAX,
):
    """Region-adaptive visual token pruning (design doc §1 ②③④⑤):
    1. Reshape the N tokens to a (side, side) grid and split into R*R square regions.
    2. Score each region by mean relevance (ranking) and feature diversity (effective_rank),
       then allocate an adaptive local budget across regions.
    3. Within each region, run the attention+diversity greedy selection with the region's
       own erank (region-aware adaptive tau), then map back to global indices.
    """
    score = ranking[0].float()   # (N,)
    # z-score so region importance is on a consistent scale whether `ranking` is raw
    # CLS attention (λ=1) or the blended query relevance (λ<1).
    score = (score - score.mean()) / (score.std() + 1e-8)
    feats = image_features[0]    # (N, D)
    N = score.shape[0]
    device = score.device
    hw = side // R

    idx2d = torch.arange(N, device=device).view(side, side)
    regions = idx2d.view(R, hw, R, hw).permute(0, 2, 1, 3).reshape(R * R, hw * hw)  # (R*R, hw*hw)

    imp, comp = [], []
    for r in range(R * R):
        idx_r = regions[r]
        imp.append(score[idx_r].mean().item())
        comp.append(effective_rank(feats[idx_r]).item())
    imp = torch.tensor(imp, device=device, dtype=torch.float32)
    comp = torch.tensor(comp, device=device, dtype=torch.float32)

    # importance + gamma * log(complexity) in log-space for numerical stability
    logits = imp + gamma * torch.log(comp + 1e-8)
    weights = torch.softmax(logits, dim=0)  # (R*R,)

    budget = allocate_budget(weights, max_tokens, mode=budget_mode)  # (R*R,), sums to max_tokens

    selected = []
    for r in range(R * R):
        b = min(int(budget[r].item()), int(regions[r].numel()))
        if b <= 0:
            continue
        idx_r = regions[r]
        score_r = score[idx_r].unsqueeze(0)      # (1, hw*hw)
        d_r = d[idx_r][:, idx_r]                 # (hw*hw, hw*hw)
        local = select_diverse_tokens_by_attention_and_distance(
            score_r, d_r,
            erank_input=comp[r].item(),
            max_tokens=b,
            static_tau=static_tau,
            erank_avg=erank_avg,
            tau_max=tau_max,
        )
        selected.append(idx_r[local])

    selected_idx = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long, device=device)

    # Safety net: guarantee exactly max_tokens (rounding / b > region-size edge cases).
    if selected_idx.numel() < max_tokens:
        chosen = set(selected_idx.tolist())
        for i in torch.argsort(score, descending=True):
            if selected_idx.numel() >= max_tokens:
                break
            ii = i.item()
            if ii not in chosen:
                selected_idx = torch.cat([selected_idx, i.unsqueeze(0)])
                chosen.add(ii)

    return selected_idx


@torch.no_grad()
def select_tokens_prunesid(patch_features, attention, max_tokens, weights=None):
    """PRUNESID (ICLR 2026) PSCA-NMS, ported to the no-CLS LLaVA-1.5 pipeline.

    Selection = principal-semantic-component grouping (sigmoid + low-rank PCA of the
    ViT penultimate patch features) -> per-group CLS-attention ranking -> intra-group
    NMS by cosine similarity -> per-group budget (floor 1, cap 5*ceil(T/64), the rest
    proportional to NMS survivors). The PCA grouping forces every semantic direction to
    be represented (coverage); attention still decides *which* token inside each group.
    Returns a (T,) tensor of global patch indices (any order; downstream applies a bool
    mask so the original spatial order is preserved).

    patch_features: (B, N, C) ViT penultimate-layer patch features (CLS already dropped).
    attention:      (B, N) CLS->patch attention from the same layer (mean over heads).
    weights:        (N,) optional per-token covariance weight (Q-PSCA). When given, each
                    token's contribution to the PCA covariance is scaled by sqrt(w), so the
                    principal semantic directions become question-adaptive. Grouping only;
                    the NMS score remains CLS attention.
    """
    x = patch_features[0].float()      # (N, C)
    attn = attention[0].float()        # (N,)
    N = x.shape[0]
    T = int(max_tokens)
    dev = x.device

    # 1) PSCA grouping: sigmoid-scale, then low-rank PCA over the token dim. The right
    #    singular vectors V are the semantic directions; each token joins argmax_j |V_ij|.
    q = max(1, int(T / 4))                                  # number of semantic directions
    standard = torch.sigmoid(x).t()                         # (C, N)
    if weights is not None:
        # Q-PSCA: scale each token's covariance contribution by sqrt(question relevance),
        # so the principal semantic directions rotate toward what the question asks about.
        w = weights.to(device=dev, dtype=torch.float32).clamp(min=0.0)   # (N,)
        standard = standard * torch.sqrt(w + 1e-8).unsqueeze(0)          # (C, N)
    _, _, V = torch.pca_lowrank(standard, q=q)              # V: (N, q), column-centered
    V = torch.abs(V)
    belong = torch.argmax(V, dim=-1)                        # (N,) group id per token

    # 2) score = CLS attention, but only inside the token's own group (masked elsewhere).
    #    (We rank by attention; |V| only determines grouping, matching the reference code.)

    # 3) within-group cosine similarity + global redundancy -> adaptive NMS threshold.
    xn = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    sim = xn @ xn.t()                                       # (N, N)
    triu = torch.triu(torch.ones_like(sim), diagonal=1).bool()
    sim_mean = (sim * triu).sum() / triu.sum()              # scalar global redundancy
    tau = (T / 32.0) * sim_mean                             # adaptive threshold
    same_group = belong.unsqueeze(1) == belong.unsqueeze(0)  # (N, N)
    group_sim = sim.clone()
    group_sim[~same_group] = 0.0                            # suppress cross-group edges

    # 4) intra-group NMS: greedy by attention, suppress cos-similar (>tau) same-group tokens.
    group_kept = [[] for _ in range(q)]
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    for i in torch.argsort(attn, descending=True).tolist():
        if not alive[i]:
            continue
        alive[i] = False
        group_kept[int(belong[i].item())].append(i)
        alive[(group_sim[i] > tau) & alive] = False

    keep_counts = torch.tensor([len(k) for k in group_kept], device=dev, dtype=torch.long)
    group_counts = torch.tensor(
        [(belong == g).sum().item() for g in range(q)], device=dev, dtype=torch.long)

    # 5) per-group budget: floor 1 (coverage), cap, rest proportional to NMS survivors.
    lower = torch.clamp(group_counts, max=1)                # 1 if group non-empty else 0
    upper = torch.minimum(
        torch.minimum(
            torch.full((q,), 5 * math.ceil(T / 64), device=dev, dtype=torch.long),
            group_counts),
        keep_counts)
    while int(upper.sum()) < T:                             # enlarge cap if NMS kept too few
        upper = torch.minimum(upper + 1, group_counts)

    other = max(int(T - lower.sum()), 0)                    # budget beyond the coverage floor
    denom = keep_counts.sum().item()
    norm = keep_counts.float() / (denom if denom > 0 else 1.0)
    exact = norm * other
    other_d = exact.floor().long()
    rem = exact - exact.floor()
    deficit = int(other - other_d.sum())
    if deficit > 0:                                          # largest-remainder rounding
        _, order = torch.sort(rem, descending=True)
        other_d[order[:deficit]] += 1
    budget = torch.minimum(other_d + lower, upper)

    sort_idx = torch.argsort(keep_counts, descending=True).tolist()
    fill = 0
    while int(budget.sum()) < T and fill < q:               # fill leftover to exactly T
        g = sort_idx[fill]
        add = min(int(upper[g] - budget[g]), int(T - budget.sum()))
        if add > 0:
            budget[g] += add
        fill += 1

    # 6) take the top-budget tokens of each group (attention order), backfill if under.
    selected = []
    for g in range(q):
        selected.extend(group_kept[g][:int(budget[g].item())])
    if len(selected) < T:
        have = set(selected)
        for i in torch.argsort(attn, descending=True).tolist():
            if len(selected) >= T:
                break
            if i not in have:
                selected.append(i)
                have.add(i)
    return torch.tensor(selected[:T], device=dev, dtype=torch.long)


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()


    def encode_images(self, images, query_embeds=None):
        image_features, image_attentions = self.get_model().get_vision_tower()(images) # (B, N, C), (B, M, N)
        B, N, C = image_features.shape

        # [RegionVTP] keep pre-projector patch features for CLIP-space query relevance
        patch_1024 = image_features

        visual_token_num = self.get_visual_token_num() # T
        image_attentions = image_attentions.mean(dim=1) # (B, N)
        image_features = self.get_model().mm_projector(image_features) # (B, N, D)

        # [RegionVTP] TokenCarve: pruning happens inside the LLM (layer-2 rank-fusion), so the
        # ViT passes its full patch set through the projector unpruned.
        if os.environ.get("LLM_LAYER_PRUNE", "0") == "1":
            full_mask = torch.ones(B, N, dtype=torch.bool, device=image_features.device)
            return image_features, full_mask, image_attentions

        if visual_token_num >=576:
            return image_features, torch.ones(B, N, dtype=torch.bool, device=image_features.device), image_attentions
        
        erank = effective_rank(image_features[0])

        dist_th_env = os.environ.get("DIST_THRESHOLD", None)
        static_tau = float(dist_th_env) if dist_th_env is not None else None

        feats_norm = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)  # (1, N, D)
        feats_norm_squeezed = feats_norm.squeeze(0)  # (N, D)
        cos_sim = torch.mm(feats_norm_squeezed, feats_norm_squeezed.t())  # (N, N)
        d = 1.0 - cos_sim   # (N, N)

        # [RegionVTP] Query-conditioned relevance: blend CLS->patch attention with question
        # embedding similarity. QUERY_LAMBDA=1.0 (default) reproduces AgilePruner exactly.
        lam = float(os.environ.get("QUERY_LAMBDA", "1.0"))
        # Q-PSCA: question-weighted PSCA grouping (PRUNESID covariance + cross-modal weight).
        qpsca = (os.environ.get("SELECTOR", "agilepruner") == "prunesid"
                 and os.environ.get("PRUNESID_QUERY", "0") == "1")
        qpsca_weights = None
        ranking = image_attentions  # (B, N) raw CLS->patch attention, used as the ranking score
        if query_embeds is not None and (lam < 1.0 or qpsca):
            # [RegionVTP] cosine in CLIP's 512-dim joint space: project pre-projector patches
            # via visual_projection and compare to the CLIP text embedding of the question.
            clip, _ = _get_clip_text(os.environ.get("CLIP_TEXT_MODEL") or self.get_vision_tower().vision_tower_name)
            vp = clip.visual_projection.weight.detach().to(device=image_features.device, dtype=torch.float32)
            v512 = patch_1024.float() @ vp.t()  # (B, N, 512)
            v512 = v512 / (v512.norm(dim=-1, keepdim=True) + 1e-8)
            q = query_embeds.float() / (query_embeds.float().norm() + 1e-8)  # (512,)
            query_sim = torch.einsum('bnd,d->bn', v512, q)  # (B, N)
            if qpsca:
                # Q-PSCA: softmax over tokens -> covariance weight (sharpness via temperature).
                t_q = float(os.environ.get("QPSPCA_TEMP", "0.5"))
                qpsca_weights = torch.softmax(query_sim[0].float() / t_q, dim=0)  # (N,)
            if lam < 1.0:
                attn = image_attentions.float()
                sim = query_sim
                attn_n = (attn - attn.mean(dim=-1, keepdim=True)) / (attn.std(dim=-1, keepdim=True) + 1e-8)
                sim_n = (sim - sim.mean(dim=-1, keepdim=True)) / (sim.std(dim=-1, keepdim=True) + 1e-8)
                ranking = lam * attn_n + (1.0 - lam) * sim_n

        # [RegionVTP] Selection strategy. SELECTOR=prunesid switches to the PSCA-NMS
        # spectral-coverage selector (PRUNESID, ICLR 2026); default is AgilePruner's
        # attention+diversity greedy. REGION_SIZE>1 still routes to the region selector.
        selector = os.environ.get("SELECTOR", "agilepruner")
        R = int(os.environ.get("REGION_SIZE", "0"))
        side = int(round(N ** 0.5))
        if selector == "prunesid":
            # PRUNESID_SPACE: "pre" = ViT penultimate (1024-dim, the reference setting),
            # "post" = post-mm_projector (4096-dim, the space that actually feeds the LLM).
            space = os.environ.get("PRUNESID_SPACE", "pre")
            feats = image_features if space == "post" else patch_1024
            token_indices = select_tokens_prunesid(
                feats, image_attentions, max_tokens=visual_token_num,
                weights=qpsca_weights,
            )
        elif R > 1 and side * side == N and side % R == 0:
            gamma = float(os.environ.get("REGION_GAMMA", "1.0"))
            budget_mode = os.environ.get("BUDGET_MODE", "waterfill")
            token_indices = select_tokens_regionwise(
                ranking, d, image_features,
                max_tokens=visual_token_num, side=side, R=R,
                gamma=gamma, budget_mode=budget_mode, static_tau=static_tau,
            )
        else:
            token_indices = select_diverse_tokens_by_attention_and_distance(
                ranking, d, erank_input=erank.item(), max_tokens=visual_token_num, static_tau=static_tau
            )

        top_indices = token_indices.unsqueeze(0)
        index_masks = torch.zeros(B, N, dtype=torch.bool, device=image_features.device) # (B, N)

        if top_indices.dim() == 2 and top_indices.size(0) == 1:
            top_indices = top_indices.expand(B, -1)

        index_masks.scatter_(1, top_indices, True) # (B, N)

        # [RegionVTP] Phase 1: ToMe-style merge — fold pruned tokens into their nearest
        # kept token instead of discarding, preserving their information. Off by default
        # so MERGE_MODE unset reproduces AgilePruner bit-exactly.
        if os.environ.get("MERGE_MODE", "off") == "nearest":
            image_features = merge_pruned_tokens(image_features, token_indices)

        global n_rank, n_sample, erank_log
        selected_erank = effective_rank(image_features[0][index_masks[0]])
        n_rank += selected_erank.item()
        n_sample += 1
        erank_log.append(selected_erank.item())

        return image_features, index_masks, image_attentions

    def token_carve_llm_prune(self, inputs_embeds, attention_mask, position_ids, sys_length, image_length):
        """TokenCarve-style LLM-layer pruning (rank-fusion + prune-then-merge).

        Runs the first `TCARVE_LAYER` decoder layers over the FULL image-token sequence to
        obtain question-aware attention + contextualized hidden states, then re-selects a
        subset of image tokens by fusing (i) the last input token's (question) attention to
        each image token [AV] and (ii) each image token's SVD row contribution C_i = sum_j
        |U_ij * sigma_j| in the contextualized space [SV]. The surviving top half is kept;
        the most-similar tokens of the bottom half are folded (mean) into their nearest kept
        token. Returns a shorter input the remaining layers process via a normal forward (a
        full re-run from layer 0), which sidesteps KV-cache surgery. Faithful to TokenCarve's
        SELECTION signal; the merge is applied at the input-embedding level rather than
        in-place at layer 2 (documented deviation).

        Assumes batch_size=1 and a single contiguous image-token block.
        """
        model = self.get_model()
        visual_token_num = self.get_visual_token_num()
        work_layer = int(os.environ.get("TCARVE_LAYER", "2"))
        rank = int(os.environ.get("TCARVE_RANK", str(int(1.5 * visual_token_num))))
        merge_nums = int(os.environ.get("TCARVE_MERGE", str(visual_token_num // 2)))
        sv_av_mode = int(os.environ.get("TCARVE_MODE", "0"))      # 0=fuse, 1=SV only, 2=AV only
        sv_av_weight = float(os.environ.get("TCARVE_WEIGHT", "0.5"))
        qadp_diversity = int(os.environ.get("QADP_DIVERSITY", "0"))  # 1 = QADP diversity coverage
        qadp_tau = float(os.environ.get("QADP_TAU", "0.2"))          # static similarity threshold
        qadp_adaptive = int(os.environ.get("QADP_ADAPTIVE", "0"))    # 1 = erank-adaptive tau (Eq. 6)
        qadp_erank_ref = float(os.environ.get("QADP_ERANK_REF", str(ERANK_AVG_REF)))
        qadp_tau_max = float(os.environ.get("QADP_TAU_MAX", str(TAU_MAX)))

        embeds = inputs_embeds
        bsz, seq_len, dim = embeds.shape
        dev = embeds.device
        dtype = embeds.dtype

        rank = max(2, min(rank, image_length))

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=dev, dtype=torch.long).unsqueeze(0)

        # 4D causal mask for the partial (prefill) forward over the full sequence.
        causal = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=dev, dtype=dtype)
        causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

        # pass 1: layers 0..work_layer-1 over the full sequence -> question-aware attn + hidden.
        hid = embeds
        attn_w = None
        for layer in model.layers[:work_layer]:
            out = layer(hid, attention_mask=causal, position_ids=position_ids,
                        use_cache=False, output_attentions=True)
            hid = out[0]
            attn_w = out[1]                                        # (B, heads, L, L)

        img_slice = slice(sys_length, sys_length + image_length)

        # AV: last input token's (question) attention to each image token, mean over heads.
        q2img = attn_w.mean(dim=1)[0, -1, img_slice]               # (image_length,)
        _, av_order = torch.sort(q2img, descending=True)

        # SV: SVD row contribution of the contextualized image tokens.
        img_hid = hid[0, img_slice, :].to(torch.float32)           # (image_length, D)
        U, S, _ = torch.linalg.svd(img_hid, full_matrices=False)
        row_contrib = (U * S.unsqueeze(0)).abs().sum(dim=1)        # (image_length,)
        _, sv_order = torch.sort(row_contrib, descending=True)

        # rank fusion: positional weighting (TokenCarve AV/SV fusion).
        m = image_length
        idx_w = torch.arange(m, 0, -1, device=dev, dtype=torch.float32)
        fused = torch.zeros(m, device=dev, dtype=torch.float32)
        fused[av_order] += idx_w * sv_av_weight
        fused[sv_order] += idx_w * (1.0 - sv_av_weight)
        if qadp_diversity == 1:
            # QADP: question-adaptive diversity coverage (select-only). Rank by the fused
            # relevance score, but greedily suppress candidates whose cosine distance to an
            # already-selected token is below `qadp_tau` in the question-conditioned LLM space
            # (layer-`work_layer` hidden states), spreading the budget across the image instead
            # of over-focusing on one region. Reuses AgilePruner's greedy selector (Eq. 6).
            norm_hid = img_hid / (img_hid.norm(dim=-1, keepdim=True) + 1e-8)
            d_hid = 1.0 - (norm_hid @ norm_hid.t())                # (m, m) cosine distance
            erank_llm = effective_rank(img_hid).item()
            if qadp_adaptive == 1:
                # AgilePruner-style per-token adaptive tau (Eq. 6) in the LLM space:
                # tau_i = order_i * (erank / erank_ref * 0.01), capped at tau_max. The
                # reference values are ViT-calibrated; expose them so they can be retuned
                # for the LLM space without touching code.
                keep_local = select_diverse_tokens_by_attention_and_distance(
                    fused.unsqueeze(0), d_hid, erank_llm,
                    max_tokens=rank, static_tau=None,
                    erank_avg=qadp_erank_ref, tau_max=qadp_tau_max,
                )
            else:
                keep_local = select_diverse_tokens_by_attention_and_distance(
                    fused.unsqueeze(0), d_hid, erank_llm,
                    max_tokens=rank, static_tau=(qadp_tau if qadp_tau > 0 else None),
                )
        elif sv_av_mode == 2:
            keep_local = av_order[:rank]
        elif sv_av_mode == 1:
            keep_local = sv_order[:rank]
        else:
            keep_local = torch.topk(fused, rank).indices
        keep_local = keep_local.to(device=dev, dtype=torch.long)   # (rank,) local image indices

        # prune-then-merge: fold the most-similar B (bottom half) into A (top half) via cosine.
        set_length = rank // 2
        A_local = keep_local[:set_length]
        B_local = keep_local[set_length:]
        img_emb = embeds[0, img_slice, :].float()                  # (image_length, D)
        norm = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
        sim = norm[B_local] @ norm[A_local].t()                    # (set_length, set_length)
        sim_max, sim_arg = sim.max(dim=-1)
        reduce_n = min(set_length, merge_nums)
        order = torch.sort(sim_max, descending=True).indices
        merge_B = B_local[order[:reduce_n]]
        merge_A = A_local[sim_arg[order[:reduce_n]]]
        remaining_B = B_local[order[reduce_n:]]

        # merged image embeddings: mean of each A and the B folded into it; kept order = A + remaining B.
        merged = img_emb.clone()
        counts = torch.ones(image_length, device=dev, dtype=torch.float32)
        merged.index_add_(0, merge_A, img_emb[merge_B])
        counts.index_add_(0, merge_A, torch.ones(reduce_n, device=dev, dtype=torch.float32))
        merged = merged / counts.unsqueeze(-1)
        final_keep_local = torch.cat([A_local, remaining_B])
        final_img_emb = merged[final_keep_local].to(dtype)         # (T_final, D)

        # global keep indices: prefix + kept image tokens + suffix.
        keep_global = torch.cat([
            torch.arange(0, sys_length, device=dev, dtype=torch.long),
            sys_length + final_keep_local,
            torch.arange(sys_length + image_length, seq_len, device=dev, dtype=torch.long),
        ])
        new_embeds = embeds[:, keep_global, :]
        new_attn_mask = attention_mask[:, keep_global] if attention_mask is not None else None
        new_pos = torch.arange(new_embeds.shape[1], device=dev, dtype=torch.long).unsqueeze(0)

        # [RegionVTP] report erank of the FINAL selection (the number that matters here).
        global n_rank, n_sample, erank_log
        er = effective_rank(final_img_emb)
        n_rank += er.item()
        n_sample += 1
        erank_log.append(er.item())

        return new_embeds, new_attn_mask, new_pos, keep_global, int(final_keep_local.numel())

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # [RegionVTP] Query embedding for query-conditioned relevance: encode the question
        # with the CLIP text tower (same checkpoint as the vision tower) into its 512-dim
        # joint space, compared by cosine to the visual_projection of pre-projector patches.
        # The question follows the last <image> token; taking tokens after it drops the
        # constant system prompt. Assumes batch_size=1.
        query_embeds = None
        qpsca = (os.environ.get("SELECTOR", "agilepruner") == "prunesid"
                 and os.environ.get("PRUNESID_QUERY", "0") == "1")
        if float(os.environ.get("QUERY_LAMBDA", "1.0")) < 1.0 or qpsca:
            img_idx = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero().flatten()
            if img_idx.numel() > 0:
                query_ids = input_ids[0, img_idx[-1] + 1:]
                if query_ids.numel() > 0:
                    text = self.tokenizer.decode(query_ids, skip_special_tokens=True)
                    model_name = os.environ.get("CLIP_TEXT_MODEL") or vision_tower.vision_tower_name
                    query_embeds = encode_query(text, model_name, device=input_ids.device, dtype=input_ids.dtype)  # (512,)

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features, index_masks, image_attns = self.encode_images(concat_images, query_embeds=query_embeds)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            index_masks = torch.split(index_masks, split_sizes, dim=0)
            # mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            mm_patch_merge_type = 'spatial' # only support 'spatial' and 'spatial_unpad'
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
                index_masks = [x.flatten(0, 1) for x in index_masks]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, (image_feature, index_mask) in enumerate(zip(image_features, index_masks)):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        base_index_mask = index_mask[0]
                        image_feature = image_feature[1:]
                        index_mask = index_mask[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            index_mask = index_mask.view(num_patch_height, num_patch_width, height, width)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous().unsqueeze(0)
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            index_mask = index_mask.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            index_mask = unpad_image(index_mask, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(*index_mask.shape[:-1], 1, dtype=torch.bool).to(index_mask.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            index_mask = index_mask.flatten(1, 2).squeeze(0)
                            image_feature = image_feature[index_mask]
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                            index_mask = index_mask.flatten(0, 3)
                            image_feature = image_feature[index_mask]
                        base_image_feature = base_image_feature[base_index_mask]
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        index_mask = index_mask[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(1, dtype=torch.bool).to(index_mask.device)
                            ), dim=0)
                        image_feature = image_feature[index_mask]
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features, index_masks, image_attns = self.encode_images(images, query_embeds=query_embeds)
            new_image_features = []
            for image_feature, index_mask in zip(image_features, index_masks):
                image_feature = image_feature[index_mask]
                new_image_features.append(image_feature)
            image_features = torch.stack(new_image_features, dim=0)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        v_token_count = image_features[0].shape[0]

        # [RegionVTP] TokenCarve: prune + merge image tokens inside the LLM (rank-fusion at an
        # early layer). Batch-size-1 / single-image assumption, true for the eval harness.
        if os.environ.get("LLM_LAYER_PRUNE", "0") == "1" and batch_size == 1:
            img_idx_all = (_input_ids[0] == IMAGE_TOKEN_INDEX).nonzero().flatten()
            if img_idx_all.numel() > 0:
                sys_length = int(img_idx_all[0].item())
                image_length = int(image_features[0].shape[0])
                new_input_embeds, attention_mask, position_ids, keep_global, v_token_count = self.token_carve_llm_prune(
                    new_input_embeds, attention_mask, position_ids, sys_length, image_length)
                new_labels_padded = new_labels_padded[:, keep_global]

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, v_token_count, image_attns

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
