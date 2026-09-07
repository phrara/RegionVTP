"""LLaVA-1.5 专属的 STC-Cacher 接入适配器（opt-in，默认关）。

把 STC 的帧间选择性重算（STC-Cacher）并进 LLaVA-1.5 的 CLIP 视觉塔，但严格做了
模型适配，保证「不覆盖原逻辑、不影响已有 benchmark」：

1. **门控**：只有 ``STC_PATCH_VISION=1`` 才注册；否则直接返回，代码路径与原来
   bit-exact 一致。
2. **注意力权重保留**：STC 原版 ``stc_sdpa_attention`` 会丢弃 attention 权重（返回
   None），而 LLaVA-1.5 / QADP 的 token 剪枝以 ``CLS→patch`` 注意力作为排序分数
   （``llava_arch.py`` 里 ``ranking = image_attentions``）。这里用
   :func:`_llava_attention_preserve` 替换 ``layer.stc_attention``，在
   ``output_attentions=True`` 时算出真实注意力权重。
3. **单图直通**：LLaVA-1.5 是单图模型，没有「上一帧」可复用；若按原样开
   ``selective``，会把相邻样本误当成相邻帧而算错。因此本适配器把策略固定为
   ``strategy="none"``（全量前向）并关掉 CUDA graph，保证开启后输出与基线一致、
   跨样本安全——真正的帧间加速留给多帧视频底座（见 QADP-streaming-融合方案.md 的
   M0/M1，那里用 ``enable_streaming_cacher`` + ``reset_streaming_cacher``）。
"""

from __future__ import annotations

import logging
import os
import types
from typing import Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)

_STC_TRUE_VALUES = {"1", "true", "yes", "on", "selective", "cacher"}


def _llava_attention_preserve(
    layer,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """SDPA + out_proj，但在 ``output_attentions=True`` 时返回真实注意力权重。

    与 STC 的 :func:`stc_sdpa_attention` 唯一区别是不丢弃注意力权重。LLaVA-1.5 的
    CLIP 塔是非因果、无 padding 的固定网格，``attention_mask`` 实际为 None，因此
    手动 softmax 路径与 HF CLIP 的 ``output_attentions=True`` 路径等价。
    """
    if not output_attentions:
        from stc.cacher.reference_forward import stc_sdpa_attention

        return stc_sdpa_attention(
            layer, query_states, key_states, value_states, attention_mask, output_attentions
        )

    attn = layer.self_attn
    head_dim = key_states.shape[-1]
    scale = head_dim ** -0.5

    attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * scale
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_weights = attn_weights.masked_fill(~attention_mask, float("-inf"))
        else:
            attn_weights = attn_weights + attention_mask
    attn_weights = torch.softmax(attn_weights, dim=-1)

    dropout = getattr(attn, "dropout", None)
    p = dropout.p if isinstance(dropout, nn.Dropout) else 0.0
    attn_weights = torch.nn.functional.dropout(attn_weights, p=p, training=layer.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    batch_size = query_states.shape[0]
    query_length = query_states.shape[-2]
    embed_dim = getattr(attn, "embed_dim", query_states.shape[1] * query_states.shape[-1])
    attn_output = attn_output.view(batch_size, query_length, embed_dim)
    attn_output = attn.out_proj(attn_output)

    return attn_output, attn_weights


def maybe_register_stc_cacher(vision_tower: nn.Module) -> nn.Module:
    """opt-in 地把 STC-Cacher 挂到 CLIP 视觉塔（STC_PATCH_VISION=1 才生效）。

    关闭（默认）时不做任何事，保证与原有代码路径 bit-exact 一致；开启时以
    ``strategy="none"``（单图全量直通）+ 注意力权重保留的方式注册，用于验证接入
    正确性。真正的选择性重算在多帧视频底座上打开（M0/M1）。
    """
    raw = os.environ.get("STC_PATCH_VISION")
    if raw is None or raw.strip().lower() not in _STC_TRUE_VALUES:
        return vision_tower

    from stc.config import STCConfig
    from stc.integrations.hf_vit import register_stc_cacher

    cfg = STCConfig.initialize_from_env()
    # 单图模型无帧间复用：强制全量直通，避免把相邻样本误当相邻帧（真正的 selective
    # 留给多帧视频底座）；单图下也无 selective 帧，CUDA graph 无收益，关掉更稳。
    cfg.cache.strategy = "none"
    cfg.cache.cuda_graph = False

    register_stc_cacher(vision_tower, kind="clip", config=cfg.cache)

    for layer in vision_tower.vision_model.encoder.layers:
        layer.stc_attention = types.MethodType(_llava_attention_preserve, layer)

    logger.info(
        "STC-Cacher registered on LLaVA-1.5 CLIP tower (kind=clip, strategy=%s, "
        "attention weights preserved). No speedup on single-image; real temporal "
        "reuse requires the multi-frame video base.",
        cfg.cache.strategy,
    )
    return vision_tower
