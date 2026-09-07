"""LLaVA-OneVision video wrapper with opt-in STC-Cacher (frame-to-frame ViT reuse).

M1 (video base + STC-Cacher) — the video counterpart of ``llava_onevision_qadp.py``
(M0, single-image QADP).  It adds the one thing the standard transformers
``get_video_features`` cannot do: drive the SigLIP tower **one frame at a time**
(B=1) so STC-Cacher can reuse stationary tokens across consecutive frames.  The
standard path batches all frames into a single ``vision_tower`` call, which the
cacher sees as one chunk and therefore always recomputes in full — no saving.

Opt-in / non-invasive (same contract as the LLaVA-1.5 CLIP adapter in
``llava/model/multimodal_encoder/stc_cacher_adapter.py``):

* The cacher is registered **only** when ``STC_PATCH_VISION=1``.  With the flag
  off, the tower runs its native forward (bit-identical baseline), so both arms
  share the same per-frame loop and the latency comparison is apples-to-apples.
* Touches only the OneVision SigLIP tower.  Never imports ``llava/model/`` (whose
  ``__init__`` crashes on transformers >= 4.40 because MPT was removed), never
  affects the LLaVA-1.5 CLIP tower or any existing benchmark.
"""

from __future__ import annotations

import torch

from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from stc import default_config, register_stc_cacher, reset_default_cache, stc_patch_vision_enabled


def load_onevision_video(model_path: str, device: str = "cuda"):
    """Load OneVision + processor, optionally patching the SigLIP tower with STC-Cacher.

    Reads ``STC_*`` env vars into the process-wide :class:`stc.STCConfig` (same as
    the STC reference ``load_model``), then registers the cacher iff
    ``STC_PATCH_VISION=1``.  Mirrors ``load_onevision_qadp`` (no ``"auto"``
    device_map, fp16, no flash-attn).
    """
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": device},
    )
    model.eval()

    cfg = default_config().initialize_from_env()
    if stc_patch_vision_enabled():
        register_stc_cacher(model.vision_tower, kind="siglip", config=cfg.cache)
        print(
            f"[STC] Cacher on SigLIP tower: strategy={cfg.cache.strategy} "
            f"update_ratio={cfg.cache.update_token_ratio} interval={cfg.cache.cache_interval}"
        )
    else:
        print("[STC] Cacher OFF (STC_PATCH_VISION unset) — pure OneVision baseline")
    reset_default_cache(0, cfg.cache.update_token_ratio)
    return model, processor


@torch.no_grad()
def encode_video_per_frame(model, pixel_values_videos, cfg=None):
    """Encode a video by driving the vision tower per frame (B=1).

    Args:
        pixel_values_videos: ``(batch, frames, C, H, W)`` — e.g. the output of
            ``processor.video_processor(frames, return_tensors="pt").pixel_values_videos``.

    Returns:
        ``(batch, frames * pooled_tokens, D)`` — same shape as the standard
        ``get_video_features`` output (projector + spatial pooling applied), but
        the tower is driven frame-by-frame so the cacher's ``chunk_idx`` advances
        and selective recompute actually fires.
    """
    cfg = cfg if cfg is not None else default_config()
    batch_size, frames = pixel_values_videos.shape[:2]

    layer = model.config.vision_feature_layer
    strategy = model.config.vision_feature_select_strategy

    per_frame = []
    for f in range(frames):
        # Advance the cacher's chunk counter before each frame: chunk 0 does a full
        # reference encode, later frames reuse it selectively (only the changed
        # ``update_token_ratio`` fraction is recomputed).
        reset_default_cache(f, cfg.cache.update_token_ratio)
        frame = pixel_values_videos[:, f]  # (batch, C, H, W)
        out = model.vision_tower(frame, output_hidden_states=True)
        if isinstance(layer, int):
            feat = out.hidden_states[layer]
        else:
            feat = torch.cat([out.hidden_states[i] for i in layer], dim=-1)
        if strategy == "default":
            feat = feat[:, 1:]  # drop the leading (CLS-like) token
        per_frame.append(feat)

    video_features = torch.cat(per_frame, dim=0)  # (batch*frames, N, D)
    video_features = model.multi_modal_projector(video_features)
    video_features = model.apply_pooling(video_features)
    video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)
    return video_features
