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


@torch.no_grad()
def merge_video_features(model, input_ids, inputs_embeds, video_features):
    """Scatter ``video_features`` into ``inputs_embeds`` at the ``<video>`` token positions.

    Mirrors the video branch of ``LlavaOnevisionModel.forward`` (transformers >= 4.45):
    the pooled per-frame features ``(batch, frames * N, D)`` get **one** trailing
    ``image_newline`` appended (``+1`` token total, not one per frame), then are scattered
    over the ``video_token_index`` slots of the prompt (which the processor has expanded to
    exactly ``frames * N + 1`` tokens).
    """
    newline = model.image_newline[None, None, :].repeat(video_features.shape[0], 1, 1)
    newline = newline.to(video_features.device)
    video_features = torch.cat((video_features, newline), dim=1).flatten(0, 1)
    video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)

    video_mask = (input_ids == model.config.video_token_index).unsqueeze(-1)
    video_mask = video_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
    return inputs_embeds.masked_scatter(video_mask, video_features)


@torch.no_grad()
def autoregressive_generate(model, inputs_embeds, max_new_tokens=128):
    """Greedy prefill + decode loop over ``model.language_model`` (mirrors M0's
    ``_autoregressive_generate``).  LLaVA-OneVision's ``generate(inputs_embeds=...)`` path
    degenerates into repetition, so we drive the language model directly and return only the
    generated token ids (no prompt prefix).
    """
    eos = model.generation_config.eos_token_id or model.config.eos_token_id
    if isinstance(eos, (list, tuple)):
        eos_set = set(int(e) for e in eos)
    elif eos is not None:
        eos_set = {int(eos)}
    else:
        eos_set = set()

    device = inputs_embeds.device
    out = model.language_model(inputs_embeds=inputs_embeds, use_cache=True)
    past_key_values = out.past_key_values
    next_token = out.logits[0, -1].argmax(dim=-1)

    generated = [int(next_token.item())]
    for _ in range(max_new_tokens - 1):
        if generated[-1] in eos_set:
            break
        out = model.language_model(
            input_ids=next_token.reshape(1, 1), use_cache=True, past_key_values=past_key_values
        )
        past_key_values = out.past_key_values
        next_token = out.logits[0, -1].argmax(dim=-1)
        generated.append(int(next_token.item()))

    return torch.tensor([generated], device=device, dtype=torch.long)


@torch.no_grad()
def run_video_qa(model, processor, frames, prompt, max_new_tokens=128, cfg=None):
    """Full end-to-end video QA: ``frames`` + ``prompt`` -> generated answer text.

    ``frames`` is anything the processor's video pipeline accepts (a uint8 array
    ``(N, H, W, 3)``, a list of PIL images, or a video path).  The SigLIP tower is driven
    per frame (B=1) by :func:`encode_video_per_frame`, so STC-Cacher is exercised when
    ``STC_PATCH_VISION=1``; with the flag off this is the pure OneVision video baseline.
    """
    cfg = cfg if cfg is not None else default_config()

    conversation = [
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=text, videos=frames, return_tensors="pt")

    input_ids = inputs["input_ids"].to(model.device)
    pixel_values_videos = inputs["pixel_values_videos"].to(model.device, model.dtype)

    video_features = encode_video_per_frame(model, pixel_values_videos, cfg=cfg)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds = merge_video_features(model, input_ids, inputs_embeds, video_features)

    out_ids = autoregressive_generate(model, inputs_embeds, max_new_tokens=max_new_tokens)
    return processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
