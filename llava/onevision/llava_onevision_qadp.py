"""LLaVA-OneVision QADP wrapper (opt-in, default = pure OneVision baseline).

This is the ONLY file in the repo that imports ``LlavaOnevisionForConditionalGeneration``
/ ``LlavaOnevisionProcessor`` (transformers >= 4.45), so it must be imported **directly**
by the OneVision scripts — never through ``llava/model/__init__.py`` (which is imported by
the LLaVA-1.5 path on transformers 4.37.2 and would break there).

Design (mirrors the proven LLaVA-1.5 ``generate`` override in
``llava/model/language_model/llava_llama.py:112-152``): instead of overriding ``forward``
(and fighting ``generate``'s KV-cache / attention-mask bookkeeping), we do the image
merge + QADP prune **up front** inside ``generate``, then hand the already-pruned
``inputs_embeds`` to the standard ``super().generate``. The sequence length is shortened
*before* ``generate`` ever sees it, so there is no cache / position / mask mismatch.

Image-token geometry (transformers 4.45+): the merge is inlined in the parent ``forward``
— ``get_image_features`` -> ``pack_image_features`` -> ``masked_scatter`` over the
``<image>`` token positions. ``pack_image_features`` appends **one** ``image_newline``
token at the end of each image's features, so a single image yields ``196`` SigLIP tokens
+ ``1`` newline = ``197`` scattered tokens. QADP prunes only the ``196`` visual tokens and
leaves the newline in the suffix (it is a separator, not a prunable patch).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from llava.onevision.qadp_core import qadp_llm_prune, read_qadp_env


def _qadp_enabled() -> bool:
    return os.environ.get("LLM_LAYER_PRUNE", "0") == "1"


class LlavaOnevisionQADP(LlavaOnevisionForConditionalGeneration):
    """``LlavaOnevisionForConditionalGeneration`` with QADP pruning on the image tokens."""

    def _locate_image_block(self, input_ids, image_features, feature_lens):
        """Return ``(sys_length, image_length)`` for a single-image batch.

        ``sys_length`` is the position of the first ``<image>`` token (== start of the
        image block in ``inputs_embeds`` after the 1:1 ``masked_scatter``). ``image_length``
        is the number of *visual* tokens (``image_features`` already carries one trailing
        ``image_newline`` per image, which we exclude so the separator is never pruned).
        """
        token_index = self.config.image_token_index
        pos = (input_ids[0] == token_index).nonzero().flatten()
        if pos.numel() == 0:
            return 0, 0
        sys_length = int(pos[0].item())
        num_images = int(feature_lens.numel()) if feature_lens is not None else 1
        image_length = int(image_features.shape[0]) - num_images
        return sys_length, image_length

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_sizes_videos: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        do_prune = (
            _qadp_enabled()
            and pixel_values is not None
            and pixel_values_videos is None  # M0: single image only, no video
            and input_ids is not None
            and input_ids.shape[0] == 1
        )

        if do_prune:
            # Replicate the parent forward's image merge to build the FULL inputs_embeds.
            inputs_embeds = self.get_input_embeddings()(input_ids)
            image_features = self.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=self.config.vision_feature_layer,
                vision_feature_select_strategy=self.config.vision_feature_select_strategy,
            )
            image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                image_newline=self.image_newline,
                vision_aspect_ratio=self.config.vision_aspect_ratio,
            )
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            sys_length, image_length = self._locate_image_block(input_ids, image_features, feature_lens)
            if image_length > 0:
                cfg = read_qadp_env(visual_token_num=image_length)
                if position_ids is None:
                    position_ids = torch.arange(
                        inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long
                    ).unsqueeze(0)
                # Qwen2 (transformers >= 4.46) requires the precomputed RoPE (cos, sin)
                # tuple; the layer no longer derives it from position_ids on its own.
                position_embeddings = self.language_model.model.rotary_emb(inputs_embeds, position_ids)
                new_embeds, new_mask, new_pos, _keep_global, _n_kept = qadp_llm_prune(
                    inputs_embeds, attention_mask, position_ids, sys_length, image_length,
                    layers=self.language_model.model.layers, cfg=cfg,
                    position_embeddings=position_embeddings,
                )
                print(
                    f"[QADP] sys_length={sys_length} image_length={image_length} "
                    f"rank={cfg.rank} -> kept={_n_kept} "
                    f"(seq {inputs_embeds.shape[1]} -> {new_embeds.shape[1]})"
                )
                inputs_embeds, attention_mask, position_ids = new_embeds, new_mask, new_pos

            # Pruned (or, if image_length==0, full) embeds -> standard generate, no pixel inputs.
            return super().generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )

        # Baseline: delegate unchanged.
        return super().generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            pixel_values_videos=pixel_values_videos,
            image_sizes_videos=image_sizes_videos,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )


def load_onevision_qadp(model_path: str, device: str = "cuda"):
    """Load the OneVision model + processor. Mirrors STC's ReKV load call (no ``"auto"``).

    Deliberately does NOT set ``attn_implementation="flash_attention_2"`` — flash attention
    cannot return attention weights, which ``qadp_llm_prune`` needs via ``output_attentions``.
    """
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    model = LlavaOnevisionQADP.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": device},
    )
    model.eval()
    return model, processor
