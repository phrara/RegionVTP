"""Minimal single-image prompt -> answer for LLaVA-OneVision + QADP.

Run from the repo root (so ``llava`` is importable), in the transformers >= 4.45 env::

    LLM_LAYER_PRUNE=1 TCARVE_RANK=64 python scripts/onevision/run_single_image_qadp.py \
        --model-path /path/to/llava-onevision-qwen2-7b-ov-hf \
        --image /path/to/img.jpg --prompt "What is shown in this image?"

Pruning is gated by ``LLM_LAYER_PRUNE=1`` (default off -> pure OneVision baseline).
Token budget is ``TCARVE_RANK`` (must be set explicitly to actually prune; the default
``1.5 * 196 = 294 -> capped to 196`` is a no-op). See ``qadp_core.read_qadp_env`` for the
full env-var surface (TCARVE_* / QADP_*).
"""

from __future__ import annotations

import argparse
import os
import sys

# 让 `llava` 包可在不 `pip install -e .` 的情况下被 import（独立环境）。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from PIL import Image

from llava.onevision.llava_onevision_qadp import load_onevision_qadp


def build_inputs(processor, image_path: str, prompt: str):
    image = Image.open(image_path).convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return inputs


def move_inputs(inputs, device: torch.device, dtype: torch.dtype):
    for k, v in list(inputs.items()):
        if not isinstance(v, torch.Tensor):
            continue
        if v.dtype in (torch.float32, torch.float16, torch.bfloat16):
            inputs[k] = v.to(device=device, dtype=dtype)
        else:
            inputs[k] = v.to(device=device)
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, processor = load_onevision_qadp(args.model_path, device=args.device)
    device = torch.device(args.device)

    inputs = build_inputs(processor, args.image, args.prompt)
    inputs = move_inputs(inputs, device, model.dtype)

    input_len = inputs["input_ids"].shape[1]
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs["pixel_values"],
        "attention_mask": inputs.get("attention_mask"),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    if "image_sizes" in inputs:
        gen_kwargs["image_sizes"] = inputs["image_sizes"]

    with torch.no_grad():
        output = model.generate(**gen_kwargs)

    # 剪枝路径下序列被缩短，真实 prompt 长度由 wrapper 记录在 _qadp_input_len；
    # 基线路径无此属性，回退到原始 input_len。
    prompt_len = getattr(model, "_qadp_input_len", input_len)
    answer = processor.decode(
        output[0, prompt_len:], skip_special_tokens=True
    ).strip()

    print("=" * 60)
    print(f"Q: {args.prompt}")
    print(f"A: {answer}")
    print("=" * 60)


if __name__ == "__main__":
    main()
