"""Minimal JSONL VQA loop with exact-match accuracy (stub for M0).

Input: one JSON per line, ``{"image": path, "question": text, "answer": gt}``.

    LLM_LAYER_PRUNE=1 TCARVE_RANK=64 python scripts/onevision/eval_vqa_qadp.py \
        --model-path /path/to/llava-onevision-qwen2-7b-ov-hf --data questions.jsonl

Only intended to confirm "pruning does not crash and stays sensible" on a handful of
examples; the real benchmarks (MME / TextVQA / ...) for OneVision come in a later milestone.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--data", required=True, help="JSONL: {image, question, answer}")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, processor = load_onevision_qadp(args.model_path, device=args.device)
    device = torch.device(args.device)

    with open(args.data, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    correct = 0
    for i, row in enumerate(rows):
        inputs = build_inputs(processor, row["image"], row["question"])
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
        pred = processor.decode(output[0, input_len:], skip_special_tokens=True).strip()

        gt = row["answer"].strip()
        ok = pred.lower() == gt.lower()
        correct += int(ok)
        print(f"[{i + 1}/{len(rows)}] pred={pred!r} gt={gt!r} {'OK' if ok else 'X'}")

    print(f"exact-match: {correct}/{len(rows)} = {correct / len(rows):.2%}")


if __name__ == "__main__":
    main()
