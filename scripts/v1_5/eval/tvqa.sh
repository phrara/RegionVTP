#!/bin/bash

CKPT="llava-v1.5-7b"
METHOD="${METHOD:-regionvtp}"
TOKEN=${1}
PARAM="n_${TOKEN}"

# CLIP text tower checkpoint (question encoder for query-conditioned relevance).
# Defaults to the local download on this server; override via the environment if needed.
export CLIP_TEXT_MODEL="${CLIP_TEXT_MODEL:-/groups/g900403/home/share/phr/models/openai/clip-vit-large-patch14-336}"

python -W ignore -m llava.eval.model_vqa_loader \
    --model-path /groups/g900403/home/share/phr/models/liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --visual-token-num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl
