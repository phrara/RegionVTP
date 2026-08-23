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
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --visual-token-num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment ${CKPT}/${METHOD}/${PARAM}

cd eval_tool

python calculation.py --results_dir answers/${CKPT}/${METHOD}/${PARAM}
