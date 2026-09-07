#!/usr/bin/env bash
# LLaVA-OneVision + QADP 入口包装（transformers>=4.45 独立环境）。
#
# 用法（先激活独立环境，再执行本脚本，其余参数原样透传给 run_single_image_qadp.py）：
#   conda activate <onevision-env>
#   LLM_LAYER_PRUNE=1 TCARVE_RANK=64 bash scripts/onevision/run_onevision.sh \
#       --model-path /path/to/llava-onevision-qwen2-7b-ov-hf \
#       --image /path/to/img.jpg --prompt "What is shown in this image?"
#
# 关键环境变量（详见 llava/model/qadp_core.py 的 read_qadp_env）：
#   LLM_LAYER_PRUNE=1   总闸（默认关 = 纯 OneVision 基线）
#   TCARVE_RANK=64      视觉 token 预算（OneVision 必须显式设，默认 294→cap 196 等效不剪）
set -euo pipefail

cd "$(dirname "$0")/../.."   # 回到仓库根，保证 `llava` 可 import

exec python scripts/onevision/run_single_image_qadp.py "$@"
