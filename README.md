# RegionVTP

**Region-adaptive visual token pruning for LVLMs (query-conditioned, training-free)** — a training-free extension of [AgilePruner](https://github.com/cvsp-lab/AgilePruner) (ICLR 2026).

## 定位

在 AgilePruner 的免训练 attention + diversity 混合剪枝之上，加入两点改进，目标是在相同压缩效率（保留 64 / 128 visual token）下超过 AgilePruner 基线（64 → 95.4%，128 → 97.4%）：

1. **Query-conditioned relevance** —— 用问题文本的 LLM 词嵌入对视觉 token 加权，让保留的 token 对齐「问题在问什么」，缓解幻觉。
2. **Region-adaptive budget** —— 把 576 token 还原成 24×24 空间网格，按区域重要性/复杂度分配局部预算，再在区域内做 attention+diversity 剪枝，强制空间覆盖。

完整设计见 vault 文档：`mllm/vi-token-reduction/RegionVTP-设计.md`。

## 状态

- ✅ **Query-conditioned relevance（M1+M2）**：`encode_images` 增加 `query_embeds` 参数；`prepare_inputs_labels_for_multimodal` 取 `<image>` 之后的问题 token 求均值得到问题向量（去掉了 system prompt 稀释）；`QUERY_LAMBDA` 开关融合。`λ=1`（默认）与原版 AgilePruner bit-exact。
- ✅ **Region-adaptive budget（M3）**：`select_tokens_regionwise(...)` + `allocate_budget(...)` 新函数；`REGION_SIZE` 开启 R×R 区域划分、局部 erank、局部预算（waterfill/softmax），区域内跑 attention+diversity 贪心。

## 计划中的消融开关（env var）

| env var | 含义 | 默认 |
|---|---|---|
| `QUERY_LAMBDA` | λ∈[0,1]，1=纯 CLS 注意力（=原版） | 1.0（关，推荐 0.5） |
| `REGION_SIZE` | R（0=关 region，退化原版） | 0（关，推荐 4） |
| `REGION_GAMMA` | complexity 指数 | 1.0 |
| `BUDGET_MODE` | waterfill / softmax | waterfill |
| `DIST_THRESHOLD` | 静态 tau（沿用 AgilePruner） | None |

硬约束：`λ=1, R=0` 必须 bit-exact 复现 AgilePruner。

## 环境

```bash
conda create -n regionvtp python=3.10 -y
conda activate regionvtp
pip install -e .
# 可选加速
pip install flash-attn --no-build-isolation
```

## 模型与数据

- 模型：`liuhaotian/llava-v1.5-7b`
- 数据：按 [EVAL.md](EVAL.md) 下载

## 评测

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope.sh 64
# 打开开关
QUERY_LAMBDA=0.5 REGION_SIZE=4 CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope.sh 64
```

## 致谢

基于 [AgilePruner](https://github.com/cvsp-lab/AgilePruner)（fork 自其 `main`），其又基于 [LLaVA](https://github.com/haotian-liu/LLaVA) 与 [FasterVLM](https://github.com/Theia-4869/FasterVLM)。许可证沿用 Apache 2.0。
