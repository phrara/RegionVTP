# STC-Cacher 融入 temp/RegionVTP 说明

> 更新时间：2026-09-04
> 目标：把 STC 的帧间选择性重算（STC-Cacher）**代码并进本项目**，做成默认关、按 LLaVA-1.5 适配的 opt-in 接入点，不影响已有 benchmark 复现。

## 一句话结论

Cacher 的**代码已并进本项目**（vendored `stc/` 包 + 一个默认关的接入点），但因为 LLaVA-1.5 是**单图模型**，没有「上一帧」可复用，所以本接入点当前是「正确性直通」形态（开启后输出与基线一致、不加速）。真正的帧间加速要等底座换成多帧视频模型（M0/M1）。

---

## 1. 改了什么（3 处，默认全部不生效）

| 文件 | 改动 | 默认状态 |
|---|---|---|
| `stc/`（新增，vendored） | 从 STC-master 复制的 cacher/pruner/core 包 | 只是多了代码，不被 import |
| `llava/model/multimodal_encoder/stc_cacher_adapter.py`（新增） | 门控 + 注意力权重保留的接入适配器 | 不调用 |
| `llava/model/multimodal_encoder/clip_encoder.py`（改 1 处） | `CLIPVisionTower.load_model` 末尾加一行门控调用 | `STC_PATCH_VISION` 未设 → 直接 return |

## 2. 为什么默认关、非侵入

- 接入点用 `STC_PATCH_VISION` 门控，未设时 `maybe_register_stc_cacher` **第一个 if 就 return**，不进 STC 任何代码，LLaVA-1.5 的加载/前向路径与原来 bit-exact 一致。
- 只 patch `CLIPVisionTower`（CLIP 塔），不碰共享 builder / mm_projector / LLM。

## 3. 怎么开启

```bash
STC_PATCH_VISION=1 python -m llava.serve.model_worker ...   # 或你的评测入口
```

| env | 作用 |
|---|---|
| `STC_PATCH_VISION` | 1=开启（默认关） |
| `STC_UPDATE_TOKEN_RATIO` | 每帧重算比例（本接入点当前 strategy=none，暂不生效） |
| `STC_CACHE_INTERVAL` | 参考帧刷新间隔（同上） |

## 4. 两个关键适配（为什么不能直接把 STC 原版拿来用）

1. **注意力权重保留**：STC 原版 `stc_sdpa_attention` 直接丢弃 attention 权重（返回 None），而 QADP 用 `CLS→patch` 注意力做 token 排序（`llava_arch.py` 的 `ranking = image_attentions`）。丢弃 → `feature_select` 里 `None[:, :, 0, 1:]` 崩溃。适配器用 `_llava_attention_preserve` 在 `output_attentions=True` 时算回真实权重。
2. **单图直通（strategy=none）**：Cacher 的 selective 重算假定「连续多帧、同一视频」。LLaVA-1.5 一次只来一张独立图片，若按原样开 selective，会把第 2 张图当成「第 1 张图的下一帧」，复用第 1 张的参考而算错。故适配器把策略固定为全量前向，跨样本安全。

## 5. 重要限制：单图模型没有帧间收益

STC-Cacher 的收益全部来自「相邻帧复用静止 token」。LLaVA-1.5 是单图模型：

- 无「上一帧」→ 永远全量前向 → **无加速**；
- 且 cacher 需要 `output_attentions=False` 的塔（它丢弃权重），与 QADP 的 `output_attentions=True` 路径天然冲突（本适配器用手动注意力绕开，但也因此没有 SDPA 加速）。

所以本接入点的价值是：**cacher 代码在项目里了、接线验证过了、开启不崩、跨样本安全**。要让它真正省 ViT 编码时间，必须换到多帧视频底座。

## 6. 下一步（真正的接入在视频底座）

见 `QADP-streaming-融合方案.md`：

- **M0**：换 LLaVA-OneVision（SigLIP 视频塔，196 token/帧）。
- **M1**：在视频塔上按 STC 官方方式 `register_stc_cacher(kind="siglip")` + `enable_streaming_cacher()` / `reset_streaming_cacher()` 逐帧推进，扫 `update_token_ratio` / `cache_interval`，拿到论文的 ViT ↓24.5%。

到 M1 时，本适配器里的 `_llava_attention_preserve` 可复用（如果将来还要在需要注意力权重的塔上开 cacher）。
