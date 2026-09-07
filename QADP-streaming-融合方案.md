# 流式方案推进计划：先接入 STC-Cacher

> 更新日期：2026-09-07。
> **本轮范围**：先把 STC-Cacher（帧间选择性重算）接入我们的视频底座，拿到帧间复用的延迟收益 + 精度保持。
> **暂缓**：STC-Pruner 帧内剪枝、以及 QADP 的问题条件化替换（列为后续阶段，不阻塞本轮）。

---

## 当前进度（截至 2026-09-07）

- ✅ **底座已定**：LLaVA-OneVision-7B（SigLIP SO400M，196 token/帧 → Qwen2-7B，28 层），`AI-ModelScope/llava-onevision-qwen2-7b-ov-hf`。
- ✅ **QADP 单图已在 OneVision 上跑通**（M0）——代码在 `temp/RegionVTP/llava/onevision/`：
  - `qadp_core.py`（纯 torch 的 QADP 核心，逐行移植 `token_carve_llm_prune`）
  - `llava_onevision_qadp.py`（`LlavaOnevisionQADP` 包装类，走 `generate` 前置剪枝）
  - 入口 `scripts/onevision/run_single_image_qadp.py` / `eval_vqa_qadp.py`；依赖 `requirements/onevision.txt`
- ✅ **独立环境**：transformers==4.49.0（LLaVA-1.5 的 4.37.2 环境一字不动，4 个 bench 锚点零风险）。
- ⏳ **待做**：视频底座 + 不压缩基线 + STC-Cacher 接入（下一里程碑 M1）。

---

## ⚠️ 硬约束：不能影响已有 benchmark 的复现结果

**任何 STC 接入都不能动已有的对照锚**（尤其 LLaVA-1.5-7B 上的 QADP：MME / TextVQA / POPE / SQA / GQA，这些数字已定型）。具体三条：

1. **按模型适配，不全局覆盖**：STC-Cacher 是 monkey-patch（替换 `layer.forward`），只 patch 视频底座（LLaVA-OneVision 的 **SigLIP** 塔），**绝不** patch LLaVA-1.5 的 **CLIP** 塔，也不改共享的模型加载路径。
2. **显式开关、默认关**：接入用 `STC_PATCH_VISION=1` 这类 env 门控，默认 off；不开开关时，代码路径与原来 **bit-exact 一致**。
3. **物理隔离优先（已落地，机制见下）**：OneVision 这条线全部收在 `temp/RegionVTP/llava/onevision/`（独立子包）+ 独立环境（transformers 4.49），**不 import `llava/model/`**（其 `__init__.py` 在 4.40+ 会因 MPT 已删而崩），LLaVA-1.5 的 `llava/model/*` 与 `llava_arch.py` **一字未改**。

> 好消息：STC 本身已经为「非侵入」做好了基础设施——`stc_patch_vision_enabled()` 读 `STC_PATCH_VISION`（默认 False），`register_stc_cacher` 会保存 `_stc_old_forward`、`unregister_stc_cacher()` 可完整还原。我们只需**确保接入点放在视频模型自己的入口里，别进共享 builder**。

---

## 0. 为什么先做 Cacher、后做 Pruner

| 维度 | STC-Cacher（本轮） | STC-Pruner / QADP 替换（暂缓） |
|---|---|---|
| 性质 | 即插即用，一行挂上 | 需移植 rank-fusion 到 Qwen2，工作量大 |
| 算法风险 | 无（纯借用，已被验证） | 有（问题条件化在流式是否仍成立待验证） |
| 见效 | 快（ViT 编码 ↓24.5%） | 慢（先要过 M0/M1 才能谈） |
| 我们是否已有 | ❌ 完全缺失（帧间复用） | ✅ 已有（静态图已验证的 QADP，单图已落到 OneVision） |

结论：**Cacher 是我们最短的一块板，借来直接补上；Pruner 是我们已有的长板（单图已验证），替换它在流式里的收益要等底座跑通后才能兑现，不急在这一轮。**

---

## 1. 底座选型（已确认）

Cacher 只在「同一视觉塔连续处理多帧」时才有收益——单图模型没有「上一帧」可复用。所以**换视频底座是绕不开的前提**。

| 路线 | 改造量 | 结论 |
|---|---|---|
| 留 LLaVA-1.5-7B + 加多帧包装 | 大（单图模型硬加视频能力） | ❌ |
| **换 LLaVA-OneVision-7B**（SigLIP，196 token/帧） | 中（STC 的 ReKV 参考集成就是它） | ✅ **已选定并跑通单图** |

理由：STC 的 ReKV 集成就是 LLaVA-OneVision + SigLIP，`register_stc_cacher(kind="siglip")` 开箱即用，数据/评测脚本现成。换底座 + 接入 Cacher 可以合并成一条线做。

---

## 2. STC-Cacher 的核心参数（接入后要调的）

| 参数 | 默认 | 含义 | 调参方向 |
|---|---|---|---|
| `update_token_ratio` | 0.25 | 每帧重算「变化最大」的 token 比例 | 越大越准、越慢 |
| `cache_interval` | 代码默认 2 / README 示例 4 | 每几帧完整刷新一次参考帧 | 越大越快、误差累积越多 |
| `share_selection` | 开 | 只用首层算「重算哪些 token」，其余层复用（**近似**） | 精度敏感时关掉 |
| `cuda_graph` | 开 | CUDA graph 回放选择性前向 | 失败自动回退 eager，一般不用管 |
| `selector_metric` | cosine | 判断「变没变」的度量 | cosine/l1/l2/dot |

> ⚠️ 注意一个坑：`cache_interval` 的**代码默认值是 2，README 的示例却是 4**，两处不一致，接入时先确认用哪个，并把它作为消融项之一。

---

## 3. 分阶段计划

### M0 — QADP 单图跑通 ✅（已完成）

- **目标**：把 LLaVA-OneVision 的加载 + QADP 剪枝在**我们的仓库里改通，先单图**。
- **改动**：
  1. 抽出模型无关的 QADP 核心 `llava/onevision/qadp_core.py`（纯 torch，不 import transformers）；
  2. 子类包装 `LlavaOnevisionForConditionalGeneration`，走 `generate` 前置剪枝（镜像 LLaVA-1.5 `llava_llama.py:112-152`），**不 override `forward`**；
  3. 独立环境（transformers==4.49.0）+ 单图入口脚本。
- **验收**：基线 sanity → 透明钩子（`TCARVE_RANK=196` 与不剪逐 token 一致）→ 真实剪枝（`TCARVE_RANK=64` 不崩、log 正确）。
- **关键坑（已记录到 memory）**：`llava/model/__init__.py` 在 4.40+ 崩（MPT 已删）；Qwen2 层直接调要传 `position_embeddings`；**anyres 视觉 token 是动态的**（大幅图多 crop，本测试图 4724 视觉 + 24 newline，不是 196/197）；`generate(inputs_embeds=...)` 会退化重复，改用**手动 prefill+decode 循环**，且 decode 的 `input_ids` 要 2D `(1,1)`（`next_token.reshape(1,1)`，不是 `.unsqueeze(0)`）。

### M1 — 视频底座 + 不压缩基线 + 接入 STC-Cacher（核心交付）

- **目标**：跑通 OVO-Bench / StreamingBench，拿到「不压缩基线」，再把 Cacher 挂上测出帧间复用收益。
- **改动（M1 第一步已落地，代码在 `llava/onevision/`）**：
  1. `llava/onevision/llava_onevision_video.py`：视频包装器 `load_onevision_video()`（门控 `register_stc_cacher(vision_tower, kind="siglip")`，仅 `STC_PATCH_VISION=1` 生效）+ `encode_video_per_frame()`（**逐帧 B=1 驱动 SigLIP 塔**、帧间 `reset_default_cache(chunk_idx)` 推进缓存——标准 `get_video_features` 是整批一次过塔，cacher 看到只有一个 chunk、永远全量重算，所以必须逐帧）；
  2. `scripts/onevision/bench_video_latency.py`：合成视频（时序冗余噪声漂移）的 ViT 编码延迟 benchmark，基线 vs +STC，直接看 ViT 那行 ↓≈24.5%；
  3. 其余（视频数据下载、真实 eval、扫参）接在延迟验收通过之后。
- **入口命令（两条，两进程对比 ViT encode 行）**：
  ```bash
  # 基线
  python scripts/onevision/bench_video_latency.py --model-path $MODEL --num-frames 16
  # +STC（必须显式设 STC_UPDATE_TOKEN_RATIO=0.25 才有收益，默认 1.0=全量重算）
  STC_PATCH_VISION=1 STC_UPDATE_TOKEN_RATIO=0.25 STC_CACHE_INTERVAL=4 \
    python scripts/onevision/bench_video_latency.py --model-path $MODEL --num-frames 16
  ```
- **验收**：
  - 不压缩基线准确率/延迟能复现（OVO 实时 64.4、ViT 编码 103.7 等）；
  - **ViT 编码延迟下降 ≈ 论文（↓24.5%）**——这是 M1 第一步的硬验收，合成视频即可测，不用下数据；
  - 准确率损失可控（论文 −1.9 是 Cacher+Pruner 一起的结果，**单独 Cacher 的损失应更小**，这一步要测出来并记录）。

### M2（暂缓）— QADP 帧内替换 STC-Pruner

问题条件化 rank-fusion 已在单图（M0）落到 Qwen2，但**流式/视频下的有效性待验证**。等 M1 稳定后再排期，独立于本轮。

### M3（暂缓）— 跨帧多样性 + 完整消融

时序锚 + 跨帧抑制 + 完整消融表。**后续阶段。**

---

## 4. 代码级接入点（本轮只涉及 Cacher）

| 动作 | 代码 |
|---|---|
| 挂 Cacher 到 SigLIP 塔 | [hf_vit.py](temp/STC-master/stc/integrations/hf_vit.py) `register_stc_cacher(kind="siglip")` |
| 逐帧推进缓存 / 每视频重置 | `stc/integrations/streaming.py` `enable_streaming_cacher()` / `reset_streaming_cacher()` |
| CUDA graph 回放（选择性帧） | [graph.py](temp/STC-master/stc/cacher/graph.py) `SelectiveCUDAGraphRunner` |
| 选择性重算的前向 | [reference_forward.py](temp/STC-master/stc/cacher/reference_forward.py) |
| 「变没变」的度量 | [selectors.py](temp/STC-master/stc/core/selectors.py) `select_dynamic_token_indices` |

> 本轮**不碰** `stc/pruner/`（STC-Pruner 留到 M2 才替换）。
> **接入位置铁律**：`register_stc_cacher` 只在视频模型的入口调用，且包在 `if stc_patch_vision_enabled():` 里。**任何改到 LLaVA-1.5 + QADP 共享代码（`llava/model/`、`llava_arch.py`）的改动都算越界**——那会动到已定型的 benchmark 复现。
> OneVision 的 QADP 入口现在在 `temp/RegionVTP/llava/onevision/`（独立子包 + 独立环境），Cacher 接入点也应落在这个子包或视频模型的自己的加载路径里。

---

## 5. 风险与坑

1. **数据 / 评测环境成本**：OVO-Bench、StreamingBench 视频下载量大、评测跑起来重，M1 一开始就要准备，别拖到后面。
2. **Cacher 只在多帧视频里有效**：单图场景无「上一帧」，Cacher 没有收益——这正是必须换视频底座的原因。
3. **`share_selection` 是精度换速度的近似**：各层共享同一组「重算 token」，精度敏感时先关掉验证。
4. **`cache_interval` 默认值两处不一致**（代码 2 / README 4），要确认并纳入消融。
5. **单卡显存**：RTX 5070 8GB 本地只够编辑/小验证，实际评测要上远程 GPU（多帧视频的 ViT + LLM prefill 显存不小）。
6. **OneVision 环境隔离**：Qwen2 底座（transformers 4.49）与 LLaVA-1.5（4.37.2）必须分环境；`llava/onevision/` 直接 import，绝不走 `llava/model/__init__.py`。

---

## 6. 里程碑总表

| 步 | 交付 | 验收 |
|---|---|---|
| M0 ✅ | QADP 单图跑通（OneVision-7B，独立 4.49 环境） | 透明钩子逐 token 一致、真实剪枝不崩 |
| M1 | 视频底座 + 不压缩基线 + STC-Cacher 接入 + 消融 | ViT 延迟 ↓24.5% 左右、精度损失可控，记录单独 Cacher 的损失 |
| M2（暂缓） | QADP 帧内替换 STC-Pruner | 同预算准确率损失压到接近 0 |
| M3（暂缓） | 跨帧多样性 + 完整消融 | 论文级 QADP-streaming 结果 |
