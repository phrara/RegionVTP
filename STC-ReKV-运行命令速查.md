# STC / ReKV 运行命令速查

> 目标：在远程 GPU（gpu10）上跑通 STC-Cacher，看到 ViT 编码的帧间复用收益（论文 ↓24.5%）。
> 只覆盖 ReKV（LLaVA-OneVision）这一条线。Cacher 的延迟收益用**合成视频**即可测，无需下载真实视频数据。

---

## 0. 两个「跑」的层次

| 层次 | 脚本 | 需要视频数据？ | 看什么 |
|---|---|---|---|
| **延迟 benchmark**（最快验收） | `speed_benchmark/run_rekv.sh` | ❌ 合成视频 | ViT / LLM 延迟下降 |
| 端到端 smoke | `scripts/eval_rekv/eval_rekv_smoke.sh` | ✅ 一段 mp4 | 全链路能跑通 |
| 精度 benchmark | `eval_offline_benchs.sh` / OVO-Bench | ✅ 完整数据集 | 准确率 |

**接入 Cacher 的验收 = 第 1 行**：跑 `run_rekv.sh`，看 ViT encode 那行的 reduction ≈ 24.5%。

---

## 1. 环境搭建

```bash
cd STC-master                          # 你 clone 的位置

conda create -n stc-rekv python=3.10 -y
conda activate stc-rekv

# PyTorch 按 gpu10 的 CUDA 版本调整（下面示例是 cu121）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -e .[hf]                   # 装 stc 包 + torch + transformers

# ⚠️ ReKV 额外依赖：REPRODUCE.md 写的 `pip install -r requirements.txt`，
#    但根目录其实没有 requirements.txt（只有 models/StreamForest/ 有一份）。
#    ReKV 模型代码依赖以下包，手动装：
pip install flash-attn decord logzero numpy tqdm
#    flash-attn 要匹配 CUDA/torch 版本；装不上先跳过，跑起来报 import 错再补。
```

---

## 2. 下载模型

用 0.5B 做 smoke / 延迟快速验证，7B 做正式结果。路径指到 gpu10 的共享模型目录：

```bash
export HF_HOME=/groups/g900403/home/share/phr/models   # 或你自己的缓存路径

# 0.5B（smoke / 延迟快速验证，显存友好）
huggingface-cli download llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
  --local-dir "$HF_HOME/llava-onevision-qwen2-0.5b-ov-hf"

# 7B（默认研究设置）
huggingface-cli download llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --local-dir "$HF_HOME/llava-onevision-qwen2-7b-ov-hf"
```

`--local-dir` 下载到非标准路径，所以要用 `REKV_LLAVA_OV_*_PATH` 显式指过去：

```bash
export REKV_LLAVA_OV_05B_PATH="$HF_HOME/llava-onevision-qwen2-0.5b-ov-hf"
export REKV_LLAVA_OV_7B_PATH="$HF_HOME/llava-onevision-qwen2-7b-ov-hf"
```

> 不设这些 PATH 变量也行——脚本会去 `$HF_HOME/hub/models--llava-hf--.../snapshots` 自动找，但前提是用 `huggingface-cli download` **不带** `--local-dir`（即走标准缓存布局）。用 `--local-dir` 就一定要设 PATH。

---

## 3. 【最快验收】延迟 benchmark（合成视频，无需视频数据）

```bash
cd STC-master
export HF_HOME=/groups/g900403/home/share/phr/models
export REKV_LLAVA_OV_7B_PATH="$HF_HOME/llava-onevision-qwen2-7b-ov-hf"   # 7B；显存不够换 0.5B
export REKV_MODEL=llava_ov_7b                                            # 或 llava_ov_0.5b

GPU=0 bash speed_benchmark/run_rekv.sh
```

脚本默认 `NUM_FRAMES=16 REPEATS=20`，先跑 baseline（`rekv`）再跑 `+STC`（`rekv_stc`），最后打印对比表：

```
 ReKV vs ReKV+STC (min over 20 reps)  |  frames=16
 --------------------------------------------------------------
  stage        baseline      +STC        speedup  reduction
  ViT encode   xxx.x  ->  xxx.x ms    x1.xx   (xx.x% down)
  LLM prefill  xxx.x  ->  xxx.x ms    x1.xx   (xx.x% down)
```

**怎么读**：
- **ViT encode 那行的 reduction = STC-Cacher 的贡献**（论文 ↓24.5%），这就是「接入 Cacher」要验收的数字。
- LLM prefill 那行 = STC-Pruner 的贡献（↓45.3%），本轮暂缓，先不纠结它。

**常用参数**：

```bash
# 只跑某一档
GPU=0 bash speed_benchmark/run_rekv.sh rekv        # baseline
GPU=0 bash speed_benchmark/run_rekv.sh rekv_stc    # +STC

# 帧数 / 重复 / 显存不足用 0.5B
GPU=0 NUM_FRAMES=64 REPEATS=20 bash speed_benchmark/run_rekv.sh
GPU=0 REKV_MODEL=llava_ov_0.5b bash speed_benchmark/run_rekv.sh

# 用真实视频替换合成视频（可选）
GPU=0 VIDEO=/path/to/clip.mp4 bash speed_benchmark/run_rekv.sh
```

> 延迟 benchmark 用的是**合成视频**（`build_synthetic_video`，带帧间相关性的噪声漂移），专门模拟高时序冗余，所以**不用下视频数据**。模型权重不影响延迟结论，快速验证可先用 0.5B。

---

## 4. 端到端 smoke（需要一段本地 mp4）

验证「模型加载 → 视频编码 → 检索 → 生成」全链路能跑通：

1. 编辑 `benchmarks/offline/smoke/smoke_rekv.json`，把第一个样本的 `video_path` 改成你本地一段 mp4 的路径。
2. 跑 baseline 和 +STC：

```bash
export HF_HOME=/groups/g900403/home/share/phr/models
export REKV_LLAVA_OV_05B_PATH="$HF_HOME/llava-onevision-qwen2-0.5b-ov-hf"
export REKV_MODEL=llava_ov_0.5b

bash scripts/eval_rekv/eval_rekv_smoke.sh rekv        # 基线
bash scripts/eval_rekv/eval_rekv_smoke.sh rekv_stc    # +STC
```

预期输出目录：`results/smoke_rekv` / `results/smoke_rekv_stc`。

> ⚠️ `eval_rekv_smoke.sh` 里 `HF_HOME` 的默认值被写死成作者机器的路径（`/apdcephfs_tj5/...`），**必须 export 覆盖**成你自己的，否则找不到模型。

---

## 5. 精度 benchmark（后续，需要真实视频数据）

### 5.1 离线 benchmark（MLVU / EgoSchema / VideoMME）

下载视频 + 标注到 `benchmarks/offline/` 下后：

```bash
CUDA_VISIBLE_DEVICES=0 \
STC_PATCH_VISION=1 \
STC_TOKEN_PER_FRAME=64 \
STC_UPDATE_TOKEN_RATIO=0.25 \
bash scripts/eval_rekv/eval_offline_benchs.sh \
  --dataset mlvu \
  --model llava_ov_7b \
  --num_gpus 1 \
  --num_processes 1 \
  --save_dir results/mlvu_rekv_stc
```

支持的 dataset 名在 `models/rekv/model/video_qa/configs.py`。

### 5.2 OVO-Bench / StreamingBench

需要 `ANNO_PATH` / `VIDEO_DIR` / `CHUNKED_DIR`，多卡跑：

```bash
export ANNO_PATH=/path/to/ovo_bench_new.json
export VIDEO_DIR=/path/to/OVO-Bench/src_videos
export CHUNKED_DIR=/path/to/OVO-Bench/chunked_videos

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 TOTAL_PROCESSES=4 \
STC_PATCH_VISION=1 STC_TOKEN_PER_FRAME=64 STC_UPDATE_TOKEN_RATIO=0.25 \
bash scripts/eval_rekv/ovobench_scripts/eval_rekv.sh
```

---

## 6. env var 速查表

| env var | 默认 | 作用 |
|---|---|---|
| `STC_PATCH_VISION` | 0 | **1=开 STC-Cacher**（0=基线，纯 LLaVA-OneVision） |
| `STC_UPDATE_TOKEN_RATIO` | 1.0 / 0.25 | Cacher 每帧重算「变化最大」token 的比例（0.25=只重算 25%） |
| `STC_CACHE_INTERVAL` | 4 | 每几帧完整刷新一次参考帧 |
| `STC_TOKEN_PER_FRAME` | 196 | Pruner 每帧 token 预算（196=全保留；64=剪枝，本轮暂缓） |
| `REKV_MODEL` | llava_ov_7b | 选模型：`llava_ov_7b` / `llava_ov_0.5b` |
| `REKV_LLAVA_OV_7B_PATH` | — | 7B 模型路径（`--local-dir` 下载时必须设） |
| `REKV_LLAVA_OV_05B_PATH` | — | 0.5B 模型路径 |
| `HF_HOME` | ~/.cache/huggingface | HF 缓存根，**必须覆盖成自己的路径** |
| `GPU` | — | 延迟 benchmark 选卡（`GPU=0`） |
| `NUM_FRAMES` | 16 | 延迟 benchmark 帧数 |
| `REPEATS` | 20 | 延迟 benchmark 重复次数 |

> 「接入 Cacher」相关的只有前三个（`STC_PATCH_VISION` / `STC_UPDATE_TOKEN_RATIO` / `STC_CACHE_INTERVAL`）；`STC_TOKEN_PER_FRAME` 是 Pruner 的，暂缓。

---

## 7. 易错点

1. **`requirements.txt` 不存在**：REPRODUCE.md 里写的 `pip install -r requirements.txt` 在根目录没有（只有 `models/StreamForest/requirements.txt`）。ReKV 依赖（flash-attn / decord / logzero）要手动装。
2. **`HF_HOME` 被写死**：`eval_rekv_smoke.sh` 默认值指向作者机器路径，必须 export 覆盖。
3. **`--local-dir` 下载要配 PATH**：用 `--local-dir` 下载到非标准路径后，一定要设 `REKV_LLAVA_OV_*_PATH`，否则脚本去 HF 缓存 snapshot 里找不到。
4. **flash-attn 装不上**：要匹配 CUDA/torch 版本；先跳过，跑起来报 import 错再针对性补 wheel。`stc` 的注意力有 torch/triton 两个实现，可能有回退路径。
5. **7B 显存**：gpu10 上跑 7B 若 OOM，先用 `REKV_MODEL=llava_ov_0.5b` 走通流程，再想办法上 7B。
6. **延迟只看 reduction 比例**：绝对延迟随 GPU 型号/负载变化，**min 值对应该卡的稳定下限**；论文的 24.5%/45.3% 是比例，复现看比例不看绝对值。

---

## 8. 最小可行路径（建议执行顺序）

```bash
# ① 装环境（1 次）
conda create -n stc-rekv python=3.10 -y && conda activate stc-rekv
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .[hf] && pip install decord logzero numpy tqdm

# ② 下模型（0.5B 先验证，够了再下 7B）
export HF_HOME=/groups/g900403/home/share/phr/models
huggingface-cli download llava-hf/llava-onevision-qwen2-0.5b-ov-hf --local-dir "$HF_HOME/llava-onevision-qwen2-0.5b-ov-hf"
export REKV_LLAVA_OV_05B_PATH="$HF_HOME/llava-onevision-qwen2-0.5b-ov-hf"

# ③ 直接看 Cacher 的 ViT 收益（合成视频，最快）
cd STC-master
GPU=0 REKV_MODEL=llava_ov_0.5b bash speed_benchmark/run_rekv.sh
```
