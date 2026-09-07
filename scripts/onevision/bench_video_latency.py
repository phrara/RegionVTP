"""STC-Cacher latency benchmark for LLaVA-OneVision (ViT encode only).

Measures the SigLIP vision-tower encode time on a synthetic (temporally
redundant) video, baseline vs +STC.  This is the M1 acceptance: the **ViT encode**
row should drop ~24.5% with the cacher (STC paper, ReKV Table 1).  Only the
vision tower is timed — the LLM-prefill / STC-Pruner reduction is M2.

Run **two processes** (one config each) and compare the ``ViT encode`` row::

    python scripts/onevision/bench_video_latency.py \\
        --model-path /path/to/llava-onevision-qwen2-7b-ov-hf --num-frames 16

    STC_PATCH_VISION=1 STC_UPDATE_TOKEN_RATIO=0.25 STC_CACHE_INTERVAL=4 \\
        python scripts/onevision/bench_video_latency.py \\
        --model-path /path/to/llava-onevision-qwen2-7b-ov-hf --num-frames 16

Use a real clip (needs ``decord``) with ``--video /path/to/clip.mp4`` instead of
the synthetic frames.  ``--num-frames`` raises the redundancy (and the saving).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# 让 `llava` 包可在不 `pip install -e .` 的情况下被 import（独立环境）。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from llava.onevision.llava_onevision_video import encode_video_per_frame, load_onevision_video
from stc import default_config, stc_patch_vision_enabled


class StageTimer:
    """Accumulate GPU time of a wrapped forward via CUDA events."""

    def __init__(self):
        self.enabled = False
        self._pairs = []

    def reset(self):
        self._pairs = []

    def wrap(self, fn):
        def wrapper(*a, **k):
            if not self.enabled:
                return fn(*a, **k)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*a, **k)
            e.record()
            self._pairs.append((s, e))
            return out

        return wrapper

    def elapsed_ms(self):
        return float(sum(s.elapsed_time(e) for s, e in self._pairs))


def build_synthetic_video(num_frames: int, size: int, seed: int = 0) -> torch.Tensor:
    """Temporally-coherent synthetic clip ``(N, H, W, 3)`` uint8.

    A base frame drifting under small per-frame noise mimics the high temporal
    redundancy of a real stream; latency is content-independent.
    """
    rng = np.random.default_rng(seed)
    frames = np.empty((num_frames, size, size, 3), dtype=np.uint8)
    cur = rng.integers(0, 256, size=(size, size, 3), dtype=np.int16)
    for i in range(num_frames):
        cur = np.clip(cur + rng.integers(-8, 9, size=cur.shape, dtype=np.int16), 0, 255)
        frames[i] = cur.astype(np.uint8)
    return torch.from_numpy(frames)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=384,
                        help="synthetic-frame resolution (processor resizes to the ViT input)")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--video", default=None, help="optional real mp4 (needs decord)")
    parser.add_argument("--sample-fps", type=float, default=0.5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: a GPU is required.", file=sys.stderr)
        return 2
    torch.set_grad_enabled(False)

    model, processor = load_onevision_video(args.model_path, device="cuda")
    cfg = default_config()

    if args.video:
        from decord import VideoReader, cpu

        vr = VideoReader(args.video, ctx=cpu(0))
        step = max(1, int(round(vr.get_avg_fps()) / args.sample_fps))
        idx = list(range(0, len(vr), step))[: args.num_frames]
        frames = torch.from_numpy(vr.get_batch(idx).asnumpy())
    else:
        frames = build_synthetic_video(args.num_frames, args.image_size)

    pixel_values_videos = processor.video_processor(frames, return_tensors="pt").pixel_values_videos
    pixel_values_videos = pixel_values_videos.to("cuda", model.dtype)
    num_frames = pixel_values_videos.shape[1]

    vit_timer = StageTimer()
    model.vision_tower.forward = vit_timer.wrap(model.vision_tower.forward)

    for _ in range(args.warmup):
        encode_video_per_frame(model, pixel_values_videos, cfg=cfg)
    torch.cuda.synchronize()

    vit_ms = []
    torch.cuda.reset_peak_memory_stats(0)
    for _ in range(args.repeats):
        vit_timer.reset()
        vit_timer.enabled = True
        encode_video_per_frame(model, pixel_values_videos, cfg=cfg)
        vit_timer.enabled = False
        torch.cuda.synchronize()
        vit_ms.append(vit_timer.elapsed_ms())

    vit_ms = np.asarray(vit_ms)
    print("=" * 66)
    print(f" STC ViT-encode latency | patch_vision={stc_patch_vision_enabled()} "
          f"update_ratio={cfg.cache.update_token_ratio} interval={cfg.cache.cache_interval}")
    print(f" frames={num_frames} repeats={args.repeats}")
    print("-" * 66)
    print(f" ViT encode : min {vit_ms.min():8.1f}  median {np.median(vit_ms):8.1f}  "
          f"mean {vit_ms.mean():.1f} ± {vit_ms.std():.1f} ms")
    print(f" Peak mem   : {torch.cuda.max_memory_allocated(0) / 1e9:8.2f} GB")
    print("=" * 66)
    print(" Compare this 'ViT encode' row against the STC_PATCH_VISION=0 run;")
    print(" target ~24.5% reduction (STC paper).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
