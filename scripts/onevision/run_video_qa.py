"""End-to-end video QA for LLaVA-OneVision, with opt-in STC-Cacher + QADP.

This is the fused streaming-pipeline entry point: a video (real mp4 / folder of frame
images / synthetic) + a text prompt -> generated answer.  It exercises the whole path
``video -> per-frame SigLIP (cacher-aware) -> projector -> pooling -> merge -> QADP prune
(opt-in) -> LLM prefill+decode``.  Two independent knobs, both default off:

* ``STC_PATCH_VISION=1`` — frame-to-frame ViT reuse (STC-Cacher).
* ``LLM_LAYER_PRUNE=1`` — intra-frame token pruning (QADP); budget via ``TCARVE_RANK`` /
  ``TCARVE_MERGE`` (set explicitly, defaults are a near no-op).

Run from the repo root (so ``llava`` and ``stc`` are importable)::

    # baseline (pure OneVision)
    python scripts/onevision/run_video_qa.py \\
        --model-path /path/to/llava-onevision-qwen2-7b-ov-hf \\
        --video /path/to/clip.mp4 --prompt "What is happening in this video?"

    # +STC-Cacher (frame-to-frame ViT reuse)
    STC_PATCH_VISION=1 STC_UPDATE_TOKEN_RATIO=0.25 STC_CACHE_INTERVAL=4 \\
        python scripts/onevision/run_video_qa.py \\
        --model-path /path/to/llava-onevision-qwen2-7b-ov-hf \\
        --video /path/to/clip.mp4 --prompt "What is happening in this video?"

    # fused: Cacher + QADP (frame-to-frame reuse AND intra-frame prune)
    STC_PATCH_VISION=1 STC_UPDATE_TOKEN_RATIO=0.25 STC_CACHE_INTERVAL=4 \\
        LLM_LAYER_PRUNE=1 TCARVE_RANK=64 \\
        python scripts/onevision/run_video_qa.py \\
        --model-path /path/to/llava-onevision-qwen2-7b-ov-hf \\
        --video /path/to/clip.mp4 --prompt "What is happening in this video?"
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from llava.onevision.llava_onevision_video import load_onevision_video, run_video_qa
from stc import default_config, stc_patch_vision_enabled


def build_synthetic_video(num_frames: int, size: int, seed: int = 0) -> np.ndarray:
    """Temporally-coherent synthetic clip ``(N, H, W, 3)`` uint8 (content-independent)."""
    rng = np.random.default_rng(seed)
    frames = np.empty((num_frames, size, size, 3), dtype=np.uint8)
    cur = rng.integers(0, 256, size=(size, size, 3), dtype=np.int16)
    for i in range(num_frames):
        cur = np.clip(cur + rng.integers(-8, 9, size=cur.shape, dtype=np.int16), 0, 255)
        frames[i] = cur.astype(np.uint8)
    return frames


def load_frames(args) -> tuple[np.ndarray, str]:
    """Return ``(frames uint8 (N,H,W,3), source_desc)`` from --video / --image-dir / synthetic."""
    if args.video:
        from decord import VideoReader, cpu

        vr = VideoReader(args.video, ctx=cpu(0))
        step = max(1, int(round(vr.get_avg_fps()) / args.sample_fps))
        idx = list(range(0, len(vr), step))[: args.num_frames]
        return vr.get_batch(idx).asnumpy(), f"{args.video} ({len(idx)} frames)"
    if args.image_dir:
        paths = sorted(glob.glob(os.path.join(args.image_dir, "*.*")))[: args.num_frames]
        from PIL import Image

        imgs = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
        return np.stack(imgs, axis=0), f"{args.image_dir} ({len(imgs)} frames)"
    return build_synthetic_video(args.num_frames, args.image_size), f"synthetic ({args.num_frames} frames)"


def main() -> int:
    logging.basicConfig(level=logging.INFO)  # 让 STC-Cacher 的 CUDA-graph 捕获日志可见
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video", default=None, help="real mp4 (needs decord)")
    parser.add_argument("--image-dir", default=None, help="folder of frame images (PIL)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=384, help="synthetic-frame resolution")
    parser.add_argument("--sample-fps", type=float, default=0.5, help="mp4 frame subsampling")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: a GPU is required.", file=sys.stderr)
        return 2
    torch.set_grad_enabled(False)

    model, processor = load_onevision_video(args.model_path, device="cuda")
    cfg = default_config()
    frames, desc = load_frames(args)

    qadp_on = os.environ.get("LLM_LAYER_PRUNE", "0") == "1"
    print("=" * 66)
    print(f" pipeline | cacher={stc_patch_vision_enabled()} "
          f"(update_ratio={cfg.cache.update_token_ratio} interval={cfg.cache.cache_interval}) "
          f"qadp={qadp_on}")
    print(f" video    | {desc}")
    print(f" prompt   | {args.prompt}")
    print("-" * 66)

    t0 = time.perf_counter()
    timing: dict = {}
    answer = run_video_qa(model, processor, frames, args.prompt,
                          max_new_tokens=args.max_new_tokens, cfg=cfg, timing=timing)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    print(f" answer   | {answer}")
    print("-" * 66)
    n_gen = timing.get("n_generated", args.max_new_tokens)
    print(f" ViT      | {timing.get('vit_ms', float('nan')):8.1f} ms  (per-frame SigLIP, cacher's effect)")
    if qadp_on:
        print(f" QADP     | {timing.get('qadp_ms', float('nan')):8.1f} ms  (partial forward + per-frame select/merge)")
    print(f" prefill  | {timing.get('prefill_ms', float('nan')):8.1f} ms  (LLM, QADP shortens the visual seq)")
    print(f" decode   | {timing.get('decode_ms', float('nan')):8.1f} ms  ({n_gen} tok, output-length-dependent)")
    print(f" wall     | {wall:8.2f} s")
    print("=" * 66)
    print(" Compare against STC_PATCH_VISION=0 / LLM_LAYER_PRUNE=0 runs for the cacher / QADP effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
