"""End-to-end video QA for LLaVA-OneVision, with opt-in STC-Cacher.

This is the M1 "complete pipeline" entry point: a video (real mp4 / folder of frame
images / synthetic) + a text prompt -> generated answer.  It exercises the whole path
``video -> per-frame SigLIP (cacher-aware) -> projector -> pooling -> LLM prefill+decode``
so you can eyeball the effect on a single local clip.

Run from the repo root (so ``llava`` and ``stc`` are importable)::

    # baseline (pure OneVision, no cacher)
    python scripts/onevision/run_video_qa.py \\
        --model-path /path/to/llava-onevision-qwen2-7b-ov-hf \\
        --video /path/to/clip.mp4 --prompt "What is happening in this video?"

    # +STC-Cacher (frame-to-frame ViT reuse)
    STC_PATCH_VISION=1 STC_UPDATE_TOKEN_RATIO=0.25 STC_CACHE_INTERVAL=4 \\
        python scripts/onevision/run_video_qa.py \\
        --model-path /path/to/llava-onevision-qwen2-7b-ov-hf \\
        --video /path/to/clip.mp4 --prompt "What is happening in this video?"
"""

from __future__ import annotations

import argparse
import glob
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

    print("=" * 66)
    print(f" pipeline | patch_vision={stc_patch_vision_enabled()} "
          f"update_ratio={cfg.cache.update_token_ratio} interval={cfg.cache.cache_interval}")
    print(f" video    | {desc}")
    print(f" prompt   | {args.prompt}")
    print("-" * 66)

    t0 = time.perf_counter()
    answer = run_video_qa(model, processor, frames, args.prompt,
                          max_new_tokens=args.max_new_tokens, cfg=cfg)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    print(f" answer   | {answer}")
    print(f" wall     | {wall:.2f} s  (ViT encode + LLM prefill/decode)")
    print("=" * 66)
    print(" Compare this against the STC_PATCH_VISION=0 run to see the cacher's effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
