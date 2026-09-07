#!/usr/bin/env python3
"""Convert AoiDragon/POPE raw annotations into the LLaVA eval layout used by
scripts/v1_5/eval/pope.sh + llava/eval/eval_pope.py.

The POPE repo (https://github.com/AoiDragon/POPE) ships three JSON arrays under
`output/coco/`:
    pope_popular.json / pope_adversarial.json / pope_random.json
each entry: {"question_id", "image", "text", "label"} (label in {"yes", "no"},
image is the COCO val2014 filename, e.g. "COCO_val2014_000000000042.jpg").

This produces, under playground/data/eval/pope/:
    llava_pope_test.jsonl      -> model question file (adds instruction suffix + category)
    coco/coco_pope_<cat>.json  -> JSONL label files eval_pope.py reads (image/text/label)
"""
import json
import os

POPE_DIR = "POPE/output/coco"          # path to the cloned POPE repo's coco outputs
OUT_DIR = "playground/data/eval/pope"

CATEGORIES = ["popular", "adversarial", "random"]

os.makedirs(os.path.join(OUT_DIR, "coco"), exist_ok=True)

qid = 0
with open(os.path.join(OUT_DIR, "llava_pope_test.jsonl"), "w") as qf:
    for cat in CATEGORIES:
        src = os.path.join(POPE_DIR, "pope_{}.json".format(cat))
        data = json.load(open(src, encoding="utf-8"))
        with open(os.path.join(OUT_DIR, "coco", "coco_pope_{}.json".format(cat)), "w") as lf:
            for d in data:
                lf.write(json.dumps({"image": d["image"], "text": d["text"], "label": d["label"]}) + "\n")
                qid += 1
                qf.write(json.dumps({
                    "question_id": qid,
                    "image": d["image"],
                    "text": d["text"] + "\nAnswer the question using a single word or phrase.",
                    "category": cat,
                }) + "\n")

print("Wrote {} questions -> {}/llava_pope_test.jsonl".format(qid, OUT_DIR))
print("Wrote label files -> {}/coco/coco_pope_*.json".format(OUT_DIR))
