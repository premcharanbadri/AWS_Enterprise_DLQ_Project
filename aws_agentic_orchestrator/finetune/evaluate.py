import json, numpy as np
from sentence_transformers import SentenceTransformer

THRESHOLD = 0.85
held_out = [json.loads(l) for l in open("finetune/holdout.jsonl")]

def cos(v1, v2):
    return float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def report(path, name):
    m = SentenceTransformer(path)
    pos = [p for p in held_out if p["label"] == 1]
    neg = [p for p in held_out if p["label"] == 0]
    hits = sum(cos(*m.encode([p["a"], p["b"]])) >= THRESHOLD for p in pos)
    false = sum(cos(*m.encode([p["a"], p["b"]])) >= THRESHOLD for p in neg)
    print(f"{name}: hit rate {hits}/{len(pos)} = {hits/len(pos):.1%} | "
          f"false matches {false}/{len(neg)} = {false/len(neg):.1%}")

report("all-MiniLM-L6-v2", "base")
report("models/query-encoder-v1", "fine-tuned")
