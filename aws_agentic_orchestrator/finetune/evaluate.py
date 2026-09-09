import json, numpy as np
from sentence_transformers import SentenceTransformer

THRESHOLD = 0.85
rows = [json.loads(l) for l in open("finetune/holdout.jsonl")]
pos = [r for r in rows if r["label"] == 1]
neg = [r for r in rows if r["label"] == 0]

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def report(path, name):
    m = SentenceTransformer(path)
    hits  = sum(cos(*m.encode([r["a"], r["b"]])) >= THRESHOLD for r in pos)
    false = sum(cos(*m.encode([r["a"], r["b"]])) >= THRESHOLD for r in neg)
    ps = [cos(*m.encode([r["a"], r["b"]])) for r in pos]
    ns = [cos(*m.encode([r["a"], r["b"]])) for r in neg]
    print(f"{name:12} hit rate {hits}/{len(pos)} = {hits/len(pos):.1%}   "
          f"false matches {false}/{len(neg)} = {false/len(neg):.1%}")
    print(f"{name:12} mean sim  positives {np.mean(ps):.3f}   negatives {np.mean(ns):.3f}   gap {np.mean(ps)-np.mean(ns):.3f}")

report("all-MiniLM-L6-v2", "base")
report("models/query-encoder-v1", "fine-tuned")

def sweep(path, name):
    m = SentenceTransformer(path)
    ps = [cos(*m.encode([r["a"], r["b"]])) for r in pos]
    ns = [cos(*m.encode([r["a"], r["b"]])) for r in neg]
    print(f"\n{name}")
    print(f"{'thresh':>7} {'hit':>7} {'false':>7}")
    for t in [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.85]:
        h = sum(s >= t for s in ps) / len(ps)
        f = sum(s >= t for s in ns) / len(ns)
        print(f"{t:>7.2f} {h:>6.1%} {f:>6.1%}")

sweep("all-MiniLM-L6-v2", "base")
sweep("models/query-encoder-v1", "fine-tuned")