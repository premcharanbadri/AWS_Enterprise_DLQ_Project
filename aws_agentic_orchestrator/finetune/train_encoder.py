import json, os
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

BASE = "all-MiniLM-L6-v2"
OUT  = "models/query-encoder-v1"
USE_LORA = True

# read the training split only — never pairs.jsonl, which contains the holdout
rows = [json.loads(l) for l in open("finetune/train.jsonl")]
train = [InputExample(texts=[r["a"], r["b"]], label=float(r["label"])) for r in rows]
print(f"training examples: {len(train)}  "
      f"(pos {sum(r['label']==1 for r in rows)}, neg {sum(r['label']==0 for r in rows)})")

model = SentenceTransformer(BASE)

if USE_LORA:
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=16, lora_alpha=32,
                     target_modules=["query", "key", "value"],
                     lora_dropout=0.05, task_type="FEATURE_EXTRACTION")
    model[0].auto_model = get_peft_model(model[0].auto_model, cfg)
    model[0].auto_model.print_trainable_parameters()

loader = DataLoader(train, shuffle=True, batch_size=16)
loss = losses.CosineSimilarityLoss(model)

model.fit(train_objectives=[(loader, loss)],
          epochs=20,
          warmup_steps=20,
          optimizer_params={'lr': 3e-4},
          show_progress_bar=True)

if USE_LORA:
    model[0].auto_model = model[0].auto_model.merge_and_unload()

os.makedirs(OUT, exist_ok=True)
model.save(OUT)
print(f"Saved to {OUT}")