import json, os
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

BASE = "all-MiniLM-L6-v2"
OUT  = "models/query-encoder-v1"
USE_LORA = True

pairs = [json.loads(l) for l in open("finetune/pairs.jsonl")]
positives = [InputExample(texts=[p["a"], p["b"]]) for p in pairs if p["label"] == 1]

model = SentenceTransformer(BASE)

if USE_LORA:
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"],
                     lora_dropout=0.1, task_type="FEATURE_EXTRACTION")
    model[0].auto_model = get_peft_model(model[0].auto_model, cfg)
    model[0].auto_model.print_trainable_parameters()

loader = DataLoader(positives, shuffle=True, batch_size=16)
loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(loader, loss)], epochs=3, warmup_steps=50, show_progress_bar=True)

os.makedirs(OUT, exist_ok=True)
model.save(OUT)
print(f"Saved to {OUT}")