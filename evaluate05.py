# pip install -U trl transformers datasets accelerate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import inspect

#MODEL_ID = "logs/qwen2.5-0.5b/run_20250901_134929_sft_10K/checkpoint-1000000"
MODEL_ID = "logs/qwen2.5-0.5b/run_20250829_115320_apo_zero_10K/checkpoint-1000000"

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
policy = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

# 1) Load UltraFeedback-binarized preference split
ds = load_dataset("json", data_files="data/evaluate.json")["train"]

# 2) Keep only preference keys; drop 'messages', scores, ids, etc.
keep = {"prompt", "chosen", "rejected"}
drop = [c for c in ds.column_names if c not in keep]
eval_ds = ds.remove_columns(drop)

# 3) Tiny dummy train set to satisfy older TRL constructors that prep both splits
dummy_train = eval_ds.select(range(1))

# 4) Config: no generation during eval; loss-only
args = DPOConfig(
    do_train=False,
    do_eval=True,
    per_device_eval_batch_size=2,
    generate_during_eval=False,   # correct flag in DPOConfig
    max_prompt_length=512,
    max_completion_length=512,
    reference_free=True,          # set False + pass ref_model if you have one
    report_to="none",
)

trainer = DPOTrainer(
    model=policy,
    args=args,
    train_dataset=dummy_train,
    eval_dataset=eval_ds,
    processing_class=tok,
)

metrics = trainer.evaluate(metric_key_prefix="dpo")
print({k: metrics[k] for k in metrics if k.startswith("dpo_") or k.startswith("eval_")})
# 'dpo_loss': 1.1628071608629485e-17, 'dpo_runtime': 9.6334, 'dpo_samples_per_second': 31.142, 'dpo_steps_per_second': 15.571, 'eval_rewards/chosen': -0.014313971623778343, 'eval_rewards/rejected': 0.11224927008152008, 'eval_rewards/accuracies': 0.18333333730697632, 'eval_rewards/margins': -0.12656323611736298, 'eval_logps/chosen': -33.470516204833984, 'eval_logps/rejected': -441.1280822753906, 'eval_logits/chosen': -2.9845526218414307, 'eval_logits/rejected': -3.325526237487793}
