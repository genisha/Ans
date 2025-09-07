# pip install -U trl transformers datasets accelerate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import inspect
import os

os.environ["WANDB_PROJECT"] = "Evaluate"
os.environ["WANDB_LOG_MODEL"] = "Qwen2.5-0.5b"

MODEL_ID = "./logs/qwen2.5-0.5b/run_20250826_192223_sft_10K/checkpoint-1000000"
#MODEL_ID = "./logs/qwen2.5-0.5b/run_20250829_115320_apo_zero_10K/checkpoint-1000000"
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
policy = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

# 1) Load UltraFeedback-binarized preference split
#ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs").select(range(10))
#ds = load_dataset('json', data_files={
#    'test': 'data/evaluate.json'
#})
ds = load_dataset("json", data_files="data/evaluate.json")["train"]

# 2) Keep only preference keys; drop 'messages', scores, ids, etc.
keep = {"prompt", "chosen", "rejected"}
drop = [c for c in ds.column_names if c not in keep]
eval_ds = ds.remove_columns(drop)

# 3) Tiny dummy train set to satisfy older TRL constructors that prep both splits
dummy_train = eval_ds.select(range(1))

dummy_train = load_dataset("json", data_files="data/evaluate.json")["train"]
eval_ds = load_dataset("json", data_files="data/evaluate_valid.json")["train"]


# 4) Config: no generation during eval; loss-only
"""
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

"""
args = DPOConfig(
    output_dir="dpo-eval-demo",
    do_train=True,                     # training must run to log stepwise eval
    do_eval=True,
    eval_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=10,
    report_to="wandb",           # or "wandb"
    logging_dir="./tb_logs",
    generate_during_eval=False,
    reference_free=True,
)




trainer = DPOTrainer(
    model=policy,
    args=args,
    train_dataset=dummy_train,
    eval_dataset=eval_ds,
    processing_class=tok,
)
trainer = DPOTrainer(model=policy, args=args, train_dataset=dummy_train, eval_dataset=eval_ds, processing_class=tok)
trainer.train()

metrics = trainer.evaluate(metric_key_prefix="dpo")
print({k: metrics[k] for k in metrics if k.startswith("dpo_") or k.startswith("eval_")})
# Read: dpo_eval_loss, dpo_rewards/accuracies, dpo_rewards/margins, dpo_rewards/chosen, dpo_rewards/rejected
# {'dpo_loss': 5.722265720367432, 'dpo_runtime': 17.2569, 'dpo_samples_per_second': 0.579, 'dpo_steps_per_second': 0.29, 'eval_rewards/chosen': -0.003398055676370859, 'eval_rewards/rejected': -0.0041963583789765835, 'eval_rewards/accuracies': 0.5, 'eval_rewards/margins': 0.0007982999086380005, 'eval_logps/chosen': -346.3999938964844, 'eval_logps/rejected': -438.79998779296875, 'eval_logits/chosen': -2.246875047683716, 'eval_logits/rejected': -1.3703124523162842}
