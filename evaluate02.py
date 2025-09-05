# pip install -U trl transformers datasets accelerate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import inspect

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
policy = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

# 1) Load UltraFeedback-binarized preference split
ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs").select(range(10))

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
# Read: dpo_eval_loss, dpo_rewards/accuracies, dpo_rewards/margins, dpo_rewards/chosen, dpo_rewards/rejected
# {'dpo_loss': 5.722265720367432, 'dpo_runtime': 17.2569, 'dpo_samples_per_second': 0.579, 'dpo_steps_per_second': 0.29, 'eval_rewards/chosen': -0.003398055676370859, 'eval_rewards/rejected': -0.0041963583789765835, 'eval_rewards/accuracies': 0.5, 'eval_rewards/margins': 0.0007982999086380005, 'eval_logps/chosen': -346.3999938964844, 'eval_logps/rejected': -438.79998779296875, 'eval_logits/chosen': -2.246875047683716, 'eval_logits/rejected': -1.3703124523162842}
