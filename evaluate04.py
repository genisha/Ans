# pip install -U trl transformers datasets accelerate wandb

import os
import wandb
import gc
import datetime
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import DPOTrainer, DPOConfig

# --- W&B setup ---
os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "Evaluate")
os.environ["WANDB_LOG_MODEL"] = os.environ.get("WANDB_LOG_MODEL", "Qwen2.5-0.5b")

RUN_NAME = "dpo-eval-qwen2.5-0.5b"
wandb_run = wandb.init(project=os.environ["WANDB_PROJECT"], name=RUN_NAME)


# Load dataset
dataset = load_dataset('json', data_files={
    'train': 'data/evaluate_valid.json',
    'valid': 'data/evaluate.json'
})

# Model directory

#model_path = "/home/genisha_admin/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
#model_path = "./logs/qwen2.5-0.5b/run_20250826_192223_sft_10K/checkpoint-1000000" #sft
model_path = "./logs/qwen2.5-0.5b/run_20250829_115320_apo_zero_10K/checkpoint-1000000" #apo_zero

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to("cuda:0")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



# Print model layer names (optional)
print("\n--- Model Layers ---")
for name, _ in model.named_parameters():
    print(name)

# Fine-tune top-N layers only
total_layers = 24
N = 2
trainable_layers = range(total_layers - N, total_layers)

for name, param in model.named_parameters():
    in_last_layers = any(f"model.layers.{i}" in name for i in trainable_layers)
    is_norm = "model.norm" in name
    is_lm_head = "lm_head" in name
    param.requires_grad = in_last_layers or is_norm or is_lm_head
# Define memory-clearing callback
class MemoryClearCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()

# Training configuration

args = DPOConfig(
    output_dir="dpo-eval-demo",
    do_train=True,                     # training must run to log stepwise eval
    do_eval=True,
    eval_strategy="steps",
    eval_steps=1,
    logging_strategy="steps",
    logging_steps=1,
    report_to="wandb",           # or "wandb"
    logging_dir="./tb_logs",
    generate_during_eval=False,
    reference_free=True,
)



# Initialize trainer
trainer = DPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'],
)

# Register memory clearing callback
trainer.add_callback(MemoryClearCallback())

# Train
try:
    trainer.train()
    print('Training completed successfully.')
except Exception as e:
    print('Training failed with exception:')
    print(e)

# Cleanup
gc.collect()
torch.cuda.empty_cache()

# Save info
print(f"Model and checkpoints saved to: {training_args.output_dir}")
print(f"Training completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
