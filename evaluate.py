import os, json, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DatasetDict
from trl import DPOConfig, DPOTrainer
from typing import List, Dict
import wandb

# ----------- CONFIG -----------
MODEL_PATH = "./qwen_model"          # path to your trained model (or a checkpoint dir)
TEST_JSON = "data/test_500.json"     # your 500-row evaluation set
PROJECT = os.environ.get("WANDB_PROJECT", "AutoReward")
RUN_NAME = "qwen_eval_500"
BATCH_SIZE = 2
MAX_PROMPT_LEN = 512
MAX_GEN_LEN = 512
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42


# ----------- LOAD MODEL & TOKENIZER -----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.to(DEVICE)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------- LOAD TEST DATA -----------
ds = load_dataset("json", data_files={"test": TEST_JSON})
test = ds["test"]

# Detect schema
features = set(test.features.keys())
is_dpo = {"prompt", "chosen", "rejected"}.issubset(features)

# ----------- W&B INIT -----------
wandb.init(project=PROJECT, name=RUN_NAME, config={
    "model_path": MODEL_PATH,
    "test_file": TEST_JSON,
    "is_dpo_format": is_dpo,
    "batch_size": BATCH_SIZE,
    "max_prompt_length": MAX_PROMPT_LEN,
    "max_gen_length": MAX_GEN_LEN
})


# ----------- PATH A: DPO-FORMAT EVAL -----------
if is_dpo:
    # Minimal DPO trainer just for evaluation
    args = DPOConfig(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=BATCH_SIZE,
        max_prompt_length=MAX_PROMPT_LEN,
        max_completion_length=MAX_GEN_LEN,
        report_to=["wandb"]
    )
    trainer = DPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=None,
        eval_dataset=test,
    )
    metrics = trainer.evaluate(eval_dataset=test)
    # W&B will already have the eval metrics; also log explicitly:
    wandb.log(metrics)
    print("DPO eval metrics:", metrics)

