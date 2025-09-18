import torch

# PyTorch version
print("PyTorch version:", torch.__version__)

# CUDA build version
print("CUDA version (built):", torch.version.cuda)

# Check if CUDA is available at runtime
print("Is CUDA available:", torch.cuda.is_available())

# If available, show the current GPU name
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



from transformers import TrainingArguments

# Path to your training output directory
output_dir = "./logs/qwen2.5-0.5b/run_20250901_131518_sft_10K_0_05/checkpoint-1000000"

training_args = torch.load(
    f"{output_dir}/training_args.bin",
    map_location="cpu",
    weights_only=False,  # <-- important
)
print(training_args)
