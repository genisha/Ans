import torch
import json
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import os

def generate_model_response(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, do_sample=True):
    """
    Generate a response from the model for a given prompt.
    
    Args:
        model: The DPO model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
    
    Returns:
        str: Generated response
    """
    device = next(model.parameters()).device
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Decode only the generated part (exclude the prompt)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_response.strip()

def calculate_dpo_loss_model_vs_chosen(model, tokenizer, prompt, model_output, chosen, beta=0.1, max_length=512):
    """
    Calculate DPO loss between model output and chosen response.
    
    Args:
        model: The DPO model
        tokenizer: The tokenizer
        prompt: Input prompt
        model_output: Model's generated response
        chosen: Preferred/chosen response
        beta: DPO beta parameter
        max_length: Maximum sequence length
    
    Returns:
        float: DPO loss value (model_output vs chosen)
    """
    device = next(model.parameters()).device
    
    # Tokenize model output and chosen responses
    model_input = tokenizer(
        prompt + model_output, 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    ).to(device)
    
    chosen_input = tokenizer(
        prompt + chosen, 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    ).to(device)
    
    prompt_length = len(tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"])
    
    with torch.no_grad():
        # Get logits for model output
        model_outputs = model(**model_input)
        model_logits = model_outputs.logits
        
        # Get logits for chosen response  
        chosen_outputs = model(**chosen_input)
        chosen_logits = chosen_outputs.logits
        
        # Calculate log probabilities for the response parts only
        model_labels = model_input["input_ids"][:, 1:]  # Shift for next token prediction
        chosen_labels = chosen_input["input_ids"][:, 1:]
        
        model_logprobs = F.log_softmax(model_logits[:, :-1, :], dim=-1)
        chosen_logprobs = F.log_softmax(chosen_logits[:, :-1, :], dim=-1)
        
        # Get log probabilities for actual tokens (response part only)
        model_log_probs = model_logprobs.gather(-1, model_labels.unsqueeze(-1)).squeeze(-1)
        chosen_log_probs = chosen_logprobs.gather(-1, chosen_labels.unsqueeze(-1)).squeeze(-1)
        
        # Sum log probabilities for response tokens only (skip prompt tokens)
        model_response_logprobs = model_log_probs[:, prompt_length-1:].sum()
        chosen_response_logprobs = chosen_log_probs[:, prompt_length-1:].sum()
        
        # Calculate DPO loss (chosen is preferred over model output)
        # Positive loss means chosen is better than model output
        logits_diff = beta * (chosen_response_logprobs - model_response_logprobs)
        dpo_loss = -F.logsigmoid(logits_diff).item()
        
    return dpo_loss

def evaluate_dpo_model_per_sample(
    model_path,
    test_dataset,
    output_file="dpo_losses_per_sample.json",
    beta=0.1,
    max_length=512,
    max_new_tokens=256,
    temperature=0.7,
    use_manual_calculation=True
):
    """
    Evaluate DPO model by comparing model outputs with chosen responses.
    
    Args:
        model_path (str): Path to your trained DPO model
        test_dataset: Your test dataset (should have 'prompt' and 'chosen' fields)
        output_file (str): Path to save the results
        beta (float): DPO beta parameter
        max_length (int): Maximum sequence length for tokenization
        max_new_tokens (int): Maximum new tokens to generate
        temperature (float): Sampling temperature for generation
        use_manual_calculation (bool): Whether to use manual DPO loss calculation
    """
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Results storage
    results = []
    
    print(f"Evaluating {len(test_dataset)} samples...")
    print("Generating model responses and calculating DPO loss vs chosen responses...")
    
    if use_manual_calculation:
        # Use manual DPO loss calculation with model generation
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating samples"):
            sample = test_dataset[idx]
            
            try:
                # Generate model response
                model_response = generate_model_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=sample['prompt'],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True
                )
                
                # Calculate DPO loss between model output and chosen response
                dpo_loss = calculate_dpo_loss_model_vs_chosen(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=sample['prompt'],
                    model_output=model_response,
                    chosen=sample['chosen'],
                    beta=beta,
                    max_length=max_length
                )
                
                sample_result = {
                    'sample_index': idx,
                    'dpo_loss': dpo_loss,
                    'prompt': sample['prompt'],
                    'model_output': model_response,
                    'chosen': sample['chosen'],
                    'rejected': sample.get('rejected', ''),  # Include if available
                }
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                sample_result = {
                    'sample_index': idx,
                    'dpo_loss': None,
                    'prompt': sample['prompt'],
                    'model_output': '',
                    'chosen': sample['chosen'],
                    'rejected': sample.get('rejected', ''),
                    'error': str(e)
                }
            
            results.append(sample_result)
            
            # Optional: Print progress every 5 samples
            if (idx + 1) % 5 == 0:
                valid_losses = [r['dpo_loss'] for r in results if r['dpo_loss'] is not None]
                if valid_losses:
                    avg_loss = np.mean(valid_losses[-5:])  # Average of last 5
                    print(f"Processed {idx + 1}/{len(test_dataset)} samples. Recent avg loss: {avg_loss:.4f}")
                    print(f"Sample {idx} - Prompt: {sample['prompt'][:50]}...")
                    print(f"Model: {results[-1]['model_output'][:100]}...")
                    print(f"Chosen: {sample['chosen'][:100]}...")
                    print(f"DPO Loss: {results[-1]['dpo_loss']:.4f}\n")
    
    else:
        print("Note: Trainer-based evaluation not implemented for model vs chosen comparison.")
        print("Using manual calculation instead...")
        return evaluate_dpo_model_per_sample(
            model_path=model_path,
            test_dataset=test_dataset,
            output_file=output_file,
            beta=beta,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_manual_calculation=True
        )
    
    # Save results to JSON file
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Also save as CSV for easier analysis
    csv_file = output_file.replace('.json', '.csv')
    print(f"Saving results to {csv_file}...")
    
    if results:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    # Print summary statistics
    losses = [r['dpo_loss'] for r in results if r['dpo_loss'] is not None]
    if losses:
        print(f"\nEvaluation Summary:")
        print(f"Total samples: {len(results)}")
        print(f"Valid losses: {len(losses)}")
        print(f"Mean DPO Loss: {np.mean(losses):.4f}")
        print(f"Std DPO Loss: {np.std(losses):.4f}")
        print(f"Min DPO Loss: {np.min(losses):.4f}")
        print(f"Max DPO Loss: {np.max(losses):.4f}")
    
    return results

def load_test_dataset(dataset_path, dataset_format="json"):
    """
    Load your test dataset. Modify this function based on your data format.
    
    Args:
        dataset_path (str): Path to your test dataset
        dataset_format (str): Format of your dataset ("json", "jsonl", "csv", etc.)
    
    Returns:
        Dataset: Loaded dataset in HuggingFace Dataset format
    """
    if dataset_format == "json":
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif dataset_format == "jsonl":
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")
    
    # Convert to HuggingFace Dataset
    # Ensure your data has the required DPO format: prompt, chosen, rejected
    return Dataset.from_list(data)

# Example usage
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "logs/qwen2.5-0.5b/run_20250901_134929_sft_10K/checkpoint-1000000" # Replace with your model path
    #MODEL_PATH = "logs/qwen2.5-0.5b/run_20250911_055537_aot_10K_new/checkpoint-500000"
    TEST_DATASET_PATH = "data/test02.json"  # Replace with your test dataset path
    OUTPUT_FILE = "./dpo_evaluation_sft.json"
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_test_dataset(TEST_DATASET_PATH, dataset_format="json")
    
    # Verify dataset format
    print(f"Dataset size: {len(test_dataset)}")
    print("Sample data keys:", test_dataset[0].keys())
    
    # Run evaluation - model output vs chosen responses
    results = evaluate_dpo_model_per_sample(
        model_path=MODEL_PATH,
        test_dataset=test_dataset,
        output_file=OUTPUT_FILE,
        beta=0.1,  # Adjust this to match your training beta
        max_length=512,  # Max length for tokenization
        max_new_tokens=256,  # Max tokens for model generation
        temperature=0.7,  # Temperature for model generation
        use_manual_calculation=True
    )
    
    print(f"Evaluation complete! Results saved to {OUTPUT_FILE}")
