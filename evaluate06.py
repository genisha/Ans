import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch.nn.functional as F
from tqdm import tqdm

class DPOLossEvaluator:
    def __init__(self, model_path, tokenizer_path=None, beta=0.1):
        """
        DPO Loss Evaluator für trainierte Modelle
        
        Args:
            model_path: Pfad zu deinem trainierten Modell
            tokenizer_path: Pfad zum Tokenizer (falls None, wird model_path verwendet)
            beta: DPO Beta Parameter (Standard: 0.1)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        
        # Modell und Tokenizer laden
        print(f"Lade Modell von: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Tokenizer padding token setzen falls nicht vorhanden
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def compute_dpo_loss(self, chosen_texts, rejected_texts, prompts):
        """
        Berechnet DPO Loss für given chosen/rejected pairs
        """
        losses = []
        
        for chosen, rejected, prompt in tqdm(zip(chosen_texts, rejected_texts, prompts), 
                                           desc="Computing DPO Loss"):
            
            # Tokenize inputs
            chosen_input = self.tokenizer(prompt + chosen, 
                                        return_tensors="pt", 
                                        padding=True, 
                                        truncation=True, 
                                        max_length=512)
            
            rejected_input = self.tokenizer(prompt + rejected, 
                                          return_tensors="pt", 
                                          padding=True, 
                                          truncation=True, 
                                          max_length=512)
            
            # Move to device
            chosen_input = {k: v.to(self.device) for k, v in chosen_input.items()}
            rejected_input = {k: v.to(self.device) for k, v in rejected_input.items()}
            
            with torch.no_grad():
                # Forward pass für chosen response
                chosen_outputs = self.model(**chosen_input, labels=chosen_input["input_ids"])
                chosen_logprobs = -chosen_outputs.loss.item()
                
                # Forward pass für rejected response  
                rejected_outputs = self.model(**rejected_input, labels=rejected_input["input_ids"])
                rejected_logprobs = -rejected_outputs.loss.item()
                
                # DPO Loss berechnen
                # DPO Loss = -log(σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x))))
                logit_diff = self.beta * (chosen_logprobs - rejected_logprobs)
                loss = -torch.log(torch.sigmoid(torch.tensor(logit_diff))).item()
                
                losses.append({
                    'dpo_loss': loss,
                    'chosen_logprob': chosen_logprobs,
                    'rejected_logprob': rejected_logprobs,
                    'logit_diff': logit_diff
                })
        
        return losses
    
    def evaluate_on_dataset(self, test_data):
        """
        Evaluiert das Modell auf Test-Dataset
        
        Args:
            test_data: Dictionary, List oder Dataset mit 'prompt', 'chosen', 'rejected' keys
        """
        if isinstance(test_data, dict):
            prompts = test_data['prompt']
            chosen = test_data['chosen'] 
            rejected = test_data['rejected']
        elif isinstance(test_data, list):
            # Annahme: Liste von Dictionaries
            prompts = [item['prompt'] for item in test_data]
            chosen = [item['chosen'] for item in test_data]
            rejected = [item['rejected'] for item in test_data]
        else:
            # Annahme: Dataset object oder ähnlich
            try:
                prompts = test_data['prompt']
                chosen = test_data['chosen']
                rejected = test_data['rejected']
            except (KeyError, TypeError):
                raise ValueError("test_data muss ein Dictionary mit 'prompt', 'chosen', 'rejected' keys sein, "
                               "eine Liste von Dictionaries, oder ein Dataset-Objekt")
        
        print(f"Evaluiere auf {len(prompts)} Beispielen...")
        
        # DPO Loss berechnen
        loss_results = self.compute_dpo_loss(chosen, rejected, prompts)
        
        return loss_results
    
    def plot_loss_analysis(self, loss_results, save_path=None):
        """
        Erstellt verschiedene Plots zur Loss-Analyse
        """
        # Extract values
        dpo_losses = [r['dpo_loss'] for r in loss_results]
        chosen_logprobs = [r['chosen_logprob'] for r in loss_results]
        rejected_logprobs = [r['rejected_logprob'] for r in loss_results]
        logit_diffs = [r['logit_diff'] for r in loss_results]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DPO Model Evaluation Results', fontsize=16)
        
        # Plot 1: DPO Loss Distribution
        axes[0, 0].hist(dpo_losses, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, 0].set_title('DPO Loss Distribution')
        axes[0, 0].set_xlabel('DPO Loss')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(dpo_losses), color='black', linestyle='--', 
                          label=f'Mean: {np.mean(dpo_losses):.4f}')
        axes[0, 0].legend()
        
        # Plot 2: Loss über Zeit/Samples
        axes[0, 1].plot(dpo_losses, alpha=0.7, color='blue')
        axes[0, 1].set_title('DPO Loss over Samples')
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('DPO Loss')
        
        # Plot 3: Chosen vs Rejected Logprobs
        axes[1, 0].scatter(chosen_logprobs, rejected_logprobs, alpha=0.6)
        axes[1, 0].plot([min(chosen_logprobs), max(chosen_logprobs)], 
                       [min(chosen_logprobs), max(chosen_logprobs)], 
                       'r--', label='x=y line')
        axes[1, 0].set_title('Chosen vs Rejected Log Probabilities')
        axes[1, 0].set_xlabel('Chosen Response Log Prob')
        axes[1, 0].set_ylabel('Rejected Response Log Prob')
        axes[1, 0].legend()
        
        # Plot 4: Logit Differences
        axes[1, 1].hist(logit_diffs, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_title('Logit Differences Distribution')
        axes[1, 1].set_xlabel('β * (log π_chosen - log π_rejected)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(logit_diffs), color='black', linestyle='--',
                          label=f'Mean: {np.mean(logit_diffs):.4f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot gespeichert unter: {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Anzahl Samples: {len(dpo_losses)}")
        print(f"Durchschnittlicher DPO Loss: {np.mean(dpo_losses):.4f}")
        print(f"Std DPO Loss: {np.std(dpo_losses):.4f}")
        print(f"Min DPO Loss: {np.min(dpo_losses):.4f}")
        print(f"Max DPO Loss: {np.max(dpo_losses):.4f}")
        print(f"Durchschnittlicher Logit Diff: {np.mean(logit_diffs):.4f}")
        print(f"Positive Logit Diffs (Modell bevorzugt chosen): {np.sum(np.array(logit_diffs) > 0)} / {len(logit_diffs)} ({100*np.sum(np.array(logit_diffs) > 0)/len(logit_diffs):.1f}%)")


# Beispiel für die Verwendung
def main():
    # 1. Pfade zu deinem Modell
    #model_path = "logs/qwen2.5-0.5b/run_20250829_115320_apo_zero_10K/checkpoint-1000000"  # Ersetze mit deinem Modell-Pfad
    model_path = "logs/qwen2.5-0.5b/run_20250901_134929_sft_10K/checkpoint-1000000"
    
    # 2. Test-Dataset erstellen (Beispiel)
    # Ersetze dies mit deinem echten Test-Dataset
    test_data = load_test_data_from_file("data/evaluate.json")
    
    # 3. Evaluator initialisieren
    evaluator = DPOLossEvaluator(model_path, beta=0.1)
    
    # 4. Evaluation durchführen
    loss_results = evaluator.evaluate_on_dataset(test_data)
    
    # 5. Plots erstellen
    evaluator.plot_loss_analysis(loss_results, save_path="dpo_evaluation.png")

# Für größere Datasets von Datei laden
def load_test_data_from_file(file_path):
    """
    Lädt Test-Data von JSON oder CSV Datei
    """
    import json
    import pandas as pd
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return {
            'prompt': df['prompt'].tolist(),
            'chosen': df['chosen'].tolist(), 
            'rejected': df['rejected'].tolist()
        }
    else:
        raise ValueError("Unterstütztes Format: .json oder .csv")

if __name__ == "__main__":
    main()
