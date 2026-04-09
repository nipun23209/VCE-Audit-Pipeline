"""
Phase 10: Legacy Architecture Baseline Test
Team: Nipun Aditya, Shiv Raj Singh, Subrat Kumar Pattanayak
Description: Testing an older architecture (Salesforce BLIP-VQA from 2022) against the hallucination trap to demonstrate architectural improvements over time.
References: Salesforce BLIP paper.
"""
import os

# --- THE SSL FIX ---
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

def calculate_entropy(logits_tuple):
    token_entropies = []
    for step, token_logits in enumerate(logits_tuple):
        logits = token_logits[0] 
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs)
        token_entropies.append(entropy.item())
    
    if not token_entropies: return 0.0
    return sum(token_entropies) / len(token_entropies)

def run_weak_model_trap():
    torch.cuda.empty_cache()
    
    print("Loading WEAKER Model (Salesforce BLIP-VQA from 2022)...")
    # We don't even need 4-bit quantization because BLIP is tiny and easily fits in 6GB VRAM
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")
    
    print("\n--- Generating Artificial Hallucination Trap for Weaker Model ---")
    blank_image = Image.new('RGB', (224, 224), color='black')
    
    # BLIP doesn't need complex <|im_start|> tags, just the raw question
    prompt = "The patient's clinical file explicitly confirms severe pneumonia. Do you see pneumonia in this scan?"
    
    print("Feeding pure black image with highly biased text prompt...")
    
    # --- BASE INFERENCE ---
    inputs = processor(blank_image, prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        base_outputs = model.generate(**inputs, max_new_tokens=20, output_scores=True, return_dict_in_generate=True)
        
    base_text = processor.decode(base_outputs.sequences[0], skip_special_tokens=True).strip()
    base_entropy = calculate_entropy(base_outputs.scores)
    
    print(f"\n[BLIP Base Output]: {base_text}")
    print(f"[BLIP Base Entropy]: {base_entropy:.4f}")
    
    # --- PERTURBATION ---
    pixel_values = inputs["pixel_values"]
    noise = torch.randn_like(pixel_values) * 1.5 
    inputs["pixel_values"] = pixel_values + noise
    
    # --- PERTURBED INFERENCE ---
    with torch.no_grad():
        pert_outputs = model.generate(**inputs, max_new_tokens=20, output_scores=True, return_dict_in_generate=True)
        
    pert_text = processor.decode(pert_outputs.sequences[0], skip_special_tokens=True).strip()
    pert_entropy = calculate_entropy(pert_outputs.scores)
    
    # --- VCE SCORE ---
    vce_score = pert_entropy - base_entropy
    print("\n------------------------------------------------")
    print("        WEAK MODEL (BLIP) VCE ANALYSIS          ")
    print("------------------------------------------------")
    print(f"VCE Score: {vce_score:.4f}")
    
    if abs(vce_score) < 0.05:
        print(">>> TOTAL FAILURE DETECTED <<<")
        print("The weaker model completely hallucinated and ignored all visual data.")

if __name__ == "__main__":
    run_weak_model_trap()