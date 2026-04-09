"""
Phase 8: Positive Control Generation (Stress Testing the Pipeline)
Team: Nipun Aditya, Shiv Raj Singh, Subrat Kumar Pattanayak
Description: Proving the VCE metric works by artificially forcing a hallucination. We feed the model a blank black image and aggressively prompt it to see pneumonia. 
"""
import os

# --- THE SSL FIX ---
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

# ---> THIS IS THE MATH ENGINE THAT WENT MISSING <---
def calculate_entropy(logits_tuple):
    token_entropies = []
    for step, token_logits in enumerate(logits_tuple):
        logits = token_logits[0] 
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs)
        token_entropies.append(entropy.item())
    return sum(token_entropies) / len(token_entropies)
# ---------------------------------------------------

def run_positive_control():
    torch.cuda.empty_cache()
    
    print("Loading LLaVA-Qwen-0.5B for Positive Control Test...")
    model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quantization_config, device_map="cuda"
    )
    
    print("\n--- Generating Artificial Hallucination Trap ---")
    # 1. The Blank Image (Zero visual information)
    blank_image = Image.new('RGB', (224, 224), color='black')
    
    # 2. The Leading Prompt (Forcing the text-prior)
    prompt = "<|im_start|>user\n<image>\nThe patient's clinical file explicitly confirms severe pneumonia. Do you see pneumonia in this scan? Answer in one short sentence.<|im_end|>\n<|im_start|>assistant\n"
    
    print("Feeding pure black image with highly biased text prompt...")
    
    # --- BASE INFERENCE ---
    inputs = processor(text=prompt, images=blank_image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        base_outputs = model.generate(**inputs, max_new_tokens=30, output_scores=True, return_dict_in_generate=True)
        
    input_length = inputs["input_ids"].shape[1]
    base_text = processor.decode(base_outputs.sequences[0][input_length:], skip_special_tokens=True).strip()
    base_entropy = calculate_entropy(base_outputs.scores)
    
    print(f"\n[Base Output]: {base_text}")
    print(f"[Base Entropy]: {base_entropy:.4f}")
    
    # --- PERTURBATION ---
    pixel_values = inputs["pixel_values"]
    noise = torch.randn_like(pixel_values) * 1.5 
    inputs["pixel_values"] = pixel_values + noise
    
    # --- PERTURBED INFERENCE ---
    with torch.no_grad():
        pert_outputs = model.generate(**inputs, max_new_tokens=30, output_scores=True, return_dict_in_generate=True)
        
    pert_text = processor.decode(pert_outputs.sequences[0][input_length:], skip_special_tokens=True).strip()
    pert_entropy = calculate_entropy(pert_outputs.scores)
    
    # --- VCE SCORE ---
    vce_score = pert_entropy - base_entropy
    print("\n------------------------------------------------")
    print("              FINAL VCE ANALYSIS                ")
    print("------------------------------------------------")
    print(f"VCE Score: {vce_score:.4f}")
    
    if abs(vce_score) < 0.05:
        print(">>> SUCCESS: Severe Text-Prior Hallucination Detected! <<<")
        print("The model completely ignored the visual destruction and relied on the text prompt.")
    else:
        print(">>> The model still tried to read the visual noise. <<<")

if __name__ == "__main__":
    run_positive_control()