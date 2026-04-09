"""
Phase 4: The Vision-Conditioned Entropy (VCE) Pipeline
Team: Nipun Aditya, Shiv Raj Singh, Subrat Kumar Pattanayak
Description: This script calculates the final VCE score by measuring the delta between the Base Entropy (clean image) and the Perturbed Entropy (image with Gaussian noise).
References: PyTorch documentation for tensor perturbation.
"""
import os

# --- THE SSL FIX ---
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

def calculate_entropy(logits_tuple):
    """Helper function to calculate average Shannon Entropy across generated tokens."""
    token_entropies = []
    for step, token_logits in enumerate(logits_tuple):
        logits = token_logits[0] 
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs)
        token_entropies.append(entropy.item())
    return sum(token_entropies) / len(token_entropies)

def run_vce_pipeline():
    torch.cuda.empty_cache()
    
    model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
    print(f"Loading {model_id}...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="cuda"
    )

    print("Loading test image...")
    image = Image.open("test_image.jpg").convert("RGB")
    
    # THE FIX: Speaking Qwen's native chat template language (ChatML format)
    prompt = "<|im_start|>user\n<image>\nDescribe what you see in this image in one sentence.<|im_end|>\n<|im_start|>assistant\n"
    
    # ---------------------------------------------------------
    # 1. BASE INFERENCE (Clean Image)
    # ---------------------------------------------------------
    print("\n--- [1] Base Inference (Clean Image) ---")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        base_outputs = model.generate(**inputs, max_new_tokens=50, output_scores=True, return_dict_in_generate=True)
        
    # Mathematically slicing out ONLY the newly generated text tokens to avoid reading the prompt's entropy
    input_length = inputs["input_ids"].shape[1]
    base_generated_ids = base_outputs.sequences[0][input_length:]
    base_text = processor.decode(base_generated_ids, skip_special_tokens=True).strip()
    
    base_entropy = calculate_entropy(base_outputs.scores)
    
    print(f"Output: {base_text}")
    print(f"Base Entropy: {base_entropy:.4f}")

    # ---------------------------------------------------------
    # 2. ADVERSARIAL PERTURBATION (Noisy Image)
    # ---------------------------------------------------------
    print("\n--- [2] Injecting Gaussian Noise ---")
    pixel_values = inputs["pixel_values"]
    
    # Injecting heavy Gaussian noise (Standard Deviation / Sigma = 1.5) to simulate visual destruction
    sigma = 1.5 
    noise = torch.randn_like(pixel_values) * sigma
    perturbed_pixel_values = pixel_values + noise
    
    inputs["pixel_values"] = perturbed_pixel_values

    print("\n--- [3] Perturbed Inference (Noisy Image) ---")
    with torch.no_grad():
        pert_outputs = model.generate(**inputs, max_new_tokens=50, output_scores=True, return_dict_in_generate=True)
        
    pert_generated_ids = pert_outputs.sequences[0][input_length:]
    pert_text = processor.decode(pert_generated_ids, skip_special_tokens=True).strip()
    
    pert_entropy = calculate_entropy(pert_outputs.scores)
    
    print(f"Output: {pert_text}")
    print(f"Perturbed Entropy: {pert_entropy:.4f}")

    # ---------------------------------------------------------
    # 3. VCE SCORE CALCULATION
    # ---------------------------------------------------------
    print("\n--- [4] Final VCE Analysis ---")
    # The core metric: Delta between noisy state and clean state
    vce_score = pert_entropy - base_entropy
    print(f"VCE Score: {vce_score:.4f}")
    print("------------------------------")

if __name__ == "__main__":
    run_vce_pipeline()