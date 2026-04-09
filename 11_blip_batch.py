"""
Phase 11: Dataset-Wide Legacy Audit
Team: Nipun Aditya, Shiv Raj Singh, Subrat Kumar Pattanayak
Description: Running the older BLIP model against the full 624-image medical dataset to prove that legacy models suffer from systemic visual collapse (Zero VCE) compared to modern edge models.
"""
import os
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering

if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

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

def run_blip_audit():
    print("Loading Legacy Model (BLIP-VQA-Base)...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")
    
    base_dir = "chest_xray/test"
    categories = ["NORMAL", "PNEUMONIA"]
    
    # BLIP's simple prompt
    prompt = "Is this chest x-ray normal or does it show signs of pneumonia?"
    
    results = []
    
    # Gather all image paths
    image_paths = []
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        if os.path.exists(cat_dir):
            for filename in os.listdir(cat_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append((category, filename, os.path.join(cat_dir, filename)))

    print(f"Found {len(image_paths)} images for BLIP Audit. Starting pipeline...")
    
    # Wrapped with tqdm for a visual progress bar in the terminal
    for category, filename, filepath in tqdm(image_paths):
        try:
            image = Image.open(filepath).convert('RGB')
            image_resized = image.resize((224, 224))
            
            # --- BASE INFERENCE ---
            inputs = processor(image_resized, prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                base_outputs = model.generate(**inputs, max_new_tokens=20, output_scores=True, return_dict_in_generate=True)
                
            base_text = processor.decode(base_outputs.sequences[0], skip_special_tokens=True).strip()
            base_entropy = calculate_entropy(base_outputs.scores)
            
            # --- PERTURBED INFERENCE ---
            pixel_values = inputs["pixel_values"]
            noise = torch.randn_like(pixel_values) * 1.5 
            inputs["pixel_values"] = pixel_values + noise
            
            with torch.no_grad():
                pert_outputs = model.generate(**inputs, max_new_tokens=20, output_scores=True, return_dict_in_generate=True)
                
            pert_text = processor.decode(pert_outputs.sequences[0], skip_special_tokens=True).strip()
            pert_entropy = calculate_entropy(pert_outputs.scores)
            
            vce_score = pert_entropy - base_entropy
            
            results.append({
                "Filename": filename,
                "Ground_Truth": category,
                "Base_Output": base_text,
                "Perturbed_Output": pert_text,
                "VCE_Score": round(vce_score, 4)
            })
            
            # Prevent memory leaks
            del inputs, base_outputs, pert_outputs, pixel_values, noise
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("blip_vce_audit_results.csv", index=False)
    print("\nAudit Complete! Saved to blip_vce_audit_results.csv")

if __name__ == "__main__":
    run_blip_audit()