"""
Phase 6: The Master VCE Batch Audit
Team: Nipun Aditya, Shiv Raj Singh, Subrat Kumar Pattanayak
Description: The core experimental script. Runs the VCE pipeline iteratively over the 624-image test set. Includes strict memory management protocols to prevent CUDA Out-Of-Memory (OOM) errors on 6GB VRAM hardware.
"""
import os

# --- THE SSL FIX ---
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from torch.utils.data import Dataset, DataLoader
import csv

# --- 1. DATASET & DATALOADER ---
class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, split="test"):
        self.root_dir = os.path.join(root_dir, split)
        self.image_paths = []
        self.labels = []
        self.classes = {"NORMAL": 0, "PNEUMONIA": 1}
        
        for class_name, class_label in self.classes.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir): continue
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpeg', '.jpg')):
                    self.image_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(class_name)

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return image, self.labels[idx], img_path

def custom_collate(batch):
    return [item[0] for item in batch], [item[1] for item in batch], [item[2] for item in batch]

# --- 2. ENTROPY MATH ---
def calculate_entropy(logits_tuple):
    token_entropies = []
    for step, token_logits in enumerate(logits_tuple):
        logits = token_logits[0] 
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs)
        token_entropies.append(entropy.item())
    return sum(token_entropies) / len(token_entropies)

# --- 3. MASTER PIPELINE ---
def run_vce_audit():
    torch.cuda.empty_cache()
    
    print("Loading Dataset...")
    dataset = ChestXRayDataset(root_dir="chest_xray", split="test")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
    
    print("Loading LLaVA-Qwen-0.5B Edge Model...")
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
    
    prompt = "<|im_start|>user\n<image>\nIs this chest X-ray normal or does it show signs of pneumonia? Answer in one short sentence.<|im_end|>\n<|im_start|>assistant\n"
    
    csv_filename = "vce_audit_results.csv"
    print(f"\nStarting VCE Audit. Results will be saved to {csv_filename}...")
    
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Ground_Truth", "Base_Output", "Perturbed_Output", "Base_Entropy", "Perturbed_Entropy", "VCE_Score"])
        
        # Loop through all 624 images in the test set
        for i, (images, labels, paths) in enumerate(dataloader):
            image = images[0]
            ground_truth = labels[0]
            filename = os.path.basename(paths[0])
            
            print(f"\nProcessing [{i+1}/{len(dataloader)}]: {filename} (Truth: {ground_truth})")
            
            # --- BASE INFERENCE ---
            inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
            with torch.no_grad():
                base_outputs = model.generate(**inputs, max_new_tokens=50, output_scores=True, return_dict_in_generate=True)
                
            input_length = inputs["input_ids"].shape[1]
            base_text = processor.decode(base_outputs.sequences[0][input_length:], skip_special_tokens=True).strip()
            base_entropy = calculate_entropy(base_outputs.scores)
            
            # --- PERTURBATION (NOISE) ---
            pixel_values = inputs["pixel_values"]
            noise = torch.randn_like(pixel_values) * 1.5 
            inputs["pixel_values"] = pixel_values + noise
            
            # --- PERTURBED INFERENCE ---
            with torch.no_grad():
                pert_outputs = model.generate(**inputs, max_new_tokens=50, output_scores=True, return_dict_in_generate=True)
                
            pert_text = processor.decode(pert_outputs.sequences[0][input_length:], skip_special_tokens=True).strip()
            pert_entropy = calculate_entropy(pert_outputs.scores)
            
            # --- VCE CALCULATION ---
            vce_score = pert_entropy - base_entropy
            print(f"Base Entropy: {base_entropy:.4f} | Perturbed Entropy: {pert_entropy:.4f} | VCE: {vce_score:.4f}")
            
            # Write row to CSV
            writer.writerow([filename, ground_truth, base_text, pert_text, round(base_entropy, 4), round(pert_entropy, 4), round(vce_score, 4)])
            
            # --- THE MEMORY LEAK FIX ---
            # 1. Force save the CSV line to the SSD instantly in case of a crash
            file.flush() 
            
            # 2. Explicitly destroy the heavy tensors from this iteration
            del inputs, base_outputs, pert_outputs, pixel_values, noise
            
            # 3. Command Windows to empty the GPU trash can before the next loop
            torch.cuda.empty_cache()

    print(f"\nAudit complete! Check {csv_filename} for the logged data.")

if __name__ == "__main__":
    run_vce_audit()