"""
Phase 2: Standard Architecture Baseline
Team: Nipun Aditya, Shiv Raj Singh, Subrat Kumar Pattanayak
Description: Testing the official LLaVA 1.5 7B model. We ultimately had to pivot away from this because the 7B parameter count was too heavy for rapid batch inference on local hardware.
References: Hugging Face Transformers documentation (LLaVA-1.5)
"""
import os

# --- THE SSL FIX WE NEED TO KEEP ---
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
# -----------------------------------

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

def run_baseline():
    # Shifting to the official, highly stable LLaVA 1.5 architecture
    model_id = "llava-hf/llava-1.5-7b-hf"
    
    print(f"Loading official {model_id}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # No bug fixes needed. The official processor config works out-of-the-box.
    processor = AutoProcessor.from_pretrained(model_id)
    
    print("Downloading weights to GPU... (This will take a few minutes for the 7B model)")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    memory_gb = model.get_memory_footprint() / (1024**3)
    print(f"\nModel Memory Footprint: {memory_gb:.2f} GB")

    print("Loading local test image...")
    image_path = "test_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"\n[!] Error: Could not find '{image_path}'.")
        return
        
    image = Image.open(image_path).convert("RGB")

    print("Sending image and prompt to LLaVA...")
    # Using the standard LLaVA 1.5 prompt template as per official docs
    prompt = "USER: <image>\nDescribe what you see in this image in one sentence.\nASSISTANT:"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    print("Generating response...")
    output = model.generate(**inputs, max_new_tokens=50)
    
    print("\n--- Model Output ---")
    response = processor.decode(output[0], skip_special_tokens=True)
    final_text = response.split("ASSISTANT:")[-1].strip()
    
    print(final_text)
    print("--------------------")

if __name__ == "__main__":
    run_baseline()