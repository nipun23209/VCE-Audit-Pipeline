"""
Phase 3: The Edge Model Pivot (Qwen-0.5B)
Team: Nipun Aditya, Shiv Raj Singh, Subrat Kumar Pattanayak
Description: This is the breakthrough script. We pivoted to the LLaVA-Qwen-0.5B model. It is lightweight enough for our hardware but maintains modern vision-language capabilities.
References: Alibaba Qwen team documentation, Hugging Face Hub
"""
import os

# --- THE SSL FIX ---
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

def extract_and_calculate_entropy():
    # Shiv Raj suggested clearing the CUDA cache here to prevent memory fragmentation
    torch.cuda.empty_cache()
    
    # THE PIVOT: An ultra-lightweight, edge-deployable 0.5B LLaVA model
    model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
    
    print(f"Loading {model_id} in 4-bit precision...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    processor = AutoProcessor.from_pretrained(model_id)
    
    print("Downloading weights to GPU... (This will load instantly)")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="cuda" # Forcing directly to GPU to bypass CPU bottlenecks
    )

    print("Loading local test image...")
    image_path = "test_image.jpg"
    image = Image.open(image_path).convert("RGB")

    print("Sending image and prompt to LLaVA...")
    # Updated prompt format tailored specifically for the Qwen-based LLaVA architecture
    prompt = "<image>\nUSER: Describe what you see in this image in one sentence.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    print("Generating response and extracting internal logits...")
    # output_scores=True is critical here; it allows us to extract the raw math for the entropy calculation
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50,
            output_scores=True, 
            return_dict_in_generate=True
        )
    
    # Decode text as usual
    response = processor.decode(outputs.sequences[0], skip_special_tokens=True)
    final_text = response.split("ASSISTANT:")[-1].strip()
    
    print(f"\n--- Model Output ---")
    print(final_text)
    
    # --- ENTROPY CALCULATION PIPELINE ---
    # Mathematical foundation based on Information Theory (Shannon Entropy)
    print("\n--- Internal Uncertainty (Entropy) ---")
    
    token_entropies = []
    
    # outputs.scores is a tuple containing a tensor of logits for each generated word/token
    for step, token_logits in enumerate(outputs.scores):
        logits = token_logits[0] 
        
        # 1. Convert raw logits to a probability distribution (0 to 1) using Softmax
        probs = F.softmax(logits, dim=-1)
        
        # 2. Calculate Shannon Entropy: -sum(P * log(P))
        # We add 1e-10 to prevent a math error (log of 0 is undefined)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs)
        
        # Store the entropy value for this specific token
        token_entropies.append(entropy.item())
        
    # Calculate the average entropy across the entire generated sentence
    avg_entropy = sum(token_entropies) / len(token_entropies)
    
    print(f"Average Base Entropy: {avg_entropy:.4f}")
    print("--------------------------------------")

if __name__ == "__main__":
    extract_and_calculate_entropy()