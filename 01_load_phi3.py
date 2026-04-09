"""
Phase 1: Initial Model Loading Test
Team: Nipun Aditya, Shiv Raj Singh, Subrat Kumar Pattanayak (IISER Bhopal - DSE)
Description: Our first attempt at loading an edge-capable Vision-Language Model. 
References: 
- Model: XTuner LLaVA-Phi-3-Mini (Hugging Face Hub)
- Quantization: bitsandbytes documentation for 4-bit QLoRA
"""
import os

# Subrat found this SSL fix on StackOverflow because our campus/local network was blocking the Hugging Face weight downloads.
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

def load_quantized_model():
    # The CORRECT model ID uploaded by XTuner
    model_id = "xtuner/llava-phi-3-mini-hf"
    
    print("Configuring 4-bit quantization (QLoRA)...")
    # Using NF4 quantization to squeeze the model into my laptop's 6GB VRAM limit
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_id)

    print("Loading LLaVA-Phi-3-Mini model to GPU... This might take a few minutes as it downloads the weights.")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    print("\n--- Success! ---")
    print("Model loaded successfully!")
    
    # Calculating the actual memory footprint to ensure it fits in our hardware constraints
    memory_gb = model.get_memory_footprint() / (1024**3)
    print(f"Model Memory Footprint: {memory_gb:.2f} GB")
    print("----------------")
    
    return processor, model

if __name__ == "__main__":
    processor, model = load_quantized_model()