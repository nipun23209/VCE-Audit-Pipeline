# VCE-Audit-Pipeline
An auditing pipeline using Vision-Conditioned Entropy (VCE) to detect text-prior hallucinations and quantify visual grounding in edge Vision-Language Models (VLMs).     (vlm, ai-safety, hallucination-detection, pytorch, medical-ai, qwen, llava)
# Auditing Multimodal Entanglement in Edge VLMs via Vision-Conditioned Entropy (VCE)

**Institution:** Indian Institute of Science Education and Research (IISER) Bhopal  
**Department:** Data Science and Engineering (DSE)  
**Team:** Nipun Aditya, Shiv Raj Singh, Subrat Kumar Pattanayak  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00)

## 📌 Executive Summary
When Vision-Language Models (VLMs) are deployed in clinical environments, they are susceptible to a critical, potentially fatal failure mode: **Text-Prior Hallucination**. This occurs when an aggressive clinical text prompt overpowers the model's vision tower, causing the AI to blindly guess a diagnosis without actually looking at the X-Ray pixels. 

This project introduces **Vision-Conditioned Entropy (VCE)**, a mathematical auditing pipeline designed to act as a "lie-detector" for edge models (like LLaVA-Qwen-0.5B). By measuring the delta in the model's internal Shannon Entropy when subjected to targeted adversarial visual noise, VCE autonomously flags hallucinations and ensures the model is safely grounded in visual reality.

---

## 🧮 The Mathematics of VCE

VCE leverages Information Theory to measure an AI's internal uncertainty. Instead of looking at *what* the AI says, we measure *how* it made its decision.

**Formula:** `VCE = Perturbed Entropy - Base Entropy`

1. **Base Entropy:** The Shannon Entropy of the generated tokens when viewing a clean image.
2. **Perturbed Entropy:** The Shannon Entropy when the image is mathematically destroyed using heavy Gaussian noise ($\sigma=1.5$).

**Diagnostic Framework:**
* **Positive VCE (> 0.0):** True Visual Grounding. The noise confused the AI, proving its vision tower is active and attempting to read the pixels.
* **Zero VCE (== 0.0):** Total Visual Collapse. The AI is completely blind and acting purely as a text chatbot.
* **Negative VCE (< 0.0):** Text-Prior Hijack / Color Bias. The noise caused the AI to panic, abandon visual reasoning, and anchor entirely to the text prompt.

---

## 📂 Repository Architecture

This pipeline is highly modular, designed to run on resource-constrained hardware (6GB VRAM) using 4-bit QLoRA quantization and strict memory management.

### Phase 1: Model Selection & Setup
* `01_load_model.py`: Initial hardware constraint testing (Phi-3).
* `02_baseline_inference.py`: Standard architecture baseline (LLaVA-1.5 7B).
* `03_entropy_calculation.py`: The pivot to the finalized edge model (**LLaVA-Qwen-0.5B**) and the implementation of the raw entropy extraction logic.

### Phase 2: The Core Engine
* `04_vce_calculation.py`: The mathematical VCE pipeline implementation.
* `05_dataset_loader.py`: Custom PyTorch dataloader for the Kaggle Chest X-Ray dataset, featuring custom collate functions for metadata tracking.

### Phase 3: Clinical Batch Auditing
* `06_batch_vce_evaluation.py`: The master script. Runs the VCE audit iteratively across 624 medical test images with aggressive VRAM garbage collection.
* `07_visualize_vce.py`: Generates distribution plots to analyze the AI's visual grounding across the clinical dataset.
* `09_sort_flagged_cases.py`: Autonomously extracts and isolates the most dangerous text-prior hallucinations (lowest VCE scores) for human radiologist review.

### Phase 4: Adversarial Stress Testing
* `08_force_hallucination.py`: The "Positive Control" test. Proves the VCE metric works by intentionally forcing a hallucination on a pure black image.
* `10_weak_model_test.py` & `11_blip_batch_audit.py`: Benchmarks the modern edge model against a legacy architecture (Salesforce BLIP, 2022) to mathematically prove dataset-wide visual collapse in older models.
* `12_cross_domain_audit.py`: A domain-agnostic stress test utilizing a localized dataset of everyday objects to expose color-biased hallucinations and validate the VCE metric outside of healthcare.

---

## 🚀 Installation & Usage

**1. Clone the repository and setup the environment:**
```bash
git clone [https://github.com/nipun23209/VCE-Audit-Pipeline.git](https://github.com/nipun23209/VCE-Audit-Pipeline.git)
cd VCE-Audit-Pipeline
pip install torch torchvision transformers pillow pandas matplotlib seaborn accelerate bitsandbytes tqdm

**2. Download the Dataset:**
Download the Chest X-Ray dataset from Kaggle. Extract the downloaded archive and ensure the dataset folder is named `chest_xray/` and is placed directly in the root directory of this repository.

**3. Run the Pipeline:**
Execute the scripts sequentially. For the primary dataset audit, run the master evaluation script:
```bash
python 06_batch_vce_evaluation.py
