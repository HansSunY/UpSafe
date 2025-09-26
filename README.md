# UpSafe℃: Upcycling for Controllable Safety in Large Language Models

---

## 1. Environment Setup

### Create Conda Environment
```bash
conda create -n upsafe python==3.10.13 -y
conda activate upsafe
```
### Install Dependencies
```bash
cd UpSafe
bash init_env.sh
```
## 2. Model Preparation
Download the base model from HuggingFace and place it in the current project root
```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./Llama-3.1-8B-Instruction
```
## 3. Training Pipeline
(The safety-critical layers are pre-scanned and defined in the config file.)
### Stage 1: Safety Experts Training
```bash
llamafactory-cli train script/s1_llama8B.yaml
```
### Stage 2: Soft Guardrail Training
```bash
llamafactory-cli train script/s2_llama8B.yaml
```
## 4. Inference
### Option1: HuggingFace Inference
```bash
llamafactory-cli chat script/infer_hf.yaml
```
### Option 2: vLLM Inference
(1) Merge with Base Model
```bash
python convert_upcycle.py \
  --adapter_path ./r1-llama-8b-s2 \
  --output_path ./llama-8b-upcycle \
  --arch llama
```
(2) Run Inference with vLLM
```bash
llamafactory-cli chat script/infer_vllm.yaml
```