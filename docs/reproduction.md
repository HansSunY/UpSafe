# Reproduction guide

Run commands from the repository root. Use separate training and inference environments.

## 1. Installation

Two separate environments are recommended (the training-side patched modeling
targets transformers ≤ 4.51.3 while the merged upcycled model's modeling file
targets transformers ≥ 4.53 / vLLM):

```bash
# --- training env ---
conda create -n upsafe-train python=3.10 -y && conda activate upsafe-train
pip install -r requirements_train.txt

# --- eval/inference env ---
conda create -n upsafe-infer python=3.10 -y && conda activate upsafe-infer
pip install -r requirements_infer.txt
```

In the **training** env, run the patch installer (installs the vendored
LLaMA-Factory, then patches `peft==0.14.0` and `transformers` *inside that env*
— backups of every replaced file are kept under `setup/backups/`):

```bash
bash setup/init_env.sh
```

> The Moe tuner registers a new `PeftType.MOE` inside peft. `init_env.sh`
> applies those patch files idempotently and verifies the imports at the end.

## 2. Data

- **Training**: STAR-1 (Wang et al., 2025) safety reasoning data — 1,000
  harmful + 915 benign samples. The exact JSON files used by our configs are
  shipped in `LLaMA-Factory/data/` with their dataset registry entries
  (`dataset_info.json`, where each dataset carries its `"safe"` tag used by
  the soft-guardrail loss).
- **Evaluation** (`data/eval/`): JailbreakBench (100), StrongREJECT (313),
  WildChat (370 sampled toxic conversations), WildJailbreak (250 curated
  adversarial prompts), XSTest (250). General-capability benchmarks (MMLU,
  MATH-500, HumanEval) follow the [simple-evals](https://github.com/openai/simple-evals)
  protocol and are run with that framework directly.

## 3. Training pipeline

### 3.0 Base models

Download the instruction-tuned backbone(s) into `./models/`, e.g.

```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./models/Llama-3.1-8B-Instruct
```

### 3.1 (Optional) Safety-critical layer scan

Pre-computed top-3 safety-critical layers used in the paper:

| Model | Layers | | Model | Layers |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 13, 17, 18 | | R1-Distill-Llama-8B | 13, 17, 18 |
| Qwen2.5-7B-Instruct | 15, 17, 18 | | R1-Distill-Qwen-7B | 13, 15, 16 |
| Qwen2.5-14B-Instruct | 23, 24, 26 | | R1-Distill-Qwen-14B | 23, 24, 26 |

Scan your own model with:

```bash
bash scan/run_scan.sh ./models/Llama-3.1-8B-Instruct scan/features_llama8b
```

Note (paper B.1): for DeepSeek-R1-Distill models, scan their *instruction-tuned
counterpart* instead of the distilled model — distillation entangles the
layer-wise safety signal.

### 3.2 Two-stage training

With the training env activated:

```bash
# stage 1 – safety experts (trains experts + routers, masks the general expert)
llamafactory-cli train upcycle/configs/llama3.1-8b/s1.yaml

# stage 2 – soft guardrail (freezes experts, trains routers only)
llamafactory-cli train upcycle/configs/llama3.1-8b/s2.yaml
```

Configs for all six paper models live in `upcycle/configs/<model>/{s1,s2}.yaml`
(R1-Distill models use the CoT variants of STAR-1 with the `deepseek3`
template). Baseline configs (SFT-only LoRA and single-stage MoE) are in
`upcycle/configs/baselines/`; additional 32B-scale models in `upcycle/configs/extra/`.

Launch multi-GPU with `FORCE_TORCHRUN=1 llamafactory-cli train <yaml>` (the
configs assume DeepSpeed ZeRO-3, 8×A100-80GB, per-device batch 4 × grad-accum 4).

### 3.3 Merge into a standalone model

```bash
python upcycle/convert_model.py \
    --adapter_path ./outputs/llama3.1-8b_s2 \
    --output_path ./models/Llama-3.1-8B-Instruct-UpSafe \
    --arch llama            # or qwen
```

This produces a portable HF checkpoint that carries the safety-temperature
routing code (`modeling_llama.py`/`modeling_qwen2.py`) via `auto_map`, and is loaded with `trust_remote_code=True`. vLLM uses its Transformers fallback for this custom architecture; compatibility must be checked in the target GPU environment.

## 4. Inference & the safety temperature

```bash
# In the patched training environment: base model + stage-2 adapter.
# This chat configuration uses the learned router without the inference-time dial.
llamafactory-cli chat upcycle/infer/chat_hf.yaml

# In the inference environment: merged model + the safety-temperature dial.
python eval/QA_gen.py --model_path ./models/Llama-3.1-8B-Instruct-UpSafe \
    --bench jbb --tau 0.5 --output_file results/llama8b_upsafe_jbb.json
```

τ is read from `safe_tempature` in the merged model's `config.json`
(default 0.5 = neutral). Change it:

```python
import json, pathlib
p = pathlib.Path("models/Llama-3.1-8B-Instruct-UpSafe/config.json")
cfg = json.loads(p.read_text())
cfg["safe_tempature"] = 0.8   # more conservative  (0.0 ... 1.0)
p.write_text(json.dumps(cfg, indent=2))
```

## 5. Evaluation

Generation is greedy with vLLM (TP via `VLLM_TP`); the judge is GPT-4o (any
OpenAI-compatible endpoint works). Configure the judge via env vars —
set credentials in your shell:

```bash
export JUDGE_BASE_URL=https://api.openai.com/v1
export JUDGE_API_KEY=sk-****
export JUDGE_MODEL=gpt-4o
```

Full loop (generate → judge) for one model:

```bash
bash eval/run_eval.sh ./models/Llama-3.1-8B-Instruct-UpSafe llama8b_upsafe
```

Or step by step:

```bash
# safety: responses on JBB / StrongREJECT / WildChat / WildJailbreak
python eval/QA_gen.py --model_path ./models/Llama-3.1-8B-Instruct-UpSafe \
    --bench jbb --tau 0.5 --output_file results/llama8b_upsafe_jbb.json
python eval/judge/QA_eval.py --QA_path results/llama8b_upsafe_jbb.json

# over-refusal: XSTest non-refusal rate
python eval/QA_gen.py --model_path ... --bench xs --tau 0.5 --output_file results/..._xs.json
python eval/judge/refuse_judge.py --QA_path results/..._xs.json
```

Safety rate = fraction of responses *not* scored 5 by the judge; XSTest reports
the 1_full_compliance rate. For a safety-temperature sweep run the loop with
`TAUS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"`.

## Hyperparameter notes

Configs follow the settings reported in the paper (Appendix A.1): lr 5e-5,
cosine schedule, warmup 0.1, per-device batch 4×4, stage-1 20 epochs with
`aux_loss_coef 0.01`, stage-2 10 epochs with `sg_loss_coef 0.1`, top-3 layers,
4 experts, top-2 routing. Two known deviations exist in the internal
development configs that produced our released checkpoints: stage-2 was run
for 20 epochs, and the Llama-3.1-8B stage-1/stage-2 data mixes slightly
differ from the appendix description — see [configuration notes](../CONFIG_NOTES.md) if you need to
reproduce the checkpoints bit-exactly.


## Local validation

In an inference environment with the dependencies installed:

```bash
python -m unittest discover -s tests -v
```

In the patched training environment:

```bash
python tests/check_training.py
```

These checks use tiny randomly initialized CPU models and mocked judge calls.
They do not run full training, access external judge APIs, or validate vLLM on CUDA.
