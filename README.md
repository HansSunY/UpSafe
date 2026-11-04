# UpSafe℃: Upcycling for Controllable Safety in Large Language Models (NeurIPS 2026)

Official implementation of **UpSafe℃** (NeurIPS 2026). UpSafe adds safety experts
to selected layers of a language model and uses a two-stage training procedure
to learn when to activate them. A safety temperature **τ** adjusts routing at
inference time without retraining.

<p align="center">
  <img src="assets/framework.png" alt="UpSafe: safety-critical layer scan, two-stage training, and inference-time safety control" width="960">
</p>

## Quick start

Run all commands from the repository root. Training requires CUDA GPUs; the
provided configurations use DeepSpeed ZeRO-3. Download the base checkpoint to
`models/Llama-3.1-8B-Instruct` before training.

```bash
conda create -n upsafe-train python=3.10 -y
conda activate upsafe-train
bash setup/init_env.sh

# Stage 1: train safety experts and routers.
llamafactory-cli train upcycle/configs/llama3.1-8b/s1.yaml

# Stage 2: freeze experts and train routers on harmful and benign examples.
llamafactory-cli train upcycle/configs/llama3.1-8b/s2.yaml

# Export the stage-2 adapter with inference-time safety control.
python upcycle/convert_model.py \
  --adapter_path outputs/llama3.1-8b_s2 \
  --output_path models/Llama-3.1-8B-Instruct-UpSafe --arch llama
```

The installer patches PEFT and Transformers **in the active training environment**
and saves the original files under `setup/backups/`. Use a dedicated environment.

## Evaluation

Use a separate environment for the merged model:

```bash
conda create -n upsafe-infer python=3.10 -y
conda activate upsafe-infer
pip install -r requirements_infer.txt

export JUDGE_API_KEY="your-api-key"
export JUDGE_BASE_URL="https://api.openai.com/v1"
export JUDGE_MODEL="gpt-4o"

bash eval/run_eval.sh models/Llama-3.1-8B-Instruct-UpSafe llama8b_upsafe
```

The default is **τ = 0.5**. Set `TAUS="0.2 0.5 0.8"` to evaluate multiple
settings, or pass `--tau` to `eval/QA_gen.py`. Higher values favor safety experts
and can increase refusal on benign prompts. The checkpoint field
`safe_tempature` retains its original spelling for compatibility.

## Code and data

| Directory | Contents |
|---|---|
| `scan/` | Safety-critical layer probing and ranking |
| `upcycle/` | MoE adapters, model export, and training configurations |
| `eval/` | Response generation and safety/refusal judges |
| `data/eval/` | JBB, StrongREJECT, WildChat, WildJailbreak, and XSTest prompts |
| `LLaMA-Factory/` | Training framework and STAR-1 training data |

Configurations cover Llama-3.1-8B, Qwen2.5-7B/14B, and their R1-Distill variants.
See the [reproduction guide](docs/reproduction.md) for layer scans, model paths,
per-model configurations, and individual evaluation commands.

## Citation

```bibtex
@InProceedings{Sun_2026_NeurIPS,
    author    = {Sun, Yuhao and Yu, Lingyun and Xu, Zhuoer and Yang, Kun and Cui, Shiwen and Zhang, Yongdong and Xie, Hongtao},
    title     = {UpSafe{$^\circ$C}: Upcycling for Controllable Safety in Large Language Models},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year      = {2026}
}
```

## Acknowledgments and license

Built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),
[PEFT](https://github.com/huggingface/peft), and
[Transformers](https://github.com/huggingface/transformers), using
[STAR-1](https://huggingface.co/datasets/safe048/StarVector-Reasoning-1k) data.
Code is licensed under [Apache-2.0](LICENSE). Third-party code, datasets, and
base models retain their respective licenses; see [LEGAL.md](LEGAL.md).
