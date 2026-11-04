#!/usr/bin/env bash
# Install and patch packages inside the currently active training environment.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python -m pip install -r "$ROOT/requirements_train.txt"
python -m pip install -e "$ROOT/LLaMA-Factory"
python "$ROOT/setup/patch_environment.py"
python - <<'PY'
from peft import MoeConfig, MoeModel, PeftType, mapping
from transformers.models.llama.modeling_llama import sg_loss_func
from transformers.models.qwen2.modeling_qwen2 import sg_loss_func as qwen_sg_loss
assert PeftType.MOE == "MOE"
assert "MOE" in mapping.PEFT_TYPE_TO_CONFIG_MAPPING
print("UpSafe training patches are ready.")
PY
