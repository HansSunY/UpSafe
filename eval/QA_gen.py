"""Generate model responses on the safety / over-refusal benchmarks with vLLM.

Works with any HF checkpoint; for merged upcycled models pass ``--tau`` to set
the safety temperature (paper Sec. 2.3) for this inference run.
Generation is greedy (top_k=1), matching the paper's evaluation protocol.

Benchmarks (see data/eval/):
  jbb        JailbreakBench      (100 misuse behaviors)
  strong     StrongREJECT        (313 policy-violating instructions)
  wildchat   WildChat            (370 sampled toxic conversations)
  wj         WildJailbreak       (250 curated adversarial prompts)
  xs         XSTest              (250 exaggerated-safety prompts)

Usage:
  python eval/QA_gen.py --model_path ./models/Llama-3.1-8B-Instruct-UpSafe \
      --bench jbb --tau 0.5 --output_file results/llama8b_upsafe_jbb.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import fire
import pandas as pd
from vllm import LLM, SamplingParams

# make `router_buffer` (imported by the shipped modeling file) resolvable
sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent.parent / "data" / "eval"

# benchmark -> (filename, prompt column). All files carry {"instruction": ...}
# style records (or Alpaca-format with instruction/input/output).
BENCHES = {
    "jbb": "jbb.json",
    "strong": "strongreject.json",
    "wildchat": "wildchat.json",
    "wj": "wildjailbreak.json",
    "xs": "xstest.json",
}


def get_input_text(tokenizer, prompt, system_prompt=None):
    if system_prompt is None:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def set_safe_tempature(model_path: Path, tau: float):
    if not 0.0 <= tau <= 1.0:
        raise ValueError("--tau must be in [0, 1].")
    return {"safe_tempature": tau}


def load_bench(bench: str):
    if bench not in BENCHES:
        raise ValueError(f"unknown bench {bench!r}, expected one of {list(BENCHES)}")
    with open(DATA_DIR / BENCHES[bench], "r") as f:
        return json.load(f)


def main_wj(model_path: str = "", bench: str = "", output_file: str = "", tau: float = -1.0,
            max_model_len: int = 10240):
    model_path = Path(model_path)
    dataset = load_bench(bench)

    # safety temperature (optional; only meaningful for merged upcycled models)
    overrides = None
    if tau != -1.0:
        overrides = set_safe_tempature(model_path, tau)

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=int(os.environ.get("VLLM_TP", "1")),
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
        hf_overrides=overrides,
    )
    tokenizer = llm.get_tokenizer()

    truncate = 2048 if bench == "wildchat" else None  # long in-the-wild conversations
    prompts = [r["instruction"][:truncate] if truncate else r["instruction"] for r in dataset]
    formatted = [get_input_text(tokenizer, p) for p in prompts]
    prompt_lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in formatted]
    if not prompt_lengths or max(prompt_lengths) >= max_model_len:
        raise ValueError("Prompts must be nonempty and shorter than max_model_len.")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_model_len - max(prompt_lengths))
    outputs = llm.generate(formatted, sampling_params)

    data = [
        dict(instruction=r["instruction"], output=o.outputs[0].text)
        for r, o in zip(dataset, outputs)
    ]

    out = Path(output_file or f"results/{bench}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"saved {len(data)} generations -> {out}")


if __name__ == "__main__":
    fire.Fire(main_wj)
