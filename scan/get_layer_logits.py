"""SS-Score scan, step 1/3: capture per-layer MLP activations for every probe prompt.

For each ``layers.<i>.mlp`` module we store the activation of the **last token**
of the chat-templated prompt (a single forward ``generate(max_new_tokens=1)``),
saved as ``<out_dir>/layers.<i>.mlp.pt`` of shape [n_prompts, hidden].

Usage:
  python scan/get_layer_logits.py --ckpt ./models/Llama-3.1-8B-Instruct \
      --csv scan/merged_dataset.csv --out_dir scan/features_llama-8b
"""
import os
import re
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main(ckpt: str = "", csv: str = "", out_dir: str = "./scan/features"):
    assert ckpt, "--ckpt is required"
    csv = csv or os.path.join(os.path.dirname(os.path.abspath(__file__)), "merged_dataset.csv")

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype="auto", device_map="cuda").cuda()
    model = model.eval()

    os.makedirs(out_dir, exist_ok=True)

    mlp_outputs = defaultdict(list)

    def save_mlp_output(layer_name, module, input, output):
        # last-token activation detached to cpu
        mlp_outputs[layer_name].append(output[0].detach().cpu()[-1, ...].unsqueeze(0))

    # forward hooks on every decoder MLP layer
    for name, module in model.named_modules():
        if re.search(r"layers\.\d+\.mlp$", name) is not None:
            module.register_forward_hook(
                lambda mod, inp, out, layer_name=name: save_mlp_output(layer_name, mod, inp, out)
            )

    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(csv)

    def get_input_text(prompt, tokenizer):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tokenizer([text], return_tensors="pt").to(model.device)

    for record in tqdm(df.to_records(), total=len(df)):
        inputs = get_input_text(record["question"], tokenizer)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1)

    for layer_name, output in mlp_outputs.items():
        output = torch.cat(output, axis=0)
        torch.save(output, os.path.join(out_dir, f"{layer_name}.pt"))
    print(f"saved {len(mlp_outputs)} layer tensors to {out_dir}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
