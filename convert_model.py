import os
import json
import argparse
from pathlib import Path
import safetensors.torch
from safetensors.torch import save_file
from transformers import AutoModel


def convert_model(adapter_path: str, output_path: str, arch: str):
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. load adapter config
    with open(adapter_path / "adapter_config.json", "r") as f:
        config = json.load(f)

    print("layers_to_transform: ", config["layers_to_transform"])

    # 2. copy config & tokenizer files (json) and symlink others
    base_model_path = Path(config["base_model_name_or_path"])
    for ele in base_model_path.glob("*"):
        if ele.suffix == ".json":
            os.system(f"cp {ele} {output_path}/")
        else:
            os.system(f"ln -s {ele} {output_path}/")

    # 3. load adapter weights
    adapter = safetensors.torch.load_file(adapter_path / "adapter_model.safetensors", device="cpu")

    adapter = {
        "model.layers" + "layers".join(k.split("layers")[1:])
        .replace("moe_experts.", "moe_experts.default.")
        .replace("moe_router_embedding.", "moe_router_embedding.default."): v
        for k, v in adapter.items()
    }

    save_file(adapter, output_path / "moe_layer.safetensors")
    new_index = {k: "moe_layer.safetensors" for k in adapter.keys()}

    # 4. extract base MLP weights for transformed layers
    base_model = AutoModel.from_pretrained(str(base_model_path), trust_remote_code=True)

    base_layer = {
        "model." + k.replace("mlp.", "mlp.expert0."): v
        for k, v in base_model.state_dict().items()
        if any(str(i) in k for i in config['layers_to_transform']) and "mlp" in k
    }
    save_file(base_layer, output_path / "base_layer.safetensors")
    new_index.update({k: "base_layer.safetensors" for k in base_layer.keys()})

    # 5. update model.safetensors.index.json
    with open(base_model_path / "model.safetensors.index.json", "r") as f:
        ori_index = json.load(f)

    ori_index['weight_map'].update(new_index)

    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(ori_index, f, indent=2)

    # 6. add modeling file
    if arch == "qwen":
        os.system(f"cp ./infer/modeling_qwen2.py {output_path}/")
        arch_name = "Qwen2"
        model_file = "modeling_qwen2"
    elif arch == "llama":
        os.system(f"cp ./infer/modeling_llama.py {output_path}/")
        arch_name = "Llama"
        model_file = "modeling_llama"
    else:
        raise ValueError("arch must be 'qwen' or 'llama'")

    # 7. update config.json
    with open(output_path / "config.json", "r") as f:
        upcycle_config = json.load(f)

    upcycle_config['architectures'] = [f'{model_file}.{arch_name}ForCausalLM']
    upcycle_config['auto_map'] = {
        'AutoConfig': f'{model_file}.{arch_name}MoEConfig',
        'AutoModelForCausalLM': f'{model_file}.{arch_name}ForCausalLM',
        'AutoModel': f'{model_file}.{arch_name}Model'
    }
    upcycle_config['layers_to_transform'] = config['layers_to_transform']
    upcycle_config['num_experts'] = config['num_experts']
    upcycle_config['topk'] = config['topk']
    upcycle_config['safe_tempature'] = 0.5

    with open(output_path / "config.json", "w") as f:
        json.dump(upcycle_config, f, indent=2)

    print(f"✅ Conversion finished! Saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert adapter into upcycled MoE model (Qwen or LLaMA)")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to adapter model folder")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save converted model")
    parser.add_argument("--arch", type=str, choices=["qwen", "llama"], required=True, help="Backbone architecture")
    args = parser.parse_args()

    convert_model(args.adapter_path, args.output_path, args.arch)
