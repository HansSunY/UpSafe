"""Export a stage-2 adapter as a portable checkpoint with safety-temperature routing."""
import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def convert_model(adapter_path, output_path, arch, safe_tempature=0.5):
    if arch not in ('llama', 'qwen'):
        raise ValueError("arch must be 'llama' or 'qwen'")
    if not 0 <= safe_tempature <= 1:
        raise ValueError('Safety temperature must be in [0, 1].')
    adapter_path = Path(adapter_path).resolve()
    output_path = Path(output_path).resolve()
    config = json.loads((adapter_path / 'adapter_config.json').read_text())
    if config.get('save_all_params', False):
        raise ValueError('This exporter expects frozen-base MoE adapters, not full-model tuning.')
    base_path = Path(config['base_model_name_or_path']).resolve()
    if not base_path.is_dir():
        raise FileNotFoundError(f'Download the base checkpoint locally first: {base_path}')
    if output_path == base_path or output_path == adapter_path:
        raise ValueError('Output must differ from the base and adapter directories.')
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError('Output directory must be empty to avoid mixing checkpoints.')
    layers = config['layers_to_transform']
    layers = [layers] if isinstance(layers, int) else list(layers)
    layers = set(map(int, layers))
    if not layers:
        raise ValueError('No upcycled layers were specified.')
    index_path = base_path / 'model.safetensors.index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        with safe_open(base_path / 'model.safetensors', framework='pt', device='cpu') as shard:
            index = {'metadata': {}, 'weight_map': {key: 'model.safetensors' for key in shard.keys()}}
    original_map = index['weight_map']
    original_keys = {
        key for key in original_map
        if (match := re.match(r'model\.layers\.(\d+)\.mlp\.', key)) and int(match[1]) in layers
    }
    if {int(key.split('.')[2]) for key in original_keys} != layers:
        raise ValueError('Some selected layers have no MLP weights in the base checkpoint.')
    adapter = {}
    for key, tensor in load_file(adapter_path / 'adapter_model.safetensors', device='cpu').items():
        # Older snapshots included frozen backbone tensors in the adapter file.
        if 'moe_experts.' not in key and 'moe_router_embedding.' not in key:
            continue
        match = re.search(r'(?:^|\.)layers\.(\d+)\.(.*)', key)
        if not match or int(match[1]) not in layers:
            raise ValueError(f'Unexpected adapter weight: {key}')
        suffix = match[2]
        for component in ('moe_experts', 'moe_router_embedding'):
            suffix = re.sub(rf'{component}\.(?!default\.)', component + '.default.', suffix)
        if 'moe_experts.' not in suffix and 'moe_router_embedding.' not in suffix:
            raise ValueError('Full-model adapters are not supported by this exporter.')
        adapter[f'model.layers.{match[1]}.{suffix}'] = tensor
    if not adapter:
        raise ValueError('The adapter contains no weights.')
    base_layers = {}
    for shard_file in sorted({original_map[key] for key in original_keys}):
        with safe_open(base_path / shard_file, framework='pt', device='cpu') as shard:
            for key in sorted(original_keys):
                if original_map[key] == shard_file:
                    base_layers[key.replace('.mlp.', '.mlp.expert0.', 1)] = shard.get_tensor(key)
    output_path.mkdir(parents=True, exist_ok=True)
    # Copy files so the exported model remains loadable after moving it elsewhere.
    for source in base_path.iterdir():
        if source.is_file() and source.name not in ('model.safetensors.index.json',):
            if source.suffix in ('.json', '.safetensors', '.model', '.txt', '.tiktoken', '.jinja'):
                destination = 'model-original.safetensors' if source.name == 'model.safetensors' else source.name
                shutil.copy2(source, output_path / destination)
    save_file(adapter, output_path / 'moe_layer.safetensors')
    save_file(base_layers, output_path / 'base_layer.safetensors')
    index['weight_map'] = {key: ('model-original.safetensors' if value == 'model.safetensors' else value)
                           for key, value in original_map.items() if key not in original_keys}
    index['weight_map'].update({key: 'moe_layer.safetensors' for key in adapter})
    index['weight_map'].update({key: 'base_layer.safetensors' for key in base_layers})
    original_bytes = index.get('metadata', {}).get('total_size')
    added_bytes = sum(value.numel() * value.element_size() for value in adapter.values())
    if original_bytes is not None:
        index.setdefault('metadata', {})['total_size'] = original_bytes + added_bytes
    elif not index_path.exists():
        with safe_open(base_path / 'model.safetensors', framework='pt', device='cpu') as shard:
            total = sum(shard.get_tensor(key).numel() * shard.get_tensor(key).element_size() for key in shard.keys())
        index.setdefault('metadata', {})['total_size'] = total + added_bytes
    (output_path / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2))
    model_file, prefix = ('modeling_llama', 'Llama') if arch == 'llama' else ('modeling_qwen2', 'Qwen2')
    shutil.copy2(Path(__file__).parent / 'infer' / f'{model_file}.py', output_path)
    merged_config = json.loads((output_path / 'config.json').read_text())
    # A distinct architecture name prevents vLLM from selecting a dense native model.
    merged_config.update({
        'architectures': [f'UpSafe{prefix}ForCausalLM'],
        'model_type': 'llamamoe' if arch == 'llama' else 'qwen2moe',
        'auto_map': {
            'AutoConfig': f'{model_file}.{prefix}MoEConfig',
            'AutoModelForCausalLM': f'{model_file}.UpSafe{prefix}ForCausalLM',
            'AutoModel': f'{model_file}.{prefix}Model',
        },
        'layers_to_transform': sorted(layers),
        'num_experts': config['num_experts'],
        'topk': config['topk'],
        'safe_tempature': safe_tempature,
    })
    (output_path / 'config.json').write_text(json.dumps(merged_config, indent=2))
    print(f'Exported checkpoint to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--arch', choices=['llama', 'qwen'], required=True)
    parser.add_argument('--safe_tempature', type=float, default=0.5)
    args = parser.parse_args()
    convert_model(**vars(args))
