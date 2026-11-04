"""Apply the UpSafe patches to the active training environment."""
import hashlib
import importlib.metadata
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch_environment(site_packages, backup_dir):
    site_packages, backup_dir = Path(site_packages), Path(backup_dir)
    replacements = {
        'peft/mapping.py': 'upcycle/peft_moe/mapping.py',
        'peft/peft_model.py': 'upcycle/peft_moe/peft_model.py',
        'peft/tuners/tuners_utils.py': 'upcycle/peft_moe/tuners_utils.py',
        'peft/utils/save_and_load.py': 'upcycle/peft_moe/save_and_load.py',
        'transformers/models/llama/modeling_llama.py': 'upcycle/models/modeling_llama.py',
        'transformers/models/qwen2/modeling_qwen2.py': 'upcycle/models/modeling_qwen2.py',
    }
    edits = {}
    for relative, statement in {
        'peft/__init__.py': 'from .tuners.moe import MoeConfig, MoeModel',
        'peft/tuners/__init__.py': 'from .moe import MoeConfig, MoeModel',
    }.items():
        content = (site_packages / relative).read_text()
        if statement not in content:
            content += '\n' + statement + '\n'
        edits[relative] = content
    relative = 'peft/utils/peft_types.py'
    content = (site_packages / relative).read_text()
    if '    MOE = "MOE"' not in content:
        anchor = '    LORA = "LORA"'
        if anchor not in content:
            raise RuntimeError('Cannot locate the PEFT 0.14.0 enum anchor.')
        content = content.replace(anchor, anchor + '\n    MOE = "MOE"', 1)
    edits[relative] = content
    relative = 'peft/utils/constants.py'
    content = (site_packages / relative).read_text()
    if 'PeftType.MOE: "moe_"' not in content:
        anchor = 'PEFT_TYPE_TO_PREFIX_MAPPING = {'
        if anchor not in content:
            raise RuntimeError('Cannot locate the PEFT adapter-prefix mapping.')
        content = content.replace(anchor, anchor + '\n    PeftType.MOE: "moe_",', 1)
    edits[relative] = content
    # Validate all targets before replacing any installed files.
    for relative in set(replacements) | set(edits):
        target = site_packages / relative
        if not target.is_file():
            raise FileNotFoundError(target)
    for relative in set(replacements) | set(edits):
        target, backup = site_packages / relative, backup_dir / relative
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
    shutil.copytree(ROOT / 'upcycle/peft_moe/moe', site_packages / 'peft/tuners/moe', dirs_exist_ok=True)
    for relative, source in replacements.items():
        shutil.copy2(ROOT / source, site_packages / relative)
    for relative, content in edits.items():
        (site_packages / relative).write_text(content)


if __name__ == '__main__':
    for package, expected in [('peft', '0.14.0'), ('transformers', '4.51.3')]:
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f'Expected {package}=={expected}; use the training requirements.')
    site_packages = Path(importlib.metadata.distribution('peft').locate_file(''))
    environment_id = hashlib.sha256(sys.prefix.encode()).hexdigest()[:12]
    backup_dir = ROOT / 'setup/backups' / environment_id
    patch_environment(site_packages, backup_dir)
    print(f'Patched {site_packages}; original files are in {backup_dir}')
