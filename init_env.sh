cd LLaMA-Factory-093/
pip install -e .
cd ..
pip install peft==0.14.0

cp -f models/modeling_llama.py /opt/conda/lib/python3.10/site-packages/transformers/models/llama/
cp -f models/modeling_qwen2.py /opt/conda/lib/python3.10/site-packages/transformers/models/qwen2/
cp -rf peft_moe/moe/ /opt/conda/lib/python3.10/site-packages/peft/tuners
cp -f peft_moe/mapping.py /opt/conda/lib/python3.10/site-packages/peft
cp -f peft_moe/peft_model.py /opt/conda/lib/python3.10/site-packages/peft
cp -f peft_moe/tuners_utils.py /opt/conda/lib/python3.10/site-packages/peft/tuners
cp -f peft_moe/save_and_load.py /opt/conda/lib/python3.10/site-packages/peft/utils

sed -i '100a \\    MoeConfig,\n    MoeModel' /opt/conda/lib/python3.10/site-packages/peft/__init__.py 
sed -i '$a\from .moe import MoeConfig, MoeModel' /opt/conda/lib/python3.10/site-packages/peft/tuners/__init__.py
sed -i '68a \\    MOE = "MOE"' /opt/conda/lib/python3.10/site-packages/peft/utils/peft_types.py
sed -i '303a \\    PeftType.MOE: "moe_",' /opt/conda/lib/python3.10/site-packages/peft/utils/constants.py