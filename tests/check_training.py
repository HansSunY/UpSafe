"""Training smoke checks; run in the patched training environment."""
import json
import tempfile
from pathlib import Path

import torch
from peft import MoeConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM


def run():
    for config_class, model_class in [(LlamaConfig, LlamaForCausalLM), (Qwen2Config, Qwen2ForCausalLM)]:
        kwargs = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
        config = config_class(**kwargs)
        base = model_class(config)
        adapter_config = MoeConfig(task_type=TaskType.CAUSAL_LM, layers_to_transform=[0],
                                   num_experts=4, topk=2, aux_loss_coef=0.01, sg_loss_coef=0)
        model = get_peft_model(base, adapter_config)
        model.train()
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        output = model(input_ids=tokens, labels=tokens, attention_mask=torch.ones_like(tokens),
                       safety=torch.tensor([False, True]))
        assert output.sg_loss is None
        assert torch.isfinite(output.loss)
        output.loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            saved = json.loads((Path(directory) / 'adapter_config.json').read_text())
            assert saved['sg_loss_coef'] is None
            stage2_config = PeftConfig.from_pretrained(directory)
            stage2_config.aux_loss_coef = None
            stage2_config.sg_loss_coef = 0.1
            stage2 = PeftModel.from_pretrained(model_class(config_class(**kwargs)), directory,
                                              config=stage2_config, is_trainable=True)
            for name, parameter in stage2.named_parameters():
                parameter.requires_grad_('router' in name)
            stage2.train()
            for batch_size in (1, 2, 4):
                tokens = torch.randint(0, 32, (batch_size, 4))
                output = stage2(input_ids=tokens, labels=tokens, attention_mask=torch.ones_like(tokens),
                                safety=torch.tensor([i % 2 == 0 for i in range(batch_size)]))
                assert output.balance_loss is None
                assert torch.isfinite(output.sg_loss)
                output.loss.backward()
                for name, parameter in stage2.named_parameters():
                    if parameter.grad is not None:
                        assert torch.isfinite(parameter.grad).all(), name
                stage2.zero_grad()
        print(f'{model_class.__name__}: stage 1 and stage 2 forward/backward, save/reload OK')


if __name__ == '__main__':
    run()
