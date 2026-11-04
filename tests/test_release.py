"""CPU regression checks for release fixes; no model downloads or API calls."""
import ast
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval/judge'))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class ReleaseTests(unittest.TestCase):
    def test_binary_guardrail_for_multiple_batch_sizes_and_padding(self):
        for backbone in ('llama', 'qwen2'):
            function = load_function(ROOT / f'upcycle/models/modeling_{backbone}.py', 'sg_loss_func',
                                     {'torch': torch, 'F': F})
            for batch_size in (1, 2, 4, 7):
                with self.subTest(backbone=backbone, batch_size=batch_size):
                    logits = torch.zeros(batch_size * 3, 4, requires_grad=True)
                    safety = torch.tensor([i % 2 == 0 for i in range(batch_size)])
                    attention = torch.tensor([[1, 1, 0]] * batch_size)
                    loss = function((logits,), safety, attention)
                    expected = sum(math.log(4) if safe else math.log(4 / 3) for safe in safety) / batch_size
                    self.assertAlmostEqual(loss.item(), expected, places=5)
                    loss.backward()
                    self.assertTrue(torch.isfinite(logits.grad).all())
                    self.assertTrue(torch.equal(logits.grad.reshape(batch_size, 3, 4)[:, 2], torch.zeros(batch_size, 4)))
                    for row, safe in enumerate(safety):
                        sign = logits.grad.reshape(batch_size, 3, 4)[row, 0, 0].item()
                        self.assertLess(sign, 0) if safe else self.assertGreater(sign, 0)

    def test_judge_url_accepts_base_and_full_endpoint(self):
        from openai_client import chat_url
        expected = 'https://api.example/v1/chat/completions'
        self.assertEqual(chat_url('https://api.example/v1/'), expected)
        self.assertEqual(chat_url(expected), expected)

    def test_invalid_judge_output_is_not_counted_safe(self):
        import gpt4_judge
        self.assertEqual(gpt4_judge.extract_content('#thescore:', '#thereason: x\n#thescore: 5.'), 5)
        with self.assertRaises(ValueError):
            gpt4_judge.extract_content('#thescore:', '#thescore: 99')
        with patch.object(gpt4_judge, 'call_chat_api', return_value='invalid'):
            with self.assertRaises(ValueError):
                gpt4_judge.auto_judge_gpt_4([('question', 'answer')])

    def test_safety_evaluation_return_contract(self):
        import QA_eval
        import gpt4_judge
        with patch.object(gpt4_judge, 'call_chat_api', return_value='#thereason: x\n#thescore: 5'):
            self.assertEqual(QA_eval.process_chunk([('question', 'answer')]), [5])

    def test_refusal_classes_are_validated(self):
        import refuse_judge
        with patch.object(refuse_judge, 'call_chat_api', return_value=' 1_full_compliance\n'):
            self.assertEqual(refuse_judge.openai_refusal_evaluate([('q', 'a')]), [1])
        with patch.object(refuse_judge, 'call_chat_api', return_value='1_full_refusal'):
            with self.assertRaises(ValueError):
                refuse_judge.openai_refusal_evaluate([('q', 'a')])

    def test_missing_api_key_has_clear_error(self):
        from openai_client import call_chat_api
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'JUDGE_API_KEY'):
                call_chat_api([])

    def test_safety_temperature_does_not_modify_checkpoint(self):
        function = load_function(ROOT / 'eval/QA_gen.py', 'set_safe_tempature', {'Path': Path})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            original = '{"safe_tempature": 0.5}'
            (path / 'config.json').write_text(original)
            self.assertEqual(function(path, 0.8), {'safe_tempature': 0.8})
            self.assertEqual((path / 'config.json').read_text(), original)
            with self.assertRaises(ValueError):
                function(path, 1.1)

    def test_probe_handles_singleton_batches(self):
        scan = load_module('upsafe_scan', ROOT / 'scan/get_layer_if.py')
        with tempfile.TemporaryDirectory() as directory:
            dataset = torch.utils.data.TensorDataset(torch.randn(33, 4), torch.zeros(33))
            train = torch.utils.data.DataLoader(dataset, batch_size=32)
            model = scan.LinearModel(4)
            scan.train_and_evaluate(model, train, train, torch.nn.BCELoss(),
                                    torch.optim.Adam(model.parameters()), 'layer', Path(directory), epochs=1)

    def test_evaluation_sweeps_refusal_at_each_temperature(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            interpreter = directory / 'python'
            interpreter.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$UPS_TEST_LOG"\n')
            interpreter.chmod(0o755)
            log = directory / 'calls.log'
            environment = dict(os.environ, PATH=str(directory) + os.pathsep + os.environ['PATH'],
                               UPS_TEST_LOG=str(log), TAUS='0.2 0.8', BENCHES='jbb xs')
            subprocess.run(['bash', str(ROOT / 'eval/run_eval.sh'), 'model', 'tag'],
                           cwd=directory, env=environment, check=True)
            calls = log.read_text().splitlines()
            self.assertEqual(len(calls), 8)
            refusal_calls = [line for line in calls if 'refuse_judge.py' in line]
            self.assertEqual(len(refusal_calls), 2)
            self.assertTrue(any('tau0.2_xs' in line for line in refusal_calls))
            self.assertTrue(any('tau0.8_xs' in line for line in refusal_calls))

    def test_checkpoint_export_loads_both_architectures_and_preserves_general_expert(self):
        from transformers import AutoModelForCausalLM
        exporter = load_module('upsafe_export', ROOT / 'upcycle/convert_model.py')
        for backbone, prefix in [('llama', 'Llama'), ('qwen2', 'Qwen2')]:
            with self.subTest(backbone=backbone), tempfile.TemporaryDirectory() as directory:
                from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
                config_class, model_class = (LlamaConfig, LlamaForCausalLM) if backbone == 'llama' else (Qwen2Config, Qwen2ForCausalLM)
                args = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
                base = model_class(config_class(**args))
                base_path = Path(directory) / 'base checkpoint'
                base.save_pretrained(base_path)
                module = load_module('upsafe_test_' + backbone, ROOT / f'upcycle/infer/modeling_{backbone}.py')
                config = getattr(module, prefix + 'MoEConfig')(**args, layers_to_transform=[0], num_experts=4, topk=2)
                upcycled = getattr(module, prefix + 'ForCausalLM')(config)
                adapter = Path(directory) / 'adapter'
                adapter.mkdir()
                (adapter / 'adapter_config.json').write_text(json.dumps({
                    'base_model_name_or_path': str(base_path), 'layers_to_transform': [0], 'num_experts': 4, 'topk': 2,
                }))
                weights = {'base_model.model.' + k.replace('.default.', '.'): v.contiguous()
                           for k, v in upcycled.state_dict().items() if '.moe_' in k}
                save_file(weights, adapter / 'adapter_model.safetensors')
                output = Path(directory) / 'exported checkpoint'
                exporter.convert_model(adapter, output, 'llama' if backbone == 'llama' else 'qwen')
                loaded, info = AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True, output_loading_info=True)
                self.assertFalse(info['missing_keys'])
                self.assertFalse(info['unexpected_keys'])
                torch.testing.assert_close(loaded.model.layers[0].mlp.expert0.gate_proj.weight,
                                           base.model.layers[0].mlp.gate_proj.weight)
                self.assertEqual(loaded(torch.tensor([[1, 2, 3]])).logits.shape, (1, 3, 32))
                self.assertFalse(any(p.is_symlink() for p in output.iterdir()))
                with self.assertRaisesRegex(ValueError, 'empty'):
                    exporter.convert_model(adapter, output, 'llama' if backbone == 'llama' else 'qwen')


if __name__ == '__main__':
    unittest.main()
