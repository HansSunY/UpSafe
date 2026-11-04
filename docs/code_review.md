# Release review

This review covers the UpSafe scripts, configurations, modified training code,
and documentation. It also checks syntax across the vendored Python source.

## Changes

- Shortened the main README and moved detailed commands into the reproduction
  guide. Added the paper's framework as PNG and PDF, and replaced the anonymous
  citation with the author list.
- Translated UpSafe-specific Chinese comments/docstrings into English. No
  Chinese Python comments remain. Upstream localization resources, Chinese model
  prompt templates, dataset text, and the upstream Chinese README are retained.
- Rebuilt the environment patch installer: correct PEFT import anchors and
  package paths, environment-specific backups with preserved directory names,
  repeatable application, and a verification step. Backups are ignored by Git.
- Fixed zero-valued loss coefficients being treated as enabled. Stage 2 no longer
  masks the general expert merely because an auxiliary coefficient is zero.
- Replaced fixed batch-size divisors in both soft-guardrail losses with the actual
  batch size. Implemented the binary negative log probability of selecting the
  general/safety expert groups, excluding padding. This corrects the previous
  cross-entropy-on-probabilities calculation. **New training runs can differ from
  the original runs; existing checkpoint weights are not modified.**
- Fixed the Trainer override to forward its arguments and honor `return_outputs`.
  Corrected inference adapter handling and full-parameter trainability handling.
- Removed a Qwen training docstring decorator that crashed module imports because
  the decorated forward method had no docstring.
- Made DeepSpeed parameter gathering conditional on partitioned parameters,
  allowing ordinary CPU inference and smoke checks. CUDA training still uses
  the provided ZeRO-3 configuration.
- Saved MoE adapters as expert/router weights rather than unnecessarily including
  the frozen backbone (unless full-model saving is requested).
- Reworked model export to use safe filesystem operations, support both sharded
  and single-file safetensors checkpoints, preserve tokenizer files, copy weights
  into a portable directory, and avoid stale original MLP entries in the index.
  Single-file base weights are renamed so the loader reads the updated index.
  A distinct UpSafe architecture alias avoids selecting a dense vLLM model.
  Full-model tuning exports are rejected explicitly rather than silently losing
  updated base weights.
- Applied evaluation temperatures as runtime overrides instead of modifying the
  source checkpoint. Validated the temperature range and generation context
  budget. Explicitly set greedy decoding.
- Fixed the safety judge's two-versus-three-value unpacking error, normalized API
  base/full URLs, bounded request retries, rejected malformed results, and
  prevented failed API calls from being counted as safe responses.
- Evaluated XSTest at every requested temperature in the evaluation loop.
- Corrected the R1-Distill-Qwen 7B/14B one-stage baseline layer lists to match the
  corresponding two-stage configurations. Set the HF adapter chat config to the
  MoE tuning type.
- Pinned PyTorch and constrained training dependencies to the vendored framework's
  supported ranges. Preserved the separate training/inference environments.

## Validation

- Python syntax and undefined-name checks across the repository.
- Shell syntax checks for installation, scanning, and evaluation entry points.
- JSON/YAML parsing, dataset registry paths and SHA-1 hashes, training dataset
  sizes, and stage-to-stage layer-list consistency.
- CPU regression suite: binary routing loss and gradients at batch sizes 1, 2,
  4, and 7; padding exclusion; singleton probe batches; invalid judge output;
  URL normalization; safety-score return values; refusal labels; temperature
  overrides; multi-temperature evaluation dispatch; and checkpoint export/reload
  for both Llama and Qwen, with no missing or unexpected weights.
- Training patch applied twice to an isolated environment. Tiny Llama and Qwen
  models complete stage-1 and stage-2 forward/backward passes and adapter
  save/reload. Stage 2 is checked at batch sizes 1, 2, and 4.

The tests use PyTorch 2.5.1, PEFT 0.14.0, Transformers 4.51.3 for training, and
Transformers 4.53.1 for inference. They do not call a judge service, download full
base models, train on CUDA, or verify full-model benchmark results. vLLM custom
model loading still requires a smoke check on the target CUDA system; its
[documented Transformers fallback](https://docs.vllm.ai/en/v0.7.2/models/supported_models.html#transformers-fallback)
is not a guarantee for every hardware/runtime combination.

## Experiment details requiring author verification

The supplied Qwen2.5-7B layer list is `15,17,18`, while R1-Distill-Qwen-7B uses
`13,15,16`. These were preserved, and their baseline configs now match them.
Confirm these values before claiming that both use exactly the same layers from
the instruction-tuned counterpart. Configuration differences recorded in
[CONFIG_NOTES.md](../CONFIG_NOTES.md) are retained; full training logs were not
available to independently establish the original experimental settings.
