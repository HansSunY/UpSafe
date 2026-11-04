# Config notes: paper vs. development configs

The released configs (`upcycle/configs/**`) are normalized to the settings
described in the paper (Appendix A.1). If you want to recover the recorded configuration of the *internal
development runs*, apply the following
deltas — they are the differences between the anonymous code snapshot and the
paper text:

| Item | Paper (released config) | Internal dev config |
|---|---|---|
| Stage-2 epochs | 10 | 20 |
| Llama-3.1-8B stage-1 dataset | `star,star_eval` (1,000 harmful) | `star,star_eval,star_anchor` (all 1,915) |
| Llama-3.1-8B stage-2 dataset | `star,star_eval,star_anchor` (1,915) | `star_eval,star_anchor` (300 harmful + 915 benign) |
| Qwen2.5-7B / -14B stage-1 dataset | `star,star_eval` (1,000 harmful) | same |
| Qwen2.5-7B / -14B stage-2 dataset | `star,star_eval,star_anchor` (1,915) | `star_eval,star_anchor` (300 + 915) |
| MoE one-stage baseline epochs | 30 | 20 (some runs) |
| loss arg name | `sg_loss_coef` | earlier snapshots called it `lpr_loss_coef` |

Everything else (lr 5e-5, warmup 0.1, cosine, batch 4×4, aux 0.01 / sg 0.1,
top-3 layers, 4 experts, top-2) matches.

## Corrections in the public code

The release review corrected fixed batch-size divisors in the soft-guardrail
loss, and now uses the binary negative log probability described by the method,
excluding padding tokens. Previous code applied cross entropy to probabilities
and assumed a fixed batch size. Existing checkpoints are unchanged; training
with corrected code should not be described as a bit-exact reproduction of the
original development runs.

Zero auxiliary-loss coefficients now disable the corresponding loss and do not
mask the general expert during stage 2. The one-stage R1-Distill-Qwen-7B/14B
baseline layer lists now match their corresponding two-stage configurations.

The layer lists for Qwen2.5-7B (15, 17, 18) and R1-Distill-Qwen-7B (13, 15, 16)
still differ in the supplied configurations. They are preserved as provided;
confirm the intended experimental setting before claiming identical layer
selection between these two models.

Training patches target transformers 4.51.3 and peft 0.14.0. The exported model
uses separate inference code. Runtime compatibility and benchmark results must
be validated on the target CUDA hardware; local code checks are not a substitute
for a full training or evaluation run.
