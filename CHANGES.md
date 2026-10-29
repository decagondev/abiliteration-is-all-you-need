# Changes from the starter `teach_abliterate.py`

## Bugs fixed
| Starter behaviour | Effect | Fix |
|---|---|---|
| Layer-selection "KL proxy" compared two generations that both had the hook attached | Always 0; the drift penalty never filtered anything | Real first-token KL, base logits vs ablated logits |
| Selection hook ablated only one layer's output | Scored a different intervention from the permanent edit | Hooks on all 49 residual writers, exactly the modules step 6 edits |
| Edited `embed_tokens` while tied to `lm_head` | Output head silently edited; final RMSNorm gain reintroduces r̂, so logits drift | Untie (clone head) before editing; save with `tie_word_embeddings=False`; test proves the difference |
| `torch.save` of whole model + tokenizer objects | Breaks on torch ≥ 2.6 (`weights_only=True`), wastes disk | Load from HF cache; only tensors/JSON saved |
| `--step N` relied on in-memory state | Running any step alone crashed | Every step loads its inputs from `out/` |
| `startswith` refusal check | Missed "Sure, but I can't…" style replies | Substring regex over first 300 chars (still a heuristic, documented) |
| Truncation to `MAX_LEN` | Could cut off the assistant prefix being measured | Reject over-long prompts instead |
| Layer-0 candidate | Means coincide exactly (same template token) → divide by zero / NaN | Detected, printed, and skipped with a reason |

## Method upgrades
- Paper's selection: bypass (rank), induce (> 0), KL (< 0.1), prune last 20% of layers; optional multi-position search.
- Logit-based refusal score: one forward pass per prompt instead of generation, so CPU selection is practical.
- Training filter by observed behavior, with a safe fallback and `--no-filter`.
- Batched, left-padded inputs; explicit system prompt; float64 means.
- `verify` step: hooked base vs edited-and-reloaded logits.
- `induce` command: activation addition on a harmless prompt.
- Variants: `--alpha`, `--method norm-preserving`, `--projected`.
- Generations hidden by default (`--reveal` to print a few after review).
- Local prompt files for offline use and new-behavior experiments.

## Tests
16 offline tests on a tiny random Qwen2: projection geometry, partial α, degenerate directions, norm preservation,
KL identity, detector, hook cleanup, zero r̂ in every layer's input under hooks, hook ≡ weight edit with tied head,
the naive tied edit failing, save/reload round-trip, refusal-score sign, and a full steps 1–8 + induce pipeline run.
