# Code: abliterate.py

One file, eight resumable steps. Each step loads its inputs from `out/` and saves its outputs there,
so any step can be re-run alone (`run --step N`) and projected in class.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v   # 16 offline tests (~2 s), no downloads
python abliterate.py doctor
./demo.sh                                 # paused walkthrough: Enter per step, s skip, q quit
python abliterate.py all                  # steps 1-8 without pauses
python abliterate.py induce               # add the direction to a harmless prompt
./chat before   |   ./chat after          # --message "…" for one-shot, --chat-tokens N reply cap
```

| Step | Command | Saves |
|---|---|---|
| 1 load | `run --step 1` | `model.json` |
| 2 data | `run --step 2` | `splits.json` (train filtered by behavior) |
| 3 capture | `run --step 3` | `acts.pt` [prompts, layers, positions, d_model] |
| 4 directions | `run --step 4` | `directions.pt` |
| 5 select | `run --step 5` | `selection.json`, `direction.pt` |
| 6 edit | `run --step 6` | `edited/` (HF model), `edit.json` |
| 7 verify | `run --step 7` | `verify.json` (hook vs reloaded-weights logit gap) |
| 8 eval | `run --step 8` | `report.json`, `private/generations.json` |

## Useful flags
Every `Config` field is a flag: `--n-train 128 --n-val 32 --n-test 100 --n-positions 5 --batch-size 16`,
`--method norm-preserving`, `--alpha 0.5`, `--projected`, `--no-filter`, `--kl-threshold 0.1`,
`--device cpu|cuda|mps`, `--out out/exp2`, `--model-id <hf id or local dir>`.
`--harmful-dataset` / `--harmless-dataset` accept a Hugging Face id or a local `.txt` (one prompt per line) or
`.jsonl` (`{"text": …}` per line), which makes the lab work offline and enables exercise E.

## Defaults
Paper-style: 128/32/100 per class, refusal-score selection with KL < 0.1, induce > 0, last 20% of layers pruned.
On a laptop CPU the full run takes minutes; lower `--n-val` for a faster live demo.

Supports HF decoder models with `model.model.embed_tokens`, `layers[i].self_attn.o_proj` and `layers[i].mlp.down_proj`
(Qwen2, Llama, Mistral, Gemma-style). Other architectures need their own `residual_writers`.

See `CHANGES.md` for what was fixed relative to the starter `teach_abliterate.py`.
