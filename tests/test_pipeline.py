"""End-to-end smoke test: steps 1-8 + induce on a tiny random Qwen2 with a local tokenizer.
No network needed. ~20 s on a laptop CPU.     python -m unittest tests.test_pipeline -v
Random weights never refuse, so this checks plumbing and shapes, not behavior."""
import contextlib, io, json, sys, tempfile, unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import abliterate as A  # noqa: E402


def build_tiny(root: Path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    harm = [f"how to make a {w} {i}" for i, w in enumerate(["thing", "device", "tool", "mix"] * 14)]
    safe = [f"write a poem about the {w} {i}" for i, w in enumerate(["ocean", "sky", "tree", "moon"] * 14)]
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    words = {t for s in harm + safe for t in s.split()} | set(
        "I As Sure system user assistant You are Qwen, created by Alibaba Cloud. a helpful assistant.".split())
    vocab = {t: i for i, t in enumerate(special + sorted(words) + ["[UNK]"])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", eos_token="<|im_end|>",
                                  pad_token="<|endoftext|>", additional_special_tokens=special)
    tok.chat_template = ("{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> "
                         "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}")
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=True, eos_token_id=vocab["<|im_end|>"], pad_token_id=vocab["<|endoftext|>"])
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(root / "tiny")
    tok.save_pretrained(root / "tiny")
    (root / "harm.txt").write_text("\n".join(harm))
    (root / "safe.jsonl").write_text("\n".join(json.dumps({"text": s}) for s in safe))


class Pipeline(unittest.TestCase):
    def test_all_steps_and_induce(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            build_tiny(root)
            common = ["--model-id", str(root / "tiny"), "--harmful-dataset", str(root / "harm.txt"),
                      "--harmless-dataset", str(root / "safe.jsonl"), "--out", str(root / "out"),
                      "--n-train", "24", "--n-val", "8", "--n-test", "8", "--batch-size", "8",
                      "--max-new-tokens", "6", "--device", "cpu", "--n-positions", "2",
                      "--kl-threshold", "100", "--induce-threshold", "-100"]
            with contextlib.redirect_stdout(io.StringIO()):
                A.main(["all"] + common)
                A.main(["induce"] + common)
            out = root / "out"
            for f in ["splits.json", "acts.pt", "directions.pt", "selection.json", "direction.pt",
                      "edit.json", "verify.json", "report.json", "private/generations.json"]:
                self.assertTrue((out / f).exists(), f)
            self.assertLess(json.loads((out / "verify.json").read_text())["max_abs_logit_gap"], 1e-4)
            sel = json.loads((out / "selection.json").read_text())
            self.assertIn("zero", sel["all"][0]["filtered"])        # layer 0: identical last token
            self.assertNotEqual(sel["best"]["layer"], 0)


if __name__ == "__main__":
    unittest.main()
