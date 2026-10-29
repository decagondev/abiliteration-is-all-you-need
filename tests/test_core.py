"""Offline tests for abliterate.py on a tiny random Qwen2 (no downloads).

    python -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import abliterate as A  # noqa: E402

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402


def tiny_qwen(tied=True, seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=tied)
    m = Qwen2ForCausalLM(cfg).eval()
    # random init leaves RMSNorm gains at 1; perturb them so the tied-head trap is visible
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.add_(0.3 * torch.randn_like(p))
    return m


IDS = torch.randint(0, 97, (2, 11), generator=torch.Generator().manual_seed(1))


def logits(m):
    with torch.no_grad():
        return m(input_ids=IDS).logits


class Geometry(unittest.TestCase):
    def test_project_removes_component_and_is_idempotent(self):
        x = torch.randn(5, 16, dtype=torch.float64)
        r = A.unit(torch.randn(16, dtype=torch.float64))
        y = A.project_out(x, r)
        self.assertLess((y @ r).abs().max().item(), 1e-12)
        self.assertTrue(torch.allclose(A.project_out(y, r), y))

    def test_project_leaves_orthogonal_part_untouched(self):
        r = A.unit(torch.randn(16, dtype=torch.float64))
        x = torch.randn(16, dtype=torch.float64)
        x_perp = A.project_out(x, r)
        self.assertTrue(torch.allclose(A.project_out(x_perp + 3.0 * r, r), x_perp))

    def test_partial_alpha(self):
        r = A.unit(torch.randn(8, dtype=torch.float64))
        x = torch.randn(8, dtype=torch.float64)
        self.assertAlmostEqual(float(A.project_out(x, r, 0.5) @ r), 0.5 * float(x @ r), places=10)

    def test_degenerate_direction_raises(self):
        with self.assertRaises(ValueError):
            A.unit(torch.zeros(8))

    def test_orthogonalize_linear_and_embedding(self):
        r = A.unit(torch.randn(8, dtype=torch.float64))
        W = torch.randn(8, 5, dtype=torch.float64)    # [d_model, d_in]
        E = torch.randn(20, 8, dtype=torch.float64)   # [vocab, d_model]
        self.assertLess(A.orthogonalize_(W, r, 0), 1e-12)
        self.assertLess(A.orthogonalize_(E, r, 1), 1e-12)
        self.assertLess((r @ W).abs().max().item(), 1e-12)
        self.assertLess((E @ r).abs().max().item(), 1e-12)

    def test_norm_preserving_keeps_norms_and_removes_r(self):
        r = A.unit(torch.randn(8, dtype=torch.float64))
        W = torch.randn(8, 5, dtype=torch.float64)
        n0 = W.norm(dim=0).clone()
        left = A.orthogonalize_(W, r, 0, norm_preserving=True)
        self.assertLess(left, 1e-12)
        self.assertTrue(torch.allclose(W.norm(dim=0), n0))

    def test_kl_identity_is_zero(self):
        z = torch.randn(4, 30)
        self.assertAlmostEqual(A.kl_first_token(z, z), 0.0, places=10)
        self.assertGreater(A.kl_first_token(z, z + torch.randn(4, 30)), 0.0)

    def test_refusal_detector(self):
        self.assertTrue(A.looks_like_refusal("I'm sorry, but I can't help with that."))
        self.assertTrue(A.looks_like_refusal("As an AI language model, I must decline."))
        self.assertFalse(A.looks_like_refusal("Sure! Here is a poem about the sea."))


class Intervention(unittest.TestCase):
    def test_writer_count(self):
        m = tiny_qwen()
        self.assertEqual(len(list(A.residual_writers(m))), 1 + 2 * m.config.num_hidden_layers)

    def test_hooks_clean_up(self):
        m = tiny_qwen()
        r = A.unit(torch.randn(32))
        before = logits(m)
        with A.ablation_hooks(m, r):
            during = logits(m)
        self.assertFalse(torch.allclose(before, during))
        self.assertTrue(torch.allclose(before, logits(m)))
        with A.addition_hook(m, 1, r, 5.0):
            self.assertFalse(torch.allclose(before, logits(m)))
        self.assertTrue(torch.allclose(before, logits(m)))

    def test_residual_stream_has_no_r_component_under_hooks(self):
        m = tiny_qwen()
        r = A.unit(torch.randn(32))
        with A.ablation_hooks(m, r), torch.no_grad():
            hs = m(input_ids=IDS, output_hidden_states=True).hidden_states
        for l in range(m.config.num_hidden_layers):         # resid_pre of every layer
            self.assertLess((hs[l] @ r).abs().max().item(), 1e-5)

    def test_hook_equals_weight_edit_with_tied_head(self):
        m = tiny_qwen(tied=True)
        r = A.unit(torch.randn(32))
        with A.ablation_hooks(m, r):
            hooked = logits(m)
        head_before = m.lm_head.weight.detach().clone()
        A.apply_edit(m, r)
        self.assertFalse(m.config.tie_word_embeddings)
        self.assertTrue(torch.equal(m.lm_head.weight, head_before), "lm_head must be preserved")
        self.assertLess((hooked - logits(m)).abs().max().item(), 1e-4)

    def test_naive_tied_edit_is_wrong(self):
        """Editing the embedding while still tied changes lm_head → logits drift."""
        m = tiny_qwen(tied=True)
        r = A.unit(torch.randn(32))
        with A.ablation_hooks(m, r):
            hooked = logits(m)
        with torch.no_grad():
            for _, mod, ax in A.residual_writers(m):
                A.orthogonalize_(mod.weight.data, r, ax)     # no untie
        self.assertGreater((hooked - logits(m)).abs().max().item(), 1e-3)

    def test_save_reload_roundtrip(self):
        m = tiny_qwen(tied=True)
        A.apply_edit(m, A.unit(torch.randn(32)))
        want = logits(m)
        with tempfile.TemporaryDirectory() as d:
            m.save_pretrained(d)
            back = Qwen2ForCausalLM.from_pretrained(d).eval()
        self.assertFalse(back.config.tie_word_embeddings)
        self.assertLess((want - logits(back)).abs().max().item(), 1e-5)

    def test_refusal_score_sign(self):
        z = torch.full((1, 10), -10.0)
        z[0, 3] = 10.0
        self.assertGreater(A.refusal_score(z, [3]).item(), 0)
        self.assertLess(A.refusal_score(z, [4]).item(), 0)


if __name__ == "__main__":
    unittest.main()
