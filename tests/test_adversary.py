import numpy as np
import torch
from melodicvae.model import CorpusAdversary
from melodicvae.train import IGNORE, adversary_targets, lambda_schedule


class _FakeDataset:

    def __init__(self, values):
        self._names = sorted(set(values))
        index = {n: i for i, n in enumerate(self._names)}
        self._codes = np.array([index[v] for v in values], dtype=np.int32)

    @property
    def fields(self):
        return ["corpus"]

    def names(self, field):
        return self._names

    def codes(self, field):
        return self._codes


def test_gradient_reversal_flips_sign():
    z = torch.randn(8, 16, requires_grad=True)
    head = CorpusAdversary(16, 3, hidden=8).eval()
    loss = head(z, 1.0).sum()
    loss.backward()
    reversed_grad = z.grad.clone()
    z.grad = None
    head(z, -1.0).sum().backward()
    assert torch.allclose(reversed_grad, -z.grad, atol=1e-06)


def test_unknown_corpus_is_ignored_and_weights_favour_rare():
    ds = _FakeDataset(["a"] * 10 + ["b"] * 2 + ["unknown"] * 3)
    codes, weights, names = adversary_targets(ds, "corpus")
    assert names == ["a", "b", "unknown"]
    assert (codes == IGNORE).sum() == 3
    assert weights[names.index("b")] > weights[names.index("a")]
    assert weights[names.index("unknown")] == 0.0


def test_ignore_index_excludes_unknown_from_the_loss():
    torch.manual_seed(0)
    logits = torch.randn(4, 3)
    y = torch.tensor([0, 1, IGNORE, IGNORE])
    full = torch.nn.functional.cross_entropy(logits, y, ignore_index=IGNORE)
    kept = torch.nn.functional.cross_entropy(logits[:2], y[:2])
    assert torch.allclose(full, kept)


def test_lambda_ramps_from_zero_to_max():
    assert lambda_schedule(0, 100, 0.3) == 0.0
    assert 0.0 < lambda_schedule(50, 100, 0.3) < 0.3
    assert lambda_schedule(1000, 100, 0.3) == lambda_schedule(100, 100, 0.3)
    assert abs(lambda_schedule(100, 100, 0.3) - 0.3) < 0.001
