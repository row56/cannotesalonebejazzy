import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.style import StyleSpace, auc, axis_auc


def _two_blobs(n=400, dim=8, sep=3.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(2 * n, dim))
    genre = np.repeat([0, 1], n)
    z[genre == 0, 0] += sep
    corpus = np.concatenate([np.repeat([0, 1], n // 2), np.full(n, 2)])
    z[corpus == 1, 1] += sep
    return (z, genre, corpus)


def test_auc_is_symmetric_and_bounded():
    pos, neg = (np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 2.0]))
    assert auc(pos, neg) == 1.0
    assert auc(neg, pos) == 0.0
    assert auc(pos, pos) == 0.5


def test_auc_is_nan_without_both_classes():
    assert np.isnan(auc(np.array([1.0]), np.array([])))


def test_axis_fit_on_a_subset_still_separates_the_heldout_corpus():
    z, genre, corpus = _two_blobs()
    fit = corpus != 1
    space = StyleSpace.fit(z[fit], genre[fit], ["a", "b"], min_count=10)
    axis = space.axes["a"]
    scores = z @ axis
    assert auc(scores[corpus == 1], scores[corpus == 2]) > 0.9
    assert 0.3 < auc(scores[corpus == 1], scores[fit & (genre == 0)]) < 0.7


def test_holdout_changes_nothing_when_the_corpus_is_redundant():
    z, genre, corpus = _two_blobs(sep=3.0)
    z[corpus == 1, 1] -= 3.0
    full = StyleSpace.fit(z, genre, ["a", "b"], min_count=10).axes["a"]
    fit = corpus != 1
    held = StyleSpace.fit(z[fit], genre[fit], ["a", "b"], min_count=10).axes["a"]
    assert float(full @ held) > 0.99


def test_axis_auc_matches_the_general_helper():
    z, genre, _ = _two_blobs()
    space = StyleSpace.fit(z, genre, ["a", "b"], min_count=10)
    axis = space.axes["a"]
    scores = z @ axis
    assert axis_auc(z, genre, axis, 0) == auc(scores[genre == 0], scores[genre != 0])
