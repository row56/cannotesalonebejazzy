import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.attributes import chromaticism, describe, note_density
from melodicvae.model import MelodyVAE, VAEConfig, vae_loss
from melodicvae.repr import HOLD, REST, pitch_to_token, skyline, tokens_to_midi, transpose
from melodicvae.style import StyleSpace


def test_skyline_takes_highest_and_marks_onsets():
    notes = [(0, 4, 60), (2, 6, 67)]
    tokens = skyline(notes, n_steps=8)
    assert tokens[0] == pitch_to_token(60)
    assert tokens[1] == HOLD
    assert tokens[2] == pitch_to_token(67)
    assert tokens[3] == HOLD
    assert tokens[6] == REST


def test_repeated_note_gets_its_own_onset():
    tokens = skyline([(0, 2, 60), (2, 4, 60)], n_steps=4)
    assert tokens[0] == pitch_to_token(60)
    assert tokens[2] == pitch_to_token(60), "repeat must not collapse into a HOLD"


def test_midi_roundtrip_preserves_pitches_and_onsets():
    pytest.importorskip("pretty_midi")
    tokens = np.array(
        [pitch_to_token(60), HOLD, pitch_to_token(64), REST, pitch_to_token(67), HOLD, HOLD, REST],
        dtype=np.int16,
    )
    pm = tokens_to_midi(tokens, grid_hz=4.0)
    notes = sorted(pm.instruments[0].notes, key=lambda n: n.start)
    assert [n.pitch for n in notes] == [60, 64, 67]
    assert [round(n.start * 4) for n in notes] == [0, 2, 4]
    assert round(notes[2].end * 4) == 7


def test_transpose_rejects_out_of_range():
    tokens = np.array([pitch_to_token(126), HOLD], dtype=np.int16)
    assert transpose(tokens, 1)[0] == pitch_to_token(127)
    assert transpose(tokens, 3) is None


def test_descriptors_are_in_range():
    tokens = np.array([pitch_to_token(60), HOLD, pitch_to_token(61), REST] * 4, dtype=np.int16)
    d = describe(tokens)
    assert 0 <= note_density(tokens) <= 1
    assert 0 <= chromaticism(tokens) <= 1
    assert set(d) == set(describe(tokens))


def test_training_step_runs_and_shrinks_loss():
    torch.manual_seed(0)
    model = MelodyVAE(VAEConfig(enc_hidden=32, dec_hidden=32, latent_dim=8))
    x = torch.randint(0, 130, (4, 16))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    first = None
    for _ in range(20):
        logits, mu, logvar, _ = model(x)
        stats = vae_loss(logits, x, mu, logvar, beta=0.1)
        opt.zero_grad()
        stats["loss"].backward()
        opt.step()
        first = first if first is not None else stats["loss"].item()
    assert stats["loss"].item() < first
    assert model.generate(torch.randn(2, 8), steps=16).shape == (2, 16)


def _two_cluster_latents(n=50, latent=16, seed=0):
    rng = np.random.default_rng(seed)
    z = np.concatenate(
        [
            rng.normal(0, 0.1, (n, latent)) + np.eye(latent)[0] * 3,
            rng.normal(0, 0.1, (n, latent)) + np.eye(latent)[1] * 3,
        ]
    )
    return (z, np.array([0] * n + [1] * n))


def test_style_space_axes():
    z, y = _two_cluster_latents()
    space = StyleSpace.fit(z, y, ["jazz", "classical"], min_count=10)
    jazz = space.axes["jazz"]
    assert np.isclose(np.linalg.norm(jazz), 1.0)
    assert jazz[0] > 0 > jazz[1]
    assert np.isclose(abs(jazz[0]), 1 / np.sqrt(2), atol=0.05)
    assert np.isclose(abs(jazz[1]), 1 / np.sqrt(2), atol=0.05)
    assert np.allclose(jazz[2:], 0, atol=0.05)
    proj = z @ jazz
    assert proj[y == 0].mean() > proj[y == 1].mean()


def test_two_labels_give_one_axis_not_two():
    z, y = _two_cluster_latents()
    space = StyleSpace.fit(z, y, ["jazz", "classical"], min_count=10)
    assert np.allclose(space.axes["jazz"], -space.axes["classical"], atol=1e-06)


def test_thin_labels_get_no_axis():
    z, y = _two_cluster_latents(n=50)
    space = StyleSpace.fit(z, y, ["jazz", "classical"], min_count=500)
    assert space.axes == {}, "a centroid over too few windows must not become an axis"
    assert space.counts == [50, 50]


def test_attribute_axis_recovers_a_planted_direction():
    rng = np.random.default_rng(1)
    latent = 12
    z = rng.normal(0, 1, (400, latent))
    direction = np.zeros(latent)
    direction[3] = 1.0
    values = z @ direction + rng.normal(0, 0.01, len(z))
    space = StyleSpace(["a"], np.zeros((1, latent)), z.mean(0))
    axis = space.attribute_axis("attr:density", z, values)
    assert abs(axis @ direction) > 0.95


def test_era_lookup_handles_accents_and_full_names():
    from melodicvae.labels import era_of

    assert era_of("Bach, Johann Sebastian") == "baroque"
    assert era_of("Dvořák") == "romantic"
    assert era_of("Claude Debussy") == "impressionist"
    assert era_of("Some Unknown Person") == "unknown"


def test_labelset_roundtrip_and_viability(tmp_path):
    from melodicvae.labels import LabelSet

    ls = LabelSet({"genre": ["jazz"] * 5 + ["classical"] * 3, "artist": ["evans"] * 5 + ["mozart"] * 3})
    assert ls.counts("genre") == {"classical": 3, "jazz": 5}
    assert ls.viable("genre", 4) == ["jazz"]
    path = tmp_path / "labels.npz"
    np.savez(path, **ls.to_npz_dict())
    back = LabelSet.from_npz(np.load(path, allow_pickle=True))
    assert back.names == ls.names
    assert np.array_equal(back.codes["artist"], ls.codes["artist"])


def test_grouped_split_never_shares_a_source_file():
    from melodicvae.data import grouped_split

    groups = np.repeat([f"piece_{i}.mid" for i in range(50)], 8)
    tr, va = grouped_split(groups, val_frac=0.2, seed=0)
    assert len(tr) + len(va) == len(groups)
    assert not set(groups[tr]) & set(groups[va])
    assert len(va) > 0 and len(tr) > 0
    assert len(np.unique(groups[va])) == 10


def test_eval_view_disables_augmentation(tmp_path):
    from melodicvae.data import MelodyDataset

    path = tmp_path / "d.npz"
    x = np.tile(np.array([62, 1, 64, 1, 0, 66, 1, 1], dtype=np.int16), (4, 8))
    np.savez(
        path,
        x=x,
        genre_codes=np.zeros(4, dtype=np.int32),
        genre_names=np.array(["jazz"], dtype=object),
    )
    ds = MelodyDataset(path, augment_transpose=5)
    ev = ds.eval_view()
    assert ev.augment == 0 and ds.augment == 5
    assert np.array_equal(ev[0][0].numpy(), x[0].astype(np.int64))


def test_estimate_tonic_finds_the_written_key():
    from melodicvae.repr import ONSET_OFFSET, estimate_tonic

    g_major = [7, 9, 11, 0, 2, 4, 6, 7, 7, 2, 7, 11, 7, 2, 7]
    tokens = np.array([ONSET_OFFSET + 60 + p for p in g_major], dtype=np.int16)
    assert estimate_tonic(tokens) == 7
    assert estimate_tonic(np.array([0, 1, 0, 1], dtype=np.int16)) is None


def test_normalise_key_moves_tonic_to_c_and_keeps_intervals():
    from melodicvae.repr import ONSET_OFFSET, estimate_tonic, normalise_key

    g_major = [7, 9, 11, 0, 2, 4, 6, 7, 7, 2, 7, 11, 7, 2, 7]
    tokens = np.array([ONSET_OFFSET + 60 + p for p in g_major], dtype=np.int16)
    out = normalise_key(tokens)
    assert estimate_tonic(out) == 0
    before = np.diff(tokens[tokens >= ONSET_OFFSET].astype(int))
    after = np.diff(out[out >= ONSET_OFFSET].astype(int))
    assert np.array_equal(before, after)
