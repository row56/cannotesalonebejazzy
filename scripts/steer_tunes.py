import json, sys
from pathlib import Path
import numpy as np, torch

R = sys.argv[1]
OUT = Path(sys.argv[2])
sys.path.insert(0, R)
sys.path.insert(0, R + '/scripts')
from melodicvae.model import MelodyVAE, VAEConfig
from melodicvae.style import StyleSpace
from melodicvae.repr import ONSET_OFFSET, HOLD, REST, normalise_key, centre_octave, tokens_to_midi
from melodicvae.attributes import describe
from build_demo import sampling_bias

N = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}


def tok(seq):
    out = []
    for n, d in seq:
        if n == 'r':
            out += [REST] * d
            continue
        p = 60 + N[n.rstrip("'")] + 12 * n.count("'")
        out += [ONSET_OFFSET + p] + [HOLD] * (d - 1)
    assert len(out) == 64, len(out)
    return centre_octave(normalise_key(np.array(out, dtype=np.int64)))


TUNES = {
    'ode_to_joy': [
        ('E', 4),
        ('E', 4),
        ('F', 4),
        ('G', 4),
        ('G', 4),
        ('F', 4),
        ('E', 4),
        ('D', 4),
        ('C', 4),
        ('C', 4),
        ('D', 4),
        ('E', 4),
        ('E', 6),
        ('D', 2),
        ('D', 8),
    ],
    'jingle_bells': [
        ('E', 4),
        ('E', 4),
        ('E', 8),
        ('E', 4),
        ('E', 4),
        ('E', 8),
        ('E', 4),
        ('G', 4),
        ('C', 6),
        ('D', 2),
        ('E', 16),
    ],
}
SIGMA, T, TOP_P, SPAN, CANDS, BPM = (1.1567, 0.9, 0.98, 12, 96, 80.0)
grid_hz = BPM / 60 * 4
b = torch.load(R + '/runs/base_v6/best.pt', map_location='cpu', weights_only=False)
m = MelodyVAE(VAEConfig(**b['cfg'].get('model', {})))
m.load_state_dict(b['model'])
m.eval()
axis = StyleSpace.load(R + '/runs/base_v6/axes_dm_heldout/styles.npz').contrast_axis('jazziness', 'jazz', 'classical')
report = {}
for name, seq in TUNES.items():
    t = tok(seq)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    tokens_to_midi(t, grid_hz=grid_hz).write(str(d / 'original.mid'))
    with torch.no_grad():
        mu, _ = m.encode(torch.tensor(t.astype(np.int64))[None])
    z0 = mu.numpy()[0]
    bias = sampling_bias(t, 0.0, SPAN, 'cpu')
    target = int((t >= ONSET_OFFSET).sum())
    rep = {'notes': target, 'orig': {k: round(float(v), 3) for k, v in describe(t).items()}}
    for alpha in (-3.0, 3.0):
        z = torch.tensor(np.atleast_2d(z0 + alpha * SIGMA * axis), dtype=torch.float32)
        cands = []
        for j in range(CANDS):
            torch.manual_seed(1000 + j)
            with torch.no_grad():
                c = m.generate(z, steps=64, temperature=T, top_p=TOP_P, logit_bias=bias).numpy()[0]
            cands.append(c)
        counts = [int((c >= ONSET_OFFSET).sum()) for c in cands]
        best = int(np.argmin([abs(n - target) for n in counts]))
        c = cands[best]
        lab = 'am3p0' if alpha < 0 else 'ap3p0'
        tokens_to_midi(c, grid_hz=grid_hz).write(str(d / f'{lab}.mid'))
        np.save(d / f'{lab}.npy', c)
        rep[lab] = {'notes': counts[best], **{k: round(float(v), 3) for k, v in describe(c).items()}}
    np.save(d / 'original.npy', t)
    report[name] = rep
    print(name, json.dumps(rep))
(OUT / 'manifest.json').write_text(json.dumps({'bpm': BPM, 'sigma': SIGMA, 'tunes': report}, indent=1))
