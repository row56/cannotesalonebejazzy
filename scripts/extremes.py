import sys, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / 'scripts'))
from melodicvae.data import MelodyDataset
from melodicvae.model import MelodyVAE, VAEConfig
from melodicvae.style import StyleSpace, encode_dataset
from melodicvae.attributes import describe
from melodicvae.repr import ONSET_OFFSET, tokens_to_midi
from build_demo import sampling_bias

# brahms and elgar pushed to +-10 sd (demo page section 2)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else R / 'artifacts/extremes'
b = torch.load(R / 'runs/base_v6/best.pt', map_location='cpu', weights_only=False)
m = MelodyVAE(VAEConfig(**b['cfg'].get('model', {}))); m.load_state_dict(b['model']); m.eval()
ax = StyleSpace.load(R / 'runs/base_v6/axes_dm_heldout/styles.npz').contrast_axis('jazziness', 'jazz', 'classical')
v7 = MelodyDataset(R / 'data/processed/train_v7.npz')
P = encode_dataset(m, DataLoader(v7, batch_size=256), 'cpu') @ ax
sig, mu = P.std(), P.mean()
seeds = {'brahms': ('data/processed/train_v7.npz', 1303), 'elgar': ('data/processed/train_v6.npz', 4524)}
alphas = [-10, -6, -3, 0, 3, 6, 10]
report = {}
for name, (data, idx) in seeds.items():
    ds = MelodyDataset(R / data); tok = ds.x[idx]
    with torch.no_grad(): z0 = m.encode(torch.tensor(tok.astype(np.int64))[None])[0].numpy()[0]
    bias = sampling_bias(tok, 0.0, 12, 'cpu'); want = int((tok[:32] >= ONSET_OFFSET).sum())
    d = out / name; d.mkdir(parents=True, exist_ok=True)
    tokens_to_midi(tok[:32], grid_hz=80 / 60 * 4).write(str(d / 'original.mid'))
    report[name] = {'start_sd': float((z0 @ ax - mu) / sig), 'original': {k: round(v, 3) for k, v in describe(tok[:32]).items()}}
    for a in alphas:
        z = z0 + a * sig * ax
        torch.manual_seed(1000 + a)
        with torch.no_grad():
            C = m.generate(torch.tensor(np.repeat(z[None], 300, 0), dtype=torch.float32), steps=32, temperature=0.9, top_p=0.98, logit_bias=bias).numpy()
        n = (C >= ONSET_OFFSET).sum(1); near = np.flatnonzero(np.abs(n - want) == np.abs(n - want).min())
        F = np.array([[describe(c)['syncopation'], describe(c)['chromaticism']] for c in C])
        # closest note count, then most typical draw
        best = C[near[np.argmin((((F[near] - F.mean(0)) / (F.std(0) + 1e-8)) ** 2).sum(1))]]
        tokens_to_midi(best, grid_hz=80 / 60 * 4).write(str(d / f'a{a:+d}.mid'))
        report[name][a] = {'pos_sd': round(report[name]['start_sd'] + a, 2), 'notes': int((best >= ONSET_OFFSET).sum()),
                           'mean_sync_all': round(float(F[:, 0].mean()), 3), 'mean_chroma_all': round(float(F[:, 1].mean()), 3)}
(out / 'report.json').write_text(json.dumps(report, indent=1))
for k, v in report.items():
    print(k, 'start', round(v['start_sd'], 2), 'orig sync', v['original']['syncopation'])
    # for a in alphas: print('  ', a, v[a])
