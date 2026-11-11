#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.attributes import chromaticism, leap_rate, mean_interval, pitch_range
from melodicvae.data import MelodyDataset
from melodicvae.model import MelodyVAE, VAEConfig
from melodicvae.repr import ONSET_OFFSET, STEPS_PER_BAR
from melodicvae.style import StyleSpace, encode_dataset
from melodicvae.train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--styles', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--alphas', type=float, nargs='*', default=[-3, -2, 0, 2, 3])
    ap.add_argument('--draws', type=int, default=200)
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--bars', type=int, default=2)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.92)
    ap.add_argument('--sigma-sample', type=int, default=20000)
    args = ap.parse_args()
    device = pick_device()
    blob = torch.load(args.ckpt, map_location=device)
    model = MelodyVAE(VAEConfig(**blob['cfg'].get('model', {}))).to(device)
    model.load_state_dict(blob['model'])
    model.eval()
    ds = MelodyDataset(args.data)
    clip_steps = min(args.bars * STEPS_PER_BAR, ds.x.shape[1])
    space = StyleSpace.load(args.styles)
    axis = space.contrast_axis('jazziness', 'jazz', 'classical')
    rng = np.random.default_rng(0)
    sample = rng.choice(len(ds), size=min(args.sigma_sample, len(ds)), replace=False)
    sub = torch.utils.data.Subset(ds, sample.tolist())
    z_sample = encode_dataset(model, DataLoader(sub, batch_size=256), device)
    sigma = float((z_sample @ axis).std())
    print(f'sigma={sigma:.3f}  norm(z) median={np.median(np.linalg.norm(z_sample, axis=1)):.2f}')
    eligible = np.array(
        [
            chromaticism(w) <= 0.12 and pitch_range(w) <= 14 and (mean_interval(w) <= 4) and (leap_rate(w) <= 0.1)
            for w in ds.x[:60000]
        ]
    )
    pool = np.flatnonzero(eligible)
    picks = rng.choice(pool, args.seeds, replace=False)
    all_counts = {a: [] for a in args.alphas}
    for rank, index in enumerate(picks):
        tokens = ds.x[index]
        with torch.no_grad():
            mu, _ = model.encode(torch.tensor(tokens.astype(np.int64), device=device).unsqueeze(0))
        z0 = mu.cpu().numpy()[0]
        target = int((tokens[:clip_steps] >= ONSET_OFFSET).sum())
        row = []
        for a in args.alphas:
            z = torch.tensor(np.atleast_2d(z0 + a * sigma * axis), dtype=torch.float32, device=device)
            counts = []
            for j in range(args.draws):
                torch.manual_seed(10000 * rank + 100 * int(a * 10) + j)
                out = (
                    model.generate(z, steps=clip_steps, temperature=args.temperature, top_p=args.top_p).cpu().numpy()[0]
                )
                counts.append(int((out >= ONSET_OFFSET).sum()))
            counts = np.array(counts)
            all_counts[a].extend(counts.tolist())
            row.append(f'a{a:+.0f}: {counts.mean():5.1f} +-{counts.std():4.1f} [{counts.min():2d},{counts.max():2d}]')
        print(f'seed{rank}  own={target:2d}  ' + '  '.join(row))
    print('\npooled over seeds')
    for a in args.alphas:
        c = np.array(all_counts[a])
        print(
            f'  a{a:+.0f}  mean {c.mean():5.1f}  sd {c.std():4.1f}  p5-p95 [{np.percentile(c, 5):.0f},{np.percentile(c, 95):.0f}]'
        )
    lo, hi = (np.array(all_counts[args.alphas[0]]), np.array(all_counts[args.alphas[-1]]))
    band = (max(lo.min(), hi.min()), min(lo.max(), hi.max()))
    if band[0] <= band[1]:
        f_lo = ((lo >= band[0]) & (lo <= band[1])).mean()
        f_hi = ((hi >= band[0]) & (hi <= band[1])).mean()
        print(
            f'\noverlap band {band[0]}-{band[1]} notes: {f_lo * 100:.1f}% of a{args.alphas[0]:+.0f} draws, {f_hi * 100:.1f}% of a{args.alphas[-1]:+.0f} draws'
        )
    else:
        print('\nno overlap at all between the extreme positions')


if __name__ == '__main__':
    main()
