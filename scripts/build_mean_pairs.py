#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--ckpt', default='runs/base_v6/best.pt')
    ap.add_argument('--styles', default='runs/base_v6/axes_dm_heldout/styles.npz')
    ap.add_argument(
        '--data', default='data/processed/train_v7.npz', help='axis-fit set: defines sigma and the two genre means'
    )
    ap.add_argument('--seed-data', default=None, help='where seeds are drawn from (default: --data)')
    ap.add_argument('--min-shift', type=float, default=1.0)
    ap.add_argument('--cross', action='store_true', help="steer each seed only to the OTHER genre's mean")
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--indices', type=int, nargs='*', default=[])
    ap.add_argument('--per-side', type=int, default=0)
    ap.add_argument('--exclude-manifest', nargs='*', default=[], help='manifests whose seed indices must not be reused')
    ap.add_argument('--bars', type=int, default=2)
    ap.add_argument('--bpm', type=float, default=80)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top-p', type=float, default=0.98)
    ap.add_argument('--register-span', type=int, default=12)
    ap.add_argument('--max-seed-chromaticism', type=float, default=0.12)
    ap.add_argument('--max-seed-range', type=float, default=12)
    ap.add_argument('--candidates', type=int, default=200)
    ap.add_argument('--match-tol', type=int, default=1)
    ap.add_argument('--seed', type=int, default=11)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo))
    from melodicvae.attributes import chromaticism, describe, pitch_range
    from melodicvae.data import MelodyDataset
    from melodicvae.model import MelodyVAE, VAEConfig
    from melodicvae.repr import ONSET_OFFSET, STEPS_PER_BAR, tokens_to_midi
    from melodicvae.style import StyleSpace, encode_dataset

    sys.path.insert(0, str(args.repo / 'scripts'))
    from build_demo import onset_counts, sampling_bias

    blob = torch.load(args.repo / args.ckpt, map_location='cpu', weights_only=False)
    model = MelodyVAE(VAEConfig(**blob['cfg'].get('model', {})))
    model.load_state_dict(blob['model'])
    model.eval()
    ds = MelodyDataset(args.repo / args.data, augment_transpose=0)
    axis = StyleSpace.load(args.repo / args.styles).contrast_axis('jazziness', 'jazz', 'classical')
    z_all = encode_dataset(model, DataLoader(ds, batch_size=256), 'cpu')
    proj = z_all @ axis
    sigma = float(proj.std())
    genre = ds.codes('genre')
    gnames = ds.names('genre')
    target = {
        'classical': float(proj[genre == gnames.index('classical')].mean()),
        'jazz': float(proj[genre == gnames.index('jazz')].mean()),
    }
    mu = float(proj.mean())
    sd = lambda v: (v - mu) / sigma
    print(
        f"sigma {sigma:.4f}  classical mean {sd(target['classical']):+.2f} SD  jazz mean {sd(target['jazz']):+.2f} SD"
    )
    sds = MelodyDataset(args.repo / (args.seed_data or args.data), augment_transpose=0)
    z_seed = z_all if args.seed_data is None else encode_dataset(model, DataLoader(sds, batch_size=256), 'cpu')
    sgenre, sgnames = (sds.codes('genre'), sds.names('genre'))
    picks = list(args.indices)
    if args.per_side:
        used, used_src = (set(), set())
        for path in args.exclude_manifest:
            seeds = json.load(open(args.repo / path))['seeds']
            used |= {s['index'] for s in seeds}
            used_src |= {s.get('source_file', '') for s in seeds}
        counts = onset_counts(sds.x)
        lo, hi = np.percentile(counts, [40, 90])
        ok = (counts >= lo) & (counts <= hi)
        ok &= np.array([chromaticism(w) for w in sds.x]) <= args.max_seed_chromaticism
        ok &= np.array([pitch_range(w) for w in sds.x]) <= args.max_seed_range
        if args.seed_data is None:
            ok &= ~np.isin(np.arange(len(sds)), list(used))
        src = np.array([str(f) for f in sds.source_file])
        ok &= ~np.isin(src, list(used_src))
        ps = sd(z_seed @ axis)
        far = np.where(sgenre == sgnames.index('classical'), sd(target['jazz']) - ps, ps - sd(target['classical']))
        ok &= far >= args.min_shift
        rng = np.random.default_rng(args.seed)
        for side in ('classical', 'jazz'):
            pool = np.flatnonzero(ok & (sgenre == sgnames.index(side)))
            chosen, seen = ([], set())
            for i in rng.permutation(pool):
                if src[i] not in seen:
                    chosen.append(int(i))
                    seen.add(src[i])
                if len(chosen) == args.per_side:
                    break
            picks += chosen
    clip_steps = args.bars * STEPS_PER_BAR
    grid_hz = args.bpm / 60.0 * 4
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        'sigma': sigma,
        'targets_sd': {k: sd(v) for k, v in target.items()},
        'bpm': args.bpm,
        'bars': args.bars,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'candidates': args.candidates,
        'seeds': [],
    }
    for rank, index in enumerate(picks):
        tokens = sds.x[index]
        z0 = z_seed[index]
        p0 = float(z0 @ axis)
        bias = sampling_bias(tokens, 0.0, args.register_span, 'cpu')
        want = int((tokens[:clip_steps] >= ONSET_OFFSET).sum())
        stem = f'seed{rank:02d}'
        (args.out / stem).mkdir(exist_ok=True)
        tokens_to_midi(tokens[:clip_steps], grid_hz=grid_hz).write(str(args.out / stem / 'original.mid'))
        side = sgnames[sgenre[index]]
        entry = {
            'id': stem,
            'index': index,
            'side': side,
            'source_file': str(sds.source_file[index]) if sds.source_file is not None else '',
            'start_sd': sd(p0),
            'clips': {},
        }
        got = []
        goals = {k: v for k, v in target.items() if not args.cross or k != side}
        for k, (name, t) in enumerate(goals.items()):
            z = z0 + (t - p0) * axis
            zt = torch.tensor(np.repeat(z[None], args.candidates, 0), dtype=torch.float32)
            torch.manual_seed(args.seed * 100000 + rank * 10 + k)
            with torch.no_grad():
                cands = model.generate(
                    zt, steps=clip_steps, temperature=args.temperature, top_p=args.top_p, logit_bias=bias
                ).numpy()
            n = (cands >= ONSET_OFFSET).sum(1)
            near = np.flatnonzero(np.abs(n - want) == np.abs(n - want).min())
            feats = np.array([[describe(c)['syncopation'], describe(c)['chromaticism']] for c in cands])
            scale = feats.std(0) + 1e-08
            dist = (((feats[near] - feats.mean(0)) / scale) ** 2).sum(1)
            best = cands[int(near[np.argmin(dist)])]
            got.append(int((best >= ONSET_OFFSET).sum()))
            tokens_to_midi(best, grid_hz=grid_hz).write(str(args.out / stem / f'{name}.mid'))
            entry['clips'][name] = {
                'file': f'{stem}/{name}.mid',
                'shift_sd': (t - p0) / sigma,
                'descriptors': {a: round(float(b), 4) for a, b in describe(best).items()},
            }
        entry['notes'] = {'target': want, 'achieved': got}
        entry['density_matched'] = max((abs(g - want) for g in got)) <= args.match_tol
        manifest['seeds'].append(entry)
        print(f"{stem} {entry['side']:<9} start {sd(p0):+.2f} SD  notes {want}->{got}  {entry['source_file'][:55]}")
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
