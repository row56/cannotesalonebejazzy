#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.attributes import DESCRIPTORS, describe_batch, leap_rate, mean_interval
from melodicvae.data import MelodyDataset
from melodicvae.model import MelodyVAE, VAEConfig
from melodicvae.style import StyleSpace, auc, axis_auc, descriptor_correlation, encode_dataset
from melodicvae.train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--field', default='genre', help='label field for categorical axes: genre|artist|era|corpus')
    ap.add_argument(
        '--min-count', type=int, default=100, help='windows required before a label value gets its own axis'
    )
    ap.add_argument(
        '--contrast', nargs='*', default=[], metavar='NAME:POS:NEG', help='signed axes, e.g. jazziness:jazz:classical'
    )
    ap.add_argument(
        '--attribute-axes', action='store_true', default=True, help='also fit a regression axis per musical descriptor'
    )
    ap.add_argument(
        '--fit-exclude',
        nargs='*',
        default=[],
        metavar='FIELD:VALUE',
        help='hold a group out of the axis fit, then report where it lands (e.g. corpus:maestro). This is the confound test: an axis fit without ever seeing that corpus should still place it on the right side if it encodes music and not the recording pipeline.',
    )
    ap.add_argument(
        '--project-out',
        nargs='*',
        default=[],
        metavar='FIELD:A:B',
        help='remove the direction between two groups from every latent before fitting, e.g. corpus:giantmidi:maestro to subtract the transcriber contrast. Estimate it on a genre other than the held-out corpus, otherwise the correction is fit on the thing it is tested against.',
    )
    ap.add_argument(
        '--project-out-attr',
        nargs='*',
        default=[],
        metavar='DESCRIPTOR',
        help="remove a musical descriptor's regression direction from every latent before fitting, e.g. note_density. Fit on the retained windows only, so a held-out corpus never informs the correction it is then judged by.",
    )
    ap.add_argument(
        '--max-mean-interval',
        type=float,
        default=0,
        help='drop every window whose average step exceeds this many semitones, before fitting and before held-out scoring. With --max-leap-rate this keeps real melodies and drops skyline tops of piano accompaniment. 0 disables',
    )
    ap.add_argument(
        '--max-leap-rate',
        type=float,
        default=0,
        help='drop every window whose share of leaps above a fifth exceeds this. 0 disables',
    )
    args = ap.parse_args()
    device = pick_device()
    blob = torch.load(args.ckpt, map_location=device)
    model = MelodyVAE(VAEConfig(**blob['cfg'].get('model', {})))
    model.load_state_dict(blob['model'])
    ds = MelodyDataset(args.data, augment_transpose=0)
    if args.field not in ds.fields:
        raise SystemExit(f'unknown field {args.field!r}; have {ds.fields}')
    z = encode_dataset(model, DataLoader(ds, batch_size=256), device)
    # print(z[:3])
    print(f'encoded {len(z)} windows into {z.shape[1]}-d latents')
    keep = np.ones(len(z), dtype=bool)
    if args.max_mean_interval > 0:
        keep &= np.array([mean_interval(w) for w in ds.x]) <= args.max_mean_interval
    if args.max_leap_rate > 0:
        keep &= np.array([leap_rate(w) for w in ds.x]) <= args.max_leap_rate
    if not keep.all():
        corpus_codes, corpus_names = (ds.codes('corpus'), ds.names('corpus'))
        for i, cname in enumerate(corpus_names):
            n = int((corpus_codes == i).sum())
            if n:
                print(f'melodic filter {cname}: kept {int(keep[corpus_codes == i].sum())} of {n}')
    z, x = (z[keep], ds.x[keep])

    def codes(field):
        return ds.codes(field)[keep]

    y, names = (codes(args.field), ds.names(args.field))
    nuisance = []
    for spec in args.project_out:
        nfield, a, b = spec.split(':')
        if nfield not in ds.fields:
            raise SystemExit(f'unknown field {nfield!r}; have {ds.fields}')
        nnames = ds.names(nfield)
        for side in (a, b):
            if side not in nnames:
                raise SystemExit(f'{side!r} is not a value of {nfield!r}; have {nnames}')
        ncodes = codes(nfield)
        u = z[ncodes == nnames.index(a)].mean(0) - z[ncodes == nnames.index(b)].mean(0)
        for prev in nuisance:
            u = u - u @ prev * prev
        norm = float(np.linalg.norm(u))
        if norm < 1e-08:
            print(f'project-out {spec}: direction is degenerate, skipping')
            continue
        u = u / norm
        before = float(z.var(0).sum())
        z = z - np.outer(z @ u, u)
        after = float(z.var(0).sum())
        nuisance.append(u)
        print(f'project-out {spec}: removed {100 * (1 - after / before):.1f}% of latent variance')
    held = {}
    fit_mask = np.ones(len(z), dtype=bool)
    for spec in args.fit_exclude:
        hfield, hvalue = spec.split(':')
        if hfield not in ds.fields:
            raise SystemExit(f'unknown field {hfield!r}; have {ds.fields}')
        hnames = ds.names(hfield)
        if hvalue not in hnames:
            raise SystemExit(f'{hvalue!r} is not a value of {hfield!r}; have {hnames}')
        mask = codes(hfield) == hnames.index(hvalue)
        held[spec] = mask
        fit_mask &= ~mask
        print(f'holding out {spec}: {int(mask.sum())} windows')
    if not fit_mask.any():
        raise SystemExit('--fit-exclude removed every window')
    descriptors = describe_batch(x)
    for dname in args.project_out_attr:
        if dname not in DESCRIPTORS:
            raise SystemExit(f'unknown descriptor {dname!r}; have {list(DESCRIPTORS)}')
        col = list(DESCRIPTORS).index(dname)
        zc = z[fit_mask] - z[fit_mask].mean(0)
        v = descriptors[fit_mask, col].astype(float)
        v = (v - v.mean()) / (v.std() + 1e-08)
        gram = zc.T @ zc + 0.001 * len(zc) * np.eye(zc.shape[1])
        u = np.linalg.solve(gram, zc.T @ v)
        for prev in nuisance:
            u = u - u @ prev * prev
        norm = float(np.linalg.norm(u))
        if norm < 1e-08:
            print(f'project-out-attr {dname}: direction is degenerate, skipping')
            continue
        u = u / norm
        before = float(z.var(0).sum())
        z = z - np.outer(z @ u, u)
        after = float(z.var(0).sum())
        nuisance.append(u)
        print(f'project-out-attr {dname}: removed {100 * (1 - after / before):.1f}% of latent variance')
    space = StyleSpace.fit(z[fit_mask], y[fit_mask], names, min_count=args.min_count)
    categorical = list(space.axes)
    print(
        f'field {args.field!r}: {len(names)} values, {len(categorical)} with >={args.min_count} windows'
        + (f' (fit on {int(fit_mask.sum())} of {len(z)})' if held else '')
    )
    for spec in args.contrast:
        name, pos, neg = spec.split(':')
        for side in (pos, neg):
            if side not in names:
                raise SystemExit(f'{side!r} is not a value of {args.field!r}; have {names}')
        space.contrast_axis(name, pos, neg)
        categorical.append(name)
    zf, yf = (z[fit_mask], y[fit_mask])
    if args.attribute_axes:
        for i, dname in enumerate(DESCRIPTORS):
            space.attribute_axis(f'attr:{dname}', zf, descriptors[fit_mask, i])
    args.out.mkdir(parents=True, exist_ok=True)
    space.save(args.out / 'styles.npz')
    report = {
        'field': args.field,
        'fit_excluded': list(held),
        'projected_out': list(args.project_out),
        'projected_out_attr': list(args.project_out_attr),
        'melodic_filter': {
            'max_mean_interval': args.max_mean_interval,
            'max_leap_rate': args.max_leap_rate,
            'kept': int(keep.sum()),
            'of': int(len(keep)),
        },
        'separability': {},
        'descriptor_correlation': {},
        'confounds': {},
        'heldout': {},
    }
    print('\n1. separability (AUC of projection, value vs rest)')
    for i, name in enumerate(names):
        if name not in space.axes:
            continue
        value = axis_auc(zf, yf, space.axes[name], i)
        report['separability'][name] = round(value, 3)
        print(f'   {name:<26} n={space.counts[i]:>6}  AUC={value:.3f}')
    print('\n2. axis -> descriptor correlation (Pearson r)')
    print('   ' + f"{'axis':<24}" + ' '.join((f'{d[:9]:>10}' for d in DESCRIPTORS)))
    for name, axis in space.axes.items():
        r = descriptor_correlation(zf, descriptors[fit_mask], axis)
        report['descriptor_correlation'][name] = {d: round(float(v), 3) for d, v in zip(DESCRIPTORS, r)}
        print(f'   {name:<24}' + ' '.join((f'{v:>10.2f}' for v in r)))
    print('\n3. confounds: |cos| between each style axis and each attribute axis')
    attr_axes = {n: v for n, v in space.axes.items() if n.startswith('attr:')}
    if attr_axes:
        for name in categorical:
            if name not in space.axes:
                continue
            sims = {a[5:]: abs(float(space.axes[name] @ v)) for a, v in attr_axes.items()}
            worst, value = max(sims.items(), key=lambda kv: kv[1])
            report['confounds'][name] = {k: round(v, 3) for k, v in sims.items()}
            flag = '  <-- largely a descriptor axis' if value > 0.7 else ''
            print(f'   {name:<24} closest={worst} |cos|={value:.2f}{flag}')
    for spec, hmask in held.items():
        print(f'\n4. held-out {spec}: projection onto axes it never helped fit')
        print(f"   {'axis':<20}" + ''.join((f"{'vs ' + n:>18}" for n in names if n in space.axes)))
        entry = {}
        for name in categorical:
            if name not in space.axes:
                continue
            scores = z @ space.axes[name]
            row, cells = ({}, [])
            for i, other in enumerate(names):
                if other not in space.axes:
                    continue
                value = auc(scores[hmask], scores[fit_mask & (y == i)])
                row[other] = round(value, 3)
                cells.append(f'{value:>18.3f}')
            entry[name] = row
            print(f'   {name:<20}' + ''.join(cells))
        report['heldout'][spec] = entry
        print(
            "   read: on the axis for its own genre, ~0.5 vs that genre's other corpus means\n   the held-out set looks like the genre; >>0.5 vs the other genre means the\n   axis transfers. Both together are the confound passing."
        )
    (args.out / 'axis_report.json').write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out / 'styles.npz'} and {args.out / 'axis_report.json'}")


if __name__ == '__main__':
    main()
