#!/usr/bin/env python3
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.labels import LabelSet

P = Path('data/processed')


def load(name, corpus=None, fmt=None):
    blob = np.load(P / f'{name}.npz', allow_pickle=True)
    labels = LabelSet.from_npz(blob)
    rows = [{f: labels.names[f][labels.codes[f][i]] for f in labels.fields} for i in range(len(blob['x']))]
    for r in rows:
        if corpus:
            r['corpus'] = corpus
        if fmt:
            r['setting'] = fmt
    return {'x': blob['x'], 'src': blob['source_file'], 'rows': rows}


def take(d, idx):
    idx = np.asarray(idx, dtype=int)
    return {'x': d['x'][idx], 'src': d['src'][idx], 'rows': [d['rows'][i] for i in idx]}


def group_split(d, field, test_frac):
    groups = defaultdict(list)
    for i, r in enumerate(d['rows']):
        groups[r[field]].append(i)
    total, n_test = (len(d['rows']), 0)
    train, test = ([], [])
    for k, (g, idx) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        if n_test / total < test_frac and k % 2 == 1:
            test.extend(idx)
            n_test += len(idx)
        else:
            train.extend(idx)
    return (take(d, sorted(train)), take(d, sorted(test)))


def cap_groups(d, field, cap, rng):
    groups = defaultdict(list)
    for i, r in enumerate(d['rows']):
        groups[r[field]].append(i)
    keep = []
    for idx in groups.values():
        keep.extend(rng.choice(idx, cap, replace=False).tolist() if len(idx) > cap else idx)
    return take(d, sorted(keep))


def subsample(d, n, rng):
    if len(d['rows']) <= n:
        return d
    return take(d, sorted(rng.choice(len(d['rows']), n, replace=False).tolist()))


def merge(parts):
    fields = ['corpus', 'genre', 'artist', 'era', 'album', 'setting']
    x = np.concatenate([p['x'] for p in parts]).astype(np.int16)
    src = np.concatenate([p['src'] for p in parts])
    labels = LabelSet({f: [r.get(f, 'unknown') for p in parts for r in p['rows']] for f in fields})
    return (x, src, labels)


def save(path, parts):
    x, src, labels = merge(parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, source_file=src, **labels.to_npz_dict())
    counts = Counter(((r['corpus'], r['genre'], r['setting']) for p in parts for r in p['rows']))
    print(f'wrote {path}: {len(x)} windows')
    for k, v in sorted(counts.items()):
        print(f'  {k[0]:<16} {k[1]:<10} {k[2]:<12} {v}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-train', type=Path, required=True)
    ap.add_argument('--out-eval', type=Path, required=True)
    ap.add_argument('--test-frac', type=float, default=0.4)
    ap.add_argument('--max-per-artist', type=int, default=600)
    ap.add_argument(
        '--with-wjazzd', action='store_true', help='add wjazzd to the jazz performance side instead of holding it out'
    )
    ap.add_argument(
        '--density-match', action='store_true', help='give all four training cells the same notes-per-window histogram'
    )
    ap.add_argument('--bin-width', type=int, default=4)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    jtd_tr, jtd_te = group_split(load('jtd_beats', fmt='performance'), 'artist', args.test_frac)
    asap_tr, asap_te = group_split(load('asap_beats', fmt='performance'), 'artist', args.test_frac)
    pdmx = load('pdmx_jazz', fmt='sheet')
    for r, s in zip(pdmx['rows'], pdmx['src']):
        if r['album'] in ('na', 'unknown', ''):
            r['album'] = str(s)
    pdmx_tr, pdmx_te = group_split(pdmx, 'album', args.test_frac)
    lieder = load('lieder', fmt='sheet')
    quartets = load('quartets', fmt='sheet')
    wjazzd = load('wjazzd_beats', fmt='performance')
    for d in (jtd_te, asap_te, pdmx_te):
        for r in d['rows']:
            r['corpus'] += '_test'
    cells = {
        ('jazz', 'sheet'): [pdmx_tr],
        ('jazz', 'performance'): [jtd_tr] + ([wjazzd] if args.with_wjazzd else []),
        ('classical', 'sheet'): [lieder],
        ('classical', 'performance'): [asap_tr],
    }
    capped = {}
    for key, parts in cells.items():
        capped[key] = [cap_groups(p, 'artist', args.max_per_artist, rng) for p in parts]
    if args.density_match:
        bins = np.arange(0, 66, args.bin_width)
        flat = {
            k: {
                'x': np.concatenate([p['x'] for p in v]),
                'src': np.concatenate([p['src'] for p in v]),
                'rows': [r for p in v for r in p['rows']],
            }
            for k, v in capped.items()
        }
        which = {k: np.digitize((d['x'] >= 2).sum(1), bins) for k, d in flat.items()}
        target = {b: min((int((w == b).sum()) for w in which.values())) for b in range(len(bins) + 1)}
        for k, d in flat.items():
            keep = []
            for b, n_b in target.items():
                pool = np.flatnonzero(which[k] == b)
                if n_b:
                    keep.extend(rng.choice(pool, n_b, replace=False).tolist())
            capped[k] = [take(d, sorted(keep))]
    sizes = {k: sum((len(p['rows']) for p in v)) for k, v in capped.items()}
    print('cell sizes after per-artist cap' + (' and density match' if args.density_match else '') + ':', sizes)
    n = min(sizes.values())
    print(f'subsampling every cell to {n} windows')
    train = []
    for key, parts in capped.items():
        tot = sum((len(p['rows']) for p in parts))
        for p in parts:
            train.append(subsample(p, int(round(n * len(p['rows']) / tot)), rng))
    save(args.out_train, train)
    held = [jtd_te, asap_te, pdmx_te, quartets] + ([] if args.with_wjazzd else [wjazzd])
    save(args.out_eval, train + held)


if __name__ == '__main__':
    main()
