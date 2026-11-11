#!/usr/bin/env python3
import argparse
import sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.labels import LabelSet
from melodicvae.repr import ONSET_OFFSET


def load(path):
    blob = np.load(path, allow_pickle=True)
    labels = LabelSet.from_npz(blob)
    rows = {f: [labels.names[f][c] for c in labels.codes[f]] for f in labels.fields}
    return (blob['x'], blob['source_file'], rows)


def cap_per_artist(artists, cap, rng):
    keep = []
    for a in set(artists):
        idx = [i for i, v in enumerate(artists) if v == a]
        keep.extend(rng.choice(idx, cap, replace=False).tolist() if len(idx) > cap else idx)
    return np.array(sorted(keep))


def density_match(jazz_counts, ref_counts, rng, bins):
    j_hist, _ = np.histogram(jazz_counts, bins)
    r_hist, _ = np.histogram(ref_counts, bins)
    r_share = r_hist / r_hist.sum()
    ok = r_share > 0
    scale = np.min(j_hist[ok] / r_share[ok])
    which = np.digitize(jazz_counts, bins) - 1
    keep = []
    for b in np.flatnonzero(ok):
        pool = np.flatnonzero(which == b)
        n = min(len(pool), int(np.floor(scale * r_share[b])))
        keep.extend(rng.choice(pool, n, replace=False).tolist())
    return np.array(sorted(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jazz', type=Path)
    ap.add_argument('classical', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-per-artist', type=int, default=600)
    ap.add_argument('--bin-width', type=int, default=4, help='onsets per histogram bin')
    ap.add_argument(
        '--no-density-match',
        dest='density_match',
        action='store_false',
        help='keep every capped jazz window; density is then controlled only by descriptor projection at axis-fit time',
    )
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    xj, sj, rj = load(args.jazz)
    xc, sc, rc = load(args.classical)
    k = cap_per_artist(rj['artist'], args.max_per_artist, rng)
    print(f'jazz per-artist cap {args.max_per_artist}: {len(xj)} -> {len(k)} windows')
    xj, sj, rj = (xj[k], sj[k], {f: [v[i] for i in k] for f, v in rj.items()})
    cj, cc = ((xj >= ONSET_OFFSET).sum(1), (xc >= ONSET_OFFSET).sum(1))
    print(f'onsets/window median: jazz {np.median(cj):.0f}, lieder {np.median(cc):.0f}')
    if args.density_match:
        bins = np.arange(0, xj.shape[1] + args.bin_width + 1, args.bin_width)
        k = density_match(cj, cc, rng, bins)
        print(f'jazz density match: {len(xj)} -> {len(k)} windows, jazz median now {np.median(cj[k]):.0f}')
        xj, sj, rj = (xj[k], sj[k], {f: [v[i] for i in k] for f, v in rj.items()})
    fields = [f for f in rj if f in rc]
    x = np.concatenate([xj, xc]).astype(np.int16)
    src = np.concatenate([sj, sc])
    labels = LabelSet({f: list(rj[f]) + list(rc[f]) for f in fields})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, x=x, source_file=src, **labels.to_npz_dict())
    print(f"wrote {args.out}: {len(x)} windows, genre {dict(Counter(labels.counts('genre')))}")
    print('top jazz artists after cap:', Counter(rj['artist']).most_common(5))


if __name__ == '__main__':
    main()
