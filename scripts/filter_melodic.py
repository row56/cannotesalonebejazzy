#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.attributes import leap_rate, mean_interval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-mean-interval', type=float, default=4.0)
    ap.add_argument('--max-leap-rate', type=float, default=0.1)
    args = ap.parse_args()
    blob = np.load(args.src, allow_pickle=True)
    x = blob['x']
    keep = np.array([mean_interval(w) <= args.max_mean_interval and leap_rate(w) <= args.max_leap_rate for w in x])
    out = {}
    for key in blob.files:
        arr = blob[key]
        out[key] = arr if key.endswith('_names') else arr[keep]
    if 'label_corpus' in blob.files:
        codes, names = (blob['label_corpus'], blob['label_corpus_names'])
        for i, name in enumerate(names):
            n = int((codes == i).sum())
            if n:
                print(f'  {name}: kept {int(keep[codes == i].sum())} of {n}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f'wrote {args.out}: {int(keep.sum())} of {len(x)} windows')


if __name__ == '__main__':
    main()
