#!/usr/bin/env python3
import argparse
import sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.labels import LabelSet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    xs, srcs, fields = ([], [], None)
    values = {}
    for path in args.inputs:
        blob = np.load(path, allow_pickle=True)
        labels = LabelSet.from_npz(blob)
        fields = labels.fields if fields is None else [f for f in fields if f in labels.fields]
        xs.append(blob['x'])
        srcs.append(blob['source_file'])
        for f in labels.fields:
            values.setdefault(f, []).append([labels.names[f][c] for c in labels.codes[f]])
    labels = LabelSet({f: [v for part in values[f] for v in part] for f in fields})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, x=np.concatenate(xs).astype(np.int16), source_file=np.concatenate(srcs), **labels.to_npz_dict()
    )
    print(f'wrote {args.out}: {sum((len(x) for x in xs))} windows')
    print('corpus:', dict(Counter({k: v for k, v in labels.counts('corpus').items() if v})))


if __name__ == '__main__':
    main()
