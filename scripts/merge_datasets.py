#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path
import numpy as np

UNKNOWN = 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    blobs = [np.load(p, allow_pickle=True) for p in args.inputs]
    seq_lens = {b['x'].shape[1] for b in blobs}
    if len(seq_lens) > 1:
        raise SystemExit(f'seq_len mismatch across inputs: {seq_lens}')
    fields = [k[len('label_') :] for k in blobs[0].files if k.startswith('label_') and (not k.endswith('_names'))]
    merged = {
        'x': np.concatenate([b['x'] for b in blobs]).astype(np.int16),
        'source_file': np.concatenate([b['source_file'] for b in blobs]),
    }
    for field in fields:
        vocab = sorted({str(n) for b in blobs for n in b.get(f'label_{field}_names', [])})
        index = {n: i for i, n in enumerate(vocab)}
        codes = []
        for b in blobs:
            key = f'label_{field}'
            if key not in b.files:
                codes.append(np.full(len(b['x']), index[UNKNOWN], dtype=np.int32))
                continue
            names = [str(n) for n in b[f'{key}_names']]
            lookup = np.array([index[n] for n in names], dtype=np.int32)
            codes.append(lookup[b[key]])
        merged[f'label_{field}'] = np.concatenate(codes)
        merged[f'label_{field}_names'] = np.array(vocab, dtype=object)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **merged)
    print(f'wrote {args.out}')
    print(f"  windows={len(merged['x'])}  seq_len={merged['x'].shape[1]}")
    for field in ('corpus', 'genre'):
        if f'label_{field}' not in merged:
            continue
        names = merged[f'label_{field}_names']
        counts = Counter((names[c] for c in merged[f'label_{field}']))
        print(f'  {field}: ' + ', '.join((f'{k}={v}' for k, v in counts.most_common())))
    if 'label_artist' in merged and len(blobs) > 1:
        per = []
        for b in blobs:
            names = [str(n) for n in b['label_artist_names']]
            per.append({names[c] for c in b['label_artist']} - {UNKNOWN})
        shared = per[-1].intersection(*per[:-1])
        print(
            f'  artists shared between last input and the rest: {len(shared)}'
            + (' -> ' + ', '.join(sorted(shared)) if shared else '')
        )


if __name__ == '__main__':
    main()
