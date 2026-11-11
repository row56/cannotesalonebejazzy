#!/usr/bin/env python3
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.data import midi_to_tokens, window
from melodicvae.labels import UNKNOWN, LabelSet, era_of
from melodicvae.repr import normalise_key

FIELDS = ['corpus', 'genre', 'artist', 'era', 'album', 'setting']


def load_metadata(raw: Path):
    meta_path = raw / 'metadata.csv'
    if meta_path.exists():
        with meta_path.open(newline='') as fh:
            rows = list(csv.DictReader(fh))
        print(f'read {len(rows)} rows from {meta_path}')
        return {raw / r['path']: {f: r.get(f, UNKNOWN) for f in FIELDS} for r in rows}
    print(f'no {meta_path}; inferring labels from directory layout')
    out = {}
    for path in raw.rglob('*'):
        if path.suffix.lower() not in {'.mid', '.midi'}:
            continue
        parts = path.relative_to(raw).parts
        genre = parts[0] if len(parts) > 1 else UNKNOWN
        corpus = parts[1] if len(parts) > 2 else genre
        artist = parts[-2].replace('_', ' ') if len(parts) > 2 else UNKNOWN
        out[path] = {
            'corpus': corpus,
            'genre': genre,
            'artist': artist,
            'era': era_of(artist) if genre == 'classical' else UNKNOWN,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=Path, default=Path('data/raw'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seq-len', type=int, default=64, help='tokens per window (64 = 4 bars)')
    ap.add_argument('--hop', type=int, default=32)
    ap.add_argument('--steps-per-beat', type=int, default=4)
    ap.add_argument('--max-per-file', type=int, default=64, help='cap windows per file so long pieces do not dominate')
    ap.add_argument(
        '--max-per-artist',
        type=int,
        default=2000,
        help='cap windows per artist so prolific performers do not define a genre; 0 disables',
    )
    ap.add_argument(
        '--no-key-norm',
        dest='key_norm',
        action='store_false',
        help='keep the original key instead of transposing each piece to C',
    )
    ap.add_argument('--min-onsets', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='debug: only read N files')
    args = ap.parse_args()
    meta = load_metadata(args.raw)
    files = sorted(meta)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f'no midi files under {args.raw}')
    windows = []
    rows = []
    sources = []
    skipped = 0
    rng = np.random.default_rng(0)
    for path in tqdm(files, desc='parsing'):
        if not path.exists():
            skipped += 1
            continue
        tokens = midi_to_tokens(path, args.steps_per_beat)
        if tokens is None:
            skipped += 1
            continue
        if args.key_norm:
            tokens = normalise_key(tokens)
        ws = window(tokens, args.seq_len, args.hop, min_onsets=args.min_onsets)
        if not ws:
            skipped += 1
            continue
        if len(ws) > args.max_per_file:
            ws = [ws[i] for i in rng.choice(len(ws), args.max_per_file, replace=False)]
        windows.extend(ws)
        rows.extend([meta[path]] * len(ws))
        sources.extend([str(path.relative_to(args.raw))] * len(ws))
    if not windows:
        raise SystemExit(
            f'every file was skipped ({skipped} of {len(files)}). Check that the midi files parse and carry a usable tempo map.'
        )
    if args.max_per_artist:
        by_artist = defaultdict(list)
        for i, r in enumerate(rows):
            by_artist[r['artist']].append(i)
        keep = []
        for idxs in by_artist.values():
            if len(idxs) > args.max_per_artist:
                idxs = rng.choice(idxs, args.max_per_artist, replace=False).tolist()
            keep.extend(idxs)
        keep.sort()
        dropped = len(windows) - len(keep)
        windows = [windows[i] for i in keep]
        rows = [rows[i] for i in keep]
        sources = [sources[i] for i in keep]
        print(f'per-artist cap {args.max_per_artist}: dropped {dropped} windows')
    x = np.stack(windows).astype(np.int16)
    labels = LabelSet({f: [r[f] for r in rows] for f in FIELDS})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, x=x, source_file=np.array(sources, dtype=object), **labels.to_npz_dict())
    print(f'\nwrote {args.out}')
    print(f'  windows={len(x)}  seq_len={x.shape[1]}  files_skipped={skipped}/{len(files)}')
    for f in FIELDS:
        counts = Counter({k: v for k, v in labels.counts(f).items() if v})
        top = counts.most_common(8)
        print(f'  {f}: {len(counts)} values, top: ' + ', '.join((f'{k}={v}' for k, v in top)))
    for f in ('artist', 'era'):
        viable = labels.viable(f, 100)
        print(f'  {f} values with >=100 windows (axis-eligible): {len(viable)}')


if __name__ == '__main__':
    main()
