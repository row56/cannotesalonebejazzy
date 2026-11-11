import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.repr import tokens_to_midi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('data/processed/train.npz'))
    ap.add_argument('--out', type=Path, default=Path('samples/listen'))
    ap.add_argument('-n', type=int, default=20)
    ap.add_argument('--field', default='genre', help='label field to filter on')
    ap.add_argument('--genre', default=None, help='value of --field to keep, e.g. jazz')
    ap.add_argument('--bpm', type=float, default=110.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    blob = np.load(args.data, allow_pickle=True)
    x, source = (blob['x'], blob['source_file'])
    idx = np.arange(len(x))
    if args.genre is not None:
        names = list(blob[f'label_{args.field}_names'])
        if args.genre not in names:
            raise SystemExit(f'unknown {args.field} {args.genre!r}; have {names}')
        idx = idx[blob[f'label_{args.field}'][idx] == names.index(args.genre)]
        if not len(idx):
            raise SystemExit(f'no windows with {args.field}={args.genre}')
    pick = np.random.default_rng(args.seed).choice(idx, size=min(args.n, len(idx)), replace=False)
    args.out.mkdir(parents=True, exist_ok=True)
    grid_hz = args.bpm / 60 * 4
    for rank, i in enumerate(pick):
        tokens_to_midi(x[i], grid_hz=grid_hz).write(str(args.out / f'{rank:02d}_idx{i}.mid'))
        print(f'{rank:02d}  {source[i]}')
    print(f'\nwrote {len(pick)} files to {args.out}')


if __name__ == '__main__':
    main()
