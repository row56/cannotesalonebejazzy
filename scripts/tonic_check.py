import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.repr import estimate_tonic

NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'none']


def tonic_histogram(x: np.ndarray) -> np.ndarray:
    counts = np.zeros(13)
    for window in x:
        tonic = estimate_tonic(window)
        counts[12 if tonic is None else tonic] += 1
    return counts / counts.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/train_v6.npz')
    ap.add_argument('--genre-a', default='classical')
    ap.add_argument('--genre-b', default='jazz')
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    x, y = (d['x'], d['label_genre'])
    names = list(d['label_genre_names'])
    ha = tonic_histogram(x[y == names.index(args.genre_a)])
    hb = tonic_histogram(x[y == names.index(args.genre_b)])
    tvd = 0.5 * np.abs(ha - hb).sum()
    print(f'{args.data}: {len(x)} windows')
    print(f"{'tonic':>6} {args.genre_a:>10} {args.genre_b:>10}")
    for i, name in enumerate(NAMES):
        print(f'{name:>6} {ha[i]:>10.3f} {hb[i]:>10.3f}')
    print(f'\ntotal variation distance {tvd:.3f}')
    print(f'best balanced accuracy from tonic alone {0.5 + tvd / 2:.3f}')


if __name__ == '__main__':
    main()
