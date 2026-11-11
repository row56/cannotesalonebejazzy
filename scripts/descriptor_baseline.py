import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.attributes import ONSET_OFFSET, beat_vacancy, syncopation

IN_CLASSICAL = ('lieder', 'asap')
IN_JAZZ = ('pdmx_jazz', 'jtd')
HELD_OUT = (
    ('quartets', 'classical'),
    ('asap_test', 'classical'),
    ('jtd_test', 'jazz'),
    ('pdmx_jazz_test', 'jazz'),
    ('wjazzd', 'jazz'),
)


def auc(a, b):
    a, b = (np.asarray(a, float), np.asarray(b, float))
    order = np.argsort(np.concatenate([a, b]), kind='mergesort')
    ranks = np.empty(len(a) + len(b), float)
    ranks[order] = np.arange(1, len(a) + len(b) + 1)
    vals = np.concatenate([a, b])[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return float((ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def onset_floor(tokens, steps_per_beat=4):
    n = int((np.asarray(tokens).ravel() >= ONSET_OFFSET).sum())
    slots = len(np.asarray(tokens).ravel()) // steps_per_beat
    return max(0, n - slots) / n if n else 0.0


def per_corpus(x, corpus, names):
    out = {}
    for ci, name in enumerate(names):
        m = corpus == ci
        if not m.sum():
            continue
        w = x[m]
        out[name] = {
            'n': int(m.sum()),
            'density': np.array([(t >= ONSET_OFFSET).sum() for t in w], float),
            'syncopation': np.array([syncopation(t) for t in w]),
            'beat_vacancy': np.array([beat_vacancy(t) for t in w]),
            'floor': np.array([onset_floor(t) for t in w]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/eval_v6.npz')
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    F = per_corpus(d['x'], d['label_corpus'], list(d['label_corpus_names']))
    print('== density and the arithmetic floor on syncopation ==')
    print(f"{'corpus':<16}{'windows':>8}{'notes/win':>11}{'syncopation':>13}{'forced min':>12}{'vacancy':>9}")
    for name in sorted(F, key=lambda k: -F[k]['density'].mean()):
        f = F[name]
        print(
            f"{name:<16}{f['n']:>8}{f['density'].mean():>11.1f}{f['syncopation'].mean():>13.3f}{f['floor'].mean():>12.3f}{f['beat_vacancy'].mean():>9.3f}"
        )
    for key in ('syncopation', 'beat_vacancy', 'density'):
        cl = np.concatenate([F[c][key] for c in IN_CLASSICAL if c in F])
        jz = np.concatenate([F[c][key] for c in IN_JAZZ if c in F])
        print(f'\n== {key} ==  in-sample separability (jazz vs classical): {auc(jz, cl):.3f}')
        print(f"{'held out':<16}{'genre':<11}{'vs classical':>13}{'vs jazz':>9}")
        for name, genre in HELD_OUT:
            if name not in F:
                continue
            v = F[name][key]
            print(f'{name:<16}{genre:<11}{auc(v, cl):>13.2f}{auc(v, jz):>9.2f}')


if __name__ == '__main__':
    main()
