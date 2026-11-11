#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.attributes import leap_rate, mean_interval
from melodicvae.data import window
from melodicvae.labels import LabelSet, era_of
from melodicvae.repr import ONSET_OFFSET, centre_octave, normalise_key, skyline

FIELDS = ['corpus', 'genre', 'artist', 'era', 'album', 'setting']
METERS = {'4/4', '2/4', '2/2'}
SPB = 4


def perf_tokens(midi_path: Path, ann: dict):
    import pretty_midi

    meters = {v[0] for v in ann['perf_time_signatures'].values()}
    if not meters or not meters <= METERS:
        return (None, 'meter')
    beats = np.asarray(ann['performance_beats'], dtype=float)
    downs = np.asarray(ann['performance_downbeats'], dtype=float)
    if len(beats) < 16 or not len(downs):
        return (None, 'beats')
    first_down = int(np.argmin(np.abs(beats - downs[0])))
    beat_idx = np.arange(len(beats)) - first_down
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return (None, 'parse')
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            if n.start < beats[0] or n.start > beats[-1]:
                continue
            s = int(round(np.interp(n.start, beats, beat_idx) * SPB))
            e = int(round(np.interp(min(n.end, beats[-1]), beats, beat_idx) * SPB))
            if s < 0:
                continue
            notes.append((s, max(e, s + 1), n.pitch))
    if len(notes) < 8:
        return (None, 'too short')
    return (skyline(notes, max((e for _, e, _ in notes)) + 1), 'ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('data/archives/asap-dataset'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seq-len', type=int, default=64)
    ap.add_argument('--hop', type=int, default=32)
    ap.add_argument('--max-per-file', type=int, default=64)
    ap.add_argument('--min-onsets', type=int, default=8)
    ap.add_argument('--max-mean-interval', type=float, default=4.0)
    ap.add_argument('--max-leap-rate', type=float, default=0.1)
    args = ap.parse_args()
    anns = json.loads((args.root / 'asap_annotations.json').read_text())
    rng = np.random.default_rng(0)
    windows, rows, sources = ([], [], [])
    status = Counter()
    on_beat = total = n_raw = 0
    for rel, ann in sorted(anns.items()):
        tokens, why = perf_tokens(args.root / rel, ann)
        status[why] += 1
        if tokens is None:
            continue
        idx = np.flatnonzero(tokens >= ONSET_OFFSET)
        on_beat += int((idx % SPB == 0).sum())
        total += len(idx)
        tokens = centre_octave(normalise_key(tokens))
        ws = window(tokens, args.seq_len, args.hop, min_onsets=args.min_onsets)
        n_raw += len(ws)
        ws = [w for w in ws if mean_interval(w) <= args.max_mean_interval and leap_rate(w) <= args.max_leap_rate]
        if not ws:
            status['no melodic windows'] += 1
            continue
        if len(ws) > args.max_per_file:
            ws = [ws[i] for i in rng.choice(len(ws), args.max_per_file, replace=False)]
        composer = rel.split('/')[0]
        meta = {
            'corpus': 'asap',
            'genre': 'classical',
            'artist': composer,
            'era': era_of(composer),
            'album': '/'.join(rel.split('/')[1:-1]),
            'setting': 'performance',
        }
        windows.extend(ws)
        rows.extend([meta] * len(ws))
        sources.extend([f'asap/{rel}'] * len(ws))
    x = np.stack(windows).astype(np.int16)
    labels = LabelSet({f: [r[f] for r in rows] for f in FIELDS})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, x=x, source_file=np.array(sources, dtype=object), **labels.to_npz_dict())
    print(f'wrote {args.out}: {len(x)} melodic windows (of {n_raw}) from {len(set(sources))} performances')
    print(f'files: {dict(status)}')
    print(f'onsets on a beat (top voice, before filtering): {on_beat / max(total, 1):.3f}')
    print('composers:', Counter((r['artist'] for r in rows)).most_common(8))


if __name__ == '__main__':
    main()
