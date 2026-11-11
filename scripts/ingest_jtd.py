#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.attributes import leap_rate, mean_interval
from melodicvae.data import window
from melodicvae.labels import LabelSet
from melodicvae.repr import ONSET_OFFSET, centre_octave, normalise_key, skyline

FIELDS = ['corpus', 'genre', 'artist', 'era', 'album', 'setting']
SPB = 4


def track_tokens(folder: Path):
    import pretty_midi

    meta = json.loads((folder / 'metadata.json').read_text().replace('NaN', 'null'))
    if meta.get('time_signature') != 4:
        return (None, 'meter', meta)
    rows = list(csv.DictReader((folder / 'beats.csv').open()))
    beats, bar_pos = ([], [])
    for r in rows:
        try:
            beats.append(float(r['beats']))
            bar_pos.append(float(r['metre_auto']))
        except (TypeError, ValueError):
            continue
    beats, bar_pos = (np.array(beats), np.array(bar_pos))
    if len(beats) < 16 or not (bar_pos == 1).any():
        return (None, 'beats', meta)
    first_down = int(np.flatnonzero(bar_pos == 1)[0])
    beat_idx = np.arange(len(beats)) - first_down
    try:
        pm = pretty_midi.PrettyMIDI(str(folder / 'piano_midi.mid'))
    except Exception:
        return (None, 'parse', meta)
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
        return (None, 'too short', meta)
    return (skyline(notes, max((e for _, e, _ in notes)) + 1), 'ok', meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('data/archives/jazz-trio-database-v02'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seq-len', type=int, default=64)
    ap.add_argument('--hop', type=int, default=32)
    ap.add_argument('--max-per-file', type=int, default=64)
    ap.add_argument('--min-onsets', type=int, default=8)
    ap.add_argument('--max-mean-interval', type=float, default=4.0)
    ap.add_argument('--max-leap-rate', type=float, default=0.1)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    windows, rows, sources = ([], [], [])
    status = Counter()
    on_beat = on_eighth = total = n_raw = 0
    for folder in sorted((p for p in args.root.iterdir() if p.is_dir())):
        tokens, why, meta = track_tokens(folder)
        status[why] += 1
        if tokens is None:
            continue
        idx = np.flatnonzero(tokens >= ONSET_OFFSET)
        on_beat += int((idx % SPB == 0).sum())
        on_eighth += int((idx % 2 == 0).sum())
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
        info = {
            'corpus': 'jtd',
            'genre': 'jazz',
            'artist': meta.get('pianist') or 'unknown',
            'era': str(meta.get('recording_year') or 'unknown'),
            'album': meta.get('album_name') or 'unknown',
            'setting': 'performance',
        }
        windows.extend(ws)
        rows.extend([info] * len(ws))
        sources.extend([f'jtd/{folder.name}'] * len(ws))
    x = np.stack(windows).astype(np.int16)
    labels = LabelSet({f: [r[f] for r in rows] for f in FIELDS})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, x=x, source_file=np.array(sources, dtype=object), **labels.to_npz_dict())
    print(f'wrote {args.out}: {len(x)} melodic windows (of {n_raw}) from {len(set(sources))} tracks')
    print(f'files: {dict(status)}')
    print(
        f'top voice before filtering: on a beat {on_beat / max(total, 1):.3f}, on an eighth {on_eighth / max(total, 1):.3f}'
    )
    print('pianists:', Counter((r['artist'] for r in rows)).most_common(8))


if __name__ == '__main__':
    main()
