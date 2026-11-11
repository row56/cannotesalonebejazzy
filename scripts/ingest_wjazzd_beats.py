#!/usr/bin/env python3
import argparse
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.data import window
from melodicvae.labels import LabelSet
from melodicvae.repr import ONSET_OFFSET, centre_octave, normalise_key, skyline

FIELDS = ['corpus', 'genre', 'artist', 'era', 'album', 'setting']
SPB = 4


def sub_step(tatum: int, division: int):
    pos = (tatum - 1) / max(division, 1)
    if division % 3 == 0 and abs(pos - 2 / 3) < 1e-06:
        pos = 0.5
    return int(round(pos * SPB))


def solo_tokens(con, melid: int):
    beats = con.execute('select onset, bar, beat from beats where melid=? order by onset', (melid,)).fetchall()
    notes = con.execute(
        'select onset, duration, pitch, bar, beat, tatum, division from melody where melid=? order by onset', (melid,)
    ).fetchall()
    if len(beats) < 8 or not notes:
        return (None, 0)
    bar0 = min((n[3] for n in notes))
    bt = np.array([b[0] for b in beats])
    bs = np.array([((b[1] - bar0) * 4 + (b[2] - 1)) * SPB for b in beats], dtype=float)
    ok = bs >= 0
    bt, bs = (bt[ok], bs[ok])
    if len(bt) < 8 or np.any(np.diff(bs) <= 0):
        return (None, 0)
    starts, ends, pitches = ([], [], [])
    for onset, dur, pitch, bar, beat, tatum, division in notes:
        if beat < 1 or beat > 4:
            continue
        s = ((bar - bar0) * 4 + (beat - 1)) * SPB + sub_step(tatum, division)
        e = int(round(np.interp(onset + dur, bt, bs)))
        starts.append(s)
        ends.append(max(e, s + 1))
        pitches.append(int(round(pitch)))
    if not starts:
        return (None, 0)
    order = np.argsort(starts, kind='stable')
    starts = np.array(starts)[order]
    ends = np.array(ends)[order]
    pitches = np.array(pitches)[order]
    ends[:-1] = np.minimum(ends[:-1], np.maximum(starts[1:], starts[:-1] + 1))
    n_steps = int(ends.max()) + 1
    trip = list(zip(starts.tolist(), ends.tolist(), pitches.tolist()))
    return (skyline(trip, n_steps), len(trip))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=Path, default=Path('data/archives/wjazzd.db'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seq-len', type=int, default=64)
    ap.add_argument('--hop', type=int, default=32, help='multiple of 16 keeps windows on bar lines')
    ap.add_argument('--max-per-file', type=int, default=64)
    ap.add_argument('--min-onsets', type=int, default=8)
    args = ap.parse_args()
    con = sqlite3.connect(args.db)
    solos = con.execute('select melid, performer, title, style, rhythmfeel, signature from solo_info').fetchall()
    rng = np.random.default_rng(0)
    windows, rows, sources = ([], [], [])
    skipped = Counter()
    on_beat, on_eighth, total = (0, 0, 0)
    for melid, performer, title, style, feel, sig in solos:
        if sig != '4/4':
            skipped['meter'] += 1
            continue
        tokens, _ = solo_tokens(con, melid)
        if tokens is None:
            skipped['no beats'] += 1
            continue
        idx = np.flatnonzero(tokens >= ONSET_OFFSET)
        on_beat += int((idx % SPB == 0).sum())
        on_eighth += int((idx % 2 == 0).sum())
        total += len(idx)
        tokens = centre_octave(normalise_key(tokens))
        ws = window(tokens, args.seq_len, args.hop, min_onsets=args.min_onsets)
        if not ws:
            skipped['no windows'] += 1
            continue
        if len(ws) > args.max_per_file:
            ws = [ws[i] for i in rng.choice(len(ws), args.max_per_file, replace=False)]
        meta = {
            'corpus': 'wjazzd',
            'genre': 'jazz',
            'artist': performer,
            'era': (style or 'unknown').lower(),
            'album': title,
            'setting': (feel or 'unknown').lower(),
        }
        windows.extend(ws)
        rows.extend([meta] * len(ws))
        sources.extend([f'wjazzd/{melid}_{performer}_{title}'] * len(ws))
    x = np.stack(windows).astype(np.int16)
    labels = LabelSet({f: [r[f] for r in rows] for f in FIELDS})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, x=x, source_file=np.array(sources, dtype=object), **labels.to_npz_dict())
    print(f'wrote {args.out}: {len(x)} windows from {len(set(sources))} solos, skipped {dict(skipped)}')
    print(
        f'onsets on a beat: {on_beat / total:.3f}, on an eighth: {on_eighth / total:.3f} (random placement on a 16th grid: 0.25 / 0.50)'
    )
    per_artist = Counter((r['artist'] for r in rows))
    print('top artists:', per_artist.most_common(6))


if __name__ == '__main__':
    main()
