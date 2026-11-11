#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.data import window
from melodicvae.labels import LabelSet
from melodicvae.repr import ONSET_OFFSET, centre_octave, normalise_key, skyline

FIELDS = ['corpus', 'genre', 'artist', 'era', 'album', 'setting']
METERS = {(4, 4), (2, 4), (2, 2)}
SPB = 4


def is_accompaniment(program: int) -> bool:
    keyboard = 0 <= program <= 7 or 16 <= program <= 23
    bass = 32 <= program <= 39
    guitar = 24 <= program <= 31
    pad = 88 <= program <= 103
    return keyboard or bass or guitar or pad


def monophony(notes) -> float:
    ns = sorted(notes, key=lambda n: n.start)
    if len(ns) < 2:
        return 0.0
    return float(np.mean([a.end <= b.start + 0.001 for a, b in zip(ns, ns[1:])]))


def melody_track(pm):
    best, score = (None, -1.0)
    for inst in pm.instruments:
        if inst.is_drum or is_accompaniment(inst.program) or len(inst.notes) < 16:
            continue
        mono = monophony(inst.notes)
        height = np.median([n.pitch for n in inst.notes])
        s = mono + 0.01 * height
        if mono >= 0.8 and s > score:
            best, score = (inst, s)
    return best


def score_tokens(path: Path):
    import pretty_midi

    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return ([], 'parse')
    inst = melody_track(pm)
    if inst is None:
        return ([], 'no melody track')
    tq = pm.resolution / SPB
    ts = sorted(pm.time_signature_changes, key=lambda t: t.time)
    if not ts:
        return ([], 'meter')
    bounds = [pm.time_to_tick(t.time) for t in ts] + [10**12]
    runs = [(bounds[i], bounds[i + 1]) for i, t in enumerate(ts) if (t.numerator, t.denominator) in METERS]
    if not runs:
        return ([], 'meter')
    out = []
    for lo, hi in runs:
        notes = []
        for n in inst.notes:
            t0, t1 = (pm.time_to_tick(n.start), pm.time_to_tick(n.end))
            if t0 < lo or t0 >= hi:
                continue
            s = int(round((t0 - lo) / tq))
            e = max(int(round((min(t1, hi) - lo) / tq)), s + 1)
            notes.append((s, e, n.pitch))
        if len(notes) >= 8:
            notes.sort()
            out.append(skyline(notes, max((e for _, e, _ in notes)) + 1))
    return (out, 'ok' if out else 'too short')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('data/archives/pdmx_jazz'))
    ap.add_argument('--meta', type=Path, default=Path('data/archives/pdmx_jazz.json'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seq-len', type=int, default=64)
    ap.add_argument('--hop', type=int, default=32)
    ap.add_argument('--max-per-file', type=int, default=64)
    ap.add_argument('--min-onsets', type=int, default=8)
    args = ap.parse_args()
    meta = json.loads(args.meta.read_text())
    rng = np.random.default_rng(0)
    windows, rows, sources = ([], [], [])
    status = Counter()
    on_beat = total = 0
    for m in meta:
        path = args.root / m['mid']
        if not path.exists():
            status['missing'] += 1
            continue
        runs, why = score_tokens(path)
        status[why] += 1
        ws = []
        for tokens in runs:
            idx = np.flatnonzero(tokens >= ONSET_OFFSET)
            on_beat += int((idx % SPB == 0).sum())
            total += len(idx)
            tokens = centre_octave(normalise_key(tokens))
            ws.extend(window(tokens, args.seq_len, args.hop, min_onsets=args.min_onsets))
        if not ws:
            if runs:
                status['no windows'] += 1
            continue
        if len(ws) > args.max_per_file:
            ws = [ws[i] for i in rng.choice(len(ws), args.max_per_file, replace=False)]
        song = ' '.join((m['song'] or 'unknown').lower().split())
        info = {
            'corpus': 'pdmx_jazz',
            'genre': 'jazz',
            'artist': m['artist'] or 'unknown',
            'era': 'unknown',
            'album': song,
            'setting': 'sheet',
        }
        windows.extend(ws)
        rows.extend([info] * len(ws))
        sources.extend([f"pdmx_jazz/{m['mid']}"] * len(ws))
    x = np.stack(windows).astype(np.int16)
    labels = LabelSet({f: [r[f] for r in rows] for f in FIELDS})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, x=x, source_file=np.array(sources, dtype=object), **labels.to_npz_dict())
    print(
        f"wrote {args.out}: {len(x)} windows from {len(set(sources))} scores ({len(set((r['album'] for r in rows)))} distinct titles)"
    )
    print(f'files: {dict(status)}')
    print(f'onsets on a beat: {on_beat / max(total, 1):.3f}')
    print('top titles:', Counter((r['album'] for r in rows)).most_common(8))


if __name__ == '__main__':
    main()
