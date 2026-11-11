#!/usr/bin/env python3
import argparse
import sys
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.data import window
from melodicvae.labels import LabelSet, era_of
from melodicvae.repr import ONSET_OFFSET, centre_octave, normalise_key, skyline

FIELDS = ['corpus', 'genre', 'artist', 'era', 'album', 'setting']
METERS = {'4/4', '2/4', '2/2'}
SPB = 4
PIANO = ('piano', 'klavier', 'pianoforte', 'harp', 'organ')


def vocal_tokens(path: str):
    warnings.filterwarnings('ignore')
    from music21 import converter

    try:
        score = converter.parse(path)
    except Exception:
        return ([], 'parse')
    voices = [p for p in score.parts if not any((k in (p.partName or '').lower() for k in PIANO))]
    if not voices:
        return ([], 'no voice')
    part = voices[0].stripTies()
    runs, cur, meter = ([], [], None)
    for m in part.getElementsByClass('Measure'):
        if m.timeSignature is not None:
            meter = m.timeSignature.ratioString
        if meter in METERS and (not (m.number == 0 and (not cur))):
            cur.append(m)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    if not runs:
        return ([], 'meter')
    out = []
    for run in runs:
        origin = float(run[0].offset)
        notes = []
        for m in run:
            for n in m.recurse().notes:
                if n.duration.isGrace or n.quarterLength <= 0:
                    continue
                off = float(m.offset) + float(n.getOffsetInHierarchy(m)) - origin
                pitch = max((p.midi for p in n.pitches))
                s = int(round(off * SPB))
                e = max(int(round((off + float(n.quarterLength)) * SPB)), s + 1)
                notes.append((s, e, pitch))
        if len(notes) >= 8:
            notes.sort()
            out.append(skyline(notes, max((e for _, e, _ in notes)) + 1))
    return (out, 'ok' if out else 'too short')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, default=Path('data/archives/lieder_mxl'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seq-len', type=int, default=64)
    ap.add_argument('--hop', type=int, default=32)
    ap.add_argument('--max-per-file', type=int, default=64)
    ap.add_argument('--min-onsets', type=int, default=8)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument(
        '--corpus',
        default='lieder',
        help='corpus label; the first non-keyboard part is the melody, so a string quartet yields its first violin',
    )
    args = ap.parse_args()
    files = sorted((str(p) for p in args.src.rglob('*.mxl')))
    rng = np.random.default_rng(0)
    windows, rows, sources = ([], [], [])
    status = Counter()
    on_beat = total = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for path, (runs, why) in zip(files, pool.map(vocal_tokens, files, chunksize=4)):
            status[why] += 1
            if not runs:
                continue
            ws = []
            for tokens in runs:
                idx = np.flatnonzero(tokens >= ONSET_OFFSET)
                on_beat += int((idx % SPB == 0).sum())
                total += len(idx)
                tokens = centre_octave(normalise_key(tokens))
                ws.extend(window(tokens, args.seq_len, args.hop, min_onsets=args.min_onsets))
            if not ws:
                status['no windows'] += 1
                continue
            if len(ws) > args.max_per_file:
                ws = [ws[i] for i in rng.choice(len(ws), args.max_per_file, replace=False)]
            rel = Path(path).relative_to(args.src)
            composer = rel.parts[0].replace('_', ' ')
            meta = {
                'corpus': args.corpus,
                'genre': 'classical',
                'artist': composer,
                'era': era_of(composer),
                'album': rel.parts[1] if len(rel.parts) > 2 else 'unknown',
                'setting': 'score',
            }
            windows.extend(ws)
            rows.extend([meta] * len(ws))
            sources.extend([f'lieder/{rel}'] * len(ws))
    x = np.stack(windows).astype(np.int16)
    labels = LabelSet({f: [r[f] for r in rows] for f in FIELDS})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, x=x, source_file=np.array(sources, dtype=object), **labels.to_npz_dict())
    print(f'wrote {args.out}: {len(x)} windows from {len(set(sources))} songs')
    print(f'files: {dict(status)}')
    print(f'onsets on a beat: {on_beat / max(total, 1):.3f}')
    print('top composers:', Counter((r['artist'] for r in rows)).most_common(6))


if __name__ == '__main__':
    main()
