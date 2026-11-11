#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path, help='directory of .mid files, searched recursively')
    ap.add_argument('--from-bpm', type=float, required=True)
    ap.add_argument('--to-bpm', type=float, required=True)
    ap.add_argument('--out', type=Path, default=None, help='default: rewrite in place')
    args = ap.parse_args()
    import pretty_midi

    scale = args.from_bpm / args.to_bpm
    midis = sorted(args.src.rglob('*.mid'))
    if not midis:
        raise SystemExit(f'no .mid files under {args.src}')
    print(f'{len(midis)} files, scaling note times by {scale:.4f} ({args.from_bpm:g} -> {args.to_bpm:g} bpm)')
    for mid in midis:
        pm = pretty_midi.PrettyMIDI(str(mid))
        for inst in pm.instruments:
            for note in inst.notes:
                note.start *= scale
                note.end *= scale
        dest = mid if args.out is None else args.out / mid.relative_to(args.src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pm.write(str(dest))
    print(f'wrote {len(midis)} files')


if __name__ == '__main__':
    main()
